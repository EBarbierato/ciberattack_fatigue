"""
QuCS_2026 — Alarm Poisoning as Cybersecurity Dynamics (CD)
State representation for the Markovian Agent Model (MAM) / CTMC simulation.

This module defines:
- Global CTMC state (attacker, alarm system, defender posture, alarm-pressure Z)
- Employee population occupancy tensor n_{T,F}
- Convenience accessors and integrity checks

Design goals:
- Keep the state strictly discrete (CTMC-friendly).
- Avoid hidden approximations: all updates are integer-valued jumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import math

from alarm_poisoning_cd.core.parameters import (
    ALARM_TYPES,
    ALARM_SYS_STATES,
    ATTACKER_STATES,
    DEFENDER_STATES,
    ModelParams,
)


TrustLevel = int   # T in {0,1,2}
FatigueLevel = int # F in {0..F_max}


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


@dataclass
class CTMCState:
    """
    Full Markov state S(t).

    attacker:  A0,A1,A2,A3
    alarm_sys: S0,S1,S2
    defender:  D0,D1,D2
    Z:         alarm-pressure bin in {0..Z_max}

    n[T][F] is represented by a flat dict mapping (T,F) -> count, with:
      T in {0,1,2}, F in {0..F_max}
    """
    t: float  # current time in hours (continuous)

    attacker: str
    alarm_sys: str
    defender: str
    Z: int

    n: Dict[Tuple[TrustLevel, FatigueLevel], int]

    def copy(self) -> "CTMCState":
        return CTMCState(
            t=self.t,
            attacker=self.attacker,
            alarm_sys=self.alarm_sys,
            defender=self.defender,
            Z=self.Z,
            n=dict(self.n),
        )

    # -------------------------
    # Integrity and dimensions
    # -------------------------

    def total_employees(self) -> int:
        return sum(self.n.values())

    def validate(self, p: ModelParams) -> None:
        if self.attacker not in ATTACKER_STATES:
            raise ValueError(f"Invalid attacker state: {self.attacker}")
        if self.alarm_sys not in ALARM_SYS_STATES:
            raise ValueError(f"Invalid alarm system state: {self.alarm_sys}")
        if self.defender not in DEFENDER_STATES:
            raise ValueError(f"Invalid defender state: {self.defender}")
        if not (0 <= self.Z <= p.defender.Z_max):
            raise ValueError(f"Z out of bounds: {self.Z}")

        # Check occupancy domain and nonnegativity
        for (T, F), c in self.n.items():
            if T not in (0, 1, 2):
                raise ValueError(f"Invalid trust level: {T}")
            if not (0 <= F <= p.trust.F_max):
                raise ValueError(f"Invalid fatigue level: {F}")
            if c < 0:
                raise ValueError(f"Negative count for (T,F)=({T},{F}): {c}")

        # Ensure all compartments exist (fill missing as zero)
        for T in (0, 1, 2):
            for F in range(p.trust.F_max + 1):
                if (T, F) not in self.n:
                    raise ValueError(f"Missing compartment (T,F)=({T},{F}) in state.n")

        tot = self.total_employees()
        if tot != p.N_employees:
            raise ValueError(f"Employee count mismatch: state has {tot}, params has {p.N_employees}")

        if self.t < 0 or not math.isfinite(self.t):
            raise ValueError(f"Invalid time t: {self.t}")

    # -------------------------
    # Convenience summaries
    # -------------------------

    def trust_marginal(self, p: ModelParams) -> Dict[int, int]:
        out = {0: 0, 1: 0, 2: 0}
        for (T, F), c in self.n.items():
            out[T] += c
        return out

    def fatigue_marginal(self, p: ModelParams) -> Dict[int, int]:
        out = {f: 0 for f in range(p.trust.F_max + 1)}
        for (T, F), c in self.n.items():
            out[F] += c
        return out

    def fraction_low_trust(self, p: ModelParams) -> float:
        lt = sum(c for (T, _F), c in self.n.items() if T == 0)
        return lt / p.N_employees

    def mean_fatigue(self, p: ModelParams) -> float:
        num = sum(F * c for (_T, F), c in self.n.items())
        return num / p.N_employees

    def fraction_high_fatigue(self, p: ModelParams, threshold: int = 4) -> float:
        hf = sum(c for (_T, F), c in self.n.items() if F >= threshold)
        return hf / p.N_employees


def initial_state(p: ModelParams) -> CTMCState:
    """
    Default initial condition:
    - attacker outside (A0)
    - alarm system intact (S0)
    - defender baseline (D0)
    - alarm pressure Z = 0
    - employees: high trust (T=2), zero fatigue (F=0)

    This matches the narrative: normal building operations before compromise.
    """
    n: Dict[Tuple[TrustLevel, FatigueLevel], int] = {}
    for T in (0, 1, 2):
        for F in range(p.trust.F_max + 1):
            n[(T, F)] = 0
    n[(2, 0)] = p.N_employees

    st = CTMCState(
        t=0.0,
        attacker="A0",
        alarm_sys="S0",
        defender="D0",
        Z=0,
        n=n,
    )
    st.validate(p)
    return st


# -------------------------
# Helper update primitives
# -------------------------

def increment_alarm_pressure(Z: int, p: ModelParams, inc: int = 1) -> int:
    return _clamp_int(Z + inc, 0, p.defender.Z_max)


def decrement_alarm_pressure(Z: int, p: ModelParams, dec: int = 1) -> int:
    return _clamp_int(Z - dec, 0, p.defender.Z_max)


def fatigue_recovery_event(state: CTMCState, p: ModelParams) -> None:
    """
    Apply a single fatigue recovery elementary event to the population.

    Event definition (CTMC elementary jump):
      pick a fatigue level f>=1 with probability proportional to f * (sum_T n[T,f])
      then move one random individual from that f to f-1, preserving trust level T.

    This event corresponds to the aggregate rates:
      for each (T,f): (T,f)->(T,f-1) at rate phi*f*n[T,f].

    IMPORTANT: This function applies exactly one jump (one individual),
    and must be called by the SSA kernel when that event channel fires.
    """
    # Build weights per compartment (T,f) with f>=1
    weights: List[Tuple[Tuple[int, int], float]] = []
    total = 0.0
    for (T, F), c in state.n.items():
        if F >= 1 and c > 0:
            w = (p.trust.phi * F) * c
            weights.append(((T, F), w))
            total += w
    if total <= 0.0:
        return  # no-op (should not be selected if SSA is correct)

    # Sample a compartment
    import random
    u = random.random() * total
    acc = 0.0
    chosen = None
    for key, w in weights:
        acc += w
        if u <= acc:
            chosen = key
            break
    if chosen is None:
        chosen = weights[-1][0]

    T, F = chosen
    # Move one person (T,F)->(T,F-1)
    state.n[(T, F)] -= 1
    state.n[(T, F - 1)] += 1


def alarm_pressure_decay_event(state: CTMCState, p: ModelParams) -> None:
    """
    Apply one elementary decay event: Z -> Z-1.

    The SSA will schedule this with total rate omega * Z.
    """
    if state.Z > 0:
        state.Z -= 1
