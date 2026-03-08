"""
QuCS_2026 — Alarm Poisoning as Cybersecurity Dynamics (CD)
Exact CTMC simulation via Gillespie / Stochastic Simulation Algorithm (SSA).

This module implements an event-driven simulation of the full Markovian Agent Model (MAM)
defined by the transition intensities in ModelParams.

Key design constraints (per your requirements):
- Continuous-time stochastic model (CTMC), no discrete-time approximation.
- Exact SSA sampling: exponential inter-event times, categorical event selection by rates.
- Population updates on alarm events use exact Binomial sampling (no rounding).
- Fatigue recovery and alarm-pressure decay are simulated as elementary CTMC jumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Callable
import math
import random

from alarm_poisoning_cd.core.parameters import (
    ALARM_TYPES,
    DEFENDER_STATES,
    ModelParams,
)
from alarm_poisoning_cd.core.state import (
    CTMCState,
    increment_alarm_pressure,
    alarm_pressure_decay_event,
    fatigue_recovery_event,
)


# -----------------------------
# Random helpers (exact SSA)
# -----------------------------

def _exp_wait(total_rate: float) -> float:
    """Sample exponential waiting time with parameter total_rate."""
    if total_rate <= 0.0:
        return math.inf
    u = random.random()
    u = max(u, 1e-15)
    return -math.log(u) / total_rate


def _pick_event(weighted: List[Tuple[str, float]]) -> str:
    """Pick one event key proportional to its positive weight."""
    total = sum(w for _, w in weighted if w > 0.0)
    if total <= 0.0:
        return "NONE"
    u = random.random() * total
    acc = 0.0
    for k, w in weighted:
        if w <= 0.0:
            continue
        acc += w
        if u <= acc:
            return k
    # numerical fallback
    for k, w in reversed(weighted):
        if w > 0:
            return k
    return "NONE"


def _binomial(n: int, p: float) -> int:
    """Exact Binomial sampling via Python's random module."""
    if n <= 0:
        return 0
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    # random.binomial not in stdlib; do an exact loop for moderate n,
    # and a BTPE-style algorithm would be overkill here.
    # N is typically <= a few hundred, so loop is exact and fine.
    x = 0
    for _ in range(n):
        if random.random() < p:
            x += 1
    return x


# -----------------------------
# Rate functions
# -----------------------------

def attacker_progression_rates(st: CTMCState, p: ModelParams) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    if st.attacker == "A0":
        rates["ATTACK_A0_A1"] = p.attacker.lambda_01
    elif st.attacker == "A1":
        rates["ATTACK_A1_A2"] = p.attacker.lambda_12
    elif st.attacker == "A2":
        rates["ATTACK_A2_A3"] = p.attacker.lambda_23
    return rates


def system_compromise_rate(st: CTMCState, p: ModelParams) -> float:
    # (S0, A1) -> S1 at kappa_comp
    if st.alarm_sys == "S0" and st.attacker in ("A1", "A2", "A3"):
        # We allow compromise once attacker is at least A1 (foothold),
        # consistent with "alarm-channel breach" being feasible with foothold.
        return p.attacker.kappa_comp
    return 0.0


def mitigation_rate(st: CTMCState, p: ModelParams) -> float:
    # S1 -> S2 at kappa_mit[D]
    if st.alarm_sys == "S1":
        return p.defender.kappa_mit[st.defender]
    return 0.0


def defender_posture_rates(st: CTMCState, p: ModelParams) -> Dict[str, float]:
    # D0->D1 and D1->D2 depend on Z; D2->D0 depends on Z threshold
    Z = st.Z
    rates: Dict[str, float] = {}
    if st.defender == "D0":
        rates["DEF_D0_D1"] = p.defender.gamma01_base + p.defender.gamma01_slope * Z
    elif st.defender == "D1":
        rates["DEF_D1_D2"] = p.defender.gamma12_base + p.defender.gamma12_slope * Z
    elif st.defender == "D2":
        rates["DEF_D2_D0"] = p.defender.gamma20 if Z <= p.defender.Z_relax_threshold else 0.0
    return rates


def alarm_pressure_decay_rate(st: CTMCState, p: ModelParams) -> float:
    # Z -> Z-1 at rate omega * Z
    return p.defender.omega * st.Z


def fatigue_recovery_total_rate(st: CTMCState, p: ModelParams) -> float:
    # Total rate sum_{T,f>=1} phi * f * n[T,f]
    tot = 0.0
    for (T, F), c in st.n.items():
        if F >= 1 and c > 0:
            tot += (p.trust.phi * F) * c
    return tot


def true_alarm_rates(st: CTMCState, p: ModelParams) -> Dict[str, float]:
    return {f"TRUE_{j}": p.hazards.lambda_true[j] for j in ALARM_TYPES}


def fake_alarm_rates(st: CTMCState, p: ModelParams) -> Dict[str, float]:
    # Only when compromised and attacker advanced
    if st.alarm_sys != "S1":
        return {}
    if st.attacker not in ("A2", "A3"):
        return {}
    base = p.fake.Lambda_fake
    return {f"FAKE_{j}": base * p.fake.pi[j] for j in ALARM_TYPES}


# -----------------------------
# Trust and fatigue update kernels (exact binomial moves)
# -----------------------------

def _g_f(f: int, slope: float) -> float:
    return 1.0 + slope * f


def _g_f_tilde(f: int, slope: float) -> float:
    return 1.0 / (1.0 + slope * f)


def trust_down_prob(j: str, T: int, F: int, D: str, p: ModelParams) -> float:
    if T <= 0:
        return 0.0
    base = p.trust.alpha_down[j]
    gf = _g_f(F, p.trust.g_f_slope)
    hD = p.trust.hD_down[D]
    qT = p.trust.qT_down[T]
    x = base * gf * hD * qT
    return 1.0 if x >= 1.0 else x


def trust_up_prob(j: str, T: int, F: int, D: str, p: ModelParams) -> float:
    if T >= 2:
        return 0.0
    base = p.trust.alpha_up[j]
    gf = _g_f_tilde(F, p.trust.g_f_tilde_slope)
    hD = p.trust.hD_up[D]
    qT = p.trust.qT_up[T]
    x = base * gf * hD * qT
    return 1.0 if x >= 1.0 else x


def apply_alarm_event(st: CTMCState, p: ModelParams, j: str, is_true: bool) -> None:
    """
    Apply the population update induced by an alarm event of type j.

    This function implements:
    - Alarm-pressure increment: Z <- min(Z+1, Z_max)
    - Fatigue exposure: for each compartment (T,F), move Binomial(n[T,F], rho_j) to (T, min(F+1,F_max))
    - Trust change:
        if fake: for each (T,F) with T>0, move Binomial(n[T,F], p_down) to (T-1,F)
        if true: for each (T,F) with T<2, move Binomial(n[T,F], p_up) to (T+1,F)

    IMPORTANT:
    - All moves are performed with exact binomial sampling.
    - Moves are sequenced to avoid double-counting:
        1) fatigue moves (producing an intermediate n')
        2) trust moves on the post-fatigue occupancy n'
      This is a modeling choice and should be documented in the paper.
    """
    if j not in ALARM_TYPES:
        raise ValueError(f"Unknown alarm type: {j}")

    # 1) increment alarm pressure
    st.Z = increment_alarm_pressure(st.Z, p, inc=1)

    rho = p.alarm_attr[j].exposure
    Fmax = p.trust.F_max

    # 2) fatigue exposure moves, computed on a snapshot to ensure consistency
    n_after = dict(st.n)
    for T in (0, 1, 2):
        # loop F descending to avoid immediate re-processing
        for F in range(Fmax, -1, -1):
            c = n_after[(T, F)]
            if c <= 0:
                continue
            exposed = _binomial(c, rho)
            if exposed <= 0:
                continue
            n_after[(T, F)] -= exposed
            F2 = F if F >= Fmax else min(F + 1, Fmax)
            n_after[(T, F2)] += exposed

    # 3) trust updates on n_after
    n_final = dict(n_after)
    D = st.defender
    if not is_true:
        # fake: downshift trust
        for F in range(Fmax + 1):
            for T in (2, 1):  # only downshift from 2->1 and 1->0
                c = n_final[(T, F)]
                if c <= 0:
                    continue
                pd = trust_down_prob(j, T, F, D, p)
                moved = _binomial(c, pd)
                if moved <= 0:
                    continue
                n_final[(T, F)] -= moved
                n_final[(T - 1, F)] += moved
    else:
        # true: upshift trust
        for F in range(Fmax + 1):
            for T in (0, 1):  # 0->1 and 1->2
                c = n_final[(T, F)]
                if c <= 0:
                    continue
                pu = trust_up_prob(j, T, F, D, p)
                moved = _binomial(c, pu)
                if moved <= 0:
                    continue
                n_final[(T, F)] -= moved
                n_final[(T + 1, F)] += moved

    st.n = n_final


# -----------------------------
# SSA kernel
# -----------------------------

@dataclass
class SSAEvent:
    """Event selected by SSA."""
    name: str
    rate: float


def build_event_rates(st: CTMCState, p: ModelParams) -> List[Tuple[str, float]]:
    """
    Build the list of all event channels with their current intensities.
    """
    rates: List[Tuple[str, float]] = []

    # attacker progression
    for k, r in attacker_progression_rates(st, p).items():
        rates.append((k, r))

    # system compromise and mitigation
    rates.append(("SYS_COMPROMISE_S0_S1", system_compromise_rate(st, p)))
    rates.append(("SYS_MITIGATE_S1_S2", mitigation_rate(st, p)))

    # defender posture evolution
    for k, r in defender_posture_rates(st, p).items():
        rates.append((k, r))

    # alarm-pressure decay
    rates.append(("PRESSURE_Z_DECAY", alarm_pressure_decay_rate(st, p)))

    # fatigue recovery (population elementary)
    rates.append(("FATIGUE_RECOVER_ONE", fatigue_recovery_total_rate(st, p)))

    # true alarms
    for k, r in true_alarm_rates(st, p).items():
        rates.append((k, r))

    # fake alarms
    for k, r in fake_alarm_rates(st, p).items():
        rates.append((k, r))

    # filter negatives just in case
    rates = [(k, r) for (k, r) in rates if r is not None and r >= 0.0]
    return rates


def apply_event(st: CTMCState, p: ModelParams, event: str) -> None:
    """
    Apply a single event channel jump to the state.
    """
    if event == "ATTACK_A0_A1":
        st.attacker = "A1"
        return
    if event == "ATTACK_A1_A2":
        st.attacker = "A2"
        return
    if event == "ATTACK_A2_A3":
        st.attacker = "A3"
        return

    if event == "SYS_COMPROMISE_S0_S1":
        st.alarm_sys = "S1"
        return
    if event == "SYS_MITIGATE_S1_S2":
        st.alarm_sys = "S2"
        return

    if event == "DEF_D0_D1":
        st.defender = "D1"
        return
    if event == "DEF_D1_D2":
        st.defender = "D2"
        return
    if event == "DEF_D2_D0":
        st.defender = "D0"
        return

    if event == "PRESSURE_Z_DECAY":
        alarm_pressure_decay_event(st, p)
        return

    if event == "FATIGUE_RECOVER_ONE":
        fatigue_recovery_event(st, p)
        return

    if event.startswith("TRUE_"):
        j = event.replace("TRUE_", "")
        apply_alarm_event(st, p, j, is_true=True)
        return

    if event.startswith("FAKE_"):
        j = event.replace("FAKE_", "")
        apply_alarm_event(st, p, j, is_true=False)
        return

    raise ValueError(f"Unknown event: {event}")


def run_ssa(
    p: ModelParams,
    st: CTMCState,
    t_end: float,
    *,
    max_events: int = 2_000_000,
    rng_seed: Optional[int] = None,
    on_step: Optional[Callable[[CTMCState, str, float], None]] = None,
) -> CTMCState:
    """
    Run the CTMC from current state st until time t_end (hours) using SSA.

    Parameters
    ----------
    p : ModelParams
        Model parameters (validated).
    st : CTMCState
        Initial state (will be mutated in-place).
    t_end : float
        Simulation horizon in hours.
    max_events : int
        Safety cap on number of jumps.
    rng_seed : Optional[int]
        Seed for reproducibility.
    on_step : Optional callback
        Called after each jump: on_step(state, event_name, event_rate).

    Returns
    -------
    CTMCState
        Final state (same object st).
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    p.validate()
    st.validate(p)

    n_events = 0
    while st.t < t_end and n_events < max_events:
        rate_list = build_event_rates(st, p)
        total_rate = sum(r for _, r in rate_list if r > 0.0)
        if total_rate <= 0.0:
            st.t = t_end
            break

        dt = _exp_wait(total_rate)
        if st.t + dt > t_end:
            st.t = t_end
            break

        # select event
        ev = _pick_event(rate_list)
        if ev == "NONE":
            st.t = t_end
            break

        # get the selected event rate (for logging only)
        ev_rate = 0.0
        for k, r in rate_list:
            if k == ev:
                ev_rate = r
                break

        st.t += dt
        apply_event(st, p, ev)

        if on_step is not None:
            on_step(st, ev, ev_rate)

        n_events += 1

    st.validate(p)
    return st
