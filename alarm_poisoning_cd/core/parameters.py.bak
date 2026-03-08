"""
QuCS_2026 — Alarm Poisoning as Cybersecurity Dynamics (CD)
Core parameter specification for the Markovian Agent Model (MAM) / CTMC simulation.

This module provides:
- Strongly-typed parameter dataclasses
- Alarm types and scenario strategies (attacker mix π)
- Default calibrated values (paper-ready)
- Validation logic (units, ranges, invariants)

All rates are per hour (h^-1). Times are hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple
import math


# -----------------------------
# Enumerations as string literals
# -----------------------------

AlarmType = str  # keep simple; validated via ALARM_TYPES keys

ALARM_TYPES: Tuple[AlarmType, ...] = (
    "fire",
    "intruder",
    "server_temp",
    "usb_malware",
    "elevator",
)

ATTACKER_STATES: Tuple[str, ...] = ("A0", "A1", "A2", "A3")
ALARM_SYS_STATES: Tuple[str, ...] = ("S0", "S1", "S2")
DEFENDER_STATES: Tuple[str, ...] = ("D0", "D1", "D2")


# -----------------------------
# Dataclasses for parameters
# -----------------------------

@dataclass(frozen=True)
class AlarmAttributes:
    """Intrinsic attributes of alarm type j."""
    severity: float        # s_j (dimensionless)
    nuisance: float        # c_j (dimensionless)
    exposure: float        # rho_j in (0,1], fraction of employees meaningfully impacted


@dataclass(frozen=True)
class TrustDynamics:
    """
    Trust update structure used on alarm events.

    Trust levels: T in {0,1,2} (low, medium, high)
    Fatigue levels: F in {0..F_max}

    p_down (fake):  alpha_down[j] * g_f(F) * hD_down(D) * qT_down(T)
    p_up (true):    alpha_up[j]   * g_f_tilde(F) * hD_up(D)   * qT_up(T)
    """
    F_max: int

    # fatigue recovery: f -> f-1 at rate phi * f
    phi: float

    # base per-event trust erosion/recovery by type
    alpha_down: Dict[AlarmType, float]
    alpha_up: Dict[AlarmType, float]

    # defender posture multipliers
    hD_down: Dict[str, float]
    hD_up: Dict[str, float]

    # trust-level multipliers (indexed by trust level integer)
    qT_down: Dict[int, float]
    qT_up: Dict[int, float]

    # fatigue shaping coefficients for the amplifiers
    g_f_slope: float               # g_f(f) = 1 + g_f_slope * f
    g_f_tilde_slope: float         # g_tilde(f) = 1 / (1 + g_f_tilde_slope * f)


@dataclass(frozen=True)
class HazardRates:
    """True hazard (real events) rates per hour."""
    lambda_true: Dict[AlarmType, float]


@dataclass(frozen=True)
class AttackerProgression:
    """
    Attacker progression rates per hour.

    A0 -> A1 at lambda_01
    A1 -> A2 at lambda_12
    A2 -> A3 at lambda_23
    Additionally: alarm system compromise requires A1 and S0, at kappa_comp
    """
    lambda_01: float
    lambda_12: float
    lambda_23: float
    kappa_comp: float  # (S0, A1) -> S1


@dataclass(frozen=True)
class DefenderAgility:
    """
    Defender posture dynamics with an alarm-pressure variable Z in {0..Z_max}.

    - Each alarm increments Z by +1 (capped).
    - Z decays via Z -> Z-1 at rate omega * Z (CTMC linear death).
    - Defender transitions depend on Z:
        D0 -> D1 rate gamma_01(Z) = gamma01_base + gamma01_slope * Z
        D1 -> D2 rate gamma_12(Z) = gamma12_base + gamma12_slope * Z
      Relaxation:
        D2 -> D0 rate gamma_20(Z) = gamma20 if Z <= Z_relax_threshold else 0
    """
    Z_max: int
    omega: float

    gamma01_base: float
    gamma01_slope: float

    gamma12_base: float
    gamma12_slope: float

    gamma20: float
    Z_relax_threshold: int

    # Alarm-system mitigation S1 -> S2 depends on defender posture
    kappa_mit: Dict[str, float]


@dataclass(frozen=True)
class FakeAlarmInjection:
    """
    Fake alarm injection model once system is compromised.

    Overall intensity Lambda_fake (h^-1) and a strategy mix π over alarm types.
    """
    Lambda_fake: float
    pi: Dict[AlarmType, float]  # must sum to 1


@dataclass(frozen=True)
class ResponseTimeModel:
    """
    Individual response-time hazard model for TRUE alarms of type j.

    mu_j(T,F,D,j) = mu0[j] * exp(beta_T*T - beta_F*F + beta_S*s_j + beta_D(D))

    Timely evacuation (for selected alarm types) is defined by:
    - deadline Delta hours (e.g., 0.05 = 3 minutes)
    - quorum m responders needed to initiate evacuation.
    """
    mu0: Dict[AlarmType, float]

    beta_T: float
    beta_F: float
    beta_S: float
    beta_D: Dict[str, float]

    Delta: float
    quorum_m: int


@dataclass(frozen=True)
class ModelParams:
    """
    Top-level parameters for the full CTMC/MAM model.
    """
    # Population
    N_employees: int

    # Alarm types
    alarm_attr: Dict[AlarmType, AlarmAttributes]

    # Processes
    hazards: HazardRates
    attacker: AttackerProgression
    defender: DefenderAgility
    trust: TrustDynamics
    fake: FakeAlarmInjection
    response: ResponseTimeModel

    # Which alarm types count as "evacuation-triggering" for the key situation metric
    evacuation_alarm_types: Tuple[AlarmType, ...] = ("fire", "intruder")

    def validate(self) -> None:
        """
        Raise ValueError if any parameter is out of admissible range or inconsistent.
        """
        # Population
        if not isinstance(self.N_employees, int) or self.N_employees <= 0:
            raise ValueError("N_employees must be a positive integer.")

        # Alarm types consistency
        for j in ALARM_TYPES:
            if j not in self.alarm_attr:
                raise ValueError(f"alarm_attr missing type '{j}'")
            if j not in self.hazards.lambda_true:
                raise ValueError(f"hazards.lambda_true missing type '{j}'")
            if j not in self.trust.alpha_down or j not in self.trust.alpha_up:
                raise ValueError(f"trust alpha missing type '{j}'")
            if j not in self.response.mu0:
                raise ValueError(f"response.mu0 missing type '{j}'")
            if j not in self.fake.pi:
                raise ValueError(f"fake.pi missing type '{j}'")

        # Attribute ranges
        for j, a in self.alarm_attr.items():
            if a.exposure <= 0.0 or a.exposure > 1.0:
                raise ValueError(f"Exposure rho must be in (0,1]; got {a.exposure} for {j}")
            if a.severity <= 0:
                raise ValueError(f"Severity must be >0; got {a.severity} for {j}")
            if a.nuisance < 0:
                raise ValueError(f"Nuisance must be >=0; got {a.nuisance} for {j}")

        # Rates nonnegative
        def _check_rate(name: str, x: float) -> None:
            if x < 0 or not math.isfinite(x):
                raise ValueError(f"{name} must be finite and >=0; got {x}")

        for j, lam in self.hazards.lambda_true.items():
            _check_rate(f"lambda_true[{j}]", lam)

        _check_rate("lambda_01", self.attacker.lambda_01)
        _check_rate("lambda_12", self.attacker.lambda_12)
        _check_rate("lambda_23", self.attacker.lambda_23)
        _check_rate("kappa_comp", self.attacker.kappa_comp)

        _check_rate("Lambda_fake", self.fake.Lambda_fake)

        # Defender and trust consistency
        if self.trust.F_max <= 0:
            raise ValueError("F_max must be >=1")
        _check_rate("phi", self.trust.phi)

        for D in DEFENDER_STATES:
            if D not in self.defender.kappa_mit:
                raise ValueError(f"defender.kappa_mit missing {D}")
            if D not in self.trust.hD_down or D not in self.trust.hD_up:
                raise ValueError(f"trust hD multipliers missing {D}")
            if D not in self.response.beta_D:
                raise ValueError(f"response.beta_D missing {D}")

        _check_rate("omega", self.defender.omega)
        if self.defender.Z_max <= 0:
            raise ValueError("Z_max must be >=1")
        if self.defender.Z_relax_threshold < 0 or self.defender.Z_relax_threshold > self.defender.Z_max:
            raise ValueError("Z_relax_threshold out of bounds")

        # π sums to 1
        s = sum(self.fake.pi.values())
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"fake.pi must sum to 1; got {s}")

        # response-time
        _check_rate("Delta", self.response.Delta)
        if self.response.Delta <= 0:
            raise ValueError("Delta must be > 0")
        if self.response.quorum_m <= 0 or self.response.quorum_m > self.N_employees:
            raise ValueError("quorum_m must be in [1, N_employees]")


# -----------------------------
# Scenario strategy factory
# -----------------------------

def strategy_fire_spam() -> Dict[AlarmType, float]:
    return {
        "fire": 0.60,
        "intruder": 0.15,
        "server_temp": 0.10,
        "usb_malware": 0.10,
        "elevator": 0.05,
    }


def strategy_low_severity_spam() -> Dict[AlarmType, float]:
    return {
        "fire": 0.01,
        "intruder": 0.04,
        "server_temp": 0.15,
        "usb_malware": 0.35,
        "elevator": 0.45,
    }


def strategy_mixed_hypothesis() -> Dict[AlarmType, float]:
    return {
        "fire": 0.05,
        "intruder": 0.15,
        "server_temp": 0.35,
        "usb_malware": 0.25,
        "elevator": 0.20,
    }


# -----------------------------
# Default parameter set (paper-ready)
# -----------------------------

def default_params(
    *,
    N_employees: int = 200,
    strategy: Optional[str] = "fire_spam",
) -> ModelParams:
    """
    Return the default calibrated parameter set.

    strategy: one of {"fire_spam","low_severity_spam","mixed"}.
    """
    alarm_attr: Dict[AlarmType, AlarmAttributes] = {
        "fire": AlarmAttributes(severity=3.0, nuisance=3.0, exposure=0.95),
        "intruder": AlarmAttributes(severity=2.5, nuisance=2.0, exposure=0.80),
        "server_temp": AlarmAttributes(severity=2.0, nuisance=2.5, exposure=0.40),
        "usb_malware": AlarmAttributes(severity=1.5, nuisance=1.0, exposure=0.25),
        "elevator": AlarmAttributes(severity=0.8, nuisance=0.6, exposure=0.35),
    }

    hazards = HazardRates(
        lambda_true={
            "fire": 0.0005,
            "intruder": 0.0020,
            "server_temp": 0.0100,
            "usb_malware": 0.0150,
            "elevator": 0.0200,
        }
    )

    attacker = AttackerProgression(
        lambda_01=0.030,
        lambda_12=0.020,
        lambda_23=0.050,
        kappa_comp=0.020,
    )

    defender = DefenderAgility(
        Z_max=12,
        omega=0.10,
        gamma01_base=0.01,
        gamma01_slope=0.015,
        gamma12_base=0.005,
        gamma12_slope=0.010,
        gamma20=0.020,
        Z_relax_threshold=2,
        kappa_mit={
            "D0": 0.005,
            "D1": 0.012,
            "D2": 0.020,
        },
    )

    trust = TrustDynamics(
        F_max=6,
        phi=0.06,
        alpha_down={
            "fire": 0.22,
            "intruder": 0.16,
            "server_temp": 0.14,
            "usb_malware": 0.10,
            "elevator": 0.06,
        },
        alpha_up={
            "fire": 0.08,
            "intruder": 0.06,
            "server_temp": 0.05,
            "usb_malware": 0.04,
            "elevator": 0.02,
        },
        hD_down={"D0": 1.00, "D1": 0.75, "D2": 0.55},
        hD_up={"D0": 1.00, "D1": 1.25, "D2": 1.45},
        qT_down={2: 1.00, 1: 1.10, 0: 0.00},
        qT_up={0: 1.00, 1: 0.85, 2: 0.00},
        g_f_slope=0.25,
        g_f_tilde_slope=0.15,
    )

    if strategy is None:
        pi = strategy_fire_spam()
    else:
        s = strategy.strip().lower()
        if s == "fire_spam":
            pi = strategy_fire_spam()
        elif s == "low_severity_spam":
            pi = strategy_low_severity_spam()
        elif s in ("mixed", "mixed_hypothesis"):
            pi = strategy_mixed_hypothesis()
        else:
            raise ValueError("Unknown strategy. Use 'fire_spam', 'low_severity_spam', or 'mixed'.")

    fake = FakeAlarmInjection(
        Lambda_fake=1.0,
        pi=pi,
    )

    response = ResponseTimeModel(
        mu0={
            "fire": 6.0,
            "intruder": 3.0,
            "server_temp": 1.0,
            "usb_malware": 0.8,
            "elevator": 0.3,
        },
        beta_T=0.55,
        beta_F=0.35,
        beta_S=0.40,
        beta_D={"D0": 0.0, "D1": 0.2, "D2": 0.4},
        Delta=0.05,          # 3 minutes
        quorum_m=8,
    )

    mp = ModelParams(
        N_employees=N_employees,
        alarm_attr=alarm_attr,
        hazards=hazards,
        attacker=attacker,
        defender=defender,
        trust=trust,
        fake=fake,
        response=response,
        evacuation_alarm_types=("fire", "intruder"),
    )
    mp.validate()
    return mp
