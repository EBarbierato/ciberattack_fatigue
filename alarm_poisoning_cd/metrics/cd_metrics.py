"""
QuCS_2026 — Cybersecurity Dynamics Metrics

This module computes macroscopic CD observables from SSA trajectories:

- Time-to-trust-collapse
- Time-to-fatigue-collapse
- Defender/attacker lag (TRAM-style LBT)
- Recovery slopes after mitigation
- Phase-transition indicators

These metrics are designed to directly answer the paper's
research questions.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ============================================================
# Loading
# ============================================================

def load_trajectory(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("time_h").reset_index(drop=True)
    return df


# ============================================================
# Collapse / tipping point metrics
# ============================================================

def first_crossing_time(
    df: pd.DataFrame,
    column: str,
    threshold: float,
) -> Optional[float]:
    """
    Return the first time column >= threshold.
    """
    sub = df[df[column] >= threshold]
    if len(sub) == 0:
        return None
    return float(sub.iloc[0]["time_h"])


def time_to_low_trust(
    df: pd.DataFrame,
    frac: float = 0.40,
) -> Optional[float]:
    """
    Time until fraction of low-trust employees exceeds frac.
    """
    return first_crossing_time(df, "low_trust_frac", frac)


def time_to_high_fatigue(
    df: pd.DataFrame,
    frac: float = 0.30,
) -> Optional[float]:
    """
    Time until fraction of highly fatigued employees exceeds frac.
    """
    return first_crossing_time(df, "high_fatigue_frac", frac)


# ============================================================
# Attacker / Defender agility
# ============================================================

ATTACKER_ORDER = {"A0": 0, "A1": 1, "A2": 2, "A3": 3}
DEFENDER_ORDER = {"D0": 0, "D1": 1, "D2": 2}


def time_to_state(
    df: pd.DataFrame,
    column: str,
    target: str,
) -> Optional[float]:
    sub = df[df[column] == target]
    if len(sub) == 0:
        return None
    return float(sub.iloc[0]["time_h"])


def attacker_time_to_A3(df: pd.DataFrame) -> Optional[float]:
    return time_to_state(df, "attacker_state", "A3")


def defender_time_to_D2(df: pd.DataFrame) -> Optional[float]:
    return time_to_state(df, "defender_state", "D2")


def lagging_behind_time(df: pd.DataFrame) -> Optional[float]:
    """
    LBT = t(D2) - t(A3).
    Positive => defender slower.
    """
    tA = attacker_time_to_A3(df)
    tD = defender_time_to_D2(df)
    if tA is None or tD is None:
        return None
    return tD - tA


# ============================================================
# Recovery dynamics
# ============================================================

def recovery_slope(
    df: pd.DataFrame,
    column: str = "low_trust_frac",
    after_state: str = "S2",
    window: float = 24.0,
) -> Optional[float]:
    """
    Estimate linear slope of column after system mitigation S2
    over the next 'window' hours.

    Negative slope => recovery.
    """
    mask = df["alarm_sys_state"] == after_state
    if not mask.any():
        return None

    t0 = df.loc[mask, "time_h"].iloc[0]
    sub = df[(df["time_h"] >= t0) & (df["time_h"] <= t0 + window)]

    if len(sub) < 5:
        return None

    x = sub["time_h"].values
    y = sub[column].values

    # simple least squares slope
    xm = x.mean()
    ym = y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = ((x - xm) ** 2).sum()
    if den == 0:
        return None
    return num / den


# ============================================================
# Phase-transition diagnostics
# ============================================================

def max_gradient(
    df: pd.DataFrame,
    column: str,
) -> float:
    """
    Maximum absolute discrete derivative d(column)/dt.
    Used to detect abrupt transitions.
    """
    dt = df["time_h"].diff()
    dy = df[column].diff()
    grad = (dy / dt).abs()
    return float(grad.max())


# ============================================================
# Batch Monte Carlo processing
# ============================================================

def summarize_runs(paths: Iterable[Path | str]) -> pd.DataFrame:
    """
    Compute all CD metrics for many trajectories.

    Returns DataFrame with one row per run.
    """
    rows: List[Dict] = []

    for p in paths:
        df = load_trajectory(p)

        row = {
            "path": str(p),
            "t_low_trust_40": time_to_low_trust(df, 0.40),
            "t_high_fatigue_30": time_to_high_fatigue(df, 0.30),
            "t_attacker_A3": attacker_time_to_A3(df),
            "t_defender_D2": defender_time_to_D2(df),
            "LBT": lagging_behind_time(df),
            "recovery_slope_low_trust": recovery_slope(df),
            "max_grad_low_trust": max_gradient(df, "low_trust_frac"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# CLI utility (optional)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute CD metrics for QuCS_2026.")
    parser.add_argument("paths", nargs="+", help="Trajectory CSV files.")
    parser.add_argument("--out", type=str, default="cd_metrics.csv")

    args = parser.parse_args()

    df = summarize_runs(args.paths)
    df.to_csv(args.out, index=False)

    print("Metrics written to", args.out)
