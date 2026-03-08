"""
QuCS_2026 — TRAM / Agility Metrics Extraction

This module extracts TRAM-style agility metrics from CTMC trajectories:

- Generation-Time (GT): time between consecutive evolutions (state changes) of a party
- Effective-Generation-Time (EGT): time between evolutions that actually improve effectiveness
- Triggering-Time (TT): time from opponent's reference generation to a response generation
- Lagging-Behind-Time (LBT): how far a party lags behind opponent w.r.t. a reference time

We implement these for:
- Attacker generations: attacker_state changes (A0->A1->A2->A3)
- Defender generations: defender_state changes (D0->D1->D2->D0 ...)
- System state for mitigation: alarm_sys_state changes (S0->S1->S2)

Assumptions:
- Trajectory CSV has columns: time_h, attacker_state, defender_state, alarm_sys_state, ...
- Rows are recorded at each CTMC jump (as produced by simulation/runner.py).
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ATTACKER_ORDER = {"A0": 0, "A1": 1, "A2": 2, "A3": 3}
DEFENDER_ORDER = {"D0": 0, "D1": 1, "D2": 2}
SYS_ORDER = {"S0": 0, "S1": 1, "S2": 2}


# ============================================================
# Loading
# ============================================================

def load_trajectory(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("time_h").reset_index(drop=True)
    return df


# ============================================================
# State-change event extraction
# ============================================================

def extract_generations(
    df: pd.DataFrame,
    column: str,
) -> List[Tuple[float, str]]:
    """
    Return list of (time, new_state) whenever df[column] changes.
    Includes the first observed state at time of first row.

    Example output:
      [(0.0,'D0'), (12.3,'D1'), (18.0,'D2'), ...]
    """
    if len(df) == 0:
        return []

    times = df["time_h"].values
    states = df[column].values

    out: List[Tuple[float, str]] = [(float(times[0]), str(states[0]))]
    for i in range(1, len(df)):
        if states[i] != states[i - 1]:
            out.append((float(times[i]), str(states[i])))
    return out


def generation_times(gens: List[Tuple[float, str]]) -> List[float]:
    """
    GT sequence: time differences between consecutive generations.
    """
    if len(gens) < 2:
        return []
    return [gens[i][0] - gens[i - 1][0] for i in range(1, len(gens))]


def effective_generations(
    gens: List[Tuple[float, str]],
    order_map: Dict[str, int],
    direction: str = "increase",
) -> List[Tuple[float, str]]:
    """
    Filter generations to only those that improve effectiveness.

    direction:
      - 'increase': keep transitions where order increases (e.g., D0->D1->D2)
      - 'decrease': keep transitions where order decreases
    """
    if len(gens) < 2:
        return gens[:]  # trivial

    out = [gens[0]]
    for i in range(1, len(gens)):
        prev = gens[i - 1][1]
        cur = gens[i][1]
        if prev not in order_map or cur not in order_map:
            continue
        if direction == "increase" and order_map[cur] > order_map[prev]:
            out.append(gens[i])
        elif direction == "decrease" and order_map[cur] < order_map[prev]:
            out.append(gens[i])
    return out


# ============================================================
# TRAM metrics: TT and LBT
# ============================================================

def triggering_time(
    opponent_gens: List[Tuple[float, str]],
    responder_gens: List[Tuple[float, str]],
    opponent_state: str,
    responder_state: str,
) -> Optional[float]:
    """
    TT: time elapsed since an opponent's reference generation (entering opponent_state)
    that may have triggered the responder entering responder_state.

    Defined as:
      TT = t_responder(responder_state) - t_opponent(opponent_state),
    where both are the first occurrence times of those states in their generation lists.

    Returns None if either state is never reached.
    """
    t_op = None
    t_rs = None

    for t, s in opponent_gens:
        if s == opponent_state:
            t_op = t
            break
    for t, s in responder_gens:
        if s == responder_state:
            t_rs = t
            break

    if t_op is None or t_rs is None:
        return None
    return t_rs - t_op


def lagging_behind_time(
    opponent_gens: List[Tuple[float, str]],
    responder_gens: List[Tuple[float, str]],
    opponent_state: str,
    responder_state: str,
) -> Optional[float]:
    """
    LBT as a special case of TT:
      LBT = t_responder(responder_state) - t_opponent(opponent_state)
    Interpreted as "how much the responder lags behind the opponent's evolution"
    relative to reference states.

    Returns None if either state is never reached.
    """
    return triggering_time(opponent_gens, responder_gens, opponent_state, responder_state)


# ============================================================
# Aggregated TRAM report for one trajectory
# ============================================================

def tram_report(df: pd.DataFrame) -> Dict[str, object]:
    """
    Produce a TRAM/agility report for one trajectory DataFrame.

    Returns a dict including:
    - attacker GT stats, defender GT stats
    - defender EGT stats (improvements only)
    - TT and LBT between attacker A3 and defender D2 (classic "lag")
    - mitigation TT: attacker A2/S1 vs system S2
    """
    atk_gens = extract_generations(df, "attacker_state")
    def_gens = extract_generations(df, "defender_state")
    sys_gens = extract_generations(df, "alarm_sys_state")

    atk_gt = generation_times(atk_gens)
    def_gt = generation_times(def_gens)

    def_eff = effective_generations(def_gens, DEFENDER_ORDER, direction="increase")
    def_egt = generation_times(def_eff)

    # canonical lag: attacker reaches A3, defender reaches D2
    lbt_A3_D2 = lagging_behind_time(atk_gens, def_gens, "A3", "D2")

    # mitigation TT: alarm system reaches S1 (compromised) -> S2 (mitigated)
    t_S1 = None
    t_S2 = None
    for t, s in sys_gens:
        if s == "S1" and t_S1 is None:
            t_S1 = t
        if s == "S2" and t_S2 is None:
            t_S2 = t
    mitigation_delay = None
    if t_S1 is not None and t_S2 is not None:
        mitigation_delay = t_S2 - t_S1

    def _stats(xs: List[float]) -> Dict[str, float]:
        if len(xs) == 0:
            return {"count": 0}
        s = sorted(xs)
        return {
            "count": float(len(xs)),
            "mean": float(sum(xs) / len(xs)),
            "min": float(s[0]),
            "max": float(s[-1]),
            "median": float(s[len(s) // 2]),
        }

    return {
        "attacker_generations": atk_gens,
        "defender_generations": def_gens,
        "system_generations": sys_gens,
        "attacker_GT": _stats(atk_gt),
        "defender_GT": _stats(def_gt),
        "defender_EGT": _stats(def_egt),
        "LBT_A3_to_D2": lbt_A3_D2,
        "mitigation_delay_S1_to_S2": mitigation_delay,
    }


# ============================================================
# Batch processing
# ============================================================

def summarize_tram(paths: List[Path | str]) -> pd.DataFrame:
    """
    Summarize TRAM metrics for multiple trajectories (Monte Carlo runs).
    """
    rows: List[Dict[str, object]] = []
    for p in paths:
        df = load_trajectory(p)
        rep = tram_report(df)

        row = {
            "path": str(p),
            "LBT_A3_to_D2": rep["LBT_A3_to_D2"],
            "mitigation_delay_S1_to_S2": rep["mitigation_delay_S1_to_S2"],
            "attacker_GT_mean": rep["attacker_GT"].get("mean") if isinstance(rep["attacker_GT"], dict) else None,
            "defender_GT_mean": rep["defender_GT"].get("mean") if isinstance(rep["defender_GT"], dict) else None,
            "defender_EGT_mean": rep["defender_EGT"].get("mean") if isinstance(rep["defender_EGT"], dict) else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute TRAM/agility metrics for QuCS_2026.")
    parser.add_argument("paths", nargs="+", help="Trajectory CSV files.")
    parser.add_argument("--out", type=str, default="tram_metrics.csv")

    args = parser.parse_args()

    out = summarize_tram([Path(p) for p in args.paths])
    out.to_csv(args.out, index=False)
    print("TRAM metrics written to", args.out)
