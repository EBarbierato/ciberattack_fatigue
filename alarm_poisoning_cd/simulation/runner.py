"""
QuCS_2026 — Alarm Poisoning as Cybersecurity Dynamics (CD)
Simulation runner.

This module:
- Instantiates default or scenario-specific parameters
- Builds the initial CTMC state
- Runs the Gillespie SSA engine
- Records full time series of CD observables
- Writes results to disk (CSV + JSON)
- Is callable as a module:

    python -m alarm_poisoning_cd.simulation.runner --horizon 168 --strategy fire_spam

Design goals:
- Paper-grade reproducibility
- No hidden approximations
- Monte Carlo ready (multiple replications supported)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from alarm_poisoning_cd.core.parameters import default_params
from alarm_poisoning_cd.core.state import CTMCState, initial_state
from alarm_poisoning_cd.models.gillespie import run_ssa


# -----------------------------
# Output helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: List[str], rows: List[Tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# -----------------------------
# Metric extraction
# -----------------------------

def snapshot_metrics(st: CTMCState, p) -> Dict[str, float]:
    """
    Extract macroscopic CD observables from the current state.
    """
    return {
        "time_h": st.t,
        "low_trust_frac": st.fraction_low_trust(p),
        "mean_fatigue": st.mean_fatigue(p),
        "high_fatigue_frac": st.fraction_high_fatigue(p),
        "attacker_state": st.attacker,
        "alarm_sys_state": st.alarm_sys,
        "defender_state": st.defender,
        "alarm_pressure_Z": st.Z,
    }


# -----------------------------
# Runner
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run QuCS_2026 CTMC simulation.")
    parser.add_argument("--horizon", type=float, default=168.0, help="Simulation horizon in hours.")
    parser.add_argument("--strategy", type=str, default="fire_spam",
                        choices=["fire_spam", "low_severity_spam", "mixed"],
                        help="Fake-alarm attacker strategy.")
    parser.add_argument("--employees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="output/results")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Build parameters and initial state
    p = default_params(N_employees=args.employees, strategy=args.strategy)
    st = initial_state(p)

    # Time series container
    rows: List[Tuple] = []

    def recorder(state: CTMCState, event: str, rate: float) -> None:
        m = snapshot_metrics(state, p)
        rows.append((
            m["time_h"],
            m["low_trust_frac"],
            m["mean_fatigue"],
            m["high_fatigue_frac"],
            m["attacker_state"],
            m["alarm_sys_state"],
            m["defender_state"],
            m["alarm_pressure_Z"],
            event,
            rate,
        ))

    # Run SSA
    run_ssa(
        p,
        st,
        t_end=args.horizon,
        rng_seed=args.seed,
        on_step=recorder,
    )

    # Write outputs
    csv_path = outdir / f"trajectory_{args.strategy}_seed{args.seed}.csv"
    header = [
        "time_h",
        "low_trust_frac",
        "mean_fatigue",
        "high_fatigue_frac",
        "attacker_state",
        "alarm_sys_state",
        "defender_state",
        "alarm_pressure_Z",
        "event",
        "event_rate",
    ]
    write_csv(csv_path, header, rows)

    # Summary JSON
    summary = {
        "strategy": args.strategy,
        "seed": args.seed,
        "horizon": args.horizon,
        "employees": args.employees,
        "final_time": st.t,
        "final_low_trust_frac": st.fraction_low_trust(p),
        "final_mean_fatigue": st.mean_fatigue(p),
        "final_alarm_pressure": st.Z,
        "final_attacker_state": st.attacker,
        "final_alarm_sys_state": st.alarm_sys,
        "final_defender_state": st.defender,
    }

    json_path = outdir / f"summary_{args.strategy}_seed{args.seed}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("Simulation complete.")
    print("CSV:", csv_path)
    print("Summary:", json_path)


if __name__ == "__main__":
    main()
