# Alarm Poisoning Cybersecurity Dynamics Model

This repository contains the simulation code accompanying the paper:

**“Crying Wolf in Cyberspace: A Cybersecurity Dynamics Study of Alarm Fatigue Attacks.”**

The project implements a **stochastic Cybersecurity Dynamics model** describing how attackers can exploit alarm fatigue by injecting false alerts into monitoring systems. The system is modeled as a **Continuous-Time Markov Chain (CTMC)** and simulated using the **Gillespie Stochastic Simulation Algorithm (SSA)**.

The simulation captures interactions between:

- attacker progression
- alarm infrastructure compromise
- defender escalation
- employee trust and fatigue dynamics
- alarm pressure accumulation

Monte Carlo simulations are used to analyze collapse dynamics and defender agility metrics.

---

# Repository Structure

```

src/
alarm_poisoning_cd/
core/        # model parameters and state representation
models/      # CTMC / Gillespie simulation engine
metrics/     # Cybersecurity Dynamics and TRAM metrics
simulation/  # simulation runner

scripts/
make_paper_figures.py
summarize_mc.py
stat_tests.py
run_mc.ps1

output/
results/     # simulation outputs
plots/       # figures used in the paper
plots_mc/    # Monte Carlo distributions

````

---

# Requirements

The code requires **Python 3.9 or newer**.

Install required packages using `pip`:

```bash
pip install numpy pandas matplotlib
````

Optional (for development):

```bash
pip install scipy
```

---

# Running a Single Simulation

You can run a single CTMC simulation using the runner module.

Example:

```bash
python -m alarm_poisoning_cd.simulation.runner --strategy fire_spam --seed 1
```

Available strategies:

```
fire_spam
low_severity_spam
mixed
```

Optional parameters:

```
--horizon     simulation time horizon (hours)
--employees   number of employees
--seed        random seed
--outdir      output directory
```

Example:

```bash
python -m alarm_poisoning_cd.simulation.runner \
    --strategy fire_spam \
    --seed 1 \
    --horizon 168 \
    --employees 200
```

Outputs:

```
output/results/trajectory_fire_spam_seed1.csv
output/results/summary_fire_spam_seed1.json
```

---

# Monte Carlo Campaign

The paper experiments use **50 simulations per strategy**.

Example workflow:

```
fire_spam seeds: 1..50
low_severity_spam seeds: 1..50
mixed seeds: 1..50
```

Total simulations:

```
150 CTMC realizations
```

You can run them manually:

```bash
for i in {1..50}; do
python -m alarm_poisoning_cd.simulation.runner --strategy fire_spam --seed $i
done
```

Repeat for other strategies.

---

# Computing Metrics

After simulations are completed, compute Cybersecurity Dynamics metrics.

Example:

```bash
python src/alarm_poisoning_cd/metrics/cd_metrics.py output/results/*.csv --out cd_metrics.csv
```

Compute TRAM agility metrics:

```bash
python src/alarm_poisoning_cd/metrics/tram.py output/results/*.csv --out tram_metrics.csv
```

---

# Generating Paper Figures

Run:

```bash
python scripts/make_paper_figures.py
```

This generates figures in:

```
output/plots/
```

Figures include:

```
fig1_disbelief_overlay.png
fig2_fatigue_overlay.png
fig3_alarm_pressure.png
fig4_time_to_collapse_hist.png
fig5_tram_lbt_hist.png
```

---

# Monte Carlo Distribution Plots

Generate statistical summaries and distribution plots:

```bash
python scripts/summarize_mc.py
```

Outputs:

```
output/plots_mc/
```

including:

```
dist_LBT_hist.png
dist_LBT_box.png
dist_t_low_trust_40_hist.png
dist_t_low_trust_40_box.png
```

---

# Statistical Tests

Pairwise statistical comparisons between strategies can be computed with:

```bash
python scripts/stat_tests.py
```

This produces:

```
stats_pairwise.csv
stats_pairwise_holm.csv
stats_pairwise.tex
```

including Mann–Whitney U tests and Cliff's delta effect sizes.

---

# Model Overview

The system state includes:

```
Attacker state A(t)
Alarm infrastructure state S(t)
Defender posture D(t)
Alarm pressure Z(t)
Employee population distribution n(T,F)
```

Employees are represented by **trust–fatigue compartments**:

```
T ∈ {0,1,2}
F ∈ {0..Fmax}
```

Events include:

```
attacker progression
alarm infrastructure compromise
defender escalation
true alarm arrival
fake alarm injection
fatigue recovery
alarm pressure decay
```

The system evolves according to a **Continuous-Time Markov Chain simulated with Gillespie SSA**.

---

