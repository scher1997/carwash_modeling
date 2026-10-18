# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run a single simulation episode
uv run python -m src.simulate

# Run Monte Carlo policy evaluation (produces results/ CSVs and figures)
uv run python -m src.evaluate

# Run parameter tuning (grid search, Optuna, sensitivity analysis)
uv run python -m src.tune
```

`src/` uses relative imports so scripts must be invoked as modules (`-m src.<name>`) from the project root, not as direct file paths.

## Architecture

The model follows Powell's Universal Modeling Language (UML) for sequential decision analytics. Each file in `src/` maps to one UML component:

| File | UML component | Role |
|---|---|---|
| `state.py` | S_t | `State` dataclass — all system state at time t |
| `exogenous.py` | W_{t+1} | Random arrivals and service draws |
| `transition.py` | S^M | Physics of the system: how state evolves given decision + exogenous |
| `objective.py` | C_t | Cost per step: wait penalty + bay operating cost |
| `policies.py` | X^π | Decision functions mapping `State → {"bays_open": int}` |
| `simulate.py` | Loop | Wires all components into a 1-minute time-stepped episode |

### Call hierarchy

```
tune.py
  └── evaluate.py        (Monte Carlo: n_samples independent seeds)
        └── simulate.run()  (single episode)
              ├── policies.py  → decision dict
              ├── exogenous.py → arrivals/completions draws
              ├── objective.py → step cost
              └── transition.py → next State
```

### Interface contract

All policies return `{"bays_open": int}`. `transition()` and `cost()` both consume this dict. When adding a new policy, register it in the `POLICIES` dict in `policies.py` — `simulate.run()` looks policies up by string key.

### Tuning modes (`tune.py`)

Three independent modes, all wrapping `evaluate_policy()`:
- **Grid search** — exhaustive enumeration of `param_grid` combinations
- **Optuna** — Bayesian (TPE) search; use when the parameter space is large or continuous
- **Sensitivity analysis** — one-at-a-time sweeps to isolate marginal impact of a single parameter

### Cost function

Profit model in `objective.contribution()`; defaults in `DEFAULT_COST_PARAMS` (placeholders, not calibrated): `price_per_wash` and `cost_per_wash` accrue on completion, `bay_cost` per staffed bay per minute (open bays plus closing bays still finishing a car), `wait_penalty` per car per minute in queue. The trace's `cost` column is `-profit`, so minimizing `total_cost` maximizes profit.
