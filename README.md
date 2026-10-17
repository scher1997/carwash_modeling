# Car Wash Modeling

A discrete-event simulation of a car wash system built using Warren Powell's
**Universal Modeling Language (UML)** for sequential decision analytics.

The goal is to model the car wash as a sequential decision problem — capturing
arrivals, queue dynamics, service decisions, and costs — so that different
operating policies can be compared objectively.

---

## Framework: Powell's Universal Modeling Language

Powell's UML decomposes any sequential decision problem into five core elements:

| Element | Symbol | Description |
|---|---|---|
| **State variable** | S_t | Everything we know at time t that is relevant to future decisions |
| **Decision variable** | x_t | The action chosen at time t |
| **Exogenous information** | W_{t+1} | New information that arrives after the decision, outside our control |
| **Transition function** | S^M | How the state evolves: S_{t+1} = S^M(S_t, x_t, W_{t+1}) |
| **Objective function** | C_t | The cost or reward earned at each step |

A **policy** X^π(S_t) maps states to decisions. The goal is to find the policy
that optimizes the cumulative objective over a time horizon.

For the car wash, this maps roughly to:
- **State**: cars in queue, number of bays open, time of day
- **Decision**: open/close a bay, pricing tier, express lane routing
- **Exogenous info**: car arrivals (random), service time draws
- **Transition**: queue updates after each arrival/departure event
- **Objective**: minimize wait time, maximize throughput, or balance both

---

## Project Structure

```
carwash_modeling/
├── src/
│   ├── state.py          # State variable definition
│   ├── exogenous.py      # Exogenous information (arrival & service processes)
│   ├── transition.py     # Transition function — how state evolves each step
│   ├── policies.py       # Decision policies (rules for choosing x_t given S_t)
│   ├── objective.py      # Cost/reward function evaluated at each time step
│   └── simulate.py       # Simulation runner — wires all components together
├── data/
│   └── (scenario configs, arrival rate tables, parameter files)
├── results/
│   └── (simulation outputs: CSVs, summary stats, raw traces)
├── notebooks/
│   └── (Quarto .qmd files for model documentation and result display)
├── .gitignore
├── requirements.txt
└── README.md
```

### `src/` — Simulation Model

Each file maps to one UML component and is designed to be imported by
`simulate.py`. They are intentionally kept thin — logic lives in the component
files, not the runner.

**`state.py`**
Defines the `State` class (or dataclass). Holds all variables that describe the
system at a point in time: queue length, bay status, cumulative stats, clock.
Nothing in here computes decisions — it only stores and reports state.

**`exogenous.py`**
Defines how random information enters the model. For the car wash this includes
the inter-arrival time distribution (e.g., Poisson arrivals) and the service
time distribution (e.g., exponential or empirical). This is where you would plug
in real arrival data if available.

**`transition.py`**
Implements `S^M(S_t, x_t, W_{t+1}) -> S_{t+1}`. Given the current state, the
decision made, and the next piece of exogenous information, returns the updated
state. This is the "physics" of the car wash — how the queue and bays actually
change.

**`policies.py`**
Defines one or more policy functions `X^π(S_t) -> x_t`. Each policy is a
different rule for operating the car wash (e.g., always keep all bays open,
open a bay when queue exceeds threshold N, cost-minimizing myopic policy).
Comparing policies is the main analytical goal of the simulation.

**`objective.py`**
Defines the cost or reward `C(S_t, x_t)` earned at each step. Could be wait
time penalty, revenue from completed washes, operating cost of open bays, or
a weighted combination. Keeping this separate makes it easy to swap objectives
without touching the rest of the model.

**`simulate.py`**
The entry point. Initializes the state, loops over the time horizon, calls
exogenous draws, applies the policy, updates state via the transition function,
and accumulates the objective. Writes results to `results/`. Run directly:
`python src/simulate.py`.

### `data/`

Input files for the simulation. This includes:
- Parameter configs (arrival rates, service rates, number of bays)
- Scenario definitions (peak hour vs. off-peak, staffing levels)
- Empirical data if real car wash data is eventually incorporated

Keep raw data immutable here — never overwrite it with script output.

### `results/`

Output from simulation runs: traces, summary statistics, and any intermediate
files produced by `simulate.py`. These files are git-ignored by default since
they are reproducible from `src/` and `data/`. Add a specific results file to
git only if it represents a canonical benchmark run.

### `notebooks/`

Quarto (`.qmd`) notebooks for documentation and visualization. Intended use:

- **Model overview notebook**: walks through the UML components with narrative
  and equations — the living paper for the model.
- **Results notebook**: loads output from `results/`, produces plots. R/GGPlot
  chunks can be mixed in here for final publication-quality visuals.
- **Policy comparison notebook**: side-by-side comparison of policy performance
  across scenarios.

Quarto supports mixing Python and R in the same document, so the simulation
code (Python) and the visualization layer (R/GGPlot) coexist cleanly.

---

## Getting Started

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the simulation
python src/simulate.py
```

Results will be written to `results/`. Open a notebook in `notebooks/` with
Quarto to explore outputs visually.

---

## Dependencies

See `requirements.txt` for the Python environment. R packages (ggplot2, etc.)
are handled separately and will be documented in the relevant notebooks when
the visualization layer is added.
