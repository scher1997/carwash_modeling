"""
Main simulation runner.

Wires together State, ExogenousInfo, transition, policy, and objective
into a single time-stepped loop. Writes results to results/.

Usage:
    python src/simulate.py
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from state import State
from exogenous import ExogenousInfo
from transition import transition
from objective import cost
from policies import POLICIES


def run(
    policy_name: str = "threshold",
    policy_params: dict = None,
    cost_params: dict = None,
    horizon: float = 480.0,    # minutes (8-hour day)
    arrival_rate: float = 10.0,
    service_rate: float = 8.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run one simulation episode and return a trace DataFrame.

    Parameters
    ----------
    policy_name : str
        Key into POLICIES dict (see policies.py).
    policy_params : dict
        Parameters passed to the policy function.
    cost_params : dict
        Parameters passed to the cost function.
    horizon : float
        Simulation length in minutes.
    arrival_rate : float
        Mean arrivals per hour.
    service_rate : float
        Mean cars served per hour per bay.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        One row per time step with state, decision, and cost columns.
    """
    policy_params = policy_params or {}
    cost_params = cost_params or {}

    policy_fn = POLICIES[policy_name]
    exo = ExogenousInfo(arrival_rate=arrival_rate, service_rate=service_rate, seed=seed)
    state = State()

    records = []
    t = 0.0
    step = 1.0  # 1-minute time steps

    while t < horizon:
        decision = policy_fn(state, policy_params)

        arrivals = int(exo.rng.poisson(exo.arrival_rate / 60.0))
        completions = min(state.bays_busy, int(exo.rng.poisson(exo.service_rate / 60.0 * state.bays_busy)))
        wait_added = state.queue_length * step

        step_cost = cost(state, decision, cost_params)

        records.append({
            "time": t,
            "queue_length": state.queue_length,
            "bays_open": state.bays_open,
            "bays_busy": state.bays_busy,
            "cars_served": state.cars_served,
            "avg_wait": state.avg_wait,
            "decision_bays": decision.get("bays_open"),
            "cost": step_cost,
        })

        exogenous_realization = {
            "arrivals": arrivals,
            "completions": completions,
            "wait_added": wait_added,
        }
        state = transition(state, decision, exogenous_realization)
        state.time = t + step
        t += step

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    df = run(policy_name="threshold", policy_params={"open_threshold": 5, "close_threshold": 2, "max_bays": 3})

    out_path = os.path.join("results", "simulation_trace.csv")
    df.to_csv(out_path, index=False)

    print(f"Simulation complete. {len(df)} steps.")
    print(f"Total cost:    {df['cost'].sum():.2f}")
    print(f"Cars served:   {df['cars_served'].iloc[-1]}")
    print(f"Avg wait:      {df['avg_wait'].iloc[-1]:.2f} min")
    print(f"Results saved: {out_path}")
