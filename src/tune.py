"""
Parameter tuning for car wash policies.

Three modes:
  1. Manual grid search  — user specifies parameter grids, all combos are evaluated
  2. Optuna optimization — automated black-box search driven by the objective function
  3. Sensitivity analysis — vary one parameter at a time to see marginal impact

Usage:
    python src/tune.py
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .evaluate import evaluate_policy, compare_policies, summary_stats


# ---------------------------------------------------------------------------
# 1. Manual grid search
# ---------------------------------------------------------------------------

def grid_search(
    policy_name: str,
    param_grid: dict,
    cost_params: dict = None,
    n_samples: int = 50,
    horizon: float = 480.0,
    arrival_rate: float = 10.0,
    service_rate: float = 8.0,
    objective: str = "total_cost",
) -> pd.DataFrame:
    """
    Exhaustively evaluate all combinations in param_grid.

    Parameters
    ----------
    param_grid : dict
        Keys are policy parameter names, values are lists of values to try.
        Example: {'open_threshold': [3, 5, 7], 'close_threshold': [1, 2]}
    objective : str
        Summary metric to rank combinations by (uses mean across replications).
        Options: total_cost, avg_wait, cars_served, avg_queue.

    Returns
    -------
    pd.DataFrame
        One row per parameter combination, sorted best-to-worst by objective.
    """
    cost_params = cost_params or {}
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    records = []
    for combo in combos:
        params = dict(zip(keys, combo))
        df = evaluate_policy(
            policy_name=policy_name,
            policy_params=params,
            cost_params=cost_params,
            n_samples=n_samples,
            horizon=horizon,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
        )
        row = {**params}
        for metric in ["total_cost", "avg_wait", "cars_served", "avg_queue", "max_queue"]:
            row[f"{metric}_mean"] = df[metric].mean()
            row[f"{metric}_std"] = df[metric].std()
        records.append(row)

    result = pd.DataFrame(records)
    ascending = objective != "cars_served"  # higher cars_served is better
    return result.sort_values(f"{objective}_mean", ascending=ascending).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Optuna automated optimization
# ---------------------------------------------------------------------------

def optuna_tune(
    policy_name: str,
    param_space: dict,
    cost_params: dict = None,
    n_trials: int = 50,
    n_samples: int = 30,
    horizon: float = 480.0,
    arrival_rate: float = 10.0,
    service_rate: float = 8.0,
    objective: str = "total_cost",
    direction: str = "minimize",
) -> tuple:
    """
    Use Optuna to search the parameter space efficiently.

    Optuna uses Bayesian optimization (TPE sampler by default) — it learns
    from past trials to focus search on promising regions. Much more efficient
    than grid search when the parameter space is large or continuous.

    Parameters
    ----------
    param_space : dict
        Parameter definitions. Each value is a dict with keys:
          - 'type': 'int', 'float', or 'categorical'
          - 'low', 'high': bounds for int/float
          - 'choices': list for categorical
        Example:
          {
            'open_threshold': {'type': 'int', 'low': 2, 'high': 10},
            'close_threshold': {'type': 'int', 'low': 1, 'high': 5},
            'max_bays': {'type': 'categorical', 'choices': [2, 3, 4]},
          }
    n_trials : int
        Number of Optuna trials (each trial = one policy configuration evaluated
        over n_samples replications).
    objective : str
        Metric to optimize (mean across replications).
    direction : str
        'minimize' or 'maximize'.

    Returns
    -------
    (study, results_df)
        study: the Optuna study object (access .best_params, .best_value, etc.)
        results_df: DataFrame of all trials with params and objective value.

    Ideas for future automation:
      - Pruning: use Optuna's MedianPruner to stop bad trials early by evaluating
        a small n_samples first, then expanding only for promising configs.
      - Multi-objective: optimize cost AND wait time simultaneously using
        optuna.create_study(directions=['minimize', 'minimize']) — gives a
        Pareto front instead of a single best solution.
      - Warm starting: seed the study with grid search results to bootstrap
        the Bayesian optimizer with known-good regions.
      - Time-varying arrival rates: extend the objective to vary arrival_rate
        across trials to find a policy robust to demand uncertainty.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    cost_params = cost_params or {}

    def _objective(trial):
        params = {}
        for name, spec in param_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        df = evaluate_policy(
            policy_name=policy_name,
            policy_params=params,
            cost_params=cost_params,
            n_samples=n_samples,
            horizon=horizon,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            base_seed=trial.number * 1000,
        )
        return df[objective].mean()

    study = optuna.create_study(direction=direction)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    records = []
    for trial in study.trials:
        row = {**trial.params, "objective_value": trial.value, "trial": trial.number}
        records.append(row)

    return study, pd.DataFrame(records).sort_values("objective_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Sensitivity analysis (one-at-a-time)
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    policy_name: str,
    base_params: dict,
    sweep_param: str,
    sweep_values: list,
    cost_params: dict = None,
    n_samples: int = 50,
    horizon: float = 480.0,
    arrival_rate: float = 10.0,
    service_rate: float = 8.0,
    objective: str = "total_cost",
) -> pd.DataFrame:
    """
    Vary one parameter while holding all others fixed at base_params.
    Shows the marginal impact of that parameter on the objective.
    """
    cost_params = cost_params or {}
    records = []

    for val in sweep_values:
        params = {**base_params, sweep_param: val}
        df = evaluate_policy(
            policy_name=policy_name,
            policy_params=params,
            cost_params=cost_params,
            n_samples=n_samples,
            horizon=horizon,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
        )
        records.append({
            sweep_param: val,
            f"{objective}_mean": df[objective].mean(),
            f"{objective}_std": df[objective].std(),
            f"{objective}_p5": df[objective].quantile(0.05),
            f"{objective}_p95": df[objective].quantile(0.95),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tuning figures
# ---------------------------------------------------------------------------

def plot_sensitivity(sensitivity_df: pd.DataFrame, sweep_param: str, objective: str = "total_cost", save_path: str = None):
    """Line plot of objective vs. swept parameter with confidence band."""
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(7, 4))

    x = sensitivity_df[sweep_param]
    y = sensitivity_df[f"{objective}_mean"]
    p5 = sensitivity_df[f"{objective}_p5"]
    p95 = sensitivity_df[f"{objective}_p95"]

    ax.plot(x, y, marker="o", color="#4C72B0")
    ax.fill_between(x, p5, p95, alpha=0.15, color="#4C72B0")
    ax.set_xlabel(sweep_param.replace("_", " ").title())
    ax.set_ylabel(objective.replace("_", " ").title())
    ax.set_title(f"Sensitivity: {objective.replace('_', ' ').title()} vs {sweep_param.replace('_', ' ').title()}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_optuna_history(study, save_path: str = None):
    """Plot Optuna optimization progress over trials."""
    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_optimization_history(study, ax=ax)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
    except Exception:
        # Fallback: manual plot
        values = [t.value for t in study.trials]
        best = np.minimum.accumulate(values) if study.direction.name == "MINIMIZE" else np.maximum.accumulate(values)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(values, alpha=0.4, label="Trial value")
        ax.plot(best, color="red", label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Objective")
        ax.set_title("Optuna Optimization History")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ---------------------------------------------------------------------------
# Entry point — demonstrates all three modes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # 1. Grid search
    print("=== Grid Search ===")
    grid = grid_search(
        policy_name="threshold",
        param_grid={
            "open_threshold": [3, 5, 7, 10],
            "close_threshold": [1, 2, 3],
            "max_bays": [2, 3],
        },
        n_samples=50,
        objective="total_cost",
    )
    grid.to_csv("results/tune_grid_search.csv", index=False)
    print(grid[["open_threshold", "close_threshold", "max_bays", "total_cost_mean", "avg_wait_mean"]].head(5).to_string(index=False))

    # 2. Sensitivity analysis on open_threshold
    print("\n=== Sensitivity: open_threshold ===")
    sens = sensitivity_analysis(
        policy_name="threshold",
        base_params={"open_threshold": 5, "close_threshold": 2, "max_bays": 3},
        sweep_param="open_threshold",
        sweep_values=list(range(1, 15)),
        n_samples=50,
    )
    sens.to_csv("results/tune_sensitivity.csv", index=False)
    plot_sensitivity(sens, "open_threshold", save_path="results/fig_sensitivity_open_threshold.png")
    print(sens.to_string(index=False))

    # 3. Optuna (fewer trials for quick demo — increase n_trials for real tuning)
    print("\n=== Optuna Optimization ===")
    study, optuna_results = optuna_tune(
        policy_name="threshold",
        param_space={
            "open_threshold": {"type": "int", "low": 1, "high": 15},
            "close_threshold": {"type": "int", "low": 1, "high": 8},
            "max_bays": {"type": "categorical", "choices": [2, 3, 4]},
        },
        n_trials=40,
        n_samples=30,
        objective="total_cost",
    )
    optuna_results.to_csv("results/tune_optuna.csv", index=False)
    plot_optuna_history(study, save_path="results/fig_optuna_history.png")
    print(f"Best params:  {study.best_params}")
    print(f"Best value:   {study.best_value:.2f}")

    print("\nAll tuning results saved to results/")
