"""
Policy evaluation via Monte Carlo simulation.

Builds on simulate.run() to produce aggregate statistics and figures
across many independent replications. Use this to compare policies
and assess variability.

Usage:
    python src/evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .simulate import run


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy_name: str,
    policy_params: dict = None,
    cost_params: dict = None,
    n_samples: int = 100,
    horizon: float = 480.0,
    arrival_rate: float = 10.0,
    service_rate: float = 8.0,
    base_seed: int = 0,
) -> pd.DataFrame:
    """
    Run n_samples independent replications of one policy.

    Each replication uses a different seed (base_seed + i) so results are
    independent but reproducible.

    Returns
    -------
    pd.DataFrame
        One row per replication with summary statistics.
    """
    policy_params = policy_params or {}
    cost_params = cost_params or {}
    records = []

    for i in range(n_samples):
        trace = run(
            policy_name=policy_name,
            policy_params=policy_params,
            cost_params=cost_params,
            horizon=horizon,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            seed=base_seed + i,
        )
        records.append({
            "replication": i,
            "total_cost": trace["cost"].sum(),
            "cars_served": trace["cars_served"].iloc[-1],
            "avg_wait": trace["avg_wait"].iloc[-1],
            "avg_queue": trace["queue_length"].mean(),
            "max_queue": trace["queue_length"].max(),
            "avg_bays_open": trace["bays_open"].mean(),
        })

    return pd.DataFrame(records)


def compare_policies(
    policies: dict,
    n_samples: int = 100,
    horizon: float = 480.0,
    arrival_rate: float = 10.0,
    service_rate: float = 8.0,
    base_seed: int = 0,
) -> pd.DataFrame:
    """
    Evaluate multiple policies under the same conditions.

    Parameters
    ----------
    policies : dict
        Keys are policy labels (str), values are dicts with keys:
          - 'name'   (str): key into POLICIES registry
          - 'params' (dict): policy_params passed to run()
          - 'cost_params' (dict, optional)

    Returns
    -------
    pd.DataFrame
        One row per (policy, replication) with summary statistics.
        Includes a 'policy' column for grouping.
    """
    frames = []
    for label, cfg in policies.items():
        df = evaluate_policy(
            policy_name=cfg["name"],
            policy_params=cfg.get("params", {}),
            cost_params=cfg.get("cost_params", {}),
            n_samples=n_samples,
            horizon=horizon,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            base_seed=base_seed,
        )
        df.insert(0, "policy", label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summary_stats(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a results DataFrame (from compare_policies) into a
    mean ± std summary table, grouped by policy.
    """
    metrics = ["total_cost", "cars_served", "avg_wait", "avg_queue", "max_queue", "avg_bays_open"]
    rows = []
    for policy, group in results.groupby("policy"):
        row = {"policy": policy}
        for m in metrics:
            row[f"{m}_mean"] = group[m].mean()
            row[f"{m}_std"] = group[m].std()
            row[f"{m}_p5"] = group[m].quantile(0.05)
            row[f"{m}_p95"] = group[m].quantile(0.95)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })


def plot_cost_distribution(results: pd.DataFrame, save_path: str = None):
    """Box plot of total cost distribution per policy."""
    _style()
    policies = results["policy"].unique()
    data = [results.loc[results["policy"] == p, "total_cost"].values for p in policies]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=policies, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6),
               medianprops=dict(color="black", linewidth=2))
    ax.set_ylabel("Total Cost")
    ax.set_title("Total Cost Distribution by Policy")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_avg_wait_distribution(results: pd.DataFrame, save_path: str = None):
    """Box plot of average wait time per policy."""
    _style()
    policies = results["policy"].unique()
    data = [results.loc[results["policy"] == p, "avg_wait"].values for p in policies]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=policies, patch_artist=True,
               boxprops=dict(facecolor="#55A868", alpha=0.6),
               medianprops=dict(color="black", linewidth=2))
    ax.set_ylabel("Average Wait (min)")
    ax.set_title("Average Customer Wait Time by Policy")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_queue_over_time(
    policy_configs: dict,
    horizon: float = 480.0,
    arrival_rate: float = 10.0,
    service_rate: float = 8.0,
    n_samples: int = 30,
    base_seed: int = 0,
    save_path: str = None,
):
    """
    Mean queue length over time with shaded 10th–90th percentile band,
    one line per policy.
    """
    _style()
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, cfg) in enumerate(policy_configs.items()):
        traces = []
        for i in range(n_samples):
            trace = run(
                policy_name=cfg["name"],
                policy_params=cfg.get("params", {}),
                horizon=horizon,
                arrival_rate=arrival_rate,
                service_rate=service_rate,
                seed=base_seed + i,
            )
            traces.append(trace["queue_length"].values)

        arr = np.array(traces)
        t = trace["time"].values
        mean = arr.mean(axis=0)
        p10 = np.percentile(arr, 10, axis=0)
        p90 = np.percentile(arr, 90, axis=0)

        c = colors[idx % len(colors)]
        ax.plot(t, mean, label=label, color=c)
        ax.fill_between(t, p10, p90, alpha=0.15, color=c)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Queue Length")
    ax.set_title("Mean Queue Length Over Time (shaded = 10th–90th pct)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_summary_bar(summary: pd.DataFrame, metric: str = "total_cost_mean", save_path: str = None):
    """Bar chart comparing policies on a single summary metric."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    err_col = metric.replace("_mean", "_std")

    ax.bar(summary["policy"], summary[metric],
           yerr=summary.get(err_col, None),
           capsize=5, color="#4C72B0", alpha=0.75, edgecolor="black")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Policy Comparison: {metric.replace('_', ' ').title()}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    policies = {
        "Always 1 Bay": {
            "name": "always_open",
            "params": {"bays": 1},
        },
        "Always 2 Bays": {
            "name": "always_open",
            "params": {"bays": 2},
        },
        "Threshold Policy": {
            "name": "threshold",
            "params": {"open_threshold": 5, "close_threshold": 2, "max_bays": 3},
        },
    }

    print("Running Monte Carlo evaluation...")
    results = compare_policies(policies, n_samples=200)
    results.to_csv("results/policy_evaluation.csv", index=False)

    stats = summary_stats(results)
    stats.to_csv("results/policy_summary.csv", index=False)
    print(stats[["policy", "total_cost_mean", "total_cost_std", "avg_wait_mean", "cars_served_mean"]].to_string(index=False))

    plot_cost_distribution(results, save_path="results/fig_cost_distribution.png")
    plot_avg_wait_distribution(results, save_path="results/fig_wait_distribution.png")
    plot_queue_over_time(policies, save_path="results/fig_queue_over_time.png")
    plot_summary_bar(stats, metric="total_cost_mean", save_path="results/fig_cost_bar.png")

    print("\nFigures saved to results/")
