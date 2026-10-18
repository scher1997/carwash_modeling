"""
Visual verification of the simulation logic.

Produces two outputs for one episode:
  1. A static "flow audit" figure: cumulative arrivals vs. departures
     (a cumulative-flow / Little's-law diagram), cars-in-system bookkeeping,
     and bay utilisation, with physical-invariant violations highlighted.
  2. An animated GIF of cars moving queue -> bays -> exit, minute by minute.

Invariants checked each step (a real car wash must satisfy all of them):
  - conservation:     arrivals_so_far - served_so_far == queue + bays_busy
  - capacity:         bays_busy <= bays_open, except closing bays finishing
                      a car (busy may exceed open, but only by draining)
  - non-negative:     queue_length >= 0
  - work-conserving:  no idle open bay while cars are waiting
  - no-vanishing:     closing bays never ejects a car mid-wash

Usage:
    uv run python -m src.visualize
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle

from .simulate import run

OUT_DIR = os.path.join("reports", "figures")


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def check_invariants(df: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per invariant (True = violated) to a trace."""
    df = df.copy()
    # Cars that arrived before step t, minus cars that left before step t.
    df["cum_arrivals"] = df["arrivals"].cumsum().shift(fill_value=0)
    df["in_system_expected"] = df["cum_arrivals"] - df["cars_served"]
    df["in_system_state"] = df["queue_length"] + df["bays_busy"]

    df["viol_conservation"] = df["in_system_expected"] != df["in_system_state"]
    # A car in service can only leave by completing, so busy_{t+1} >= busy_t - completions_t.
    still_in_service = (df["bays_busy"] - df["completions"]).shift(fill_value=0)
    # Over capacity is only legal while closing bays drain (no new starts).
    over = df["bays_busy"] > df["bays_open"]
    df["viol_capacity"] = (over & (df["bays_busy"] > still_in_service)) | (df["bays_busy"] < 0)
    df["viol_negative"] = df["queue_length"] < 0
    df["viol_idle_bay"] = (df["bays_busy"] < df["bays_open"]) & (df["queue_length"] > 0)
    df["viol_vanish"] = df["bays_busy"] < still_in_service
    return df


def summarize_invariants(df: pd.DataFrame) -> dict:
    cols = [c for c in df.columns if c.startswith("viol_")]
    return {c.removeprefix("viol_"): int(df[c].sum()) for c in cols}


# ---------------------------------------------------------------------------
# Static flow audit
# ---------------------------------------------------------------------------

def _ensure_checked(df: pd.DataFrame) -> pd.DataFrame:
    """Run check_invariants() if it hasn't been, so plots never show a false 'OK'."""
    return df if "viol_conservation" in df.columns else check_invariants(df)


def plot_flow_audit(df: pd.DataFrame, path: str) -> None:
    df = _ensure_checked(df)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.4]})
    t = df["time"]

    # 1. Cumulative flow: the vertical gap between the curves is cars in system.
    ax = axes[0]
    ax.step(t, df["cum_arrivals"], where="post", label="Cumulative arrivals", color="#2a6fdb")
    ax.step(t, df["cars_served"], where="post", label="Cumulative departures", color="#e07b00")
    ax.fill_between(t, df["cars_served"], df["cum_arrivals"], step="post",
                    color="#2a6fdb", alpha=0.12, label="Cars in system (gap)")
    ax.set_ylabel("Cars")
    ax.set_title("Cumulative flow: arrivals vs departures")
    ax.legend(loc="upper left", frameon=False)

    # 2. Bookkeeping: what the state says is in the system vs what flows imply.
    ax = axes[1]
    ax.step(t, df["in_system_expected"], where="post", color="black", lw=1.5,
            label="Expected (arrivals - departures)")
    ax.step(t, df["in_system_state"], where="post", color="#c0392b", lw=1.2, ls="--",
            label="State (queue + bays_busy)")
    bad = df["viol_conservation"]
    ax.scatter(t[bad], df.loc[bad, "in_system_state"], s=6, color="#c0392b", zorder=3)
    ax.set_ylabel("Cars in system")
    ax.set_title(f"Conservation check - {int(bad.sum())} of {len(df)} steps disagree")
    ax.legend(loc="upper left", frameon=False)

    # 3. Bays: open vs busy, with idle-while-queued steps marked.
    ax = axes[2]
    ax.step(t, df["bays_open"], where="post", color="#7f8c8d", label="Bays open")
    ax.step(t, df["bays_busy"], where="post", color="#27ae60", label="Bays busy")
    idle = df["viol_idle_bay"]
    ax.scatter(t[idle], df.loc[idle, "bays_busy"], s=8, color="#c0392b", zorder=3,
               label="Idle bay while cars wait")
    ax.set_ylabel("Bays")
    ax.set_xlabel("Minute")
    ax.set_yticks(range(0, int(df["bays_open"].max()) + 2))
    ax.legend(loc="upper left", frameon=False, ncol=3)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animated flow
# ---------------------------------------------------------------------------

def animate_flow(df: pd.DataFrame, path: str, minutes: int = 120, fps: int = 6) -> None:
    """Animate queue -> bays -> exit for the first `minutes` steps."""
    df = _ensure_checked(df)
    frames = df.iloc[:minutes].reset_index(drop=True)
    max_bays = int(df["bays_open"].max())
    max_queue = max(10, int(frames["queue_length"].max()))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def draw(i):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.2)
        ax.axis("off")
        r = frames.iloc[i]
        q, busy, open_ = int(r["queue_length"]), int(r["bays_busy"]), int(r["bays_open"])

        # Entrance: arrivals this minute
        ax.text(0.4, 3.85, "Arrivals", ha="center", fontsize=9, color="#555")
        for k in range(int(r["arrivals"])):
            ax.add_patch(Circle((0.4, 3.3 - 0.35 * k), 0.13, color="#2a6fdb"))

        # Queue lane
        ax.add_patch(Rectangle((1.0, 1.6), 3.8, 0.9, fill=False, ec="#999", ls="--"))
        ax.text(2.9, 2.7, f"Queue: {q}", ha="center", fontsize=10)
        slot = 3.8 / max_queue
        for k in range(q):
            ax.add_patch(Circle((4.8 - slot * (k + 0.5), 2.05), min(0.16, slot * 0.4),
                                color="#2a6fdb"))
        ax.annotate("", xy=(5.3, 2.05), xytext=(4.85, 2.05),
                    arrowprops=dict(arrowstyle="->", color="#999"))

        # Bays
        for b in range(max_bays):
            y = 3.2 - b * 1.1
            is_open = b < open_
            is_busy = b < busy
            ec = "#333" if is_open else "#ccc"
            fc = "#eafaf0" if is_busy else ("white" if is_open else "#f2f2f2")
            ax.add_patch(Rectangle((5.4, y - 0.4), 1.6, 0.8, fc=fc, ec=ec, lw=1.5))
            closing = is_busy and not is_open  # finishing its last car
            if closing:
                ec, fc = "#e07b00", "#fff4e5"
                ax.add_patch(Rectangle((5.4, y - 0.4), 1.6, 0.8, fc=fc, ec=ec, lw=1.5))
            label = ("closing" if closing else "busy") if is_busy else ("idle" if is_open else "closed")
            ax.text(6.2, y + 0.5, f"Bay {b + 1}: {label}", ha="center", fontsize=8,
                    color=ec)
            if is_busy:
                ax.add_patch(Circle((6.2, y), 0.18, color="#e07b00" if closing else "#27ae60"))

        # Exit
        ax.annotate("", xy=(8.0, 2.05), xytext=(7.1, 2.05),
                    arrowprops=dict(arrowstyle="->", color="#999"))
        ax.text(8.9, 2.4, "Served", ha="center", fontsize=9, color="#555")
        ax.text(8.9, 1.85, f"{int(r['cars_served'])}", ha="center", fontsize=16, color="#e07b00")
        ax.text(8.9, 1.4, f"+{int(r['completions'])} this min", ha="center", fontsize=8,
                color="#555")

        # Status line with invariant flags for this step
        flags = [n.removeprefix("viol_") for n in frames.columns
                 if n.startswith("viol_") and bool(r[n])]
        status = "OK" if not flags else "VIOLATION: " + ", ".join(flags)
        ax.text(0.1, 0.25, f"t = {int(r['time'])} min", fontsize=10)
        ax.text(9.9, 0.25, status, ha="right", fontsize=9,
                color="#27ae60" if not flags else "#c0392b")
        ax.text(0.1, 0.7, f"expected in system: {int(r['in_system_expected'])}   "
                          f"state says: {int(r['in_system_state'])}", fontsize=8, color="#555")

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    df = run(policy_name="threshold",
             policy_params={"open_threshold": 5, "close_threshold": 2, "max_bays": 3})
    df = check_invariants(df)

    print("Invariant violations (steps):")
    for name, n in summarize_invariants(df).items():
        print(f"  {name:<13} {n:>4} / {len(df)}")

    audit_path = os.path.join(OUT_DIR, "flow_audit.png")
    gif_path = os.path.join(OUT_DIR, "flow_animation.gif")
    plot_flow_audit(df, audit_path)
    animate_flow(df, gif_path)
    print(f"Saved: {audit_path}\nSaved: {gif_path}")
