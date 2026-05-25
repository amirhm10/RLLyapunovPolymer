"""Create reproducible figure assets for the CSChE Lyapunov/RL slide draft.

The script reads only existing report and result artifacts. It writes copied or
new presentation figures under ``csche/figures``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "csche" / "figures"
REPORT_FIG = ROOT / "report" / "figures" / "2026-05-20_gamma099_eps1e2_latest_analysis"

COLD = ROOT / "results" / "ColdStart" / "20260520_204513" / "bounded_hard_u_prev_0p1_xs_prev_0p1"
PRE = ROOT / "results" / "Pretrain" / "20260520_205230" / "bounded_hard_u_prev_0p1_xs_prev_0p1"
DIRECT = ROOT / "results" / "directLyap" / "20260520_204510" / "lyap_mix_u0p1_x0p1_lex"
DIRECT_MPC = ROOT / "results" / "directLyap" / "20260520_204510" / "mpc_only"

MAROON = "#7a003c"
BLUE = "#1c4a75"
GREEN = "#2c765c"
RED = "#9c3a34"
GRAY = "#6b7280"


def copy_if_exists(src: Path, dst_name: str) -> None:
    if src.exists():
        shutil.copy2(src, OUT / dst_name)


def load_metrics() -> dict:
    metrics_path = REPORT_FIG / "metrics_gamma099_eps1e2_latest.json"
    if metrics_path.exists():
        text = metrics_path.read_text(encoding="utf-8")
        text = text.replace("NaN", "null")
        return json.loads(text)

    # Fallback path if the report-side derived metrics are not present.
    rows = {}
    for label, table in {
        "Cold RL": COLD.parent / "comparison_table.csv",
        "Pretrained RL": PRE.parent / "comparison_table.csv",
        "Direct LMPC": DIRECT.parent / "comparison_table.csv",
    }.items():
        df = pd.read_csv(table)
        if label == "Cold RL":
            row = df[df["case_name"] != "mpc_only"].iloc[0]
        elif label == "Pretrained RL":
            row = df[df["case_name"] != "mpc_only"].iloc[0]
        else:
            row = df[df["case_name"] == "lyap_mix_u0p1_x0p1_lex"].iloc[0]
        rows[label] = {
            "reward_mean": float(row["reward_mean"]),
            "mean_rmse": float(row["output_rmse_mean"]),
            "ms_per_step": 1000.0 * float(row["wall_clock_seconds_per_step"]),
            "fallback_rate": float(row.get("fallback_rate", 0.0)),
            "actual_intervention_rate": float(row.get("actual_intervention_rate", 0.0)),
        }
    return rows


def plot_slide_summary(metrics: dict) -> None:
    labels = ["Cold RL", "Pretrained RL", "Direct LMPC"]
    rmse = [metrics[k]["mean_rmse"] for k in labels]
    runtime = [metrics[k]["ms_per_step"] for k in labels]
    intervention = [
        100.0 * metrics["Cold RL"]["fallback_rate"],
        100.0 * metrics["Pretrained RL"]["fallback_rate"],
        100.0 * metrics["Direct LMPC"].get("actual_intervention_rate", 0.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1))
    colors = [BLUE, GREEN, RED]
    panels = [
        ("Full-horizon mean RMSE", rmse, "scaled output RMSE"),
        ("Gate/fallback use", intervention, "percent of steps"),
        ("Runtime", runtime, "ms per control step"),
    ]
    for ax, (title, vals, ylabel) in zip(axes, panels):
        ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="x", rotation=22, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02 * max(vals), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Latest analyzed 300-episode disturbance run", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "csche_key_result_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_target_selector_mechanism() -> None:
    data = np.load(DIRECT / "arrays.npz", allow_pickle=True)
    n_ep = 800
    start = data["y_target_phys_store"].shape[0] - n_ep
    end = data["y_target_phys_store"].shape[0]
    t = np.arange(n_ep)

    y = data["y_system"][start:end, :]
    y_sp = data["y_tracking_phys_store"][start:end, :]
    y_s = data["y_target_phys_store"][start:end, :]
    residual = data["target_residual_total_norm"][start:end]
    margin = data["contraction_margin"][start:end]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.6), sharex=True)
    names = [r"$\eta$", r"$T$"]
    units = ["viscosity-like output", "K"]
    for idx in range(2):
        ax = axes[idx]
        ax.plot(t, y_sp[:, idx], color="black", linewidth=1.4, label=r"raw setpoint $y_{sp}$")
        ax.plot(t, y_s[:, idx], color=MAROON, linewidth=1.2, label=r"admissible target $y_s$")
        ax.plot(t, y[:, idx], color=BLUE if idx == 0 else GREEN, linewidth=1.1, label=r"plant output $y$")
        ax.set_ylabel(f"{names[idx]}\n{units[idx]}", fontsize=9)
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(ncol=3, loc="upper center", fontsize=8, frameon=True)

    ax = axes[2]
    ax.plot(t, residual, color=RED, linewidth=1.1, label="target residual")
    ax2 = ax.twinx()
    ax2.plot(t, margin, color=GRAY, linewidth=0.9, alpha=0.85, label="contraction margin")
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    ax2.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("Step in final episode", fontsize=9)
    ax.set_ylabel("target residual", fontsize=9)
    ax2.set_ylabel("Lyapunov margin", fontsize=9)
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper center", ncol=2, fontsize=8, frameon=True)

    fig.suptitle("Target selector mechanism in the final direct-LMPC episode", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "csche_target_selector_mechanism.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_authority_phase_summary() -> None:
    phase_rows = []
    phases = [
        ("BC 1-20", 1, 20),
        ("Handoff 21-25", 21, 25),
        ("Online 26-299", 26, 299),
        ("Final eval 300", 300, 300),
    ]
    for label, folder in [("Cold RL", COLD), ("Pretrained RL", PRE)]:
        df = pd.read_csv(folder / "episode_table.csv")
        for phase, lo, hi in phases:
            sub = df[(df["episode"] >= lo) & (df["episode"] <= hi)]
            phase_rows.append(
                {
                    "case": label,
                    "phase": phase,
                    "fallback_rate": 100.0 * sub["fallback_count"].sum() / sub["n_steps"].sum(),
                    "rmse": sub["output_rmse_mean"].mean(),
                }
            )
    rows = pd.DataFrame(phase_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    x = np.arange(len(phases))
    width = 0.36
    for j, case in enumerate(["Cold RL", "Pretrained RL"]):
        vals = [rows[(rows.case == case) & (rows.phase == p[0])]["fallback_rate"].iloc[0] for p in phases]
        axes[0].bar(x + (j - 0.5) * width, vals, width=width, label=case, color=[BLUE, GREEN][j])
        vals2 = [rows[(rows.case == case) & (rows.phase == p[0])]["rmse"].iloc[0] for p in phases]
        axes[1].bar(x + (j - 0.5) * width, vals2, width=width, label=case, color=[BLUE, GREEN][j])
    for ax, title, ylabel in [
        (axes[0], "Fallback frequency by learning phase", "fallback percent"),
        (axes[1], "Tracking by learning phase", "mean episode RMSE"),
    ]:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in phases], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "csche_phase_authority_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_episode_tracking_summary() -> None:
    cases = [
        ("Cold RL", COLD / "arrays.npz", BLUE),
        ("Pretrained RL", PRE / "arrays.npz", GREEN),
        ("Direct LMPC", DIRECT / "arrays.npz", RED),
    ]
    n_ep = 800
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.8), gridspec_kw={"width_ratios": [2.2, 1.0]})
    t = np.arange(n_ep)
    tail_rows = []

    y_sp_phys = None
    for label, path, color in cases:
        data = np.load(path, allow_pickle=True)
        start = data["y_system"].shape[0] - 1 - n_ep
        y = data["y_system"][start : start + n_ep, :]
        if "y_sp_phys_store" in data.files:
            y_sp = data["y_sp_phys_store"][start : start + n_ep, :]
        else:
            y_sp = data["y_tracking_phys_store"][start : start + n_ep, :]
        y_sp_phys = y_sp if y_sp_phys is None else y_sp_phys
        for idx in range(2):
            axes[idx, 0].plot(t, y[:, idx], color=color, linewidth=1.05, label=label)
        tail_err = np.abs(y[-100:, :] - y_sp[-100:, :]).mean(axis=0)
        tail_rows.append((label, tail_err[0], tail_err[1], color))

    out_names = [r"$\eta$", r"$T$"]
    ylabels = ["viscosity-like output", "temperature (K)"]
    for idx in range(2):
        ax = axes[idx, 0]
        ax.plot(t, y_sp_phys[:, idx], color="black", linewidth=1.35, linestyle="--", label="raw setpoint")
        ax.set_ylabel(f"{out_names[idx]}\n{ylabels[idx]}", fontsize=9)
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(ncol=4, fontsize=7.8, loc="upper center", frameon=True)
        if idx == 1:
            ax.set_xlabel("Step in final episode", fontsize=9)

    labels = [r[0] for r in tail_rows]
    colors = [r[3] for r in tail_rows]
    eta_tail = [r[1] for r in tail_rows]
    temp_tail = [r[2] for r in tail_rows]
    axes[0, 1].bar(labels, eta_tail, color=colors, edgecolor="black", linewidth=0.5)
    axes[1, 1].bar(labels, temp_tail, color=colors, edgecolor="black", linewidth=0.5)
    for ax, title, vals in [
        (axes[0, 1], r"Final 100-step $\eta$ offset", eta_tail),
        (axes[1, 1], r"Final 100-step $T$ offset", temp_tail),
    ]:
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("mean abs. error", fontsize=8.5)
        ax.tick_params(axis="x", rotation=22, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.025 * max(vals), f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Final evaluation episode: tracking and steady offset", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "csche_final_episode_tracking_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    for name in [
        "performance_runtime_summary.png",
        "rl_authority_diagnostics.png",
        "tail_offset_comparison.png",
        "last_episode_tracking_primary_methods.png",
        "mpc_only_would_be_activation.png",
        "target_diagnostics_summary.png",
        "activation_contraction_episode_counts.png",
        "reward_penalty_scale.png",
    ]:
        copy_if_exists(REPORT_FIG / name, name)

    copy_if_exists(ROOT / "StatsControl2026" / "figures" / "logo.png", "logo.png")
    copy_if_exists(ROOT / "StatsControl2026" / "figures" / "CSTR.png", "CSTR.png")

    metrics = load_metrics()
    plot_slide_summary(metrics)
    plot_target_selector_mechanism()
    plot_authority_phase_summary()
    plot_final_episode_tracking_summary()


if __name__ == "__main__":
    main()
