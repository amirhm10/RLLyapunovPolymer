"""Analyze pretrained online TD3 critic-reset and handoff-calibration runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "online_pretrained_critic_reset_analysis_2026-06-12.md"
FIG_DIR = ROOT / "report" / "figures" / "2026-06-12_online_pretrained_critic_reset_analysis"
TABLE_DIR = ROOT / "report" / "tables" / "2026-06-12_online_pretrained_critic_reset_analysis"


CASES = [
    ("LMPC pretrained + gate", "OnlineTD3_LMPCPretrained_SafetyGate", True),
    ("LMPC pretrained no gate", "OnlineTD3_LMPCPretrained_NoSafetyGate", False),
    ("OF-MPC pretrained + gate", "OnlineTD3_OFMPCPretrained_SafetyGate", True),
    ("OF-MPC pretrained no gate", "OnlineTD3_OFMPCPretrained_NoSafetyGate", False),
]

BATCHES = {
    "old_noise": {
        "label": "bounded mixed, no reset, BC std 0.02",
        "runs": {
            "LMPC pretrained + gate": "20260611_000544",
            "LMPC pretrained no gate": "20260611_000541",
            "OF-MPC pretrained + gate": "20260611_000552",
            "OF-MPC pretrained no gate": "20260611_000548",
        },
    },
    "low_noise": {
        "label": "bounded mixed, no reset, BC std 0",
        "runs": {
            "LMPC pretrained + gate": "20260612_011534",
            "LMPC pretrained no gate": "20260612_011530",
            "OF-MPC pretrained + gate": "20260612_011542",
            "OF-MPC pretrained no gate": "20260612_011538",
        },
    },
    "critic_reset": {
        "label": "bounded mixed, critic reset, BC std 1e-4",
        "runs": {
            "LMPC pretrained + gate": "20260612_130549",
            "LMPC pretrained no gate": "20260612_130546",
            "OF-MPC pretrained + gate": "20260612_130557",
            "OF-MPC pretrained no gate": "20260612_130553",
        },
    },
    "handoff_calibrated": {
        "label": "critic reset, eps 1e-2, 10-episode actor-frozen handoff",
        "runs": {
            "LMPC pretrained + gate": "20260612_205458",
            "LMPC pretrained no gate": "20260612_205455",
            "OF-MPC pretrained + gate": "20260612_205504",
            "OF-MPC pretrained no gate": "20260612_205501",
        },
    },
    "handoff_eps1e3": {
        "label": "critic reset, eps 1e-3, 10-episode actor-frozen handoff",
        "runs": {
            "LMPC pretrained + gate": "20260612_231616",
            "LMPC pretrained no gate": "20260612_231608",
            "OF-MPC pretrained + gate": "20260612_231623",
            "OF-MPC pretrained no gate": "20260612_231619",
        },
    },
}

CASE_TO_ROOT = {case: root for case, root, _gate in CASES}
CASE_TO_GATE = {case: gate for case, _root, gate in CASES}
CASE_ORDER = [case for case, _root, _gate in CASES]
TARGET_BATCH = "handoff_eps1e3"
REFERENCE_BATCHES = ("handoff_calibrated", "critic_reset", "low_noise", "old_noise")
PHASE_ORDER = ["BC", "handoff", "early full", "mid full", "tail 50"]
PLOT_BATCHES = ("old_noise", "low_noise", "critic_reset", "handoff_calibrated", "handoff_eps1e3")
FIG_PREFIX = "pretrained_handoff_eps1e3"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def rel_report(path: Path) -> str:
    return path.relative_to(REPORT.parent).as_posix()


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if 0 < abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(title for _key, title in columns) + " |",
        "| " + " | ".join([":---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _title in columns) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stop_col(episode: pd.DataFrame) -> str:
    if "step_stop_exclusive" in episode.columns:
        return "step_stop_exclusive"
    if "step_end_exclusive" in episode.columns:
        return "step_end_exclusive"
    raise KeyError("No episode stop column found.")


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def safe_max(values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def arr_or_fill(arrays: np.lib.npyio.NpzFile, name: str, n: int, fill: float = 0.0) -> np.ndarray:
    if name in arrays.files:
        return np.asarray(arrays[name], dtype=float).reshape(-1)[:n]
    return np.full(n, fill, dtype=float)


def phase_slices(episode: pd.DataFrame, phase_cfg: dict[str, Any]) -> list[tuple[str, pd.DataFrame]]:
    n_ep = int(episode["episode"].max())
    bc_eps = max(0, int(phase_cfg.get("behavior_clone_teacher_episodes", 20)))
    handoff_eps = max(0, int(phase_cfg.get("handoff_episodes", 5)))
    handoff_start = bc_eps + 1
    handoff_end = bc_eps + handoff_eps
    early_start = handoff_end + 1
    early_end = min(n_ep, early_start + 49)
    mid_start = early_end + 1
    mid_end = max(mid_start, n_ep - 50)
    return [
        ("BC", episode.loc[episode["episode"].between(1, bc_eps)]),
        ("handoff", episode.loc[episode["episode"].between(handoff_start, handoff_end)]),
        ("early full", episode.loc[episode["episode"].between(early_start, early_end)]),
        ("mid full", episode.loc[episode["episode"].between(mid_start, mid_end)]),
        ("tail 50", episode.loc[episode["episode"].between(max(1, n_ep - 49), n_ep)]),
    ]


def load_run(case: str, batch: str, run_name: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    run_dir = ROOT / "results" / CASE_TO_ROOT[case] / run_name
    arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
    episode = pd.read_csv(run_dir / "episode_table.csv")
    summary = read_json(run_dir / "summary.json")
    run_summary = read_json(run_dir / "run_summary.json")
    cfg = run_summary.get("config", {})
    phase_cfg = cfg.get("training_phase_config", {})
    end = stop_col(episode)
    n = int(episode[end].max())

    reward_no_penalty = arr_or_fill(arrays, "reward_no_penalty", n)
    reward = arr_or_fill(arrays, "rewards", n)
    fallback_penalty = arr_or_fill(arrays, "fallback_penalty", n)
    actual_intervention = arr_or_fill(arrays, "actual_intervention_flags", n)
    diagnostic_unsafe = arr_or_fill(arrays, "diagnostic_unsafe_flags", n)
    accepted = arr_or_fill(arrays, "accepted_flags", n)
    fallback_verified = arr_or_fill(arrays, "fallback_verified_flags", n)
    teacher_gap = arr_or_fill(arrays, "bc_teacher_gap_inf", n, fill=np.nan)
    handoff_gap = arr_or_fill(arrays, "handoff_candidate_gap_inf", n, fill=np.nan)
    executed_gap = arr_or_fill(arrays, "executed_action_gap_inf", n, fill=np.nan)

    u_phys = np.asarray(arrays["u_applied_phys"], dtype=float)
    du = np.diff(u_phys, axis=0) if u_phys.ndim == 2 and u_phys.shape[0] > 1 else np.empty((0,))

    worst = episode.loc[episode["reward_no_penalty_mean"].idxmin()]
    row = {
        "case": case,
        "batch": batch,
        "batch_label": BATCHES[batch]["label"],
        "run_dir": rel(run_dir),
        "n_steps": n,
        "n_episodes": int(episode["episode"].max()),
        "episode_len": int(episode["n_steps"].iloc[0]),
        "initial_agent": Path(str(cfg.get("initial_agent_path") or "")).name,
        "teacher_source": cfg.get("teacher_source"),
        "safety_gate": bool(cfg.get("safety_gate_active", CASE_TO_GATE[case])),
        "target_selector_variant": cfg.get("target_selector_variant"),
        "rho_lyap": cfg.get("rho_lyap"),
        "lyap_eps": cfg.get("lyap_eps"),
        "pretrained_critic_reset": run_summary.get("pretrained_critic_reset", cfg.get("pretrained_critic_reset")),
        "critic_loaded_from_checkpoint": cfg.get("critic_loaded_from_checkpoint"),
        "bc_noise": phase_cfg.get("bc_behavior_noise"),
        "bc_std": phase_cfg.get("bc_exploration_std"),
        "bc_update_mode": phase_cfg.get("bc_update_mode"),
        "handoff_episodes": phase_cfg.get("handoff_episodes"),
        "handoff_update_mode": phase_cfg.get("handoff_update_mode"),
        "handoff_actor_bc_updates_per_step": phase_cfg.get("handoff_actor_bc_updates_per_step"),
        "handoff_std_end": phase_cfg.get("handoff_exploration_std_end"),
        "full_std_start": phase_cfg.get("full_rl_exploration_std_start"),
        "mean_reward_no_penalty": safe_mean(reward_no_penalty),
        "mean_training_reward": safe_mean(reward),
        "mean_fallback_penalty": safe_mean(fallback_penalty),
        "summary_output_rmse_mean": float(summary.get("output_rmse_mean", np.nan)),
        "tail50_reward_no_penalty": float(episode.tail(50)["reward_no_penalty_mean"].mean()),
        "tail50_training_reward": float(episode.tail(50)["reward_mean"].mean()),
        "tail50_output_rmse_mean": float(episode.tail(50)["output_rmse_mean"].mean()),
        "actual_intervention_rate": safe_mean(actual_intervention),
        "diagnostic_unsafe_rate": safe_mean(diagnostic_unsafe),
        "accepted_rate": safe_mean(accepted),
        "fallback_verified_rate": safe_mean(fallback_verified),
        "mean_abs_du_phys": safe_mean(np.abs(du)) if du.size else float("nan"),
        "max_abs_du_phys": safe_max(np.abs(du)) if du.size else float("nan"),
        "teacher_gap_mean": safe_mean(teacher_gap),
        "handoff_gap_mean": safe_mean(handoff_gap),
        "executed_gap_mean": safe_mean(executed_gap),
        "worst_episode": int(worst["episode"]),
        "worst_reward_no_penalty": float(worst["reward_no_penalty_mean"]),
        "worst_output_rmse_mean": float(worst["output_rmse_mean"]),
    }

    phase_rows: list[dict[str, Any]] = []
    for phase, part in phase_slices(episode, phase_cfg):
        if part.empty:
            continue
        start = int(part["step_start"].iloc[0])
        finish = int(part[end].iloc[-1])
        phase_rows.append(
            {
                "case": case,
                "batch": batch,
                "phase": phase,
                "n_episodes": int(len(part)),
                "reward_no_penalty": float(part["reward_no_penalty_mean"].mean()),
                "training_reward": float(part["reward_mean"].mean()),
                "fallback_penalty": float(part["fallback_penalty_mean"].mean()),
                "output_rmse": float(part["output_rmse_mean"].mean()),
                "actual_intervention_rate": float(part["actual_intervention_rate"].mean()),
                "diagnostic_unsafe_rate": float(part["diagnostic_unsafe_rate"].mean()),
                "teacher_gap": safe_mean(teacher_gap[start:finish]),
                "handoff_gap": safe_mean(handoff_gap[start:finish]),
                "executed_gap": safe_mean(executed_gap[start:finish]),
            }
        )

    meta = {
        "run_dir": run_dir,
        "arrays": arrays,
        "episode": episode,
        "summary": summary,
        "run_summary": run_summary,
        "config": cfg,
    }
    return row, pd.DataFrame(phase_rows), meta


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    phase_frames: list[pd.DataFrame] = []
    meta: dict[str, dict[str, Any]] = {}
    for case in CASE_ORDER:
        for batch, batch_info in BATCHES.items():
            run_name = batch_info["runs"][case]
            row, phase_df, case_meta = load_run(case, batch, run_name)
            rows.append(row)
            phase_frames.append(phase_df)
            meta[f"{case}::{batch}"] = case_meta

    metrics = pd.DataFrame(rows)
    phases = pd.concat(phase_frames, ignore_index=True)
    deltas: list[dict[str, Any]] = []
    for case in CASE_ORDER:
        target = metrics.loc[(metrics["case"] == case) & (metrics["batch"] == TARGET_BATCH)].iloc[0]
        for reference in REFERENCE_BATCHES:
            ref = metrics.loc[(metrics["case"] == case) & (metrics["batch"] == reference)].iloc[0]
            deltas.append(
                {
                    "case": case,
                    "comparison": f"{TARGET_BATCH} - {reference}",
                    "delta_mean_reward_no_penalty": target["mean_reward_no_penalty"] - ref["mean_reward_no_penalty"],
                    "delta_tail50_reward_no_penalty": target["tail50_reward_no_penalty"] - ref["tail50_reward_no_penalty"],
                    "delta_tail50_rmse": target["tail50_output_rmse_mean"] - ref["tail50_output_rmse_mean"],
                    "delta_actual_intervention_pp": 100.0
                    * (target["actual_intervention_rate"] - ref["actual_intervention_rate"]),
                    "delta_diagnostic_unsafe_pp": 100.0
                    * (target["diagnostic_unsafe_rate"] - ref["diagnostic_unsafe_rate"]),
                    "delta_mean_abs_du_phys": target["mean_abs_du_phys"] - ref["mean_abs_du_phys"],
                }
            )
            for phase in PHASE_ORDER:
                r_phase = phases.loc[
                    (phases["case"] == case)
                    & (phases["batch"] == TARGET_BATCH)
                    & (phases["phase"] == phase)
                ].iloc[0]
                ref_phase = phases.loc[
                    (phases["case"] == case)
                    & (phases["batch"] == reference)
                    & (phases["phase"] == phase)
                ].iloc[0]
                deltas[-1][f"delta_{phase}_reward"] = (
                    r_phase["reward_no_penalty"] - ref_phase["reward_no_penalty"]
                )
                deltas[-1][f"delta_{phase}_rmse"] = r_phase["output_rmse"] - ref_phase["output_rmse"]
    return metrics, phases, pd.DataFrame(deltas), meta


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().to_numpy()


def make_figures(metrics: pd.DataFrame, phases: pd.DataFrame, deltas: pd.DataFrame, meta: dict[str, dict[str, Any]]) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    current = metrics.loc[metrics["batch"] == TARGET_BATCH].set_index("case").loc[CASE_ORDER]
    current_phases = phases.loc[phases["batch"] == TARGET_BATCH].copy()
    colors = {
        "LMPC pretrained + gate": "#4c78a8",
        "LMPC pretrained no gate": "#54a24b",
        "OF-MPC pretrained + gate": "#e45756",
        "OF-MPC pretrained no gate": "#f58518",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True)
    for ax, case in zip(axes.flat, CASE_ORDER):
        episode = meta[f"{case}::{TARGET_BATCH}"]["episode"]
        y = moving_average(episode["reward_no_penalty_mean"].to_numpy(), window=5)
        ax.plot(episode["episode"], y, color=colors[case], linewidth=1.8)
        ax.axvspan(1, 20, color="#e8f0fb", alpha=0.45, linewidth=0)
        ax.axvspan(21, 30, color="#fff0cc", alpha=0.45, linewidth=0)
        ax.set_title(case)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward no penalty, 5-episode mean")
    axes[1, 0].set_ylabel("Reward no penalty, 5-episode mean")
    fig.suptitle("Current pretrained online TD3 reward traces, eps 1e-3", y=0.995)
    fig.tight_layout()
    paths["reward_traces"] = FIG_DIR / f"{FIG_PREFIX}_reward_traces.png"
    fig.savefig(paths["reward_traces"], dpi=180)
    plt.close(fig)

    phase_summary = (
        current_phases.pivot(index="case", columns="phase", values="reward_no_penalty")
        .reindex(index=CASE_ORDER, columns=PHASE_ORDER)
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    values = phase_summary.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(PHASE_ORDER)), PHASE_ORDER, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(CASE_ORDER)), CASE_ORDER)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, fmt(values[i, j], 1), ha="center", va="center", fontsize=8)
    ax.set_title("Current batch reward by phase")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Reward no penalty")
    fig.tight_layout()
    paths["phase_summary"] = FIG_DIR / f"{FIG_PREFIX}_phase_reward_heatmap.png"
    fig.savefig(paths["phase_summary"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    x = np.arange(len(CASE_ORDER))
    axes[0].bar(x, current["tail50_reward_no_penalty"], color=[colors[c] for c in CASE_ORDER])
    axes[0].set_xticks(x, CASE_ORDER, rotation=35, ha="right")
    axes[0].set_ylabel("Tail50 reward no penalty")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, current["tail50_output_rmse_mean"], color=[colors[c] for c in CASE_ORDER])
    axes[1].set_xticks(x, CASE_ORDER, rotation=35, ha="right")
    axes[1].set_ylabel("Tail50 output RMSE")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Tail performance for strict-epsilon calibrated handoff batch")
    fig.tight_layout()
    paths["tail_summary"] = FIG_DIR / f"{FIG_PREFIX}_tail_summary.png"
    fig.savefig(paths["tail_summary"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), sharex=True)
    for ax, case in zip(axes.flat, CASE_ORDER):
        episode = meta[f"{case}::{TARGET_BATCH}"]["episode"]
        local = episode.loc[episode["episode"].between(18, 45)]
        ax.plot(local["episode"], local["reward_no_penalty_mean"], marker="o", label="reward", color=colors[case])
        ax2 = ax.twinx()
        ax2.plot(local["episode"], local["output_rmse_mean"], marker="s", label="RMSE", color="#c94f4f")
        ax.axvspan(21, 30, color="#fff0cc", alpha=0.55, linewidth=0)
        ax.set_title(case)
        ax.grid(True, alpha=0.25)
        if ax in axes[:, 0]:
            ax.set_ylabel("Reward no penalty")
        if ax in axes[:, 1]:
            ax2.set_ylabel("Output RMSE")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Strict-epsilon handoff transition zoom", y=0.995)
    fig.tight_layout()
    paths["handoff_zoom"] = FIG_DIR / f"{FIG_PREFIX}_handoff_zoom.png"
    fig.savefig(paths["handoff_zoom"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    x = np.arange(len(CASE_ORDER))
    axes[0].bar(x, 100.0 * current["actual_intervention_rate"], color=[colors[c] for c in CASE_ORDER])
    axes[1].bar(x, 100.0 * current["diagnostic_unsafe_rate"], color=[colors[c] for c in CASE_ORDER])
    for ax in axes:
        ax.set_xticks(x, CASE_ORDER, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Actual safety intervention %")
    axes[1].set_ylabel("Diagnostic would-activate %")
    axes[0].set_title("Safety-gate runners")
    axes[1].set_title("No-gate monitor signal")
    fig.tight_layout()
    paths["safety"] = FIG_DIR / f"{FIG_PREFIX}_safety_diagnostics.png"
    fig.savefig(paths["safety"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12.2, 9.5), sharex=True)
    for row_idx, case in enumerate(CASE_ORDER):
        arrays = meta[f"{case}::{TARGET_BATCH}"]["arrays"]
        y = np.asarray(arrays["y_system"], dtype=float)[:-1]
        sp = np.asarray(arrays["y_sp_phys_store"], dtype=float)
        n = min(y.shape[0], sp.shape[0])
        start = max(0, n - 8000)
        t = np.arange(start, n)
        for j, ylabel in enumerate(("Output 1", "Output 2")):
            ax = axes[row_idx, j]
            ax.plot(t, y[start:n, j], linewidth=1.0, label="plant")
            ax.plot(t, sp[start:n, j], linestyle="--", linewidth=1.2, label="setpoint")
            ax.set_title(f"{case} - {ylabel}")
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and j == 0:
                ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    fig.suptitle("Strict-epsilon calibrated handoff tail tracking snapshots, final 10 episodes", y=0.995)
    fig.tight_layout()
    paths["tracking"] = FIG_DIR / f"{FIG_PREFIX}_tail_tracking.png"
    fig.savefig(paths["tracking"], dpi=180)
    plt.close(fig)

    return paths


def make_tables(metrics: pd.DataFrame, phases: pd.DataFrame, deltas: pd.DataFrame) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    current_metrics = metrics.loc[metrics["batch"] == TARGET_BATCH].copy()
    current_phases = phases.loc[phases["batch"] == TARGET_BATCH].copy()
    paths = {
        "metrics": TABLE_DIR / f"{FIG_PREFIX}_current_metrics.csv",
        "phases": TABLE_DIR / f"{FIG_PREFIX}_current_phase_metrics.csv",
    }
    current_metrics.to_csv(paths["metrics"], index=False)
    current_phases.to_csv(paths["phases"], index=False)
    return paths


def report_text(metrics: pd.DataFrame, phases: pd.DataFrame, deltas: pd.DataFrame, figs: dict[str, Path], tables: dict[str, Path]) -> str:
    current = metrics.loc[metrics["batch"] == TARGET_BATCH].set_index("case").loc[CASE_ORDER]
    current_phase = phases.loc[phases["batch"] == TARGET_BATCH]
    config = current.iloc[0]

    data_rows = []
    perf_rows = []
    phase_rows = []
    transition_rows = []
    for case in CASE_ORDER:
        r = current.loc[case]
        data_rows.append(
            {
                "Case": case,
                "Run": str(r["run_dir"]).split("/")[-1],
                "Agent": r["initial_agent"],
                "Teacher": r["teacher_source"],
                "Gate": "active" if r["safety_gate"] else "monitor only",
                "Handoff": int(r["handoff_episodes"]),
            }
        )
        perf_rows.append(
            {
                "Case": case,
                "Mean Rnp": fmt(r["mean_reward_no_penalty"]),
                "Tail Rnp": fmt(r["tail50_reward_no_penalty"]),
                "Tail RMSE": fmt(r["tail50_output_rmse_mean"]),
                "Gate %": fmt(100.0 * r["actual_intervention_rate"]),
                "Diag %": fmt(100.0 * r["diagnostic_unsafe_rate"]),
                "Worst ep": int(r["worst_episode"]),
            }
        )
        row = {"Case": case}
        for phase in ("BC", "handoff", "early full", "tail 50"):
            part = current_phase.loc[
                (current_phase["case"] == case) & (current_phase["phase"] == phase),
                "reward_no_penalty",
            ]
            row[phase] = fmt(part.iloc[0]) if not part.empty else "-"
        phase_rows.append(row)
        transition_rows.append(
            {
                "Case": case,
                "Worst ep": int(r["worst_episode"]),
                "Worst Rnp": fmt(r["worst_reward_no_penalty"]),
                "Worst RMSE": fmt(r["worst_output_rmse_mean"]),
                "Mean abs du": fmt(r["mean_abs_du_phys"]),
            }
        )

    best_tail_reward_case = str(current["tail50_reward_no_penalty"].idxmax())
    best_tail_rmse_case = str(current["tail50_output_rmse_mean"].idxmin())
    best_tail_reward = current.loc[best_tail_reward_case, "tail50_reward_no_penalty"]
    best_tail_rmse = current.loc[best_tail_rmse_case, "tail50_output_rmse_mean"]
    gate_cases = current.loc[current["safety_gate"]]
    no_gate_cases = current.loc[~current["safety_gate"]]
    max_gate_rate = float(gate_cases["actual_intervention_rate"].max()) if not gate_cases.empty else 0.0
    max_diag_rate = float(no_gate_cases["diagnostic_unsafe_rate"].max()) if not no_gate_cases.empty else 0.0

    return rf"""# Pretrained Online TD3 Critic-Reset Final Batch Analysis

Date: 2026-06-12, current-only update 2026-06-13

## Question

This report now keeps only the final pretrained online TD3 setup:
pretrained actor loading, critic reset, tiny BC exploration (`1e-4`),
10-episode actor-frozen handoff, bounded-mixed Direct LMPC target selector, and
`lyap_eps=1e-3`.

Short answer: the handoff catastrophe is fixed for this final setup. The
handoff window stays on an ordinary reward scale in all four pretrained runs,
and no run shows the earlier collapse behavior. The best tail reward is
`{fmt(best_tail_reward)}` from `{best_tail_reward_case}`, and the lowest tail
RMSE is `{fmt(best_tail_rmse)}` from `{best_tail_rmse_case}`.

## Data Used

{md_table(data_rows, [
        ("Case", "Case"),
        ("Run", "Run"),
        ("Agent", "Agent"),
        ("Teacher", "Teacher"),
        ("Gate", "Gate"),
        ("Handoff", "Handoff"),
    ])}

All four runs used:

- disturbance-only plant mode, 300 episodes, and 400-step setpoint blocks;
- `target_selector_variant = bounded_mixed_u0p1_x0p1`;
- `rho_lyap={fmt(config['rho_lyap'])}`, `lyap_eps={fmt(config['lyap_eps'])}`;
- pretrained actor loaded from checkpoint and critic reset before online TD3;
- BC update mode `critic_td_plus_actor_bc`;
- handoff update mode `{config['handoff_update_mode']}`;
- handoff actor BC updates per step `{fmt(config['handoff_actor_bc_updates_per_step'], 0)}`;
- full-RL exploration std decays from `{fmt(config['full_std_start'])}` to `{fmt(config['handoff_std_end'])}`.

## Method

The TD3 state remains the scaled augmented observer state, setpoint, and
previous input:

$$
s_k =
\left[
\operatorname{{scale}}(\hat z_k)^\top,
\operatorname{{scale}}(y_{{sp,k}})^\top,
\operatorname{{scale}}(u_{{k-1}})^\top
\right]^\top .
$$

The actor output $a_k\in[-1,1]^{{n_u}}$ is mapped to the admissible input
deviation interval by

$$
u_k^\pi =
u_{{\min}} + \frac{{a_k+1}}{{2}}(u_{{\max}}-u_{{\min}}).
$$

During BC, replay stores the executed teacher-plus-noise transition while the
actor demo buffer stores the clean teacher action:

$$
u_k^{{\mathrm{{exec}}}} = u_k^T+\xi_k,\qquad
\xi_k\sim\mathcal N(0,10^{{-4}}I).
$$

During handoff, the executed action is the teacher-policy blend

$$
u_k^{{\mathrm{{exec}}}} =
\alpha_k u_k^T + (1-\alpha_k)u_k^\pi ,
\qquad \alpha_k \downarrow 0,
$$

and actor-gradient TD3 updates remain off. The critic learns on the blended
closed-loop distribution, while the actor is still supervised toward the clean
teacher action. Full TD3 actor-gradient updates begin after handoff.

The safety-gate certificate remains the model-based practical contraction test

$$
V(\hat x_{{k+1}}-x_s)
\le
\rho V(\hat x_k-x_s)+\epsilon ,
$$

with $\rho=0.99$ and $\epsilon=10^{{-3}}$.

## Current Batch Performance

{md_table(perf_rows, [
        ("Case", "Case"),
        ("Mean Rnp", "Mean Rnp"),
        ("Tail Rnp", "Tail Rnp"),
        ("Tail RMSE", "Tail RMSE"),
        ("Gate %", "Gate %"),
        ("Diag %", "Diag %"),
        ("Worst ep", "Worst ep"),
    ])}

![Tail summary]({rel_report(figs['tail_summary'])})

![Reward traces]({rel_report(figs['reward_traces'])})

The final setup is controlled across all four pretrained cases. The no-gate
runners have the best tail reward, while the safety-gate runners are more
conservative because fallback is actually applied. The maximum actual
intervention rate among gated runs is `{fmt(100.0 * max_gate_rate)}%`. The
maximum no-gate diagnostic would-activate rate is `{fmt(100.0 * max_diag_rate)}%`.

## Phase Behavior

{md_table(phase_rows, [
        ("Case", "Case"),
        ("BC", "BC"),
        ("handoff", "Handoff"),
        ("early full", "Early"),
        ("tail 50", "Tail50"),
    ])}

![Phase reward heatmap]({rel_report(figs['phase_summary'])})

The handoff phase is no longer the failure point. Handoff rewards remain near
the BC reward scale for the no-gate runners and are moderately lower for the
gated runners, which is consistent with a stricter safety certificate rather
than a learning collapse.

{md_table(transition_rows, [
        ("Case", "Case"),
        ("Worst ep", "Worst ep"),
        ("Worst Rnp", "Worst Rnp"),
        ("Worst RMSE", "Worst RMSE"),
        ("Mean abs du", "Mean abs du"),
    ])}

![Handoff zoom]({rel_report(figs['handoff_zoom'])})

The remaining transients are localized to the handoff/release window. The next
algorithmic risk is therefore not BC or handoff itself, but the release into
full actor-gradient TD3 under a stricter safety filter.

## Safety And Monitor Behavior

![Safety diagnostics]({rel_report(figs['safety'])})

The safety-gate runners show actual interventions because Direct LMPC can
replace a TD3 candidate action. The no-gate runners show zero actual
intervention by construction, while their Direct LMPC monitor signal records
how often the gate would have been active. This separation should stay in all
future reports: actual fallback and diagnostic would-activate are different
quantities.

The Direct LMPC fallback tracks the raw setpoint in the MPC objective, but the
Lyapunov certificate is centered on the bounded mixed target $(x_s,u_s,y_s)$.
Therefore tracking quality and certificate activity remain separate metrics.

## Tail Tracking

![Tail tracking]({rel_report(figs['tracking'])})

The final tracking snapshots show the same conclusion as the scalar metrics:
all four runs recover after the handoff/release window and settle into a
similar tail regime, with the no-gate cases retaining the best reward.

## Interpretation

The final setup is good enough to close the handoff-catastrophe debugging loop.

First, critic reset should stay. It keeps the useful pretrained actor prior and
removes the offline-to-online Q mismatch.

Second, the 10-episode actor-frozen handoff should stay for pretrained agents.
It gives the critic time to learn on the blended online distribution before the
actor follows TD3 gradients.

Third, `lyap_eps=1e-3` is usable with the calibrated handoff. It keeps the
safety filter active without reintroducing the catastrophic handoff reward
failure.

## Recommended Next Experiment

Keep this final setup fixed and test only a post-handoff release refinement:

1. Keep critic reset, BC std `1e-4`, 10-episode handoff, and `lyap_eps=1e-3`.
2. Add a 3-5 episode post-handoff actor-gradient ramp or delayed actor-gradient
   start.
3. Compare episodes 21-40, tail-50 reward, actual intervention rate, diagnostic
   would-activate rate, and mean absolute input movement.

## Report Artifacts

- Metrics table: `{rel(tables['metrics'])}`
- Phase table: `{rel(tables['phases'])}`
- Figures: `{rel(FIG_DIR)}`

## Limitations

- These are single-seed training runs, not seed-averaged final evidence.
- `reward_no_penalty` is the fairer control-performance metric; training reward
  includes gate/fallback shaping for safety-gate runs.
- Frozen saved-agent evaluation is still needed before claiming final
  deployment performance.
"""

    current = metrics.loc[metrics["batch"] == TARGET_BATCH].set_index("case").loc[CASE_ORDER]
    previous = metrics.loc[metrics["batch"] == "handoff_calibrated"].set_index("case").loc[CASE_ORDER]
    critic_reset = metrics.loc[metrics["batch"] == "critic_reset"].set_index("case").loc[CASE_ORDER]
    delta_prev = deltas.loc[deltas["comparison"] == f"{TARGET_BATCH} - handoff_calibrated"].set_index("case").loc[CASE_ORDER]
    delta_critic = deltas.loc[deltas["comparison"] == f"{TARGET_BATCH} - critic_reset"].set_index("case").loc[CASE_ORDER]
    delta_low = deltas.loc[deltas["comparison"] == f"{TARGET_BATCH} - low_noise"].set_index("case").loc[CASE_ORDER]
    current_phase = phases.loc[phases["batch"] == TARGET_BATCH]
    previous_phase = phases.loc[phases["batch"] == "handoff_calibrated"]
    critic_phase = phases.loc[phases["batch"] == "critic_reset"]

    data_rows = []
    for case in CASE_ORDER:
        r = current.loc[case]
        data_rows.append(
            {
                "Case": case,
                "Run": str(r["run_dir"]).split("/")[-1],
                "Agent": r["initial_agent"],
                "Teacher": r["teacher_source"],
                "Gate": "active" if r["safety_gate"] else "monitor only",
                "Handoff": int(r["handoff_episodes"]),
            }
        )

    perf_rows = []
    for case in CASE_ORDER:
        r = current.loc[case]
        perf_rows.append(
            {
                "Case": case,
                "Mean Rnp": fmt(r["mean_reward_no_penalty"]),
                "Tail Rnp": fmt(r["tail50_reward_no_penalty"]),
                "Tail RMSE": fmt(r["tail50_output_rmse_mean"]),
                "Gate %": fmt(100.0 * r["actual_intervention_rate"]),
                "Diag %": fmt(100.0 * r["diagnostic_unsafe_rate"]),
                "Worst ep": int(r["worst_episode"]),
            }
        )

    delta_rows = []
    for case in CASE_ORDER:
        d = delta_prev.loc[case]
        delta_rows.append(
            {
                "Case": case,
                "Mean Rnp": fmt(d["delta_mean_reward_no_penalty"]),
                "Handoff": fmt(d["delta_handoff_reward"]),
                "Early": fmt(d["delta_early full_reward"]),
                "Tail Rnp": fmt(d["delta_tail50_reward_no_penalty"]),
                "Tail RMSE": fmt(d["delta_tail50_rmse"]),
                "Gate pp": fmt(d["delta_actual_intervention_pp"]),
                "Diag pp": fmt(d["delta_diagnostic_unsafe_pp"]),
            }
        )

    phase_rows = []
    for case in CASE_ORDER:
        row = {"Case": case}
        for phase in ("BC", "handoff", "early full", "tail 50"):
            value = current_phase.loc[
                (current_phase["case"] == case) & (current_phase["phase"] == phase),
                "reward_no_penalty",
            ].iloc[0]
            row[phase] = fmt(value)
        phase_rows.append(row)

    transition_rows = []
    for case in CASE_ORDER:
        r = current.loc[case]
        prev = previous.loc[case]
        prev_handoff = previous_phase.loc[
            (previous_phase["case"] == case) & (previous_phase["phase"] == "handoff"),
            "reward_no_penalty",
        ].iloc[0]
        cur_handoff = current_phase.loc[
            (current_phase["case"] == case) & (current_phase["phase"] == "handoff"),
            "reward_no_penalty",
        ].iloc[0]
        transition_rows.append(
            {
                "Case": case,
                "Prev worst": int(prev["worst_episode"]),
                "New worst": int(r["worst_episode"]),
                "Prev handoff": fmt(prev_handoff),
                "New handoff": fmt(cur_handoff),
                "New worst R": fmt(r["worst_reward_no_penalty"]),
            }
        )

    low_rows = []
    for case in CASE_ORDER:
        d = delta_low.loc[case]
        low_rows.append(
            {
                "Case": case,
                "Mean Rnp": fmt(d["delta_mean_reward_no_penalty"]),
                "Early": fmt(d["delta_early full_reward"]),
                "Tail Rnp": fmt(d["delta_tail50_reward_no_penalty"]),
                "Tail RMSE": fmt(d["delta_tail50_rmse"]),
            }
        )

    config = current.iloc[0]
    lmpc_gate = current.loc["LMPC pretrained + gate"]
    lmpc_nogate = current.loc["LMPC pretrained no gate"]
    of_gate = current.loc["OF-MPC pretrained + gate"]
    of_nogate = current.loc["OF-MPC pretrained no gate"]
    best_tail_reward_case = str(current["tail50_reward_no_penalty"].idxmax())
    best_tail_rmse_case = str(current["tail50_output_rmse_mean"].idxmin())
    best_tail_reward = current.loc[best_tail_reward_case, "tail50_reward_no_penalty"]
    best_tail_rmse = current.loc[best_tail_rmse_case, "tail50_output_rmse_mean"]
    max_gate_rate = max(lmpc_gate["actual_intervention_rate"], of_gate["actual_intervention_rate"])

    return f"""# Pretrained Online TD3 Critic-Reset Batch Analysis

Date: 2026-06-12, extended 2026-06-13

## Question

This report now includes the stricter `lyap_eps=1e-3` rerun of the four
pretrained online TD3 cases. This is the batch with pretrained actor loading,
critic reset, tiny BC exploration (`1e-4`), 10-episode actor-frozen handoff,
and the bounded-mixed Direct LMPC target selector.

Short answer: returning to `lyap_eps=1e-3` keeps the calibrated-handoff fix.
The catastrophic OF-MPC handoff failure does not return. Compared with the
relaxed `lyap_eps=1e-2` batch, the stricter certificate increases actual gate
and monitor activation as expected, but it does not create the old multi-thousand
handoff collapse. The best tail performance in this single-seed batch is
`{best_tail_reward_case}`, while the gated runs are slightly more conservative.

## Paper-Consistency Frame

The interpretation remains aligned with the process-control framing in our
paper and the close comparator by Khodaverdian et al.: MPC/OF-MPC provide the
engineering reference behavior, TD3 supplies a learned policy prior, and the
Lyapunov gate is a supervisory certification layer rather than a reward-shaping
device. Because the certificate is computed for the identified
output-disturbance model and a bounded mixed Direct LMPC target, the stability
claim should remain practical and model-based:

$$
V(\\hat x_{{k+1}}-x_s)
\\le
\\rho V(\\hat x_k-x_s)+\\epsilon .
$$

For this new pretrained batch, $\\rho=0.99$ and
$\\epsilon={fmt(config['lyap_eps'])}$. This restores the default bounded-mixed
Direct LMPC tolerance after the temporary relaxed `1e-2` experiment. The safety
language should therefore emphasize the same practical, model-based first-step
contraction tube used by the rest of the bounded-mixed runs.

## Data Used

| Case | Run | Agent | Teacher | Gate | Handoff |
| :--- | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {row['Case']} | {row['Run']} | {row['Agent']} | {row['Teacher']} | {row['Gate']} | {row['Handoff']} |"
        for row in data_rows
    ) + f"""

All four current runs used:

- disturbance-only plant mode, 300 episodes, and 400-step setpoint blocks;
- bounded mixed Direct LMPC target selector, `bounded_mixed_u0p1_x0p1`;
- `rho_lyap={fmt(config['rho_lyap'])}`, `lyap_eps={fmt(config['lyap_eps'])}`;
- pretrained actor loaded from checkpoint and critic reset before online TD3;
- BC update mode `critic_td_plus_actor_bc`;
- handoff update mode `{config['handoff_update_mode']}`;
- handoff actor BC updates per step `{fmt(config['handoff_actor_bc_updates_per_step'], 0)}`.

The direct reference is the immediately previous relaxed-epsilon calibrated
batch, `handoff_calibrated`: same critic reset, same 10-episode handoff, same
actor-frozen handoff updates, but `lyap_eps=1e-2`. The older `critic_reset`
batch remains the stress-test reference because it used `lyap_eps=1e-3`, only a
5-episode handoff, and full TD3 actor-gradient updates during handoff.

## Method Reconstruction

The TD3 state remains the scaled augmented observer state, setpoint, and
previous input:

$$
s_k =
\\left[
\\operatorname{{scale}}(\\hat z_k)^\\top,
\\operatorname{{scale}}(y_{{sp,k}})^\\top,
\\operatorname{{scale}}(u_{{k-1}})^\\top
\\right]^\\top .
$$

The actor output $a_k\\in[-1,1]^{{n_u}}$ is mapped to the admissible input
deviation interval by

$$
u_k^\\pi =
u_{{\\min}} + \\frac{{a_k+1}}{{2}}(u_{{\\max}}-u_{{\\min}}).
$$

The online BC phase is now:

$$
u_k^{{\\mathrm{{exec}}}} =
u_k^T + \\xi_k,
\\qquad
\\xi_k\\sim\\mathcal N(0,10^{{-4}}I),
$$

with replay receiving $(s_k,a_k^{{\\mathrm{{exec}}}},r_k,s_{{k+1}})$ for critic TD
learning, while the actor-demo buffer receives the clean teacher action
$a_k^T$. The actor BC loss is

$$
\\mathcal L_{{BC}}(\\theta)=
\\mathbb E_{{(s,a_T)\\sim\\mathcal D_{{BC}}}}
\\left\\|\\pi_\\theta(s)-a_T\\right\\|_2^2 .
$$

The calibrated handoff phase executes

$$
u_k^{{\\mathrm{{exec}}}} =
\\alpha_k u_k^T + (1-\\alpha_k)u_k^\\pi,
\\qquad
\\alpha_k \\downarrow 0,
$$

but keeps TD3 actor-gradient updates off. During handoff, the critic still
learns from the executed blended transitions and the actor remains supervised
toward the clean teacher action. Full TD3 actor-gradient updates begin only
after handoff.

## Current Batch Performance

{md_table(perf_rows, [
        ("Case", "Case"),
        ("Mean Rnp", "Mean Rnp"),
        ("Tail Rnp", "Tail Rnp"),
        ("Tail RMSE", "Tail RMSE"),
        ("Gate %", "Gate %"),
        ("Diag %", "Diag %"),
        ("Worst ep", "Worst ep"),
    ])}

![Tail summary]({rel_report(figs['tail_summary'])})

The four current runs remain in a controlled regime. The best tail reward is
`{fmt(best_tail_reward)}` from `{best_tail_reward_case}`, and the lowest tail
RMSE is `{fmt(best_tail_rmse)}` from `{best_tail_rmse_case}`. The safety-gate
versions are more conservative under the stricter epsilon, with actual
interventions up to `{fmt(100.0 * max_gate_rate)}%`, but the old OF-MPC handoff
collapse does not reappear.

![Reward traces]({rel_report(figs['reward_traces'])})

## Change From Relaxed-Epsilon Calibrated Batch

Positive reward deltas are better. Negative RMSE deltas are better. This table
isolates the epsilon change because both batches use critic reset, tiny BC
noise, 10-episode handoff, and actor-gradient freezing during handoff.

{md_table(delta_rows, [
        ("Case", "Case"),
        ("Mean Rnp", "Mean Rnp"),
        ("Handoff", "Handoff"),
        ("Early", "Early"),
        ("Tail Rnp", "Tail Rnp"),
        ("Tail RMSE", "Tail RMSE"),
        ("Gate pp", "Gate pp"),
        ("Diag pp", "Diag pp"),
    ])}

![Phase delta]({rel_report(figs['phase_delta'])})

The stricter epsilon mainly changes safety activity, not the qualitative
learning story. Relative to the relaxed batch, the handoff reward changes by
`{fmt(delta_prev.loc['LMPC pretrained + gate']['delta_handoff_reward'])}` for
LMPC + gate, `{fmt(delta_prev.loc['LMPC pretrained no gate']['delta_handoff_reward'])}`
for LMPC no gate, `{fmt(delta_prev.loc['OF-MPC pretrained + gate']['delta_handoff_reward'])}`
for OF-MPC + gate, and `{fmt(delta_prev.loc['OF-MPC pretrained no gate']['delta_handoff_reward'])}`
for OF-MPC no gate. These are ordinary-scale changes, not a return to the
previous catastrophic handoff behavior.

For no-gate runners, the executed trajectories are unchanged by epsilon because
Direct LMPC is monitor-only. Their reward and RMSE rows therefore match the
relaxed batch, while diagnostic would-activate rates increase.

## Phase Diagnosis

{md_table(phase_rows, [
        ("Case", "Case"),
        ("BC", "BC"),
        ("handoff", "Handoff"),
        ("early full", "Early"),
        ("tail 50", "Tail50"),
    ])}

The BC phase remains teacher-dominated and almost identical across the four
runs. The 10-episode handoff is still controlled under `lyap_eps=1e-3`. The
original 5-episode critic-reset batch had OF-MPC handoff averages of roughly
`{fmt(critic_phase.loc[(critic_phase['case'] == 'OF-MPC pretrained + gate') & (critic_phase['phase'] == 'handoff'), 'reward_no_penalty'].iloc[0])}`
with gate and
`{fmt(critic_phase.loc[(critic_phase['case'] == 'OF-MPC pretrained no gate') & (critic_phase['phase'] == 'handoff'), 'reward_no_penalty'].iloc[0])}`
without gate; this rerun is nowhere near that failure mode. The remaining
transients are localized to the handoff/release window: the no-gate cases and
LMPC-gated case still peak at episode 31, while the OF-MPC-gated case peaks
inside handoff at episode 25.

{md_table(transition_rows, [
        ("Case", "Case"),
        ("Prev worst", "eps1e2 worst"),
        ("New worst", "New worst"),
        ("Prev handoff", "eps1e2 handoff"),
        ("New handoff", "New handoff"),
        ("New worst R", "New worst R"),
    ])}

![Handoff zoom]({rel_report(figs['handoff_zoom'])})

Mechanistically, the actor-frozen handoff is still doing what it was supposed
to do. The critic learns on the blended distribution before the actor is allowed
to follow TD3 policy gradients. The stricter epsilon does not undo that fix,
although it makes the safety layer more active after the policy starts moving
away from the teacher.

## Safety And Monitor Behavior

![Safety diagnostics]({rel_report(figs['safety'])})

Returning to `lyap_eps=1e-3` makes the safety certificate stricter than the
relaxed `1e-2` batch. The no-gate diagnostic would-activate rates are now
`{fmt(100.0*lmpc_nogate['diagnostic_unsafe_rate'])}%` for LMPC-pretrained no
gate and `{fmt(100.0*of_nogate['diagnostic_unsafe_rate'])}%` for
OF-MPC-pretrained no gate. The safety-gate runners therefore apply more
fallback than the relaxed batch, but the intervention rate remains a reported
deployment tradeoff rather than a training failure by itself.

The safety gate still does not optimize only for raw tracking. The Direct LMPC
fallback solves a tracking problem toward the raw setpoint in the objective,
but the contraction certificate is centered on the bounded mixed target
$(x_s,u_s,y_s)$. Therefore tracking quality and safety-certificate activity
must remain separate reported quantities.

## Tail Tracking

![Tail tracking]({rel_report(figs['tracking'])})

The tail tracking plots support the scalar metrics: the learned controllers
recover after the transition and settle into a similar regime. The no-gate
runners retain slightly better tail reward, while safety-gate runners buy a
deployment mechanism with a small performance cost.

## Context Against The No-Reset Low-Noise Batch

{md_table(low_rows, [
        ("Case", "Case"),
        ("Mean Rnp", "Mean Rnp"),
        ("Early", "Early"),
        ("Tail Rnp", "Tail Rnp"),
        ("Tail RMSE", "Tail RMSE"),
    ])}

Against the no-reset low-noise batch, the new setup remains strongly better in
the early online learning region. This preserves the earlier critic-reset
conclusion: the pretrained actor is useful, but the offline critic should not
be trusted as the initial online Q-function for the shaped closed-loop reward.

## Interpretation

The new evidence supports five conclusions.

First, critic reset should stay. It avoids the offline-to-online Q mismatch.

Second, the calibrated handoff should stay for pretrained agents. It directly
removed the OF-MPC handoff collapse.

Third, the handoff fix survives the return to `lyap_eps=1e-3`. This matters
because it separates the handoff improvement from the temporary relaxed
certificate.

Fourth, the stricter epsilon increases actual or diagnostic gate activity. That
is expected and should be reported as a control/safety tradeoff, not hidden
inside reward.

Fifth, the remaining weak point is the handoff/release boundary, especially for
gated runs under the stricter certificate. That is now a smaller and more
localized issue than the previous handoff failure.

## Recommended Next Experiment

Keep the current `lyap_eps=1e-3` setup as the preferred stability-consistent
pretrained online schedule and test one focused refinement:

1. Keep critic reset, BC std `1e-4`, and 10-episode calibrated handoff.
2. Add a 3-5 episode post-handoff actor-gradient ramp:
   critic TD remains active, actor BC may decay, and TD3 actor-gradient updates
   start at reduced frequency or after a short delay.
3. Compare specifically episodes 21-40, tail-50 reward, diagnostic unsafe rate,
   and actual gate intervention rate.
4. Keep `lyap_eps=1e-3` fixed during this ramp test so the release mechanism is
   isolated from the safety-tube parameter.

The current result is good enough that I would avoid changing BC noise again
until the episode-31 release behavior is understood.

## Report Artifacts

- Metrics table: `{rel(tables['metrics'])}`
- Phase table: `{rel(tables['phases'])}`
- Delta table: `{rel(tables['deltas'])}`
- Figures: `{rel(FIG_DIR)}`

## Limitations

- These are single-seed training runs, not seed-averaged final evidence.
- The latest strict-epsilon batch is a clean comparison against the relaxed
  calibrated batch for epsilon, but all runs are still single-seed.
- `reward_no_penalty` is the fairer control-performance metric; training reward
  includes gate/fallback shaping.
- Frozen saved-agent evaluation is still needed before claiming final
  deployment performance.
"""


def main() -> None:
    metrics, phases, deltas, meta = collect()
    tables = make_tables(metrics, phases, deltas)
    figs = make_figures(metrics, phases, deltas, meta)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(metrics, phases, deltas, figs, tables), encoding="utf-8")
    print(f"Wrote {rel(REPORT)}")
    print(f"Wrote {rel(FIG_DIR)}")
    print(f"Wrote {rel(TABLE_DIR)}")


if __name__ == "__main__":
    main()
