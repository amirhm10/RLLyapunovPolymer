"""Analyze online TD3 GART safety-gate and fallback results.

This script reads the latest completed cold-start and OF-MPC-pretrained online
TD3 runs, computes tracking and gate/fallback diagnostics, and writes compact
tables plus figures used by the corresponding Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "report" / "figures" / "2026-06-17_online_gart_gate_fallback"

RUNNERS = [
    {
        "family": "Cold start",
        "case": "Cold start + gate",
        "runner": "OnlineTD3_ColdStart_SafetyGate",
        "gate": True,
        "slug": "cold_gate",
    },
    {
        "family": "Cold start",
        "case": "Cold start no gate",
        "runner": "OnlineTD3_ColdStart_NoSafetyGate",
        "gate": False,
        "slug": "cold_no_gate",
    },
    {
        "family": "OF-MPC pretrained",
        "case": "OF-MPC pretrained + gate",
        "runner": "OnlineTD3_OFMPCPretrained_SafetyGate",
        "gate": True,
        "slug": "ofmpc_gate",
    },
    {
        "family": "OF-MPC pretrained",
        "case": "OF-MPC pretrained no gate",
        "runner": "OnlineTD3_OFMPCPretrained_NoSafetyGate",
        "gate": False,
        "slug": "ofmpc_no_gate",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_complete_run(runner: str, min_steps: int = 200_000) -> Path:
    runner_dir = RESULTS / runner
    candidates: list[tuple[str, Path]] = []
    for child in runner_dir.iterdir():
        if not child.is_dir():
            continue
        summary = child / "summary.json"
        arrays = child / "arrays.npz"
        episode = child / "episode_table.csv"
        run_summary = child / "run_summary.json"
        if not (summary.exists() and arrays.exists() and episode.exists() and run_summary.exists()):
            continue
        try:
            data = _read_json(summary)
        except Exception:
            continue
        if int(data.get("n_steps", 0)) >= min_steps:
            candidates.append((child.name, child))
    if not candidates:
        raise FileNotFoundError(f"No complete run found for {runner}")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite))


def _safe_min(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmin(finite))


def _sum_count(summary: dict[str, Any], *keys: str) -> int:
    return int(sum(int(summary.get(key, 0) or 0) for key in keys))


def _mode_count(summary: dict[str, Any], mode: str) -> int:
    return int((summary.get("mode_counts") or {}).get(mode, 0) or 0)


def _phase_bounds(config: dict[str, Any], episode_len: int, n_steps: int) -> dict[str, tuple[int, int]]:
    phase = config.get("training_phase_config") or {}
    n_episodes = int(config.get("n_episodes", max(1, n_steps // max(episode_len, 1))))
    bc_episodes = int(phase.get("behavior_clone_teacher_episodes", 0) or 0)
    handoff_episodes = int(phase.get("handoff_episodes", 0) or 0)
    bc_stop = min(bc_episodes * episode_len, n_steps)
    handoff_stop = min((bc_episodes + handoff_episodes) * episode_len, n_steps)
    tail_start = max(0, (n_episodes - 50) * episode_len)
    return {
        "BC": (0, bc_stop),
        "handoff": (bc_stop, handoff_stop),
        "full_RL": (handoff_stop, n_steps),
        "last_50": (tail_start, n_steps),
    }


def _phase_for_episode(episode: int, config: dict[str, Any]) -> str:
    phase = config.get("training_phase_config") or {}
    bc = int(phase.get("behavior_clone_teacher_episodes", 0) or 0)
    handoff = int(phase.get("handoff_episodes", 0) or 0)
    if int(episode) <= bc:
        return "BC"
    if int(episode) <= bc + handoff:
        return "handoff"
    return "full_RL"


def _output_error(arrays: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_minus_y_sp_phys_store" in arrays.files:
        return np.asarray(arrays["y_minus_y_sp_phys_store"], dtype=float)
    return np.asarray(arrays["y_system"][1:], dtype=float) - np.asarray(arrays["y_sp_phys_store"], dtype=float)


def _target_error(arrays: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_minus_y_target_phys_store" in arrays.files:
        return np.asarray(arrays["y_minus_y_target_phys_store"], dtype=float)
    return np.asarray(arrays["y_system"][1:], dtype=float) - np.asarray(arrays["y_target_phys_store"], dtype=float)


def _rmse(err: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nanmean(np.asarray(err, dtype=float) ** 2, axis=0))


def _mae(err: np.ndarray) -> np.ndarray:
    return np.nanmean(np.abs(np.asarray(err, dtype=float)), axis=0)


def _max_abs(err: np.ndarray) -> np.ndarray:
    return np.nanmax(np.abs(np.asarray(err, dtype=float)), axis=0)


def _tail(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
    return np.asarray(arr)[int(start) : int(stop)]


def _npz_series(arrays: np.lib.npyio.NpzFile, name: str, n_steps: int, default: float = np.nan) -> np.ndarray:
    if name in arrays.files:
        return np.asarray(arrays[name], dtype=float)
    return np.full(int(n_steps), float(default), dtype=float)


def _phase_metric_rows(case: dict[str, Any], arrays: np.lib.npyio.NpzFile, config: dict[str, Any]) -> list[dict[str, Any]]:
    n_steps = int(arrays["rewards"].shape[0])
    episode_len = int(n_steps // max(int(config.get("n_episodes", 300)), 1))
    bounds = _phase_bounds(config, episode_len, n_steps)
    rows = []
    err_sp = _output_error(arrays)
    err_target = _target_error(arrays)
    target_quality_ok = _npz_series(arrays, "target_quality_ok_flags", n_steps)
    candidate_contraction_ok = _npz_series(arrays, "candidate_first_step_lyap_ok_flags", n_steps)
    for phase, (start, stop) in bounds.items():
        if stop <= start:
            continue
        sl = slice(start, stop)
        rmse_sp = _rmse(err_sp[sl])
        rmse_target = _rmse(err_target[sl])
        rows.append(
            {
                "case": case["case"],
                "family": case["family"],
                "gate": bool(case["gate"]),
                "phase": phase,
                "step_start": int(start),
                "step_stop": int(stop),
                "n_steps": int(stop - start),
                "reward_mean": _safe_mean(arrays["rewards"][sl]),
                "reward_no_penalty_mean": _safe_mean(arrays["reward_no_penalty"][sl]),
                "fallback_penalty_mean": _safe_mean(arrays["fallback_penalty"][sl]),
                "fallback_penalty_sum": float(np.nansum(arrays["fallback_penalty"][sl])),
                "rmse_eta": float(rmse_sp[0]),
                "rmse_T": float(rmse_sp[1]),
                "rmse_mean": float(np.nanmean(rmse_sp)),
                "target_tracking_rmse_mean": float(np.nanmean(rmse_target)),
                "diagnostic_unsafe_rate": _safe_mean(arrays["diagnostic_unsafe_flags"][sl]),
                "actual_intervention_rate": _safe_mean(arrays["actual_intervention_flags"][sl]),
                "fallback_verified_rate": _safe_mean(arrays["fallback_verified_flags"][sl]),
                "reward_fallback_active_rate": _safe_mean(arrays["reward_fallback_active_flags"][sl]),
                "target_success_rate": _safe_mean(arrays["target_success_flags"][sl]),
                "target_quality_ok_rate": _safe_mean(target_quality_ok[sl]),
                "candidate_contraction_ok_rate": _safe_mean(candidate_contraction_ok[sl]),
                "applied_contraction_ok_rate": _safe_mean(
                    arrays["first_step_contraction_satisfied_applied_flags"][sl]
                ),
                "executed_action_gap_inf_mean": _safe_mean(arrays["executed_action_gap_inf"][sl]),
                "executed_action_gap_inf_max": _safe_max(arrays["executed_action_gap_inf"][sl]),
                "target_mismatch_inf_mean": _safe_mean(arrays["target_mismatch_inf"][sl]),
                "target_mismatch_inf_max": _safe_max(arrays["target_mismatch_inf"][sl]),
                "contraction_margin_candidate_max": _safe_max(arrays["contraction_margin_candidate"][sl]),
                "contraction_margin_candidate_p95": float(
                    np.nanpercentile(arrays["contraction_margin_candidate"][sl], 95)
                ),
            }
        )
    return rows


def _load_gate_columns(step_path: Path) -> pd.DataFrame:
    needed = {
        "correction_mode",
        "reject_reason",
        "candidate_bounds_ok",
        "candidate_move_ok",
        "candidate_lyap_ok",
        "candidate_first_step_lyap_ok",
        "actual_intervention",
        "fallback_mpc_active",
        "fallback_verified",
        "fallback_solver_status",
        "solver_status",
        "target_failure",
        "target_success",
        "target_rejection_reason",
        "target_usable_for_lmpc",
        "target_error_inf",
        "target_mismatch_inf",
        "contraction_margin_candidate",
        "final_lyap_ok",
        "reward_fallback_active",
        "policy_phase",
    }
    return pd.read_csv(step_path, usecols=lambda name: name in needed, low_memory=False)


def _gate_detail_rows(case: dict[str, Any], step: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = max(int(len(step)), 1)
    for col, label in [
        ("correction_mode", "correction_mode"),
        ("reject_reason", "reject_reason"),
        ("fallback_solver_status", "fallback_solver_status"),
        ("target_rejection_reason", "target_rejection_reason"),
    ]:
        if col not in step.columns:
            continue
        counts = step[col].fillna("none").astype(str).value_counts()
        for value, count in counts.items():
            rows.append(
                {
                    "case": case["case"],
                    "family": case["family"],
                    "gate": bool(case["gate"]),
                    "field": label,
                    "value": value,
                    "count": int(count),
                    "rate": float(count / n),
                }
            )
    return rows


def _summary_row(case: dict[str, Any], run_dir: Path, summary: dict[str, Any], arrays: np.lib.npyio.NpzFile, config: dict[str, Any]) -> dict[str, Any]:
    err_sp = _output_error(arrays)
    err_target = _target_error(arrays)
    rmse_sp = _rmse(err_sp)
    rmse_target = _rmse(err_target)
    mae_sp = _mae(err_sp)
    max_sp = _max_abs(err_sp)
    n_steps = int(summary.get("n_steps", arrays["rewards"].shape[0]))
    section16_projection = _mode_count(summary, "gart_section16_projected")
    verified_fallback = _mode_count(summary, "fallback_mpc_verified")
    target_hold = _mode_count(summary, "gart_target_not_usable_hold_prev")
    solver_hold = _mode_count(summary, "gart_solver_fail_hold_prev")
    intervention_count = int(round(float(summary.get("actual_intervention_rate", 0.0) or 0.0) * n_steps))
    diagnostic_unsafe_count = int(round(float(summary.get("diagnostic_unsafe_rate", 0.0) or 0.0) * n_steps))
    candidate_contraction_ok = _npz_series(arrays, "candidate_first_step_lyap_ok_flags", n_steps)
    return {
        "case": case["case"],
        "family": case["family"],
        "gate": bool(case["gate"]),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "controller_mode": config.get("controller_mode"),
        "teacher_source": config.get("teacher_source"),
        "fallback_controller": config.get("fallback_controller"),
        "pretrain_source": config.get("pretrain_source"),
        "actor_loaded_from_checkpoint": config.get("actor_loaded_from_checkpoint"),
        "pretrained_critic_reset": config.get("pretrained_critic_reset"),
        "n_steps": n_steps,
        "n_episodes": int(config.get("n_episodes", 0) or 0),
        "episode_len": int(n_steps // max(int(config.get("n_episodes", 1) or 1), 1)),
        "reward_mean": float(summary.get("reward_mean", np.nan)),
        "reward_no_penalty_mean": float(summary.get("reward_no_penalty_mean", np.nan)),
        "reward_augmented_mean": float(summary.get("reward_augmented_mean", np.nan)),
        "fallback_penalty_mean": float(summary.get("fallback_penalty_mean", np.nan)),
        "fallback_penalty_sum": float(summary.get("fallback_penalty_sum", np.nan)),
        "rmse_eta": float(rmse_sp[0]),
        "rmse_T": float(rmse_sp[1]),
        "rmse_mean": float(np.nanmean(rmse_sp)),
        "mae_eta": float(mae_sp[0]),
        "mae_T": float(mae_sp[1]),
        "max_abs_eta": float(max_sp[0]),
        "max_abs_T": float(max_sp[1]),
        "target_tracking_rmse_mean": float(np.nanmean(rmse_target)),
        "target_tracking_rmse_eta": float(rmse_target[0]),
        "target_tracking_rmse_T": float(rmse_target[1]),
        "target_success_rate": float(summary.get("target_quality_ok_rate", np.nan)),
        "target_solver_success_rate": float(summary.get("n_target_success", 0) / n_steps),
        "target_failure_count": int(summary.get("n_target_failures", 0) or 0),
        "target_failure_rate": float(int(summary.get("n_target_failures", 0) or 0) / n_steps),
        "diagnostic_unsafe_count": diagnostic_unsafe_count,
        "diagnostic_unsafe_rate": float(summary.get("diagnostic_unsafe_rate", np.nan)),
        "actual_intervention_count": intervention_count,
        "actual_intervention_rate": float(summary.get("actual_intervention_rate", np.nan)),
        "section16_projection_count": section16_projection,
        "section16_projection_rate": float(section16_projection / n_steps),
        "verified_fallback_count": verified_fallback,
        "verified_fallback_rate": float(verified_fallback / n_steps),
        "target_hold_prev_count": target_hold,
        "target_hold_prev_rate": float(target_hold / n_steps),
        "solver_hold_prev_count": solver_hold,
        "solver_hold_prev_rate": float(solver_hold / n_steps),
        "accepted_candidate_count": _mode_count(summary, "accepted_candidate"),
        "accepted_candidate_rate": float(_mode_count(summary, "accepted_candidate") / n_steps),
        "reward_fallback_active_rate": _safe_mean(arrays["reward_fallback_active_flags"]),
        "executed_action_gap_inf_mean": _safe_mean(arrays["executed_action_gap_inf"]),
        "executed_action_gap_inf_max": _safe_max(arrays["executed_action_gap_inf"]),
        "candidate_contraction_ok_rate": _safe_mean(candidate_contraction_ok),
        "applied_contraction_ok_rate": _safe_mean(arrays["first_step_contraction_satisfied_applied_flags"]),
        "candidate_contraction_margin_p95": float(np.nanpercentile(arrays["contraction_margin_candidate"], 95)),
        "candidate_contraction_margin_max": _safe_max(arrays["contraction_margin_candidate"]),
        "applied_contraction_margin_max": _safe_max(arrays["contraction_margin_applied"]),
        "target_mismatch_inf_mean": _safe_mean(arrays["target_mismatch_inf"]),
        "target_mismatch_inf_max": _safe_max(arrays["target_mismatch_inf"]),
        "d_s_minus_dhat_inf_max": float(summary.get("d_s_minus_dhat_inf_max", np.nan)),
        "wall_clock_seconds_per_step": float(summary.get("wall_clock_seconds_per_step", np.nan)),
        "mode_counts": json.dumps(summary.get("mode_counts", {}), sort_keys=True),
        "solver_status_counts": json.dumps(summary.get("solver_status_counts", {}), sort_keys=True),
    }


def _add_phase_lines(ax: plt.Axes, config: dict[str, Any]) -> None:
    phase = config.get("training_phase_config") or {}
    bc = int(phase.get("behavior_clone_teacher_episodes", 0) or 0)
    handoff = int(phase.get("handoff_episodes", 0) or 0)
    for idx, (x, label) in enumerate([(bc, "BC end"), (bc + handoff, "handoff end")]):
        if x > 0:
            ax.axvline(x, color="0.25", linestyle="--", linewidth=1.0, alpha=0.65)
            ax.text(
                x + 0.6,
                0.96 - 0.13 * idx,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                fontsize=8,
                color="0.25",
            )


def _plot_episode_rates(episodes: dict[str, pd.DataFrame], configs: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {
        "Cold start + gate": "#2f6f9f",
        "Cold start no gate": "#8a4f7d",
        "OF-MPC pretrained + gate": "#29745a",
        "OF-MPC pretrained no gate": "#b35c2e",
    }
    for family, ax in zip(["Cold start", "OF-MPC pretrained"], axes):
        for case, episode in episodes.items():
            if family not in case:
                continue
            rate_col = "actual_intervention_rate" if "+ gate" in case else "diagnostic_unsafe_rate"
            label = "actual intervention" if "+ gate" in case else "would-be unsafe"
            ax.plot(
                episode["episode"],
                episode[rate_col],
                linewidth=1.4,
                color=colors[case],
                label=f"{case}: {label}",
            )
        gate_case = f"{family} + gate" if family == "Cold start" else "OF-MPC pretrained + gate"
        _add_phase_lines(ax, configs[gate_case])
        ax.set_title(f"{family}: gate activity by episode")
        ax.set_ylabel("rate per episode")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "episode_gate_activity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_episode_performance(episodes: dict[str, pd.DataFrame], configs: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {
        "Cold start + gate": "#2f6f9f",
        "Cold start no gate": "#8a4f7d",
        "OF-MPC pretrained + gate": "#29745a",
        "OF-MPC pretrained no gate": "#b35c2e",
    }
    for case, episode in episodes.items():
        axes[0].plot(episode["episode"], episode["output_rmse_mean"], linewidth=1.3, color=colors[case], label=case)
        axes[1].plot(
            episode["episode"],
            episode["reward_no_penalty_mean"],
            linewidth=1.3,
            color=colors[case],
            label=case,
        )
    _add_phase_lines(axes[0], configs["Cold start + gate"])
    _add_phase_lines(axes[1], configs["Cold start + gate"])
    axes[0].set_ylabel("mean output RMSE")
    axes[1].set_ylabel("reward_no_penalty mean")
    axes[1].set_xlabel("episode")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "episode_tracking_reward.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_last_episode_tracking(arrays_by_case: dict[str, np.lib.npyio.NpzFile], configs: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    families = [
        ("Cold start", "Cold start + gate", "Cold start no gate"),
        ("OF-MPC pretrained", "OF-MPC pretrained + gate", "OF-MPC pretrained no gate"),
    ]
    labels = ["eta", "T"]
    for col, (family, gate_case, no_case) in enumerate(families):
        config = configs[gate_case]
        n_steps = arrays_by_case[gate_case]["rewards"].shape[0]
        episode_len = int(n_steps // max(int(config.get("n_episodes", 300)), 1))
        start = n_steps - episode_len
        stop = n_steps
        t = np.arange(episode_len)
        sp = np.asarray(arrays_by_case[gate_case]["y_sp_phys_store"])[start:stop]
        y_gate = np.asarray(arrays_by_case[gate_case]["y_system"])[start + 1 : stop + 1]
        y_no = np.asarray(arrays_by_case[no_case]["y_system"])[start + 1 : stop + 1]
        for row in range(2):
            ax = axes[row, col]
            ax.plot(t, y_gate[:, row], color="#2f6f9f", linewidth=1.4, label="gate")
            ax.plot(t, y_no[:, row], color="#b35c2e", linewidth=1.2, alpha=0.85, label="no gate")
            ax.step(t, sp[:, row], where="post", color="black", linewidth=1.1, linestyle="--", label="setpoint")
            ax.set_title(f"{family}: {labels[row]} last episode")
            ax.set_ylabel(labels[row])
            ax.grid(True, linestyle="--", alpha=0.3)
            if row == 0:
                ax.legend(loc="best", fontsize=8)
            if row == 1:
                ax.set_xlabel("step within episode")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "last_episode_tracking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gate_mode_counts(summary: pd.DataFrame) -> None:
    gate_summary = summary[summary["gate"]].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(gate_summary))
    accepted = gate_summary["accepted_candidate_count"].to_numpy(dtype=float)
    projected = gate_summary.get("section16_projection_count", pd.Series(0.0, index=gate_summary.index)).to_numpy(dtype=float)
    verified = gate_summary["verified_fallback_count"].to_numpy(dtype=float)
    target_hold = gate_summary["target_hold_prev_count"].to_numpy(dtype=float)
    solver_hold = gate_summary["solver_hold_prev_count"].to_numpy(dtype=float)
    ax.bar(x, accepted, label="accepted TD3 candidate", color="#7aa6c2")
    ax.bar(x, projected, bottom=accepted, label="Section 16 QCQP projection", color="#b790d4")
    ax.bar(x, verified, bottom=accepted + projected, label="verified GART-LMPC fallback", color="#4d8f6f")
    ax.bar(x, target_hold, bottom=accepted + projected + verified, label="target-not-usable hold previous", color="#d1914f")
    ax.bar(
        x,
        solver_hold,
        bottom=accepted + projected + verified + target_hold,
        label="solver-fail hold previous",
        color="#b55a5a",
    )
    ax.set_xticks(x, gate_summary["case"], rotation=15, ha="right")
    ax.set_ylabel("step count")
    ax.set_title("Safety-gate mode counts")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_mode_counts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    phase_rows = []
    gate_detail_rows = []
    episodes: dict[str, pd.DataFrame] = {}
    arrays_by_case: dict[str, np.lib.npyio.NpzFile] = {}
    configs: dict[str, dict[str, Any]] = {}
    source_rows = []

    for case in RUNNERS:
        run_dir = _latest_complete_run(case["runner"])
        summary = _read_json(run_dir / "summary.json")
        run_summary = _read_json(run_dir / "run_summary.json")
        config = dict(run_summary.get("config") or {})
        arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
        episode = pd.read_csv(run_dir / "episode_table.csv")
        episode["case"] = case["case"]
        episode["family"] = case["family"]
        episode["gate"] = bool(case["gate"])
        episode["phase"] = episode["episode"].apply(lambda ep: _phase_for_episode(int(ep), config))
        episodes[case["case"]] = episode
        arrays_by_case[case["case"]] = arrays
        configs[case["case"]] = config
        summary_rows.append(_summary_row(case, run_dir, summary, arrays, config))
        phase_rows.extend(_phase_metric_rows(case, arrays, config))
        gate_detail_rows.extend(_gate_detail_rows(case, _load_gate_columns(run_dir / "step_table.csv")))
        source_rows.append(
            {
                "case": case["case"],
                "runner": case["runner"],
                "run_dir": str(run_dir.relative_to(ROOT)),
                "summary": str((run_dir / "summary.json").relative_to(ROOT)),
                "arrays": str((run_dir / "arrays.npz").relative_to(ROOT)),
                "episode_table": str((run_dir / "episode_table.csv").relative_to(ROOT)),
                "step_table": str((run_dir / "step_table.csv").relative_to(ROOT)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    phase_df = pd.DataFrame(phase_rows)
    gate_detail_df = pd.DataFrame(gate_detail_rows)
    episode_df = pd.concat(episodes.values(), ignore_index=True)
    source_df = pd.DataFrame(source_rows)

    summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    phase_df.to_csv(OUT_DIR / "phase_metrics.csv", index=False)
    gate_detail_df.to_csv(OUT_DIR / "gate_detail_counts.csv", index=False)
    episode_df.to_csv(OUT_DIR / "episode_metrics.csv", index=False)
    source_df.to_csv(OUT_DIR / "source_runs.csv", index=False)

    _plot_episode_rates(episodes, configs)
    _plot_episode_performance(episodes, configs)
    _plot_last_episode_tracking(arrays_by_case, configs)
    _plot_gate_mode_counts(summary_df)

    with (OUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_runs": source_rows,
                "summary_records": summary_df.to_dict(orient="records"),
                "phase_records": phase_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(f"Wrote analysis artifacts to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
