"""Extended analysis for the bounded-mixed online disturbance runner batch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.of_mpc_td3_workflow import build_polymer_setup, jsonable
from utils.scaling_helpers import apply_min_max, reverse_min_max


RESULTS = ROOT / "results"
OUT_DIR = ROOT / "report" / "figures" / "2026-06-11_online_disturbance_bounded_mixed_analysis"
REPORT_PATH = ROOT / "report" / "online_disturbance_bounded_mixed_8_runner_analysis_2026-06-11.md"

PRIMARY_TARGET_MODE = "bounded"
PRIMARY_TARGET_VARIANT = "bounded_mixed_u0p1_x0p1"
COMPARATOR_TARGET_MODE = "governed_reference"
MIN_FULL_STEPS = 200_000

RUNNERS = [
    ("LMPC pretrained + gate", "OnlineTD3_LMPCPretrained_SafetyGate", "online_gate"),
    ("OF-MPC pretrained + gate", "OnlineTD3_OFMPCPretrained_SafetyGate", "online_gate"),
    ("LMPC pretrained no gate", "OnlineTD3_LMPCPretrained_NoSafetyGate", "online_no_gate"),
    ("OF-MPC pretrained no gate", "OnlineTD3_OFMPCPretrained_NoSafetyGate", "online_no_gate"),
    ("Cold start + gate", "OnlineTD3_ColdStart_SafetyGate", "online_gate"),
    ("Cold start no gate", "OnlineTD3_ColdStart_NoSafetyGate", "online_no_gate"),
    ("Direct LMPC baseline", "DirectLMPCDisturbance", "baseline"),
    ("OF-MPC baseline", "OffsetFreeMPCDisturbance", "baseline"),
]

CASE_ORDER = [item[0] for item in RUNNERS]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_paths(run_dir: Path) -> tuple[Path, Path | None]:
    summary = run_dir / "summary.json"
    run_summary = run_dir / "run_summary.json"
    return summary, run_summary if run_summary.exists() else None


def _run_config(run_dir: Path) -> dict[str, Any]:
    summary_path, run_summary_path = _summary_paths(run_dir)
    summary = _read_json(summary_path) if summary_path.exists() else {}
    run_summary = _read_json(run_summary_path) if run_summary_path else {}
    cfg = dict(run_summary.get("config", {}))
    controller = summary.get("controller", {})
    for key in ["target_mode", "target_selector_variant", "target_config"]:
        if key not in cfg:
            cfg[key] = run_summary.get(key, summary.get(key, controller.get(key)))
    return cfg


def _n_steps(run_dir: Path) -> int:
    summary_path, _ = _summary_paths(run_dir)
    if not summary_path.exists():
        return 0
    summary = _read_json(summary_path)
    for key in ["n_steps", "wall_clock_n_steps", "nFE"]:
        if summary.get(key) is not None:
            return int(summary[key])
    return 0


def _latest_full_run(runner_root: Path, *, target_mode: str) -> Path | None:
    candidates: list[Path] = []
    if not runner_root.exists():
        return None
    for child in runner_root.iterdir():
        if not child.is_dir():
            continue
        if not (child / "summary.json").exists() or not (child / "arrays.npz").exists():
            continue
        if _n_steps(child) < MIN_FULL_STEPS:
            continue
        cfg = _run_config(child)
        if str(cfg.get("target_mode", "")).strip().lower() == target_mode:
            candidates.append(child)
    return sorted(candidates, key=lambda path: path.name)[-1] if candidates else None


def _episode_end_column(episode: pd.DataFrame) -> str:
    if "step_stop_exclusive" in episode.columns:
        return "step_stop_exclusive"
    if "step_end_exclusive" in episode.columns:
        return "step_end_exclusive"
    raise KeyError("Episode table has no exclusive stop column.")


def _physical_setpoints(arrays: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_sp_phys_store" in arrays.files:
        return np.asarray(arrays["y_sp_phys_store"], dtype=float)
    y_sp = np.asarray(arrays["y_sp_steps" if "y_sp_steps" in arrays.files else "y_sp"], dtype=float)
    data_min = np.asarray(arrays["data_min"], dtype=float)
    data_max = np.asarray(arrays["data_max"], dtype=float)
    n_inputs = int(np.asarray(arrays["u_applied_phys"]).shape[1])
    setup = build_polymer_setup()
    y_ss_scaled = apply_min_max(setup.steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    return reverse_min_max(y_sp + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def _physical_target(arrays: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "y_target_phys_store" in arrays.files:
        return np.asarray(arrays["y_target_phys_store"], dtype=float)
    if not {"y_target_store", "data_min", "data_max", "u_applied_phys"}.issubset(set(arrays.files)):
        return None
    y_target = np.asarray(arrays["y_target_store"], dtype=float)
    data_min = np.asarray(arrays["data_min"], dtype=float)
    data_max = np.asarray(arrays["data_max"], dtype=float)
    n_inputs = int(np.asarray(arrays["u_applied_phys"]).shape[1])
    setup = build_polymer_setup()
    y_ss_scaled = apply_min_max(setup.steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    return reverse_min_max(y_target + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])


def _output_error_phys(arrays: np.lib.npyio.NpzFile) -> np.ndarray:
    y = np.asarray(arrays["y_system"], dtype=float)[1:]
    sp = _physical_setpoints(arrays)
    return y - sp


def _array_or_zeros(arrays: np.lib.npyio.NpzFile, name: str, n: int) -> np.ndarray:
    if name in arrays.files:
        return np.asarray(arrays[name], dtype=float).reshape(-1)[:n]
    return np.zeros(n, dtype=float)


def _array_or_nan(arrays: np.lib.npyio.NpzFile, name: str, n: int) -> np.ndarray:
    if name in arrays.files:
        return np.asarray(arrays[name], dtype=float).reshape(-1)[:n]
    return np.full(n, np.nan, dtype=float)


def _safe_mean(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _safe_max(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _sum_column(df: pd.DataFrame, name: str) -> float:
    if name not in df.columns:
        return 0.0
    return float(df[name].fillna(0.0).sum())


def _mean_column(df: pd.DataFrame, name: str) -> float:
    if name not in df.columns or df.empty:
        return float("nan")
    return float(df[name].mean())


def _enrich_episode_table(episode: pd.DataFrame, arrays: np.lib.npyio.NpzFile, err: np.ndarray) -> pd.DataFrame:
    episode = episode.copy()
    end_col = _episode_end_column(episode)
    n = err.shape[0]
    rewards = _array_or_zeros(arrays, "rewards", n)
    reward_no_penalty = _array_or_zeros(arrays, "reward_no_penalty", n)
    fallback_penalty = _array_or_zeros(arrays, "fallback_penalty", n)
    diagnostic_unsafe = _array_or_zeros(arrays, "diagnostic_unsafe_flags", n)
    actual_intervention = _array_or_zeros(arrays, "actual_intervention_flags", n)
    fallback_verified = _array_or_zeros(arrays, "fallback_verified_flags", n)

    if "reward_mean" not in episode.columns:
        episode["reward_mean"] = [
            float(np.mean(rewards[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "reward_no_penalty_mean" not in episode.columns:
        episode["reward_no_penalty_mean"] = [
            float(np.mean(reward_no_penalty[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "fallback_penalty_mean" not in episode.columns:
        episode["fallback_penalty_mean"] = [
            float(np.mean(fallback_penalty[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "output_rmse_mean" not in episode.columns:
        rmses = [
            np.sqrt(np.mean(err[int(start) : int(stop)] ** 2, axis=0))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
        episode["output0_rmse"] = [float(row[0]) for row in rmses]
        episode["output1_rmse"] = [float(row[1]) for row in rmses]
        episode["output_rmse_mean"] = [float(np.mean(row)) for row in rmses]
    if "diagnostic_unsafe_rate" not in episode.columns:
        episode["diagnostic_unsafe_rate"] = [
            float(np.mean(diagnostic_unsafe[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "actual_intervention_rate" not in episode.columns:
        episode["actual_intervention_rate"] = [
            float(np.mean(actual_intervention[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "fallback_count" not in episode.columns:
        episode["fallback_count"] = [
            int(np.sum(actual_intervention[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    if "fallback_verified_count" not in episode.columns:
        episode["fallback_verified_count"] = [
            int(np.sum(fallback_verified[int(start) : int(stop)]))
            for start, stop in zip(episode["step_start"], episode[end_col])
        ]
    for name in [
        "solver_fail_hold_prev_count",
        "target_fail_hold_prev_count",
        "fallback_unverified_count",
        "target_failure_count",
    ]:
        if name not in episode.columns:
            episode[name] = 0
    return episode


def _load_case(label: str, dirname: str, family: str, run_dir: Path, batch: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    summary = _read_json(run_dir / "summary.json")
    run_summary = _read_json(run_dir / "run_summary.json") if (run_dir / "run_summary.json").exists() else {}
    cfg = _run_config(run_dir)
    arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
    episode = pd.read_csv(run_dir / "episode_table.csv")
    err = _output_error_phys(arrays)
    episode = _enrich_episode_table(episode, arrays, err)
    n = err.shape[0]

    rewards = _array_or_zeros(arrays, "rewards", n)
    reward_no_penalty = _array_or_zeros(arrays, "reward_no_penalty", n)
    fallback_penalty = _array_or_zeros(arrays, "fallback_penalty", n)
    actual_intervention = _array_or_zeros(arrays, "actual_intervention_flags", n)
    diagnostic_unsafe = _array_or_zeros(arrays, "diagnostic_unsafe_flags", n)
    fallback_verified = _array_or_zeros(arrays, "fallback_verified_flags", n)
    target_residual = _array_or_nan(arrays, "target_residual_total_norm", n)
    target_us_u_ref = _array_or_nan(arrays, "target_us_u_ref_inf", n)
    target_xs_x_ref = _array_or_nan(arrays, "target_xs_x_ref_inf", n)
    target_u_ref_active = _array_or_zeros(arrays, "target_u_ref_active_flags", n)
    target_x_ref_active = _array_or_zeros(arrays, "target_x_ref_active_flags", n)
    contraction_margin = _array_or_nan(arrays, "contraction_margin", n)

    output_rmse = np.sqrt(np.mean(err**2, axis=0))
    output_mae = np.mean(np.abs(err), axis=0)
    tail = episode.tail(min(50, len(episode)))
    last = episode.tail(1)
    best_idx = episode["reward_no_penalty_mean"].idxmax()
    best = episode.loc[best_idx]
    safety_gate = bool(cfg.get("safety_gate_active", False))

    row = {
        "case": label,
        "runner_dir": dirname,
        "family": family,
        "batch": batch,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "n_steps": int(n),
        "n_episodes": int(len(episode)),
        "target_mode": cfg.get("target_mode"),
        "target_selector_variant": cfg.get("target_selector_variant") or "",
        "pretrain_source": cfg.get("pretrain_source") or "",
        "teacher_source": cfg.get("teacher_source") or "",
        "safety_gate": safety_gate,
        "reward_mean": _safe_mean(rewards),
        "reward_no_penalty_mean": _safe_mean(reward_no_penalty),
        "fallback_penalty_mean": _safe_mean(fallback_penalty),
        "reward_penalty_gap": _safe_mean(reward_no_penalty) - _safe_mean(rewards),
        "output0_rmse": float(output_rmse[0]),
        "output1_rmse": float(output_rmse[1]),
        "output_rmse_mean": float(np.mean(output_rmse)),
        "output0_mae": float(output_mae[0]),
        "output1_mae": float(output_mae[1]),
        "output_mae_mean": float(np.mean(output_mae)),
        "output_max_abs_error": float(np.max(np.abs(err))),
        "tail50_reward_mean": _mean_column(tail, "reward_mean"),
        "tail50_reward_no_penalty": _mean_column(tail, "reward_no_penalty_mean"),
        "tail50_fallback_penalty": _mean_column(tail, "fallback_penalty_mean"),
        "tail50_output_rmse_mean": _mean_column(tail, "output_rmse_mean"),
        "tail50_actual_intervention_rate": _mean_column(tail, "actual_intervention_rate"),
        "tail50_diagnostic_unsafe_rate": _mean_column(tail, "diagnostic_unsafe_rate"),
        "last_reward_no_penalty": float(last["reward_no_penalty_mean"].iloc[0]),
        "last_output_rmse_mean": float(last["output_rmse_mean"].iloc[0]),
        "last_actual_intervention_rate": float(last["actual_intervention_rate"].iloc[0]),
        "last_diagnostic_unsafe_rate": float(last["diagnostic_unsafe_rate"].iloc[0]),
        "best_episode": int(best["episode"]),
        "best_reward_no_penalty": float(best["reward_no_penalty_mean"]),
        "best_output_rmse_mean": float(best["output_rmse_mean"]),
        "actual_intervention_rate": _safe_mean(actual_intervention),
        "actual_gate_intervention_rate": _safe_mean(actual_intervention) if safety_gate else 0.0,
        "diagnostic_unsafe_rate": _safe_mean(diagnostic_unsafe),
        "fallback_verified_rate": _safe_mean(fallback_verified),
        "actual_intervention_count": int(np.nansum(actual_intervention)) if safety_gate else 0,
        "fallback_verified_count": int(np.nansum(fallback_verified)) if safety_gate else 0,
        "fallback_count": int(_sum_column(episode, "fallback_count")) if safety_gate else 0,
        "solver_fail_hold_prev_count": int(_sum_column(episode, "solver_fail_hold_prev_count")) if safety_gate else 0,
        "target_fail_hold_prev_count": int(_sum_column(episode, "target_fail_hold_prev_count")) if safety_gate else 0,
        "target_residual_mean": _safe_mean(target_residual),
        "target_residual_max": _safe_max(target_residual),
        "target_us_u_ref_inf_mean": _safe_mean(target_us_u_ref),
        "target_us_u_ref_inf_max": _safe_max(target_us_u_ref),
        "target_xs_x_ref_inf_mean": _safe_mean(target_xs_x_ref),
        "target_xs_x_ref_inf_max": _safe_max(target_xs_x_ref),
        "target_u_ref_active_rate": _safe_mean(target_u_ref_active),
        "target_x_ref_active_rate": _safe_mean(target_x_ref_active),
        "contraction_margin_min": float(np.nanmin(contraction_margin)) if not np.all(np.isnan(contraction_margin)) else np.nan,
        "wall_clock_seconds": float(summary.get("wall_clock_seconds", np.nan)),
        "checkpoint_selector_note": cfg.get("pretrained_checkpoint_selector_note", ""),
    }

    phase_frames = []
    for phase, mask in [
        ("BC", episode["episode"].between(1, 20)),
        ("handoff", episode["episode"].between(21, 25)),
        ("full TD3", episode["episode"].between(26, 300)),
        ("tail 50", episode["episode"].between(max(1, len(episode) - 49), len(episode))),
    ]:
        part = episode.loc[mask]
        if part.empty:
            continue
        phase_frames.append(
            {
                "case": label,
                "batch": batch,
                "phase": phase,
                "n_episodes": int(len(part)),
                "reward_mean": _mean_column(part, "reward_mean"),
                "reward_no_penalty_mean": _mean_column(part, "reward_no_penalty_mean"),
                "fallback_penalty_mean": _mean_column(part, "fallback_penalty_mean"),
                "output_rmse_mean": _mean_column(part, "output_rmse_mean"),
                "actual_intervention_rate": _mean_column(part, "actual_intervention_rate"),
                "diagnostic_unsafe_rate": _mean_column(part, "diagnostic_unsafe_rate"),
                "fallback_count": int(_sum_column(part, "fallback_count")) if safety_gate else 0,
                "fallback_verified_count": int(_sum_column(part, "fallback_verified_count")) if safety_gate else 0,
                "solver_fail_hold_prev_count": int(_sum_column(part, "solver_fail_hold_prev_count")) if safety_gate else 0,
            }
        )

    metadata = {
        "episode": episode,
        "arrays_path": run_dir / "arrays.npz",
        "run_summary": run_summary,
        "config": cfg,
    }
    return row, pd.DataFrame(phase_frames), metadata


def collect_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    phase_rows: list[pd.DataFrame] = []
    block_rows: list[dict[str, Any]] = []
    manifest: dict[str, dict[str, str]] = {}

    for label, dirname, family in RUNNERS:
        manifest[label] = {}
        for batch, target_mode in [("bounded_mixed", PRIMARY_TARGET_MODE), ("governed_reference", COMPARATOR_TARGET_MODE)]:
            run_dir = _latest_full_run(RESULTS / dirname, target_mode=target_mode)
            if run_dir is None:
                continue
            manifest[label][batch] = str(run_dir.relative_to(ROOT))
            row, phases, _meta = _load_case(label, dirname, family, run_dir, batch)
            rows.append(row)
            phase_rows.append(phases)

            arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
            err = _output_error_phys(arrays)
            n = err.shape[0]
            rewards = _array_or_zeros(arrays, "rewards", n)
            reward_no_penalty = _array_or_zeros(arrays, "reward_no_penalty", n)
            actual_intervention = _array_or_zeros(arrays, "actual_intervention_flags", n)
            diagnostic_unsafe = _array_or_zeros(arrays, "diagnostic_unsafe_flags", n)
            steps = np.arange(n)
            episode_index = steps // 800 + 1
            block_index = (steps % 800) // 400
            for window, base_mask in [
                ("all episodes", np.ones(n, dtype=bool)),
                ("tail 50", episode_index >= max(1, int(np.max(episode_index)) - 49)),
            ]:
                for block_id, block_label in [(0, "S1 high"), (1, "S2 low")]:
                    mask = base_mask & (block_index == block_id)
                    rmse = np.sqrt(np.mean(err[mask] ** 2, axis=0))
                    block_rows.append(
                        {
                            "case": label,
                            "batch": batch,
                            "window": window,
                            "block": block_label,
                            "n_steps": int(np.sum(mask)),
                            "reward_mean": float(np.mean(rewards[mask])),
                            "reward_no_penalty_mean": float(np.mean(reward_no_penalty[mask])),
                            "output0_rmse": float(rmse[0]),
                            "output1_rmse": float(rmse[1]),
                            "output_rmse_mean": float(np.mean(rmse)),
                            "actual_intervention_rate": float(np.mean(actual_intervention[mask]))
                            if family == "online_gate"
                            else 0.0,
                            "diagnostic_unsafe_rate": float(np.mean(diagnostic_unsafe[mask])),
                        }
                    )

    metrics = pd.DataFrame(rows)
    phases = pd.concat(phase_rows, ignore_index=True) if phase_rows else pd.DataFrame()
    blocks = pd.DataFrame(block_rows)

    bounded = metrics.loc[metrics["batch"] == "bounded_mixed"].set_index("case")
    governed = metrics.loc[metrics["batch"] == "governed_reference"].set_index("case")
    shared = [case for case in CASE_ORDER if case in bounded.index and case in governed.index]
    compare_rows = []
    for case in shared:
        b = bounded.loc[case]
        g = governed.loc[case]
        compare_rows.append(
            {
                "case": case,
                "bounded_run_dir": b["run_dir"],
                "governed_run_dir": g["run_dir"],
                "delta_reward_no_penalty": b["reward_no_penalty_mean"] - g["reward_no_penalty_mean"],
                "delta_reward": b["reward_mean"] - g["reward_mean"],
                "delta_fallback_penalty": b["fallback_penalty_mean"] - g["fallback_penalty_mean"],
                "delta_output_rmse_mean": b["output_rmse_mean"] - g["output_rmse_mean"],
                "delta_tail50_reward_no_penalty": b["tail50_reward_no_penalty"] - g["tail50_reward_no_penalty"],
                "delta_tail50_output_rmse_mean": b["tail50_output_rmse_mean"] - g["tail50_output_rmse_mean"],
                "delta_actual_gate_intervention_rate": b["actual_gate_intervention_rate"] - g["actual_gate_intervention_rate"],
                "delta_diagnostic_unsafe_rate": b["diagnostic_unsafe_rate"] - g["diagnostic_unsafe_rate"],
                "delta_target_residual_max": b["target_residual_max"] - g["target_residual_max"],
            }
        )
    comparison = pd.DataFrame(compare_rows)
    return metrics, phases, blocks, comparison, manifest


def _style_axes(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def _moving_average(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _episode_for(label: str, batch: str, manifest: dict[str, dict[str, str]]) -> pd.DataFrame:
    run_dir = ROOT / manifest[label][batch]
    arrays = np.load(run_dir / "arrays.npz", allow_pickle=True)
    episode = pd.read_csv(run_dir / "episode_table.csv")
    return _enrich_episode_table(episode, arrays, _output_error_phys(arrays))


def plot_bounded_overview(metrics: pd.DataFrame) -> None:
    bounded = metrics.loc[metrics["batch"] == "bounded_mixed"].copy()
    bounded["case"] = pd.Categorical(bounded["case"], CASE_ORDER, ordered=True)
    bounded = bounded.sort_values("tail50_reward_no_penalty")
    colors = np.where(bounded["family"].eq("baseline"), "#7b3294", np.where(bounded["safety_gate"], "#31688e", "#35b779"))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), constrained_layout=True)
    axes[0].barh(bounded["case"], bounded["tail50_reward_no_penalty"], color=colors)
    axes[0].set_xlabel("Tail-50 mean reward_no_penalty")
    axes[0].set_title("Late control reward, no fallback penalty")
    _style_axes(axes[0])
    rmse_order = bounded.sort_values("tail50_output_rmse_mean", ascending=False)
    colors2 = np.where(rmse_order["family"].eq("baseline"), "#7b3294", np.where(rmse_order["safety_gate"], "#31688e", "#35b779"))
    axes[1].barh(rmse_order["case"], rmse_order["tail50_output_rmse_mean"], color=colors2)
    axes[1].set_xlabel("Tail-50 physical output RMSE")
    axes[1].set_title("Late physical tracking")
    _style_axes(axes[1])
    fig.suptitle("Bounded-mixed selector batch: late-phase performance")
    fig.savefig(OUT_DIR / "bounded_tail_performance_overview.png", dpi=220)
    plt.close(fig)


def plot_selector_deltas(comparison: pd.DataFrame) -> None:
    order = comparison.copy()
    order["case"] = pd.Categorical(order["case"], CASE_ORDER, ordered=True)
    order = order.sort_values("case")
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), constrained_layout=True)
    axes[0].bar(x, order["delta_reward_no_penalty"], color="#31688e")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order["case"], rotation=30, ha="right")
    axes[0].set_title("Bounded-mixed minus governed-reference reward")
    axes[0].set_ylabel("Delta mean reward_no_penalty")
    _style_axes(axes[0])
    axes[1].bar(x - 0.18, 100 * order["delta_actual_gate_intervention_rate"], width=0.36, label="actual gate", color="#31688e")
    axes[1].bar(x + 0.18, 100 * order["delta_diagnostic_unsafe_rate"], width=0.36, label="diagnostic", color="#35b779")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order["case"], rotation=30, ha="right")
    axes[1].set_ylabel("Delta rate (percentage points)")
    axes[1].set_title("Selector effect on gate/monitor activity")
    axes[1].legend(frameon=False)
    _style_axes(axes[1])
    fig.savefig(OUT_DIR / "bounded_vs_governed_selector_deltas.png", dpi=220)
    plt.close(fig)


def plot_safety_activity(metrics: pd.DataFrame) -> None:
    bounded = metrics.loc[metrics["batch"] == "bounded_mixed"].copy()
    bounded["case"] = pd.Categorical(bounded["case"], CASE_ORDER, ordered=True)
    bounded = bounded.sort_values("case")
    x = np.arange(len(bounded))
    fig, ax1 = plt.subplots(figsize=(12.8, 5.4), constrained_layout=True)
    ax1.bar(x - 0.18, 100 * bounded["actual_gate_intervention_rate"], width=0.36, label="actual gate intervention", color="#31688e")
    ax1.bar(x + 0.18, 100 * bounded["diagnostic_unsafe_rate"], width=0.36, label="diagnostic would activate", color="#35b779")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bounded["case"], rotation=32, ha="right")
    ax1.set_ylabel("Rate (% of steps)")
    ax1.set_title("Bounded-mixed actual safety actions versus monitor-only failures")
    _style_axes(ax1)
    ax1.legend(loc="upper left", frameon=False)
    ax2 = ax1.twinx()
    ax2.plot(x, bounded["fallback_penalty_mean"], color="#b35806", marker="o", linewidth=2, label="fallback penalty mean")
    ax2.set_ylabel("Mean fallback penalty")
    ax2.spines["top"].set_visible(False)
    ax2.legend(loc="upper right", frameon=False)
    fig.savefig(OUT_DIR / "bounded_safety_activity_and_penalty.png", dpi=220)
    plt.close(fig)


def plot_fallback_breakdown(metrics: pd.DataFrame) -> None:
    gates = metrics.loc[(metrics["batch"] == "bounded_mixed") & metrics["safety_gate"]].copy()
    x = np.arange(len(gates))
    fig, ax = plt.subplots(figsize=(9.6, 4.9), constrained_layout=True)
    ax.bar(x, gates["fallback_verified_count"], label="verified Direct LMPC fallback", color="#31688e")
    ax.bar(x, gates["solver_fail_hold_prev_count"], bottom=gates["fallback_verified_count"], label="hold previous after solver issue", color="#b35806")
    ax.set_xticks(x)
    ax.set_xticklabels(gates["case"], rotation=25, ha="right")
    ax.set_ylabel("Count over 240000 steps")
    ax.set_title("Bounded-mixed safety-gate fallback / hold-prev breakdown")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.savefig(OUT_DIR / "bounded_fallback_breakdown.png", dpi=220)
    plt.close(fig)


def plot_episode_trends(manifest: dict[str, dict[str, str]], metric: str, outfile: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    palette = {
        "LMPC pretrained + gate": "#31688e",
        "OF-MPC pretrained + gate": "#1f77b4",
        "LMPC pretrained no gate": "#35b779",
        "OF-MPC pretrained no gate": "#2ca25f",
        "Cold start + gate": "#5e3c99",
        "Cold start no gate": "#8c6d31",
        "Direct LMPC baseline": "#7b3294",
        "OF-MPC baseline": "#c2a5cf",
    }
    fig, ax = plt.subplots(figsize=(12.5, 5.7), constrained_layout=True)
    for case in CASE_ORDER:
        if "bounded_mixed" not in manifest.get(case, {}):
            continue
        df = _episode_for(case, "bounded_mixed", manifest)
        lw = 2.2 if "baseline" not in case.lower() else 1.5
        alpha = 0.92 if "baseline" not in case.lower() else 0.65
        ax.plot(df["episode"], _moving_average(df[metric], 5), label=case, color=palette.get(case), linewidth=lw, alpha=alpha)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=2, fontsize=8.4, frameon=False)
    _style_axes(ax)
    fig.savefig(OUT_DIR / outfile, dpi=220)
    plt.close(fig)


def plot_gate_selector_trends(manifest: dict[str, dict[str, str]]) -> None:
    cases = ["LMPC pretrained + gate", "OF-MPC pretrained + gate", "Cold start + gate"]
    fig, axes = plt.subplots(len(cases), 1, figsize=(12.4, 8.2), sharex=True, constrained_layout=True)
    for ax, case in zip(axes, cases):
        for batch, color, style in [
            ("bounded_mixed", "#31688e", "-"),
            ("governed_reference", "#b35806", "--"),
        ]:
            if batch not in manifest.get(case, {}):
                continue
            df = _episode_for(case, batch, manifest)
            ax.plot(df["episode"], _moving_average(df["reward_no_penalty_mean"], 5), color=color, linestyle=style, linewidth=1.9, label=batch)
        ax.set_title(case)
        ax.set_ylabel("reward_no_penalty")
        _style_axes(ax)
        ax.legend(frameon=False, loc="lower right")
    axes[-1].set_xlabel("Episode")
    fig.savefig(OUT_DIR / "gate_reward_trends_bounded_vs_governed.png", dpi=220)
    plt.close(fig)


def plot_last_episode_outputs(manifest: dict[str, dict[str, str]]) -> None:
    selected = [
        "OF-MPC pretrained + gate",
        "OF-MPC pretrained no gate",
        "Cold start + gate",
        "Cold start no gate",
        "Direct LMPC baseline",
        "OF-MPC baseline",
    ]
    palette = {
        "OF-MPC pretrained + gate": "#1f77b4",
        "OF-MPC pretrained no gate": "#2ca25f",
        "Cold start + gate": "#5e3c99",
        "Cold start no gate": "#8c6d31",
        "Direct LMPC baseline": "#7b3294",
        "OF-MPC baseline": "#c2a5cf",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.5), sharex=True, constrained_layout=True)
    first_sp = None
    for case in selected:
        if "bounded_mixed" not in manifest.get(case, {}):
            continue
        arrays = np.load(ROOT / manifest[case]["bounded_mixed"] / "arrays.npz", allow_pickle=True)
        y = np.asarray(arrays["y_system"], dtype=float)[1:]
        sp = _physical_setpoints(arrays)
        if first_sp is None:
            first_sp = sp
        start = max(0, len(y) - 800)
        t = np.arange(len(y) - start)
        axes[0].plot(t, y[start:, 0], color=palette.get(case), linewidth=1.7, alpha=0.9, label=case)
        axes[1].plot(t, y[start:, 1], color=palette.get(case), linewidth=1.7, alpha=0.9, label=case)
    if first_sp is not None:
        start = max(0, len(first_sp) - 800)
        t = np.arange(len(first_sp) - start)
        axes[0].plot(t, first_sp[start:, 0], color="black", linestyle="--", linewidth=1.5, label="setpoint")
        axes[1].plot(t, first_sp[start:, 1], color="black", linestyle="--", linewidth=1.5, label="setpoint")
    axes[0].set_ylabel("Output 0")
    axes[1].set_ylabel("Output 1")
    axes[1].set_xlabel("Last episode step")
    axes[0].legend(ncol=3, fontsize=7.8, frameon=False)
    for ax in axes:
        _style_axes(ax)
    fig.suptitle("Bounded-mixed last episode physical output tracking")
    fig.savefig(OUT_DIR / "bounded_last_episode_outputs.png", dpi=220)
    plt.close(fig)


def plot_target_diagnostics(metrics: pd.DataFrame) -> None:
    bounded = metrics.loc[metrics["batch"] == "bounded_mixed"].copy()
    bounded["case"] = pd.Categorical(bounded["case"], CASE_ORDER, ordered=True)
    bounded = bounded.sort_values("case")
    x = np.arange(len(bounded))
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.6), constrained_layout=True)
    axes[0].bar(x - 0.18, bounded["target_residual_max"], width=0.36, color="#7b3294", label="max target residual")
    axes[0].bar(x + 0.18, bounded["target_us_u_ref_inf_max"], width=0.36, color="#31688e", label="max |u_s-u_ref|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bounded["case"], rotation=30, ha="right")
    axes[0].set_title("Bounded selector target residual and input anchor distance")
    axes[0].legend(frameon=False)
    _style_axes(axes[0])
    axes[1].bar(x - 0.18, 100 * bounded["target_u_ref_active_rate"], width=0.36, color="#31688e", label="u_ref active")
    axes[1].bar(x + 0.18, 100 * bounded["target_x_ref_active_rate"], width=0.36, color="#35b779", label="x_ref active")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bounded["case"], rotation=30, ha="right")
    axes[1].set_ylabel("Active rate (%)")
    axes[1].set_title("Regularization terms are active on most bounded-target steps")
    axes[1].legend(frameon=False)
    _style_axes(axes[1])
    fig.savefig(OUT_DIR / "bounded_target_diagnostics.png", dpi=220)
    plt.close(fig)


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if spec == "str":
        return str(value)
    if spec == "int":
        return f"{int(round(float(value)))}"
    if spec == "pct":
        return f"{100.0 * float(value):.2f}"
    return format(float(value), spec)


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    align = "| " + " | ".join(":--" if fmt == "str" else "--:" for _, _, fmt in columns) + " |"
    rows = [header, align]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_fmt(row.get(col), fmt) for col, _, fmt in columns) + " |")
    return "\n".join(rows)


def _figure(path: str) -> str:
    return f"figures/2026-06-11_online_disturbance_bounded_mixed_analysis/{path}"


def write_report(metrics: pd.DataFrame, phases: pd.DataFrame, blocks: pd.DataFrame, comparison: pd.DataFrame, manifest: dict[str, dict[str, str]]) -> None:
    bounded = metrics.loc[metrics["batch"] == "bounded_mixed"].copy()
    bounded["case"] = pd.Categorical(bounded["case"], CASE_ORDER, ordered=True)
    bounded = bounded.sort_values("case")
    comp = comparison.copy()
    comp["case"] = pd.Categorical(comp["case"], CASE_ORDER, ordered=True)
    comp = comp.sort_values("case")

    tail_table = bounded[
        [
            "case",
            "reward_no_penalty_mean",
            "reward_mean",
            "fallback_penalty_mean",
            "tail50_reward_no_penalty",
            "tail50_output_rmse_mean",
            "actual_gate_intervention_rate",
            "diagnostic_unsafe_rate",
        ]
    ].copy()

    safety = bounded[
        [
            "case",
            "actual_intervention_count",
            "fallback_verified_count",
            "solver_fail_hold_prev_count",
            "actual_gate_intervention_rate",
            "diagnostic_unsafe_rate",
            "fallback_penalty_mean",
        ]
    ].copy()

    target = bounded[
        [
            "case",
            "target_residual_max",
            "target_us_u_ref_inf_mean",
            "target_us_u_ref_inf_max",
            "target_u_ref_active_rate",
            "target_x_ref_active_rate",
        ]
    ].copy()

    phase_view = phases.loc[
        (phases["batch"] == "bounded_mixed")
        & phases["case"].isin(
            [
                "LMPC pretrained + gate",
                "OF-MPC pretrained + gate",
                "OF-MPC pretrained no gate",
                "Cold start + gate",
                "Cold start no gate",
            ]
        )
        & phases["phase"].isin(["BC", "full TD3", "tail 50"])
    ].copy()
    phase_view["case_phase"] = phase_view["case"] + " - " + phase_view["phase"]

    block_tail = blocks.loc[(blocks["batch"] == "bounded_mixed") & (blocks["window"] == "tail 50")].copy()
    block_tail = block_tail.loc[
        block_tail["case"].isin(["OF-MPC pretrained + gate", "OF-MPC pretrained no gate", "Direct LMPC baseline", "OF-MPC baseline"])
    ].copy()
    block_tail["case_block"] = block_tail["case"] + " - " + block_tail["block"]

    best_reward = bounded.sort_values("tail50_reward_no_penalty", ascending=False).iloc[0]
    best_rmse = bounded.sort_values("tail50_output_rmse_mean", ascending=True).iloc[0]
    strongest_monitor = bounded.sort_values("diagnostic_unsafe_rate", ascending=False).iloc[0]
    gate_comp = comp.loc[comp["case"].str.contains("gate", case=False, na=False) & ~comp["case"].str.contains("no gate", case=False, na=False)]

    overall_best = bounded.sort_values("reward_no_penalty_mean", ascending=False).iloc[0]

    text = f"""# Bounded-Mixed Selector Online Disturbance Runner Analysis

Date: 2026-06-11

## Objective

This report analyzes the eight disturbance-only runners after switching the Direct LMPC target selector from the June 10 governed-reference selector to the previous bounded mixed selector. The bounded mixed selector uses `target_mode=\"bounded\"`, `u_ref_weight=0.1`, `x_ref_weight=0.1`, `rho_lyap=0.99`, `lyap_eps=1e-3`, and `lyap_tol=1e-10`.

The central question is whether the older bounded selector restores the behavior that previously looked better: more meaningful monitor activity, better fallback behavior, and improved tracking relative to the governed-reference batch.

## Data Used

Primary bounded-mixed runs:

{markdown_table(pd.DataFrame([{'case': k, 'run_dir': v.get('bounded_mixed', '')} for k, v in manifest.items()]), [('case', 'Case', 'str'), ('run_dir', 'Run directory', 'str')])}

Comparator governed-reference runs were the latest full runs under the same result roots with `target_mode=\"governed_reference\"`.

## Method Summary

All runs use the polymer CSTR in disturbance mode with 300 episodes and 400-step setpoint blocks. The online TD3 action is represented in normalized action coordinates and mapped to scaled input-deviation coordinates before either gate evaluation or plant execution.

For safety-gate runs, the executed input is

$$
u_k =
\\begin{{cases}}
u_k^{{\\mathrm{{TD3}}}}, & V(x_{{k+1}}; x_s) \\le \\rho V(x_k; x_s) + \\epsilon, \\\\
u_k^{{\\mathrm{{LMPC}}}}, & \\text{{otherwise}},
\\end{{cases}}
$$

where $(x_s,u_s,y_s)$ is selected by the bounded output-disturbance target problem. For no-gate runs, $u_k=u_k^{{\\mathrm{{TD3}}}}$ is always executed, and the same Direct LMPC check is logged as diagnostic-only. Therefore no-gate control performance should not change when only the diagnostic target selector changes, but monitor rates can change.

## Main Bounded-Mixed Results

The best overall control reward is `{overall_best['case']}` with mean `reward_no_penalty = {overall_best['reward_no_penalty_mean']:.3f}`. The best late control reward is `{best_reward['case']}` with tail-50 `reward_no_penalty = {best_reward['tail50_reward_no_penalty']:.3f}`. The best late physical RMSE is `{best_rmse['case']}` with tail-50 mean output RMSE `{best_rmse['tail50_output_rmse_mean']:.3f}`. The strongest overall no-gate monitor activity is `{strongest_monitor['case']}` with diagnostic unsafe rate `{100.0 * strongest_monitor['diagnostic_unsafe_rate']:.2f}%`.

![Bounded tail performance]({_figure('bounded_tail_performance_overview.png')})

{markdown_table(tail_table, [
    ('case', 'Case', 'str'),
    ('reward_no_penalty_mean', 'Reward no penalty', '.3f'),
    ('reward_mean', 'Training reward', '.3f'),
    ('fallback_penalty_mean', 'Fallback penalty', '.3f'),
    ('tail50_reward_no_penalty', 'Tail50 no penalty', '.3f'),
    ('tail50_output_rmse_mean', 'Tail50 RMSE', '.3f'),
    ('actual_gate_intervention_rate', 'Actual gate %', 'pct'),
    ('diagnostic_unsafe_rate', 'Diag unsafe %', 'pct'),
])}

Interpretation:

- The OF-MPC-pretrained no-gate TD3 remains the strongest learned controller by overall reward. In the tail, cold-start no-gate is slightly better, but the margin is small enough that a seed repeat matters.
- The OF-MPC-pretrained safety-gate run is close behind the no-gate learned controllers and is clearly better than the MPC baselines by late reward/RMSE. It pays fallback penalties in training reward, so `reward_no_penalty` is the fair control-performance comparison.
- LMPC-pretrained no-gate remains poor because it is still the old governed-reference LMPC-pretrained checkpoint. The online selector change does not regenerate that checkpoint.
- Direct LMPC and OF-MPC baselines are very close in physical RMSE under this two-setpoint disturbed schedule.

## Bounded-Mixed Versus Governed-Reference

![Selector deltas]({_figure('bounded_vs_governed_selector_deltas.png')})

{markdown_table(comp, [
    ('case', 'Case', 'str'),
    ('delta_reward_no_penalty', 'Delta no penalty', '.3f'),
    ('delta_reward', 'Delta training', '.3f'),
    ('delta_fallback_penalty', 'Delta penalty', '.3f'),
    ('delta_output_rmse_mean', 'Delta RMSE', '.4f'),
    ('delta_actual_gate_intervention_rate', 'Delta actual gate %', 'pct'),
    ('delta_diagnostic_unsafe_rate', 'Delta diag %', 'pct'),
])}

Interpretation:

- Pretrained safety-gate runs improve in both `reward_no_penalty` and logged training reward under bounded mixed. They also have lower fallback penalty and lower actual intervention rate than the governed-reference batch.
- Cold-start safety worsens in `reward_no_penalty` and physical RMSE even though the fallback penalty and intervention rate are lower. That points to fallback/target quality and learning trajectory, not merely penalty accounting.
- No-gate control rewards are exactly unchanged, which is a useful sanity check because the Direct LMPC selector is diagnostic-only in those runners. Diagnostic unsafe rates decrease, so in this batch the bounded mixed selector is less restrictive for the same executed no-gate actions than the governed-reference monitor.
- Direct LMPC and OF-MPC baselines are almost unchanged, which means the main selector effect is in how the online gate accepts/rejects learned exploratory actions.

## Safety-Gate Mechanics

![Safety activity]({_figure('bounded_safety_activity_and_penalty.png')})

![Fallback breakdown]({_figure('bounded_fallback_breakdown.png')})

{markdown_table(safety, [
    ('case', 'Case', 'str'),
    ('actual_intervention_count', 'Actual int.', 'int'),
    ('fallback_verified_count', 'Verified fallback', 'int'),
    ('solver_fail_hold_prev_count', 'Hold-prev', 'int'),
    ('actual_gate_intervention_rate', 'Actual gate %', 'pct'),
    ('diagnostic_unsafe_rate', 'Diag unsafe %', 'pct'),
    ('fallback_penalty_mean', 'Penalty mean', '.3f'),
])}

The console phrase `fallback / hold-prev` combines verified Direct LMPC fallback and hold-previous events after target or solver issues. In this bounded-mixed batch, most safety-gate corrections are verified fallbacks, but LMPC-pretrained safety still has nontrivial hold-prev counts. That is why comparing only the printed fallback ratio can hide the correction quality.

## Learning Phase Behavior

![Reward trends]({_figure('bounded_episode_reward_no_penalty_trends.png')})

![RMSE trends]({_figure('bounded_episode_output_rmse_trends.png')})

![Gate selector trends]({_figure('gate_reward_trends_bounded_vs_governed.png')})

{markdown_table(phase_view, [
    ('case_phase', 'Case phase', 'str'),
    ('reward_no_penalty_mean', 'Reward no penalty', '.3f'),
    ('fallback_penalty_mean', 'Penalty', '.3f'),
    ('output_rmse_mean', 'RMSE', '.3f'),
    ('actual_intervention_rate', 'Actual gate %', 'pct'),
    ('diagnostic_unsafe_rate', 'Diag %', 'pct'),
])}

During BC, the behavior action is teacher plus exploration and the actor is supervised toward the clean teacher action. During full TD3, exploration is added to the policy action before the gate/diagnostic check. The bounded-mixed selector mostly changes what the gate judges safe and what fallback it computes. It does not change no-gate execution.

## Setpoint-Block And Tracking Evidence

![Last episode outputs]({_figure('bounded_last_episode_outputs.png')})

{markdown_table(block_tail, [
    ('case_block', 'Case block', 'str'),
    ('reward_no_penalty_mean', 'Tail reward', '.3f'),
    ('output0_rmse', 'Output0 RMSE', '.3f'),
    ('output1_rmse', 'Output1 RMSE', '.3f'),
    ('output_rmse_mean', 'Mean RMSE', '.3f'),
    ('actual_intervention_rate', 'Actual gate %', 'pct'),
    ('diagnostic_unsafe_rate', 'Diag %', 'pct'),
])}

The last-episode traces show that the best learned no-gate policy tracks tightly, while the safety-gate and baseline controllers remain more conservative around transitions. For paper comparisons, this supports reporting both tracking RMSE and safety activity rather than relying only on reward.

## Target Diagnostics

![Target diagnostics]({_figure('bounded_target_diagnostics.png')})

{markdown_table(target, [
    ('case', 'Case', 'str'),
    ('target_residual_max', 'Residual max', '.3f'),
    ('target_us_u_ref_inf_mean', 'us-uref mean', '.3f'),
    ('target_us_u_ref_inf_max', 'us-uref max', '.3f'),
    ('target_u_ref_active_rate', 'u_ref active %', 'pct'),
    ('target_x_ref_active_rate', 'x_ref active %', 'pct'),
])}

The bounded selector is active in the intended way: the input-reference and state-reference regularizers are nonzero on most steps. This confirms that the batch is not secretly running the governed-reference path.

## Main Interpretation

The bounded-mixed selector is defensible and useful, but the new evidence does not say it is universally better. It improves the pretrained safety-gate cases and gives clearer Direct LMPC diagnostic fields. It does not change no-gate execution, lowers no-gate diagnostic unsafe rates, and worsens the cold-start safety-gate reward/no-penalty metrics in this batch.

The most likely mechanism is policy quality. With a pretrained or OF-MPC-shaped candidate policy, the bounded selector provides a practical admissible target and fallback that catches unsafe exploratory steps without dominating the controller. With a weak cold-start policy, fewer interventions are not automatically better because the accepted exploratory actions and the fallback targets can still steer learning toward a poorer trajectory.

## Bugs, Inconsistencies, And Risks

- The LMPC-pretrained online runs still load an old LMPC checkpoint unless a new full `PretrainTD3LyapunovMPC.py` production run has been generated after the bounded-mixed pretraining change.
- No-gate reward equality across selectors is expected, not a bug. Only diagnostic fields should move.
- Baseline `actual_intervention_flags` are not equivalent to safety-gate fallback events, so the report uses actual gate intervention only for online safety-gate cases.
- The analysis uses a single seed/batch. Treat rankings as batch evidence, not a statistical conclusion.

## Recommended Next Experiment

1. Run a new full LMPC pretraining job with the bounded-mixed selector, then rerun the two LMPC-pretrained online runners. The current LMPC-pretrained checkpoint was trained under the old selector.
2. Add a paired two-seed or three-seed repeat for the OF-MPC-pretrained safety and no-gate runners. This will test whether the OF-MPC-pretrained no-gate advantage is robust or a single-seed artifact.
3. For cold start, test a longer OF-MPC teacher phase or a gentler transition into policy-controlled full TD3. The current bounded-mixed safety gate still leads to a worse learning path when the candidate policy is weak.

## Generated Artifacts

- Metrics CSV: `{_figure('bounded_metrics.csv')}`
- Phase CSV: `{_figure('bounded_phase_metrics.csv')}`
- Setpoint-block CSV: `{_figure('bounded_setpoint_block_metrics.csv')}`
- Selector comparison CSV: `{_figure('bounded_vs_governed_comparison.csv')}`
- Run manifest: `{_figure('run_manifest.json')}`
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, phases, blocks, comparison, manifest = collect_data()
    metrics.to_csv(OUT_DIR / "bounded_metrics.csv", index=False)
    phases.to_csv(OUT_DIR / "bounded_phase_metrics.csv", index=False)
    blocks.to_csv(OUT_DIR / "bounded_setpoint_block_metrics.csv", index=False)
    comparison.to_csv(OUT_DIR / "bounded_vs_governed_comparison.csv", index=False)
    with (OUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(manifest), handle, indent=2)

    plot_bounded_overview(metrics)
    plot_selector_deltas(comparison)
    plot_safety_activity(metrics)
    plot_fallback_breakdown(metrics)
    plot_episode_trends(
        manifest,
        "reward_no_penalty_mean",
        "bounded_episode_reward_no_penalty_trends.png",
        "5-episode smoothed reward_no_penalty",
        ylim=(-160, 5),
    )
    plot_episode_trends(
        manifest,
        "output_rmse_mean",
        "bounded_episode_output_rmse_trends.png",
        "5-episode smoothed physical output RMSE",
        ylim=(0, 1.9),
    )
    plot_gate_selector_trends(manifest)
    plot_last_episode_outputs(manifest)
    plot_target_diagnostics(metrics)
    write_report(metrics, phases, blocks, comparison, manifest)
    print(f"Wrote report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote figures/tables: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
