"""Build a shareable research bundle for LMPC target-selector diagnosis."""

from __future__ import annotations

import csv
import base64
import html
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.of_mpc_td3_workflow import build_polymer_setup, load_of_mpc_system_data

BUNDLE_DIR = ROOT / "report" / "bundles" / "2026-06-13_lmpc_target_selector_research_bundle"
FIG_DIR = BUNDLE_DIR / "figures"
TABLE_DIR = BUNDLE_DIR / "tables"

LMPC_RUNS = [
    {
        "label": "governed_reference_256",
        "short_label": "Governed 256",
        "pretrain": ROOT / "results" / "PretrainLMPC" / "20260610_005100",
        "comparison": ROOT / "results" / "PretrainLMPCComparison" / "20260610_173925",
        "expert": "direct_lmpc",
    },
    {
        "label": "bounded_mixed_256",
        "short_label": "Bounded 256",
        "pretrain": ROOT / "results" / "PretrainLMPC" / "20260611_003808",
        "comparison": ROOT / "results" / "PretrainLMPCComparison" / "20260612_004517",
        "expert": "direct_lmpc",
    },
    {
        "label": "bounded_mixed_512",
        "short_label": "Bounded 512",
        "pretrain": ROOT / "results" / "PretrainLMPC" / "20260612_011323",
        "comparison": ROOT / "results" / "PretrainLMPCComparison" / "20260613_003113",
        "expert": "direct_lmpc",
    },
]

OF_MPC_REFERENCE = {
    "label": "of_mpc_reference_256",
    "short_label": "OF-MPC 256",
    "pretrain": ROOT / "results" / "PretrainOFMPC" / "20260610_005048",
    "comparison": ROOT / "results" / "PretrainOFMPCComparison" / "20260610_154032",
    "expert": "offset_free_mpc",
}

LATEST_RUN = LMPC_RUNS[-1]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def repo_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def bundle_rel(path: Path) -> str:
    return path.relative_to(BUNDLE_DIR).as_posix()


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


def comparison_records(run: dict[str, Any]) -> list[dict[str, Any]]:
    return read_json(run["comparison"] / "comparison_metrics.json")["records"]


def load_lmpc_pretrain_row(run: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(run["pretrain"] / "summary.json")
    config = read_json(run["pretrain"] / "config.json")
    loss = read_json(run["pretrain"] / "loss_summary.json")
    controller = summary.get("controller") or config.get("controller", {})
    run_cfg = config.get("run_config", {})
    actor_layers = run_cfg.get("actor_layer_sizes") or summary.get("actor_layers_used")
    critic_layers = run_cfg.get("critic_layer_sizes") or summary.get("critic_layers_used")
    label_summary = summary.get("label_diagnostics_summary", {})
    return {
        "label": run["label"],
        "short_label": run["short_label"],
        "pretrain_dir": repo_rel(run["pretrain"]),
        "checkpoint": summary.get("checkpoint_path"),
        "actor_layers": json.dumps(actor_layers),
        "critic_layers": json.dumps(critic_layers),
        "lmpc_samples": summary.get("lmpc_samples"),
        "steady_samples": summary.get("steady_samples"),
        "target_mode": controller.get("target_mode"),
        "target_selector_variant": controller.get("target_selector_variant") or controller.get("target_mode"),
        "target_config": json.dumps(controller.get("target_config", {})),
        "rho_lyap": controller.get("rho_lyap"),
        "lyap_eps": controller.get("lyap_eps"),
        "acceptance_rate": label_summary.get("acceptance_rate"),
        "solve_success_rate": label_summary.get("solve_success_rate"),
        "broad_acceptance_rate": label_summary.get("broad_acceptance_rate"),
        "steady_acceptance_rate": label_summary.get("steady_acceptance_rate"),
        "reward_mean": summary.get("reward_stats", {}).get("mean"),
        "reward_std": summary.get("reward_stats", {}).get("std"),
        "actor_bc_loss_last": loss.get("series", {}).get("actor_bc_losses", {}).get("last"),
        "actor_bc_loss_mean": loss.get("series", {}).get("actor_bc_losses", {}).get("mean"),
        "critic_loss_last": loss.get("series", {}).get("critic_losses", {}).get("last"),
        "elapsed_hours": None if summary.get("elapsed_seconds") is None else float(summary["elapsed_seconds"]) / 3600.0,
    }


def load_of_mpc_reference_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_json(OF_MPC_REFERENCE["comparison"] / "comparison_metrics.json")["records"]
    comparison_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for rec in records:
        mode = rec["mode"]
        comparison_rows.append(
            {
                "run_label": OF_MPC_REFERENCE["label"],
                "short_label": OF_MPC_REFERENCE["short_label"],
                "mode": mode,
                "controller": "td3",
                "reward_mean": rec["rl_reward_mean"],
                "mean_rmse": rec["rl_mean_rmse"],
                "eta_rmse": rec["rl_eta_rmse"],
                "T_rmse": rec["rl_T_rmse"],
                "mean_abs_du": rec["rl_mean_abs_du"],
            }
        )
        comparison_rows.append(
            {
                "run_label": OF_MPC_REFERENCE["label"],
                "short_label": OF_MPC_REFERENCE["short_label"],
                "mode": mode,
                "controller": "offset_free_mpc",
                "reward_mean": rec["of_mpc_reward_mean"],
                "mean_rmse": rec["of_mpc_mean_rmse"],
                "eta_rmse": rec["of_mpc_eta_rmse"],
                "T_rmse": rec["of_mpc_T_rmse"],
                "mean_abs_du": rec["of_mpc_mean_abs_du"],
            }
        )
        gap_rows.append(
            {
                "run_label": OF_MPC_REFERENCE["label"],
                "short_label": OF_MPC_REFERENCE["short_label"],
                "mode": mode,
                "expert": "offset_free_mpc",
                "td3_minus_expert_reward": rec["rl_reward_mean"] - rec["of_mpc_reward_mean"],
                "td3_minus_expert_mean_rmse": rec["rl_mean_rmse"] - rec["of_mpc_mean_rmse"],
                "td3_abs_du_ratio": rec["rl_mean_abs_du"] / rec["of_mpc_mean_abs_du"],
                "td3_mean_rmse": rec["rl_mean_rmse"],
                "expert_mean_rmse": rec["of_mpc_mean_rmse"],
                "td3_reward": rec["rl_reward_mean"],
                "expert_reward": rec["of_mpc_reward_mean"],
            }
        )
    return comparison_rows, gap_rows


def collect_lmpc_comparison_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for run in LMPC_RUNS:
        records = comparison_records(run)
        for rec in records:
            row = {
                "run_label": run["label"],
                "short_label": run["short_label"],
                "mode": rec["mode"],
                "controller": rec["controller"],
                "reward_mean": rec.get("reward_mean"),
                "mean_rmse": rec.get("mean_rmse"),
                "eta_rmse": rec.get("eta_rmse"),
                "T_rmse": rec.get("T_rmse"),
                "mean_abs_du": rec.get("mean_abs_du"),
                "target_success_rate": rec.get("target_success_rate"),
                "solver_success_rate": rec.get("solver_success_rate"),
                "contraction_satisfied_rate": rec.get("contraction_satisfied_rate"),
                "target_stage_counts": json.dumps(rec.get("target_stage_counts", {})),
            }
            comparison_rows.append(row)
        for mode in sorted({rec["mode"] for rec in records}):
            td3 = next(rec for rec in records if rec["mode"] == mode and rec["controller"] == "td3")
            expert = next(rec for rec in records if rec["mode"] == mode and rec["controller"] == run["expert"])
            gap_rows.append(
                {
                    "run_label": run["label"],
                    "short_label": run["short_label"],
                    "mode": mode,
                    "expert": run["expert"],
                    "td3_minus_expert_reward": td3["reward_mean"] - expert["reward_mean"],
                    "td3_minus_expert_mean_rmse": td3["mean_rmse"] - expert["mean_rmse"],
                    "td3_abs_du_ratio": td3["mean_abs_du"] / expert["mean_abs_du"],
                    "td3_mean_rmse": td3["mean_rmse"],
                    "expert_mean_rmse": expert["mean_rmse"],
                    "td3_reward": td3["reward_mean"],
                    "expert_reward": expert["reward_mean"],
                }
            )
    return comparison_rows, gap_rows


def dev_to_phys_converter(pretrain_dir: Path):
    config = read_json(pretrain_dir / "config.json")
    setup = build_polymer_setup()
    system_data = load_of_mpc_system_data(setup)
    data_min = np.asarray(system_data["data_min"], dtype=float)
    data_max = np.asarray(system_data["data_max"], dtype=float)
    y_lo = data_min[-2:]
    y_hi = data_max[-2:]
    y_ss = np.asarray(config["system"]["steady_states"]["y_ss"], dtype=float)
    y_ss_scaled = (y_ss - y_lo) / (y_hi - y_lo)

    def convert(y_dev: np.ndarray) -> np.ndarray:
        return (np.asarray(y_dev, dtype=float) + y_ss_scaled) * (y_hi - y_lo) + y_lo

    return convert


def numeric_info_array(infos: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for info in infos:
        value = info.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def target_diagnostics_for_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    records = comparison_records(run)
    convert = dev_to_phys_converter(run["pretrain"])
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec["controller"] != "direct_lmpc":
            continue
        artifact = ROOT / rec["artifact_path"]
        bundle = read_pickle(artifact)
        infos = bundle.get("target_info_storage", [])
        stages = Counter(str(info.get("solve_stage")) for info in infos)
        mismatch_dev: list[np.ndarray] = []
        mismatch_phys: list[np.ndarray] = []
        target_rate = numeric_info_array(infos, "target_rate_inf")
        exact_violation = numeric_info_array(infos, "exact_bound_violation_inf")
        u_ref_gap = numeric_info_array(infos, "us_u_ref_inf")
        headroom = numeric_info_array(infos, "input_headroom_min")
        active_lower: list[float] = []
        active_upper: list[float] = []
        for info in infos:
            if info.get("y_s") is not None and info.get("y_sp") is not None:
                y_s = np.asarray(info["y_s"], dtype=float).reshape(-1)
                y_sp = np.asarray(info["y_sp"], dtype=float).reshape(-1)
                n = min(y_s.size, y_sp.size)
                mismatch_dev.append(y_s[:n] - y_sp[:n])
                mismatch_phys.append(convert(y_s[:n]) - convert(y_sp[:n]))
            if info.get("bounded_active_lower_mask") is not None:
                active_lower.append(float(np.sum(np.asarray(info["bounded_active_lower_mask"], dtype=bool))))
            if info.get("bounded_active_upper_mask") is not None:
                active_upper.append(float(np.sum(np.asarray(info["bounded_active_upper_mask"], dtype=bool))))
        dev = np.asarray(mismatch_dev, dtype=float)
        phys = np.asarray(mismatch_phys, dtype=float)
        abs_dev = np.abs(dev)
        abs_phys = np.abs(phys)
        rows.append(
            {
                "run_label": run["label"],
                "short_label": run["short_label"],
                "mode": rec["mode"],
                "target_mode": bundle.get("target_mode"),
                "use_target_output_for_tracking": bundle.get("use_target_output_for_tracking"),
                "stage_counts": json.dumps(dict(stages)),
                "exact_or_governed_count": stages.get("frozen_output_disturbance_exact_bounded", 0)
                + stages.get("governed_reference_target", 0),
                "bounded_ls_count": stages.get("frozen_output_disturbance_bounded_ls", 0),
                "target_mismatch_dev_eta_mean_abs": float(np.nanmean(abs_dev[:, 0])) if abs_dev.size else np.nan,
                "target_mismatch_dev_T_mean_abs": float(np.nanmean(abs_dev[:, 1])) if abs_dev.size else np.nan,
                "target_mismatch_dev_inf_p95": float(np.nanpercentile(np.max(abs_dev, axis=1), 95)) if abs_dev.size else np.nan,
                "target_mismatch_dev_inf_max": float(np.nanmax(np.max(abs_dev, axis=1))) if abs_dev.size else np.nan,
                "target_mismatch_phys_eta_mean_abs": float(np.nanmean(abs_phys[:, 0])) if abs_phys.size else np.nan,
                "target_mismatch_phys_T_mean_abs": float(np.nanmean(abs_phys[:, 1])) if abs_phys.size else np.nan,
                "target_mismatch_phys_eta_max_abs": float(np.nanmax(abs_phys[:, 0])) if abs_phys.size else np.nan,
                "target_mismatch_phys_T_max_abs": float(np.nanmax(abs_phys[:, 1])) if abs_phys.size else np.nan,
                "target_rate_inf_mean": float(np.nanmean(target_rate)) if target_rate.size else np.nan,
                "target_rate_inf_p95": float(np.nanpercentile(target_rate, 95)) if target_rate.size else np.nan,
                "exact_bound_violation_mean": float(np.nanmean(exact_violation)) if exact_violation.size else np.nan,
                "exact_bound_violation_p95": float(np.nanpercentile(exact_violation, 95)) if exact_violation.size else np.nan,
                "us_u_ref_inf_mean": float(np.nanmean(u_ref_gap)) if u_ref_gap.size else np.nan,
                "us_u_ref_inf_p95": float(np.nanpercentile(u_ref_gap, 95)) if u_ref_gap.size else np.nan,
                "input_headroom_min_mean": float(np.nanmean(headroom)) if headroom.size else np.nan,
                "active_lower_count_mean": float(np.nanmean(active_lower)) if active_lower else np.nan,
                "active_upper_count_mean": float(np.nanmean(active_upper)) if active_upper else np.nan,
            }
        )
    return rows


def label_failure_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = read_json(run["pretrain"] / "label_diagnostics.json")
    rows: list[dict[str, Any]] = []
    for group in ("broad", "steady"):
        section = diagnostics.get(group, {})
        failures = section.get("failure_reasons", {}) or {}
        total_failures = sum(int(v) for v in failures.values())
        for reason, count in sorted(failures.items(), key=lambda item: int(item[1]), reverse=True):
            rows.append(
                {
                    "run_label": run["label"],
                    "short_label": run["short_label"],
                    "group": group,
                    "failure_reason": reason,
                    "count": int(count),
                    "share_of_failures": 0.0 if total_failures == 0 else int(count) / float(total_failures),
                }
            )
    return rows


def collect_scaler_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in LMPC_RUNS:
        config = read_json(run["pretrain"] / "config.json")
        comparison_summary = read_json(run["comparison"] / "summary.json")
        controller = config.get("controller", {})
        system = config.get("system", {})
        scaling = comparison_summary.get("scaling", {})
        min_max = system.get("min_max_dict", {})
        setpoint_scaler = (
            controller.get("setpoint_scaler_y_phys")
            or scaling.get("setpoint_scaler_y_phys")
            or []
        )
        comparison_setpoints = scaling.get("comparison_setpoint_y_phys") or controller.get("rollout_setpoint_y_phys") or []
        inside_scaler = None
        if setpoint_scaler and comparison_setpoints:
            bounds = np.asarray(setpoint_scaler, dtype=float)
            setpoints = np.asarray(comparison_setpoints, dtype=float)
            lo = np.minimum(bounds[0], bounds[1])
            hi = np.maximum(bounds[0], bounds[1])
            inside_scaler = bool(np.all(setpoints >= lo) and np.all(setpoints <= hi))
        rows.append(
            {
                "run_label": run["label"],
                "short_label": run["short_label"],
                "pretrain_dir": repo_rel(run["pretrain"]),
                "comparison_dir": repo_rel(run["comparison"]),
                "setpoint_bounds_source": scaling.get("setpoint_bounds_source"),
                "state_bounds_source": scaling.get("state_bounds_source"),
                "setpoint_scaler_y_phys": json.dumps(setpoint_scaler),
                "comparison_setpoint_y_phys": json.dumps(comparison_setpoints),
                "comparison_setpoints_inside_scaler": inside_scaler,
                "y_sp_min_scaled_dev": json.dumps(min_max.get("y_sp_min")),
                "y_sp_max_scaled_dev": json.dumps(min_max.get("y_sp_max")),
                "u_min_scaled_dev": json.dumps(min_max.get("u_min")),
                "u_max_scaled_dev": json.dumps(min_max.get("u_max")),
                "u_min_phys": json.dumps(controller.get("u_min_phys")),
                "u_max_phys": json.dumps(controller.get("u_max_phys")),
            }
        )
    return rows


def collect_all() -> dict[str, Any]:
    pretrain_rows = [load_lmpc_pretrain_row(run) for run in LMPC_RUNS]
    comparison_rows, gap_rows = collect_lmpc_comparison_rows()
    of_rows, of_gap_rows = load_of_mpc_reference_rows()
    comparison_rows.extend(of_rows)
    gap_rows.extend(of_gap_rows)
    target_rows: list[dict[str, Any]] = []
    for run in LMPC_RUNS:
        target_rows.extend(target_diagnostics_for_run(run))
    failure_rows: list[dict[str, Any]] = []
    for run in LMPC_RUNS:
        failure_rows.extend(label_failure_rows(run))
    return {
        "pretrain": pd.DataFrame(pretrain_rows),
        "comparison": pd.DataFrame(comparison_rows),
        "gap": pd.DataFrame(gap_rows),
        "target": pd.DataFrame(target_rows),
        "failures": pd.DataFrame(failure_rows),
        "scaler_consistency": pd.DataFrame(collect_scaler_rows()),
    }


def make_figures(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    comparison = data["comparison"]
    gap = data["gap"]
    target = data["target"]
    failures = data["failures"]

    disturb_gap = gap.loc[gap["mode"] == "disturb"].copy()
    order = ["governed_reference_256", "bounded_mixed_256", "bounded_mixed_512", "of_mpc_reference_256"]
    disturb_gap["order"] = disturb_gap["run_label"].map({name: idx for idx, name in enumerate(order)})
    disturb_gap = disturb_gap.sort_values("order")
    x = np.arange(len(disturb_gap))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    labels = disturb_gap["short_label"].tolist()
    axes[0].bar(x, disturb_gap["td3_minus_expert_mean_rmse"], color="#4c78a8")
    axes[0].set_ylabel("TD3 - expert mean RMSE")
    axes[1].bar(x, disturb_gap["td3_minus_expert_reward"], color="#f58518")
    axes[1].set_ylabel("TD3 - expert reward")
    axes[2].bar(x, disturb_gap["td3_abs_du_ratio"], color="#54a24b")
    axes[2].set_ylabel("TD3 / expert mean abs du")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Disturbance rollout imitation gap")
    fig.tight_layout()
    paths["disturbance_gap"] = FIG_DIR / "disturbance_imitation_gap.png"
    fig.savefig(paths["disturbance_gap"], dpi=180)
    plt.close(fig)

    latest = comparison.loc[comparison["run_label"] == LATEST_RUN["label"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for idx, mode in enumerate(["nominal", "disturb"]):
        part = latest.loc[latest["mode"] == mode]
        xi = np.arange(len(part))
        axes[idx].bar(xi, part["mean_rmse"], color=["#e45756", "#4c78a8", "#72b7b2"][: len(part)])
        axes[idx].set_xticks(xi, part["controller"], rotation=25, ha="right")
        axes[idx].set_ylabel("Mean RMSE")
        axes[idx].set_title(f"Latest 512x5 comparison: {mode}")
        axes[idx].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    paths["latest_rmse"] = FIG_DIR / "latest_512_comparison_rmse.png"
    fig.savefig(paths["latest_rmse"], dpi=180)
    plt.close(fig)

    target_dist = target.loc[target["mode"] == "disturb"].copy()
    target_dist["order"] = target_dist["run_label"].map({name: idx for idx, name in enumerate([r["label"] for r in LMPC_RUNS])})
    target_dist = target_dist.sort_values("order")
    x = np.arange(len(target_dist))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    axes[0].bar(x - 0.18, target_dist["target_mismatch_phys_eta_mean_abs"], width=0.36, label="eta", color="#4c78a8")
    axes[0].bar(x + 0.18, target_dist["target_mismatch_phys_T_mean_abs"], width=0.36, label="T", color="#f58518")
    axes[0].set_ylabel("Mean abs target-setpoint mismatch, physical")
    axes[0].legend()
    axes[1].bar(x, target_dist["target_mismatch_dev_inf_p95"], color="#e45756")
    axes[1].set_ylabel("95th percentile mismatch, scaled-deviation inf norm")
    for ax in axes:
        ax.set_xticks(x, target_dist["short_label"], rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Disturbance Direct LMPC target mismatch")
    fig.tight_layout()
    paths["target_mismatch"] = FIG_DIR / "target_mismatch_diagnostics.png"
    fig.savefig(paths["target_mismatch"], dpi=180)
    plt.close(fig)

    stage_rows = []
    for _, row in target_dist.iterrows():
        counts = json.loads(row["stage_counts"])
        total = max(1, sum(int(v) for v in counts.values()))
        stage_rows.append(
            {
                "label": row["short_label"],
                "exact_or_governed": row["exact_or_governed_count"] / total,
                "bounded_ls": row["bounded_ls_count"] / total,
            }
        )
    stage_df = pd.DataFrame(stage_rows)
    x = np.arange(len(stage_df))
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x, 100.0 * stage_df["exact_or_governed"], label="exact/governed", color="#4c78a8")
    ax.bar(x, 100.0 * stage_df["bounded_ls"], bottom=100.0 * stage_df["exact_or_governed"], label="bounded LS", color="#f58518")
    ax.set_xticks(x, stage_df["label"], rotation=25, ha="right")
    ax.set_ylabel("Stage share (%)")
    ax.set_title("Disturbance target-selector stage usage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    paths["stage_usage"] = FIG_DIR / "target_selector_stage_usage.png"
    fig.savefig(paths["stage_usage"], dpi=180)
    plt.close(fig)

    latest_fail = failures.loc[
        (failures["run_label"] == LATEST_RUN["label"]) & (failures["group"] == "broad")
    ].head(8)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(latest_fail))
    ax.barh(y, latest_fail["count"], color="#e45756")
    ax.set_yticks(y, latest_fail["failure_reason"])
    ax.invert_yaxis()
    ax.set_xlabel("Rejected broad-label attempts")
    ax.set_title("Latest 512x5 LMPC label rejection reasons")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    paths["failures"] = FIG_DIR / "latest_label_failure_reasons.png"
    fig.savefig(paths["failures"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(11.6, 8.2), sharex=True)
    latest_records = comparison_records(LATEST_RUN)
    bundles = {}
    for controller in ("td3", "direct_lmpc", "offset_free_mpc"):
        rec = next(rec for rec in latest_records if rec["mode"] == "disturb" and rec["controller"] == controller)
        bundles[controller] = read_pickle(ROOT / rec["artifact_path"])
    convert = dev_to_phys_converter(LATEST_RUN["pretrain"])
    n = int(min(bundle["y_sp"].shape[0] for bundle in bundles.values()))
    t = np.arange(n)
    y_sp_phys = convert(bundles["direct_lmpc"]["y_sp"][:n])
    output_labels = ["eta", "T"]
    for j in range(2):
        ax = axes[j]
        for controller, bundle in bundles.items():
            ax.plot(t, bundle["y_system"][:n, j], linewidth=1.0, label=controller)
        ax.plot(t, y_sp_phys[:, j], color="black", linestyle="--", linewidth=1.1, label="setpoint")
        ax.set_ylabel(output_labels[j])
        ax.grid(True, alpha=0.25)
    input_labels = ["Qc", "Qm"]
    for j in range(2):
        ax = axes[j + 2]
        for controller, bundle in bundles.items():
            ax.plot(t, bundle["u_applied_phys"][:n, j], linewidth=1.0, label=controller)
        ax.set_ylabel(input_labels[j])
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8, loc="best")
    axes[-1].set_xlabel("Step")
    fig.suptitle("Latest 512x5 disturbance rollout: TD3 versus Direct LMPC and OF-MPC")
    fig.tight_layout()
    paths["rollout"] = FIG_DIR / "latest_512_disturbance_rollout_overlay.png"
    fig.savefig(paths["rollout"], dpi=180)
    plt.close(fig)
    return paths


def make_tables(data: dict[str, pd.DataFrame]) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in data.items():
        path = TABLE_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    source_rows = [
        {
            "artifact": "latest_lmpc_pretrain",
            "path": repo_rel(LATEST_RUN["pretrain"]),
            "description": "Latest 512x5 bounded-mixed LMPC pretraining bundle.",
        },
        {
            "artifact": "latest_lmpc_comparison",
            "path": repo_rel(LATEST_RUN["comparison"]),
            "description": "Latest 512x5 comparison against Direct LMPC and OF-MPC.",
        },
        {
            "artifact": "governed_reference_pretrain",
            "path": repo_rel(LMPC_RUNS[0]["pretrain"]),
            "description": "Historical governed-reference LMPC pretraining bundle.",
        },
        {
            "artifact": "governed_reference_comparison",
            "path": repo_rel(LMPC_RUNS[0]["comparison"]),
            "description": "Historical governed-reference comparison bundle.",
        },
        {
            "artifact": "of_mpc_reference_comparison",
            "path": repo_rel(OF_MPC_REFERENCE["comparison"]),
            "description": "Positive-control OF-MPC TD3 pretraining comparison.",
        },
    ]
    path = TABLE_DIR / "source_artifacts.csv"
    write_csv(path, source_rows)
    paths["source_artifacts"] = path
    return paths


def report_text(data: dict[str, pd.DataFrame], figs: dict[str, Path], tables: dict[str, Path]) -> str:
    pretrain = data["pretrain"]
    gap = data["gap"]
    target = data["target"]
    failures = data["failures"]
    scaler = data["scaler_consistency"]

    pretrain_rows = []
    for _, row in pretrain.iterrows():
        pretrain_rows.append(
            {
                "Run": row["short_label"],
                "Layers": row["actor_layers"],
                "Samples": fmt(row["lmpc_samples"], 0),
                "Selector": row["target_selector_variant"],
                "Accept": fmt(row["acceptance_rate"]),
                "BC last": fmt(row["actor_bc_loss_last"], 2),
            }
        )

    scaler_rows = []
    for _, row in scaler.iterrows():
        scaler_rows.append(
            {
                "Run": row["short_label"],
                "SP scaler": row["setpoint_scaler_y_phys"],
                "Comparison SP": row["comparison_setpoint_y_phys"],
                "Inside": row["comparison_setpoints_inside_scaler"],
                "u phys": f"{row['u_min_phys']} to {row['u_max_phys']}",
            }
        )

    gap_rows = []
    for _, row in gap.loc[gap["mode"] == "disturb"].iterrows():
        gap_rows.append(
            {
                "Run": row["short_label"],
                "Expert": row["expert"],
                "TD3 RMSE": fmt(row["td3_mean_rmse"]),
                "Expert RMSE": fmt(row["expert_mean_rmse"]),
                "RMSE gap": fmt(row["td3_minus_expert_mean_rmse"]),
                "Reward gap": fmt(row["td3_minus_expert_reward"]),
                "dU ratio": fmt(row["td3_abs_du_ratio"]),
            }
        )

    target_rows = []
    for _, row in target.loc[target["mode"] == "disturb"].iterrows():
        target_rows.append(
            {
                "Run": row["short_label"],
                "Stages": row["stage_counts"],
                "eta mismatch": fmt(row["target_mismatch_phys_eta_mean_abs"]),
                "T mismatch": fmt(row["target_mismatch_phys_T_mean_abs"]),
                "p95 dev": fmt(row["target_mismatch_dev_inf_p95"]),
                "max T mismatch": fmt(row["target_mismatch_phys_T_max_abs"]),
            }
        )

    latest_failure_rows = []
    latest_fail = failures.loc[
        (failures["run_label"] == LATEST_RUN["label"]) & (failures["group"] == "broad")
    ].head(8)
    for _, row in latest_fail.iterrows():
        latest_failure_rows.append(
            {
                "Failure": row["failure_reason"],
                "Count": fmt(row["count"], 0),
                "Share": fmt(100.0 * row["share_of_failures"]),
            }
        )

    latest_gap = gap.loc[(gap["run_label"] == LATEST_RUN["label"]) & (gap["mode"] == "disturb")].iloc[0]
    bounded_256_gap = gap.loc[(gap["run_label"] == "bounded_mixed_256") & (gap["mode"] == "disturb")].iloc[0]
    governed_gap = gap.loc[(gap["run_label"] == "governed_reference_256") & (gap["mode"] == "disturb")].iloc[0]
    of_gap = gap.loc[(gap["run_label"] == "of_mpc_reference_256") & (gap["mode"] == "disturb")].iloc[0]
    latest_pretrain = pretrain.loc[pretrain["label"] == LATEST_RUN["label"]].iloc[0]
    latest_target = target.loc[(target["run_label"] == LATEST_RUN["label"]) & (target["mode"] == "disturb")].iloc[0]

    return f"""# LMPC Target Selector Research Bundle

Date: 2026-06-13

## Purpose

This bundle summarizes why the latest Direct LMPC TD3 pretraining result is
still not convincing, even with a much larger `[512, 512, 512, 512, 512]`
actor/critic. It is intended to be shared with ChatGPT or a deep-research
agent to search for better target-selector designs.

## Executive Finding

The result does not look like a network-capacity problem. The latest 512x5
bounded-mixed LMPC actor reaches a very small final BC loss
(`{fmt(latest_pretrain['actor_bc_loss_last'], 2)}`), but its disturbance
rollout is worse than the older 256x3 bounded-mixed actor and worse than the
historical governed-reference actor. In the disturbance comparison, the latest
TD3 policy has mean RMSE `{fmt(latest_gap['td3_mean_rmse'])}` versus Direct
LMPC `{fmt(latest_gap['expert_mean_rmse'])}`, and it moves the inputs
`{fmt(latest_gap['td3_abs_du_ratio'])}x` as much as the expert.

The governed-reference selector was also not a sufficient answer. Its
disturbance TD3-vs-expert RMSE gap was `{fmt(governed_gap['td3_minus_expert_mean_rmse'])}`,
while bounded-mixed 256 had gap `{fmt(bounded_256_gap['td3_minus_expert_mean_rmse'])}`
and bounded-mixed 512 has gap `{fmt(latest_gap['td3_minus_expert_mean_rmse'])}`.
By contrast, the OF-MPC pretrained TD3 positive control has a disturbance gap
of only `{fmt(of_gap['td3_minus_expert_mean_rmse'])}`.

![Disturbance imitation gap]({bundle_rel(figs['disturbance_gap'])})

## Runs In This Bundle

{md_table(pretrain_rows, [
        ("Run", "Run"),
        ("Layers", "Layers"),
        ("Samples", "Samples"),
        ("Selector", "Selector"),
        ("Accept", "Accept"),
        ("BC last", "BC last"),
    ])}

The latest run used:

- `target_mode = bounded`
- `target_selector_variant = bounded_mixed_u0p1_x0p1`
- `target_config = {{\"u_ref_weight\": 0.1, \"x_ref_weight\": 0.1}}`
- `rho_lyap = 0.99`
- `lyap_eps = 1e-3`
- `predict_h = 9`, `cont_h = 3`
- `use_target_output_for_tracking = False`

## Scaling Contract Check

The LMPC pretraining and comparison runs use the same TD3 scaled-deviation
state/action contract. The comparison setpoints are inside the exported
physical setpoint scaler for all LMPC runs, so the latest failure is not
explained by the earlier setpoint-range mismatch problem.

{md_table(scaler_rows, [
        ("Run", "Run"),
        ("SP scaler", "SP scaler"),
        ("Comparison SP", "Comparison SP"),
        ("Inside", "Inside"),
        ("u phys", "u phys"),
    ])}

## Mathematical Reconstruction

The target selector is solving a steady target for the output-disturbance model
in scaled deviation coordinates. With augmented observer state
$\\hat z_k=[\\hat x_k^\\top,\\hat d_k^\\top]^\\top$, the target satisfies

$$
x_s = A x_s + B u_s, \\qquad
y_s = C x_s + d_s,
$$

with $d_s=\\hat d_k$ and input bounds $u_{{\\min}}\\le u_s\\le u_{{\\max}}$.

The bounded-mixed selector first tries the exact raw-setpoint steady target. If
the exact input is outside bounds, it solves a bounded least-squares target with
small anchoring penalties,

$$
\\min_{{x_s,u_s}}
\\left\\|y_s-y_{{sp}}\\right\\|^2
 + 0.1\\left\\|u_s-u_{{k-1}}\\right\\|^2
 + 0.1\\left\\|x_s-x_{{s,k-1}}\\right\\|^2,
$$

subject to the steady-state equations and input bounds. The Direct LMPC
tracking objective still tracks the raw setpoint because
`use_target_output_for_tracking=False`, but the Lyapunov certificate is centered
on $(x_s,u_s,y_s)$.

For the governed-reference selector, the target command $r_s$ is itself
governed before solving the steady target. That made the target smoother and
often feasible, but it also means the certified target can be away from the raw
setpoint.

## Comparison Performance

{md_table(gap_rows, [
        ("Run", "Run"),
        ("Expert", "Expert"),
        ("TD3 RMSE", "TD3 RMSE"),
        ("Expert RMSE", "Expert RMSE"),
        ("RMSE gap", "RMSE gap"),
        ("Reward gap", "Reward gap"),
        ("dU ratio", "dU ratio"),
    ])}

![Latest 512 comparison](figures/latest_512_comparison_rmse.png)

The latest 512x5 model is not an improvement over the 256x3 bounded-mixed
model. Its final BC loss is lower, but deterministic closed-loop comparison is
worse. This is the strongest evidence that lower supervised loss on the broad
offline replay distribution is not the same as learning the closed-loop LMPC
expert behavior that matters.

![Latest rollout overlay](figures/latest_512_disturbance_rollout_overlay.png)

## Target Selector Diagnostics

{md_table(target_rows, [
        ("Run", "Run"),
        ("Stages", "Stages"),
        ("eta mismatch", "eta mismatch"),
        ("T mismatch", "T mismatch"),
        ("p95 dev", "p95 dev"),
        ("max T mismatch", "max T mismatch"),
    ])}

![Target mismatch diagnostics](figures/target_mismatch_diagnostics.png)

![Target stage usage](figures/target_selector_stage_usage.png)

For the latest bounded-mixed disturbance Direct LMPC baseline, the exact raw
setpoint target is usable in only `{int(latest_target['exact_or_governed_count'])}`
of 1600 steps. The bounded least-squares selector is used in
`{int(latest_target['bounded_ls_count'])}` steps. Its mean physical
target-setpoint mismatch is `{fmt(latest_target['target_mismatch_phys_eta_mean_abs'])}`
in eta and `{fmt(latest_target['target_mismatch_phys_T_mean_abs'])}` in T, with
maximum T mismatch `{fmt(latest_target['target_mismatch_phys_T_max_abs'])}`.

This does not mean Direct LMPC itself is bad. Direct LMPC tracks well in the
comparison. The problem is that the offline supervised actor sees a label map
generated by a target-selection plus tracking plus Lyapunov-feasibility
pipeline. That map is much more conditional and less smooth than the OF-MPC
expert map.

## Label Rejection Pattern

{md_table(latest_failure_rows, [
        ("Failure", "Failure"),
        ("Count", "Count"),
        ("Share", "Share %"),
    ])}

![Latest label failures](figures/latest_label_failure_reasons.png)

The latest broad label pool accepts about `{fmt(latest_pretrain['acceptance_rate'])}`
of attempts overall. The largest rejected class is
`tracking:optimal:dyn_residual`, which means the optimizer status can be
acceptable but the post-check rejects the candidate because it does not satisfy
the model consistency check. This creates a conditional replay set: the actor
sees successful labels but not the surrounding feasibility boundary.

## Why The Target Selector Is The Main Suspect

1. Scaling is consistent across the pretraining and comparison contracts.
2. Direct LMPC and OF-MPC baselines track almost identically in the comparison.
3. OF-MPC TD3 imitation is excellent under the same TD3 state/action dimensions.
4. LMPC TD3 imitation is poor for governed-reference, bounded-mixed 256, and
   bounded-mixed 512.
5. The latest larger network reduces supervised loss but worsens rollout
   behavior, so architecture size alone is not the bottleneck.
6. Both target selectors can move the certified target away from the raw
   setpoint. This is acceptable for practical Lyapunov certification, but it
   creates a hard expert map for offline actor imitation.

## Research Directions To Explore

The next target-selector search should focus on making the expert map smoother,
more closed-loop relevant, and less sensitive to target-stage switches:

- A two-layer selector that first minimizes raw output mismatch, then only uses
  $u_{{k-1}}$ and $x_{{s,k-1}}$ as true tie-breakers.
- A reference-governor selector with an explicit bound on target movement and a
  reported raw-setpoint tracking loss.
- A multi-step reachable target selector instead of a steady target only.
- A soft Lyapunov/filter formulation that returns a correction direction and
  margin rather than a hard accept/reject label.
- DAgger-style relabeling on states visited by the current actor instead of only
  broad-uniform synthetic states.
- A selector-quality gate for pretraining labels, so labels with large
  target-setpoint mismatch or large target jumps are either separated, weighted,
  or excluded from actor BC.

## Bundle Files

- Pretraining summary table: `{bundle_rel(tables['pretrain'])}`
- Comparison metrics table: `{bundle_rel(tables['comparison'])}`
- TD3 expert gap table: `{bundle_rel(tables['gap'])}`
- Target diagnostics table: `{bundle_rel(tables['target'])}`
- Label failure table: `{bundle_rel(tables['failures'])}`
- Scaling consistency table: `{bundle_rel(tables['scaler_consistency'])}`
- Source artifact paths: `{bundle_rel(tables['source_artifacts'])}`
- Deep research prompt: `deep_research_prompt.md`

## What To Hand To A Research Agent

Give the agent this whole folder and ask it to read `README.md`,
`deep_research_prompt.md`, and the CSV files under `tables/`. The raw run
artifacts remain in `results/` and are referenced in `tables/source_artifacts.csv`.
"""


def prompt_text() -> str:
    return """# Deep Research Prompt: Better Direct LMPC Target Selector

I am working on safe RL/MPC for a polymer CSTR. The current Direct LMPC TD3
pretraining workflow is not giving a good pretrained actor, even after
increasing the actor and critic to `[512, 512, 512, 512, 512]`.

Please use the attached bundle to help design a better target selector.

## Evidence To Use

- `README.md`: high-level diagnosis.
- `tables/pretrain.csv`: pretraining configurations and losses.
- `tables/comparison.csv`: closed-loop comparison metrics.
- `tables/gap.csv`: TD3-vs-expert gaps.
- `tables/target.csv`: Direct LMPC target-selector diagnostics.
- `tables/failures.csv`: LMPC label rejection reasons.
- `figures/*.png`: visual summaries.

## Important Facts

- OF-MPC TD3 pretraining works well under the same plant, scaler, TD3 state/action dimensions, and comparison setpoints.
- Direct LMPC and OF-MPC baselines track almost identically.
- LMPC TD3 imitation is poor for both governed-reference and bounded-mixed selectors.
- Increasing the network to 512x5 lowered supervised BC loss but worsened closed-loop comparison.
- The bounded-mixed selector often uses a bounded least-squares target instead of the exact raw-setpoint steady target.
- The governed-reference selector also produced poor LMPC-TD3 imitation.
- Direct LMPC uses the selected target for Lyapunov certification but still tracks the raw setpoint in the MPC objective.

## Research Questions

1. What target-selector formulations are common in offset-free tracking MPC,
   reference governors, command governors, and Lyapunov MPC when the raw
   setpoint may be unreachable?
2. How can a target selector preserve practical Lyapunov contraction while
   producing a smoother expert action map for offline RL imitation?
3. Should the selector be lexicographic: first minimize raw setpoint mismatch,
   then use input/state regularization only as a tie-breaker?
4. Would a multi-step reachable reference or dynamic reference governor be more
   suitable than a steady target selector?
5. How should rejected/infeasible LMPC label regions be represented during
   offline pretraining?
6. What label-quality metrics should be logged and filtered before actor BC?
7. What concrete ablation plan should be run next?

Please produce a literature-backed target-selector redesign plan with equations,
implementation-level details, and a small ablation matrix.
"""


def _inline_image(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _html_table_from_csv(path: Path, max_rows: int | None = None) -> str:
    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_html(index=False, escape=True, border=0, classes="data-table")


def _markdown_to_simple_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    in_list = False
    in_code = False
    code_lines: list[str] = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    def flush_code() -> None:
        nonlocal in_code, code_lines
        if in_code:
            output.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
            code_lines = []
            in_code = False

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
            else:
                close_list()
                in_code = True
                code_lines = []
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            close_list()
            continue
        if line.startswith("# "):
            close_list()
            output.append(f"<h1>{html.escape(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            close_list()
            output.append(f"<h2>{html.escape(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            close_list()
            output.append(f"<h3>{html.escape(line[4:].strip())}</h3>")
        elif line.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{html.escape(line[2:].strip())}</li>")
        elif line[0:2].isdigit() and ". " in line[:5]:
            close_list()
            output.append(f"<p>{html.escape(line)}</p>")
        else:
            close_list()
            output.append(f"<p>{html.escape(line)}</p>")
    flush_code()
    close_list()
    return "\n".join(output)


def html_text(data: dict[str, pd.DataFrame], figs: dict[str, Path], tables: dict[str, Path]) -> str:
    readme = (BUNDLE_DIR / "README.md").read_text(encoding="utf-8")
    prompt = prompt_text()
    figure_blocks = []
    for title, key in [
        ("Disturbance Imitation Gap", "disturbance_gap"),
        ("Latest 512x5 RMSE Comparison", "latest_rmse"),
        ("Latest 512x5 Disturbance Rollout", "rollout"),
        ("Target Mismatch Diagnostics", "target_mismatch"),
        ("Target Selector Stage Usage", "stage_usage"),
        ("Latest Label Failure Reasons", "failures"),
    ]:
        path = figs[key]
        figure_blocks.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f"<img src=\"{_inline_image(path)}\" alt=\"{html.escape(title)}\"></section>"
        )

    table_blocks = []
    for title, key in [
        ("Pretraining Summary", "pretrain"),
        ("Closed-Loop Comparison Metrics", "comparison"),
        ("TD3 Versus Expert Gaps", "gap"),
        ("Target Diagnostics", "target"),
        ("Label Failure Reasons", "failures"),
        ("Scaling Consistency", "scaler_consistency"),
        ("Source Artifacts", "source_artifacts"),
    ]:
        table_blocks.append(
            f"<section><h2>{html.escape(title)}</h2>{_html_table_from_csv(tables[key])}</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LMPC Target Selector Research Bundle</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
      margin: 0;
      color: #17202a;
      background: #f7f8fa;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 24px 72px;
      background: white;
    }}
    h1, h2, h3 {{ color: #102033; }}
    h1 {{ font-size: 2rem; border-bottom: 2px solid #d7dde7; padding-bottom: 0.4rem; }}
    h2 {{ margin-top: 2rem; border-bottom: 1px solid #e3e7ee; padding-bottom: 0.25rem; }}
    code, pre {{
      background: #f0f3f7;
      border-radius: 4px;
    }}
    code {{ padding: 0.1rem 0.25rem; }}
    pre {{ padding: 1rem; overflow-x: auto; }}
    img {{
      display: block;
      max-width: 100%;
      margin: 12px 0 28px;
      border: 1px solid #d7dde7;
      border-radius: 6px;
      background: white;
    }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.86rem;
      margin: 0.5rem 0 1.5rem;
    }}
    .data-table th, .data-table td {{
      border: 1px solid #d7dde7;
      padding: 0.35rem 0.45rem;
      vertical-align: top;
    }}
    .data-table th {{
      background: #eef2f7;
      text-align: left;
      position: sticky;
      top: 0;
    }}
    .note {{
      background: #fff8e5;
      border-left: 4px solid #d59b00;
      padding: 0.8rem 1rem;
      margin: 1rem 0;
    }}
  </style>
</head>
<body>
<main>
  <h1>LMPC Target Selector Research Bundle</h1>
  <p class="note">Single-file HTML export generated from the local bundle on 2026-06-13. Images are embedded, and source tables are included below for easy sharing.</p>
  <section>
    <h2>Deep Research Prompt</h2>
    {_markdown_to_simple_html(prompt)}
  </section>
  <section>
    <h2>Report</h2>
    {_markdown_to_simple_html(readme)}
  </section>
  {"".join(figure_blocks)}
  {"".join(table_blocks)}
</main>
</body>
</html>
"""


def main() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_all()
    tables = make_tables(data)
    figs = make_figures(data)
    (BUNDLE_DIR / "README.md").write_text(report_text(data, figs, tables), encoding="utf-8")
    (BUNDLE_DIR / "deep_research_prompt.md").write_text(prompt_text(), encoding="utf-8")
    (BUNDLE_DIR / "lmpc_target_selector_research_bundle.html").write_text(
        html_text(data, figs, tables),
        encoding="utf-8",
    )
    print(f"Wrote {repo_rel(BUNDLE_DIR)}")
    print(f"Wrote {repo_rel(BUNDLE_DIR / 'README.md')}")
    print(f"Wrote {repo_rel(BUNDLE_DIR / 'deep_research_prompt.md')}")
    print(f"Wrote {repo_rel(BUNDLE_DIR / 'lmpc_target_selector_research_bundle.html')}")


if __name__ == "__main__":
    main()
