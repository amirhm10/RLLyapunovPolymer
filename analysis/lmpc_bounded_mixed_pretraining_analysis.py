"""Analyze bounded-mixed LMPC pretraining against OF-MPC pretraining.

The script is intentionally report-oriented: it reads saved experiment bundles,
checks scaling/config consistency, writes compact CSV summaries, and creates a
Markdown report with figures.
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

LMPC_NEW_PRETRAIN = REPO_ROOT / "results" / "PretrainLMPC" / "20260611_003808"
LMPC_OLD_PRETRAIN = REPO_ROOT / "results" / "PretrainLMPC" / "20260610_005100"
OF_PRETRAIN = REPO_ROOT / "results" / "PretrainOFMPC" / "20260610_005048"

LMPC_NEW_COMPARE = REPO_ROOT / "results" / "PretrainLMPCComparison" / "20260612_004517"
LMPC_OLD_COMPARE = REPO_ROOT / "results" / "PretrainLMPCComparison" / "20260610_173925"
OF_COMPARE = REPO_ROOT / "results" / "PretrainOFMPCComparison" / "20260610_154032"

BASELINE_CACHE = REPO_ROOT / "results" / "PretrainLMPCComparison" / "baselines"
DIRECT_LMPC_DISTURB_BASELINE = (
    BASELINE_CACHE
    / "direct_lmpc_disturb_n2_len400_disturb_before_q5_1_r1_1_rho0p99_eps0p001_"
    "target_bounded_bounded_mixed_u0p1_x0p1_u0p1_x0p1.pickle"
)
OF_MPC_DISTURB_BASELINE = (
    BASELINE_CACHE
    / "offset_free_mpc_disturb_n2_len400_disturb_before_q5_1_r1_1_rho0p99_eps0p001_"
    "target_bounded_bounded_mixed_u0p1_x0p1_u0p1_x0p1.pickle"
)

REPORT_PATH = REPO_ROOT / "report" / "lmpc_bounded_mixed_pretraining_analysis_2026-06-12.md"
FIG_DIR = REPO_ROOT / "report" / "figures" / "2026-06-12_lmpc_bounded_mixed_pretraining_analysis"
TABLE_DIR = REPO_ROOT / "report" / "tables" / "2026-06-12_lmpc_bounded_mixed_pretraining_analysis"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def rel_report(path: Path) -> str:
    return path.relative_to(REPORT_PATH.parent).as_posix()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if 0 < abs(number) < 1e-4:
        return f"{number:.3e}"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 10:
        return f"{number:.3f}"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [title for _, title in columns]
    aligns = [":---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        values = [str(row.get(key, "")) for key, _title in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def nested_get(obj: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = obj
    for key in keys:
        if isinstance(cur, dict):
            if key not in cur:
                return None
            cur = cur[key]
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return None
    return cur


def as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def arrays_match(a: Any, b: Any, *, atol: float = 1e-10) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return bool(np.allclose(as_array(a), as_array(b), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return a == b


def summarize_vector(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        return fmt(float(arr), digits=digits)
    rounded = np.round(arr.astype(float), digits)
    return np.array2string(rounded, separator=", ", max_line_width=120)


def load_pretrain_summary(name: str, path: Path) -> dict[str, Any]:
    summary = read_json(path / "summary.json")
    config = read_json(path / "config.json")
    loss = read_json(path / "loss_summary.json")
    label = read_json(path / "label_diagnostics.json") if (path / "label_diagnostics.json").exists() else {}
    return {
        "name": name,
        "path": path,
        "summary": summary,
        "config": config,
        "loss": loss,
        "label": label,
    }


def normalize_lmpc_metrics(compare_dir: Path, experiment: str) -> list[dict[str, Any]]:
    records = read_json(compare_dir / "comparison_metrics.json")["records"]
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "experiment": experiment,
                "mode": record["mode"],
                "controller": record["controller"],
                "reward_mean": float(record["reward_mean"]),
                "mean_rmse": float(record["mean_rmse"]),
                "eta_rmse": float(record["eta_rmse"]),
                "T_rmse": float(record["T_rmse"]),
                "mean_abs_du": float(record["mean_abs_du"]),
                "eta_iae": float(record["eta_iae"]),
                "T_iae": float(record["T_iae"]),
            }
        )
    return rows


def normalize_of_metrics(compare_dir: Path) -> list[dict[str, Any]]:
    records = read_json(compare_dir / "comparison_metrics.json")["records"]
    rows: list[dict[str, Any]] = []
    for record in records:
        for source, controller in (("rl", "td3"), ("of_mpc", "offset_free_mpc")):
            rows.append(
                {
                    "experiment": "OF-MPC pretrained TD3",
                    "mode": record["mode"],
                    "controller": controller,
                    "reward_mean": float(record[f"{source}_reward_mean"]),
                    "mean_rmse": float(record[f"{source}_mean_rmse"]),
                    "eta_rmse": float(record[f"{source}_eta_rmse"]),
                    "T_rmse": float(record[f"{source}_T_rmse"]),
                    "mean_abs_du": float(record[f"{source}_mean_abs_du"]),
                    "eta_iae": float(record[f"{source}_eta_iae"]),
                    "T_iae": float(record[f"{source}_T_iae"]),
                }
            )
    return rows


def compare_scaling() -> list[dict[str, Any]]:
    lmpc_config = read_json(LMPC_NEW_PRETRAIN / "config.json")
    of_config = read_json(OF_PRETRAIN / "config.json")
    lmpc_compare = read_json(LMPC_NEW_COMPARE / "summary.json")
    of_compare = read_json(OF_COMPARE / "summary.json")

    items = [
        (
            "training min_max_dict",
            nested_get(lmpc_config, ["system", "min_max_dict"]),
            nested_get(of_config, ["system", "min_max_dict"]),
        ),
        (
            "training TD3 setpoint scaler",
            nested_get(lmpc_config, ["controller", "setpoint_scaler_y_phys"]),
            nested_get(of_config, ["controller", "setpoint_y_phys"]),
        ),
        (
            "state bounds source",
            nested_get(lmpc_compare, ["scaling", "state_bounds_source"]),
            nested_get(of_compare, ["scaling", "state_bounds_source"]),
        ),
        (
            "setpoint bounds source",
            nested_get(lmpc_compare, ["scaling", "setpoint_bounds_source"]),
            nested_get(of_compare, ["scaling", "setpoint_bounds_source"]),
        ),
        (
            "physical input lower bounds",
            nested_get(lmpc_config, ["controller", "u_min_phys"]),
            nested_get(of_config, ["controller", "u_min_phys"]),
        ),
        (
            "physical input upper bounds",
            nested_get(lmpc_config, ["controller", "u_max_phys"]),
            nested_get(of_config, ["controller", "u_max_phys"]),
        ),
        (
            "MPC output weights",
            nested_get(lmpc_config, ["controller", "Qy_mpc_diag"]),
            nested_get(of_config, ["controller", "Q_mpc"]),
        ),
        (
            "MPC input weights",
            nested_get(lmpc_config, ["controller", "Rdu_mpc_diag"]),
            nested_get(of_config, ["controller", "R_mpc"]),
        ),
        (
            "TD3 actor hidden layers",
            nested_get(lmpc_config, ["td3", "actor_hidden"]),
            nested_get(of_config, ["td3", "actor_hidden"]),
        ),
        (
            "TD3 critic hidden layers",
            nested_get(lmpc_config, ["td3", "critic_hidden"]),
            nested_get(of_config, ["td3", "critic_hidden"]),
        ),
        (
            "comparison setpoint scaler",
            nested_get(lmpc_compare, ["scaling", "setpoint_scaler_y_phys"]),
            nested_get(of_compare, ["scaling", "setpoint_scaler_y_phys"]),
        ),
        (
            "comparison y_sp_min",
            nested_get(lmpc_compare, ["scaling", "y_sp_min"]),
            nested_get(of_compare, ["scaling", "y_sp_min"]),
        ),
        (
            "comparison y_sp_max",
            nested_get(lmpc_compare, ["scaling", "y_sp_max"]),
            nested_get(of_compare, ["scaling", "y_sp_max"]),
        ),
        (
            "comparison rollout setpoints",
            nested_get(lmpc_compare, ["scaling", "comparison_setpoint_y_phys"]),
            nested_get(of_compare, ["scaling", "comparison_setpoint_y_phys"]),
        ),
    ]
    rows = []
    for item, lmpc_value, of_value in items:
        rows.append(
            {
                "item": item,
                "match": "yes" if arrays_match(lmpc_value, of_value) else "NO",
                "lmpc": summarize_vector(lmpc_value) if not isinstance(lmpc_value, str) else lmpc_value,
                "of_mpc": summarize_vector(of_value) if not isinstance(of_value, str) else of_value,
            }
        )
    return rows


def pretraining_rows() -> list[dict[str, Any]]:
    runs = [
        load_pretrain_summary("LMPC bounded-mixed", LMPC_NEW_PRETRAIN),
        load_pretrain_summary("LMPC governed-ref old", LMPC_OLD_PRETRAIN),
        load_pretrain_summary("OF-MPC", OF_PRETRAIN),
    ]
    rows: list[dict[str, Any]] = []
    for run in runs:
        summary = run["summary"]
        config = run["config"]
        loss = run["loss"]
        label = run["label"]
        controller = config.get("controller", {})
        if run["name"].startswith("OF-MPC"):
            selector = "offset_free_mpc"
        else:
            selector = (
                summary.get("target_selector_variant")
                or controller.get("target_selector_variant")
                or controller.get("target_mode")
                or "direct_lmpc"
            )
        rows.append(
            {
                "run": run["name"],
                "samples": int(summary.get("buffer_size", summary.get("total_samples", 0))),
                "accepted_rate": label.get("acceptance_rate"),
                "solve_success_rate": label.get("solve_success_rate"),
                "actor_bc_last": nested_get(loss, ["series", "actor_bc_losses", "last"]),
                "critic_last": nested_get(loss, ["series", "critic_losses", "last"]),
                "reward_mean": nested_get(summary, ["reward_stats", "mean"]),
                "action_std_1": nested_get(summary, ["action_stats", "std", 0]),
                "action_std_2": nested_get(summary, ["action_stats", "std", 1]),
                "selector": selector,
            }
        )
    return rows


def failure_rows() -> list[dict[str, Any]]:
    label = read_json(LMPC_NEW_PRETRAIN / "label_diagnostics.json")
    failure = Counter(label.get("broad", {}).get("failure_reasons", {}))
    total_failed = sum(failure.values())
    rows = []
    for reason, count in failure.most_common(8):
        rows.append(
            {
                "reason": reason,
                "count": int(count),
                "share_of_failures": float(count / total_failed) if total_failed else 0.0,
            }
        )
    return rows


def formatted_performance_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interesting = [
        row
        for row in metrics
        if row["experiment"] in {"LMPC bounded-mixed TD3", "OF-MPC pretrained TD3"}
        and row["controller"] in {"td3", "direct_lmpc", "offset_free_mpc"}
    ]
    order = {
        ("LMPC bounded-mixed TD3", "td3"): 0,
        ("LMPC bounded-mixed TD3", "direct_lmpc"): 1,
        ("LMPC bounded-mixed TD3", "offset_free_mpc"): 2,
        ("OF-MPC pretrained TD3", "td3"): 3,
        ("OF-MPC pretrained TD3", "offset_free_mpc"): 4,
    }
    interesting.sort(key=lambda r: (r["mode"], order.get((r["experiment"], r["controller"]), 99)))
    rows = []
    for row in interesting:
        rows.append(
            {
                "mode": row["mode"],
                "controller": f"{row['experiment']} / {row['controller']}",
                "reward": fmt(row["reward_mean"]),
                "mean_rmse": fmt(row["mean_rmse"]),
                "eta_rmse": fmt(row["eta_rmse"]),
                "T_rmse": fmt(row["T_rmse"]),
                "mean_abs_du": fmt(row["mean_abs_du"]),
            }
        )
    return rows


def td3_gap_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["experiment"], row["controller"], row["mode"]): row for row in metrics}
    rows = []
    for mode in ("nominal", "disturb"):
        lmpc_td3 = by_key[("LMPC bounded-mixed TD3", "td3", mode)]
        direct = by_key[("LMPC bounded-mixed TD3", "direct_lmpc", mode)]
        of_td3 = by_key[("OF-MPC pretrained TD3", "td3", mode)]
        of_base = by_key[("OF-MPC pretrained TD3", "offset_free_mpc", mode)]
        rows.append(
            {
                "mode": mode,
                "lmpc_td3_rmse_gap": lmpc_td3["mean_rmse"] - direct["mean_rmse"],
                "of_td3_rmse_gap": of_td3["mean_rmse"] - of_base["mean_rmse"],
                "lmpc_td3_reward_gap": lmpc_td3["reward_mean"] - direct["reward_mean"],
                "of_td3_reward_gap": of_td3["reward_mean"] - of_base["reward_mean"],
                "lmpc_td3_du_ratio": lmpc_td3["mean_abs_du"] / direct["mean_abs_du"],
                "of_td3_du_ratio": of_td3["mean_abs_du"] / of_base["mean_abs_du"],
            }
        )
    return rows


def load_loss_series(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv_dicts(path / "loss_arrays.csv")
    actor = []
    critic = []
    for row in rows:
        actor_text = row.get("actor_bc") or row.get("actor_bc_losses") or row.get("actor_losses") or row.get("actor")
        critic_text = row.get("critic") or row.get("critic_losses")
        if actor_text not in (None, ""):
            actor.append(float(actor_text))
        if critic_text not in (None, ""):
            critic.append(float(critic_text))
    return {"actor": np.asarray(actor, dtype=float), "critic": np.asarray(critic, dtype=float)}


def bundle_y(bundle: dict[str, Any]) -> np.ndarray:
    return np.asarray(bundle.get("y_system", bundle.get("y")), dtype=float)


def bundle_u(bundle: dict[str, Any]) -> np.ndarray:
    return np.asarray(bundle.get("u_applied_phys", bundle.get("u")), dtype=float)


def y_sp_deviation_to_phys(y_sp_dev: np.ndarray) -> np.ndarray:
    scaling = read_json(LMPC_NEW_COMPARE / "summary.json")["scaling"]
    phys_range = np.asarray(scaling["setpoint_scaler_y_phys"], dtype=float)
    dev_min = np.asarray(scaling["y_sp_min"], dtype=float)
    dev_max = np.asarray(scaling["y_sp_max"], dtype=float)
    frac = (np.asarray(y_sp_dev, dtype=float) - dev_min) / (dev_max - dev_min)
    return phys_range[0, :] + frac * (phys_range[1, :] - phys_range[0, :])


def plot_metric_bars(metrics: list[dict[str, Any]]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in metrics
        if row["mode"] == "disturb"
        and (
            (row["experiment"] == "LMPC bounded-mixed TD3" and row["controller"] in {"td3", "direct_lmpc"})
            or (row["experiment"] == "OF-MPC pretrained TD3" and row["controller"] in {"td3", "offset_free_mpc"})
        )
    ]
    labels = []
    rmse = []
    reward = []
    du = []
    for row in rows:
        label = {
            ("LMPC bounded-mixed TD3", "td3"): "LMPC-TD3",
            ("LMPC bounded-mixed TD3", "direct_lmpc"): "Direct LMPC",
            ("OF-MPC pretrained TD3", "td3"): "OF-TD3",
            ("OF-MPC pretrained TD3", "offset_free_mpc"): "OF-MPC",
        }[(row["experiment"], row["controller"])]
        labels.append(label)
        rmse.append(row["mean_rmse"])
        reward.append(row["reward_mean"])
        du.append(row["mean_abs_du"])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    colors = ["#4575b4", "#91bfdb", "#d73027", "#fc8d59"]
    for ax, values, title, ylabel in (
        (axes[0], rmse, "Disturbance Tracking", "mean output RMSE"),
        (axes[1], reward, "Quadratic Reward", "mean reward"),
        (axes[2], du, "Input Movement", "mean |du|"),
    ):
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    path = FIG_DIR / "disturbance_metric_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_old_new_lmpc(metrics: list[dict[str, Any]]) -> Path:
    rows = [
        row
        for row in metrics
        if row["controller"] == "td3"
        and row["experiment"] in {"LMPC bounded-mixed TD3", "LMPC governed-ref old TD3"}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), constrained_layout=True)
    for ax, mode in zip(axes, ("nominal", "disturb")):
        mode_rows = [row for row in rows if row["mode"] == mode]
        labels = ["Old governed" if "governed" in row["experiment"] else "New bounded" for row in mode_rows]
        rmse = [row["mean_rmse"] for row in mode_rows]
        reward = [row["reward_mean"] for row in mode_rows]
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(x - width / 2, rmse, width, label="mean RMSE", color="#4c78a8")
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, reward, width, label="reward", color="#f58518")
        ax.set_title(mode.capitalize())
        ax.set_xticks(x, labels, rotation=15)
        ax.set_ylabel("mean RMSE")
        ax2.set_ylabel("mean reward")
        ax.grid(axis="y", alpha=0.25)
    handles = axes[0].patches[:1] + axes[0].twinx().patches[:1] if False else None
    path = FIG_DIR / "old_vs_new_lmpc_td3.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_losses() -> Path:
    series = {
        "LMPC bounded": load_loss_series(LMPC_NEW_PRETRAIN),
        "LMPC old": load_loss_series(LMPC_OLD_PRETRAIN),
        "OF-MPC": load_loss_series(OF_PRETRAIN),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4), constrained_layout=True)
    for name, data in series.items():
        if data["actor"].size:
            axes[0].plot(data["actor"], label=name)
        if data["critic"].size:
            axes[1].plot(data["critic"], label=name)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title("Actor BC Loss")
    axes[1].set_title("Critic Loss")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    path = FIG_DIR / "pretraining_loss_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_failure_reasons(rows: list[dict[str, Any]]) -> Path:
    labels = [row["reason"].replace("tracking:", "") for row in rows]
    counts = [row["count"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.6, 3.8), constrained_layout=True)
    ax.barh(labels[::-1], counts[::-1], color="#7b3294")
    ax.set_title("LMPC bounded-mixed label failures")
    ax.set_xlabel("candidate count")
    ax.grid(axis="x", alpha=0.25)
    path = FIG_DIR / "lmpc_label_failure_reasons.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_disturbance_rollout() -> Path:
    runs = {
        "LMPC-TD3": read_pickle(LMPC_NEW_COMPARE / "td3_results_disturb.pickle"),
        "Direct LMPC": read_pickle(DIRECT_LMPC_DISTURB_BASELINE),
        "OF-TD3": read_pickle(OF_COMPARE / "td3_results_disturb.pickle"),
        "OF-MPC": read_pickle(OF_MPC_DISTURB_BASELINE),
    }
    t = np.arange(runs["LMPC-TD3"]["y_sp"].shape[0])
    y_sp_phys = y_sp_deviation_to_phys(runs["LMPC-TD3"]["y_sp"])
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 5.6), sharex=True, constrained_layout=True)
    output_names = ["eta", "T"]
    colors = {
        "LMPC-TD3": "#4575b4",
        "Direct LMPC": "#91bfdb",
        "OF-TD3": "#d73027",
        "OF-MPC": "#fc8d59",
    }
    for j, ax in enumerate(axes):
        ax.step(t, y_sp_phys[:, j], where="post", color="black", linewidth=1.4, label="setpoint")
        for name, bundle in runs.items():
            y = bundle_y(bundle)
            y_plot = y[: t.size, j] if y.shape[0] == t.size else y[1 : t.size + 1, j]
            ax.plot(t, y_plot, label=name, color=colors[name], linewidth=1.0, alpha=0.9)
        ax.set_ylabel(output_names[j])
        ax.grid(alpha=0.25)
    axes[0].set_title("Disturbance rollout tracking")
    axes[-1].set_xlabel("time step")
    axes[0].legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    path = FIG_DIR / "disturbance_rollout_tracking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_disturbance_inputs() -> Path:
    runs = {
        "LMPC-TD3": read_pickle(LMPC_NEW_COMPARE / "td3_results_disturb.pickle"),
        "Direct LMPC": read_pickle(DIRECT_LMPC_DISTURB_BASELINE),
        "OF-TD3": read_pickle(OF_COMPARE / "td3_results_disturb.pickle"),
        "OF-MPC": read_pickle(OF_MPC_DISTURB_BASELINE),
    }
    t = np.arange(bundle_u(runs["LMPC-TD3"]).shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 5.2), sharex=True, constrained_layout=True)
    input_names = ["Qc", "Qm"]
    colors = {
        "LMPC-TD3": "#4575b4",
        "Direct LMPC": "#91bfdb",
        "OF-TD3": "#d73027",
        "OF-MPC": "#fc8d59",
    }
    for j, ax in enumerate(axes):
        for name, bundle in runs.items():
            u = bundle_u(bundle)
            ax.plot(t, u[:, j], label=name, color=colors[name], linewidth=1.0, alpha=0.9)
        ax.set_ylabel(input_names[j])
        ax.grid(alpha=0.25)
    axes[0].set_title("Disturbance rollout applied inputs")
    axes[-1].set_xlabel("time step")
    axes[0].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    path = FIG_DIR / "disturbance_rollout_inputs.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_report() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = (
        normalize_lmpc_metrics(LMPC_NEW_COMPARE, "LMPC bounded-mixed TD3")
        + normalize_lmpc_metrics(LMPC_OLD_COMPARE, "LMPC governed-ref old TD3")
        + normalize_of_metrics(OF_COMPARE)
    )
    scaling_rows = compare_scaling()
    pretrain = pretraining_rows()
    failures = failure_rows()
    gaps = td3_gap_rows(metrics)

    write_csv(TABLE_DIR / "comparison_metrics_long.csv", metrics)
    write_csv(TABLE_DIR / "scaler_consistency.csv", scaling_rows)
    write_csv(TABLE_DIR / "pretraining_summary.csv", pretrain)
    write_csv(TABLE_DIR / "lmpc_failure_reasons.csv", failures)
    write_csv(TABLE_DIR / "td3_expert_gaps.csv", gaps)

    metric_fig = plot_metric_bars(metrics)
    old_new_fig = plot_old_new_lmpc(metrics)
    loss_fig = plot_losses()
    failure_fig = plot_failure_reasons(failures)
    tracking_fig = plot_disturbance_rollout()
    input_fig = plot_disturbance_inputs()

    perf_rows = formatted_performance_rows(metrics)
    perf_table = markdown_table(
        perf_rows,
        [
            ("mode", "Mode"),
            ("controller", "Controller"),
            ("reward", "Reward"),
            ("mean_rmse", "Mean RMSE"),
            ("eta_rmse", "eta RMSE"),
            ("T_rmse", "T RMSE"),
            ("mean_abs_du", "Mean abs du"),
        ],
    )
    gap_rows = [
        {
            "mode": row["mode"],
            "lmpc_rmse_gap": fmt(row["lmpc_td3_rmse_gap"]),
            "of_rmse_gap": fmt(row["of_td3_rmse_gap"]),
            "lmpc_reward_gap": fmt(row["lmpc_td3_reward_gap"]),
            "of_reward_gap": fmt(row["of_td3_reward_gap"]),
            "lmpc_du_ratio": fmt(row["lmpc_td3_du_ratio"]),
            "of_du_ratio": fmt(row["of_td3_du_ratio"]),
        }
        for row in gaps
    ]
    gap_table = markdown_table(
        gap_rows,
        [
            ("mode", "Mode"),
            ("lmpc_rmse_gap", "LMPC TD3 RMSE Gap"),
            ("of_rmse_gap", "OF TD3 RMSE Gap"),
            ("lmpc_reward_gap", "LMPC TD3 Reward Gap"),
            ("of_reward_gap", "OF TD3 Reward Gap"),
            ("lmpc_du_ratio", "LMPC abs du ratio"),
            ("of_du_ratio", "OF abs du ratio"),
        ],
    )
    pretrain_table = markdown_table(
        [
            {
                "run": row["run"],
                "samples": f"{row['samples']:,}",
                "accepted": fmt(row["accepted_rate"]),
                "solve": fmt(row["solve_success_rate"]),
                "actor": fmt(row["actor_bc_last"], digits=6),
                "critic": fmt(row["critic_last"]),
                "reward": fmt(row["reward_mean"]),
                "selector": row["selector"],
            }
            for row in pretrain
        ],
        [
            ("run", "Run"),
            ("samples", "Samples"),
            ("accepted", "Accept Rate"),
            ("solve", "Solve Rate"),
            ("actor", "Actor BC Last"),
            ("critic", "Critic Last"),
            ("reward", "Label Reward Mean"),
            ("selector", "Selector"),
        ],
    )
    scaling_table = markdown_table(
        [{"item": row["item"], "match": row["match"]} for row in scaling_rows],
        [("item", "Contract Item"), ("match", "Match")],
    )
    failure_table = markdown_table(
        [
            {
                "reason": row["reason"],
                "count": f"{row['count']:,}",
                "share": fmt(row["share_of_failures"]),
            }
            for row in failures
        ],
        [("reason", "Failure Reason"), ("count", "Count"), ("share", "Share")],
    )

    lmpc_config = read_json(LMPC_NEW_PRETRAIN / "config.json")
    lmpc_summary = read_json(LMPC_NEW_PRETRAIN / "summary.json")
    lmpc_controller_config = lmpc_config.get("controller", {})
    compare_summary = read_json(LMPC_NEW_COMPARE / "summary.json")
    online_runner_note = (
        "`OnlineTD3_LMPCPretrained_SafetyGate.py` is a thin entrypoint into "
        "`utils.online_disturbance_runner.main_lmpc_pretrained_safety_gate()`. "
        "That shared runner resolves the newest checkpoint under `results/PretrainLMPC`, "
        "checks the TD3 setpoint scaler against the default polymer scaler, and records "
        "`target_selector_variant=bounded_mixed_u0p1_x0p1` in new run summaries."
    )

    report = f"""# Bounded-Mixed LMPC Pretraining Analysis

Date: 2026-06-12

This report analyzes the new Direct LMPC-pretrained TD3 checkpoint:

- LMPC pretraining: `{rel(LMPC_NEW_PRETRAIN)}`
- LMPC comparison: `{rel(LMPC_NEW_COMPARE)}`
- OF-MPC reference pretraining: `{rel(OF_PRETRAIN)}`
- OF-MPC reference comparison: `{rel(OF_COMPARE)}`
- Previous governed-reference LMPC comparison: `{rel(LMPC_OLD_COMPARE)}`

## Executive Takeaway

The new LMPC checkpoint is correctly labeled as the previous bounded-mixed selector
run and uses the same TD3 scaler/range contract as OF-MPC pretraining and comparison.
I do not see a scaler mismatch.

The weak result is mainly an imitation/generalization problem. The Direct LMPC expert
baseline and the OF-MPC baseline are nearly identical in the comparison rollouts, but
the LMPC-trained TD3 actor is much farther from its expert than the OF-MPC-trained TD3
actor is from OF-MPC. More uniform replay samples alone already helped the label pool
size, but the actor still underfits the harder LMPC action map.

![Disturbance metric bars]({rel_report(metric_fig)})

## Performance Snapshot

{perf_table}

## TD3-To-Expert Gap

The table below compares each pretrained actor against its own expert baseline. Lower
RMSE gap is better; reward gap is TD3 reward minus expert reward, so values closer to
zero are better.

{gap_table}

The disturbance case is the important one: LMPC-TD3 has a mean RMSE gap of
{fmt(gaps[1]['lmpc_td3_rmse_gap'])}, while OF-TD3 has a gap of
{fmt(gaps[1]['of_td3_rmse_gap'])}. LMPC-TD3 also moves the inputs about
{fmt(gaps[1]['lmpc_td3_du_ratio'])}x as much as Direct LMPC, whereas OF-TD3 is almost
matched to OF-MPC on movement.

![Disturbance rollout tracking]({rel_report(tracking_fig)})

![Disturbance rollout inputs]({rel_report(input_fig)})

## Old LMPC Versus New LMPC

The bounded-mixed run is better labeled and better aligned with the current online
gate/diagnostic selector, and it uses {int(lmpc_summary['buffer_size']):,} replay
samples. However, the deterministic comparison metrics do not yet show a control
performance win over the older governed-reference LMPC checkpoint.

![Old versus new LMPC TD3]({rel_report(old_new_fig)})

This means the new selector alignment is the right experimental hygiene, but the actor
still needs a better way to learn the LMPC label map.

## Pretraining And Label Diagnostics

{pretrain_table}

![Pretraining losses]({rel_report(loss_fig)})

The new bounded-mixed LMPC actor BC loss reaches {fmt(pretrain[0]['actor_bc_last'], digits=6)}.
That is small in absolute terms, but it is still much larger than the OF-MPC actor BC
loss ({fmt(pretrain[2]['actor_bc_last'], digits=6)}). Since both use the same
`[256, 256, 256]` actor/critic architecture, the difference is evidence that the LMPC
expert action map is more difficult to approximate, not evidence of different scalers.

## LMPC Label Failure Pattern

{failure_table}

![LMPC label failure reasons]({rel_report(failure_fig)})

The largest rejected class is `tracking:optimal:dyn_residual`. These are not ordinary
bad setpoints; they are samples for which the tracking solve status can look acceptable
but the post-check rejects the candidate. This creates a conditional dataset: the actor
only sees successful LMPC labels, while the boundaries near rejected regions are sparse
and likely non-smooth.

## Scaling And Range Audit

{scaling_table}

The detailed scaler values are exported to `{rel_report(TABLE_DIR / 'scaler_consistency.csv')}`.
The important constants are:

- TD3 setpoint scaler physical range:
  `{summarize_vector(nested_get(compare_summary, ['scaling', 'setpoint_scaler_y_phys']))}`
- Comparison setpoints:
  `{summarize_vector(nested_get(compare_summary, ['scaling', 'comparison_setpoint_y_phys']))}`
- `y_sp_min`: `{summarize_vector(nested_get(compare_summary, ['scaling', 'y_sp_min']))}`
- `y_sp_max`: `{summarize_vector(nested_get(compare_summary, ['scaling', 'y_sp_max']))}`
- LMPC selector:
  `target_mode={lmpc_summary.get('target_mode')}`,
  `target_selector_variant={lmpc_summary.get('target_selector_variant')}`,
  `target_config={lmpc_summary.get('target_config')}`,
  `rho_lyap={lmpc_controller_config.get('rho_lyap')}`,
  `lyap_eps={lmpc_controller_config.get('lyap_eps')}`.

{online_runner_note}

## Are LMPC And OF-MPC Using The Same Exact Thing?

They use the same plant, observer/scaling assets, TD3 state/action dimensions, action
scaler, setpoint scaler, comparison setpoints, MPC objective weights, and TD3 network
size. They are not the same expert-label generator:

- OF-MPC pretraining solves the offset-free MPC label directly over uniformly sampled
  augmented states, setpoints, and previous inputs.
- LMPC pretraining solves the bounded Direct LMPC target plus tracking problem, then
  keeps only candidates that pass target, tracking, bound, residual, and first-step
  contraction checks.
- LMPC labels therefore include target-stage switching and Lyapunov feasibility
  boundaries that OF-MPC labels do not have.

The baseline comparison confirms this distinction. Direct LMPC and OF-MPC themselves
track almost identically, but their offline imitation problems are not equally easy.

## Why The Result Is Still Not Good Enough

1. The LMPC expert map is more nonlinear and piecewise than OF-MPC because the target
   selector can switch stages and the contraction check imposes a hard boundary.
2. The replay distribution is broad-uniform over the full scaler box. That is good for
   coverage, but it spends a huge sample budget away from the closed-loop trajectories
   that matter most.
3. Rejected LMPC candidates leave sparse coverage near the safety/feasibility boundary.
   The actor receives successful labels but not a smooth description of what happens
   just outside the accepted set.
4. The actor BC loss is the clearest symptom. With the same architecture and more
   samples than OF-MPC, LMPC still has a substantially larger final BC loss.
5. The LMPC-TD3 actor is too aggressive in disturbance rollouts, as shown by its high
   mean input movement.

## What To Try Next

I would not first extend the physical/scaler range. The current replay already covers
the full TD3 scaler box, and the online/comparison setpoints are inside that envelope.
Going wider would mostly teach the actor outside the scale contract used online.

Recommended order:

1. Add a validation split for LMPC pretraining labels and report actor MSE by target
   stage, action saturation, and distance to the comparison rollout distribution.
2. Add targeted/stratified replay around closed-loop states, setpoint transitions, and
   accepted samples close to contraction or input-bound margins.
3. Increase actor capacity as an ablation, for example `[512, 512, 512]`, but pair it
   with validation curves. Network size is plausible, but the current evidence says
   data geometry and label complexity are at least as important.
4. Consider residualizing the TD3 action around OF-MPC or Direct LMPC for the LMPC
   pretraining task. The baselines are already good; learning a correction may be
   easier than imitating the full hard-switched LMPC map.
5. If the goal is online safety-gated TD3 rather than standalone offline imitation,
   evaluate whether the LMPC-pretrained actor improves after the warm-start BC/handoff
   schedule, because the online gate may correct exactly the regions the offline actor
   struggles with.

## Exported Tables

- `{rel_report(TABLE_DIR / 'comparison_metrics_long.csv')}`
- `{rel_report(TABLE_DIR / 'td3_expert_gaps.csv')}`
- `{rel_report(TABLE_DIR / 'pretraining_summary.csv')}`
- `{rel_report(TABLE_DIR / 'lmpc_failure_reasons.csv')}`
- `{rel_report(TABLE_DIR / 'scaler_consistency.csv')}`
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    build_report()
    print(f"Wrote {REPORT_PATH}")
