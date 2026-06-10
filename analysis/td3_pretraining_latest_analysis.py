from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.of_mpc_td3_workflow import (
    build_polymer_setup,
    load_of_mpc_system_data,
    y_sp_phys_from_scaled,
)


OF_PRETRAIN_DIR = REPO_ROOT / "results" / "PretrainOFMPC" / "20260610_005048"
LMPC_PRETRAIN_DIR = REPO_ROOT / "results" / "PretrainLMPC" / "20260610_005100"
OF_COMPARISON_DIR = REPO_ROOT / "results" / "PretrainOFMPCComparison" / "20260610_154032"
LMPC_COMPARISON_DIR = REPO_ROOT / "results" / "PretrainLMPCComparison" / "20260610_173925"
FIG_DIR = REPO_ROOT / "report" / "figures" / "2026-06-10_td3_pretraining_full_scale"


COLORS = {
    "OF-TD3": "#0072B2",
    "OF-MPC": "#009E73",
    "LMPC-TD3": "#D55E00",
    "Direct LMPC": "#CC79A7",
}


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        return None if not np.isfinite(as_float) else as_float
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd._libs.missing.NAType):
        return None
    return value


def normalize_rollout(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    if "y" in payload:
        y = np.asarray(payload["y"], dtype=float)
        u = np.asarray(payload["u"], dtype=float)
    elif "y_mpc" in payload:
        y = np.asarray(payload["y_mpc"], dtype=float)
        u = np.asarray(payload["u_mpc"], dtype=float)
    else:
        y = np.asarray(payload["y_system"], dtype=float)
        u = np.asarray(payload["u_applied_phys"], dtype=float)
    return {
        "y": y,
        "u": u,
        "y_sp_scaled": np.asarray(payload["y_sp"], dtype=float),
        "rewards": np.asarray(payload["rewards"], dtype=float),
    }


def physical_setpoint(y_sp_scaled: np.ndarray) -> np.ndarray:
    setup = build_polymer_setup()
    system_data = load_of_mpc_system_data(setup)
    return y_sp_phys_from_scaled(
        y_sp_scaled,
        steady_states=setup.steady_states,
        data_min=system_data["data_min"],
        data_max=system_data["data_max"],
        inputs_number=2,
    )


def load_rollouts(mode: str) -> dict[str, dict[str, np.ndarray]]:
    of_td3 = load_pickle(OF_COMPARISON_DIR / f"td3_results_{mode}.pickle")
    of_baseline = load_pickle(
        REPO_ROOT
        / "results"
        / "PretrainOFMPCComparison"
        / "baselines"
        / f"mpc_results_{mode}_n2_len400.pickle"
    )
    lmpc_td3 = load_pickle(LMPC_COMPARISON_DIR / f"td3_results_{mode}.pickle")
    direct_lmpc = load_pickle(
        REPO_ROOT
        / "results"
        / "PretrainLMPCComparison"
        / "baselines"
        / f"direct_lmpc_{mode}_n2_len400_disturb_before_q5_1_r1_1_rho0p99_eps0p005.pickle"
    )
    return {
        "OF-TD3": normalize_rollout(of_td3),
        "OF-MPC": normalize_rollout(of_baseline),
        "LMPC-TD3": normalize_rollout(lmpc_td3),
        "Direct LMPC": normalize_rollout(direct_lmpc),
    }


def compact_metric_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in load_json(OF_COMPARISON_DIR / "comparison_metrics.json")["records"]:
        rows.append(
            {
                "mode": record["mode"],
                "controller": "OF-TD3",
                "reward_mean": record["rl_reward_mean"],
                "mean_rmse": record["rl_mean_rmse"],
                "eta_rmse": record["rl_eta_rmse"],
                "T_rmse": record["rl_T_rmse"],
                "eta_iae": record["rl_eta_iae"],
                "T_iae": record["rl_T_iae"],
                "mean_abs_du": record["rl_mean_abs_du"],
            }
        )
        rows.append(
            {
                "mode": record["mode"],
                "controller": "OF-MPC",
                "reward_mean": record["of_mpc_reward_mean"],
                "mean_rmse": record["of_mpc_mean_rmse"],
                "eta_rmse": record["of_mpc_eta_rmse"],
                "T_rmse": record["of_mpc_T_rmse"],
                "eta_iae": record["of_mpc_eta_iae"],
                "T_iae": record["of_mpc_T_iae"],
                "mean_abs_du": record["of_mpc_mean_abs_du"],
            }
        )

    lmpc_records = load_json(LMPC_COMPARISON_DIR / "comparison_metrics.json")["records"]
    for record in lmpc_records:
        if record["controller"] == "offset_free_mpc":
            continue
        label = "LMPC-TD3" if record["controller"] == "td3" else "Direct LMPC"
        rows.append(
            {
                "mode": record["mode"],
                "controller": label,
                "reward_mean": record["reward_mean"],
                "mean_rmse": record["mean_rmse"],
                "eta_rmse": record["eta_rmse"],
                "T_rmse": record["T_rmse"],
                "eta_iae": record["eta_iae"],
                "T_iae": record["T_iae"],
                "mean_abs_du": record["mean_abs_du"],
                "solver_success_rate": record.get("solver_success_rate"),
                "contraction_satisfied_rate": record.get("contraction_satisfied_rate"),
                "diagnostic_unsafe_count": record.get("diagnostic_unsafe_count"),
            }
        )

    df = pd.DataFrame(rows)
    order = ["OF-TD3", "OF-MPC", "LMPC-TD3", "Direct LMPC"]
    df["controller"] = pd.Categorical(df["controller"], categories=order, ordered=True)
    return df.sort_values(["mode", "controller"]).reset_index(drop=True)


def matched_gap_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ["nominal", "disturb"]:
        mode_df = metrics[metrics["mode"] == mode].set_index("controller")
        pairs = [
            ("OF-TD3", "OF-MPC", "OF-TD3 vs OF-MPC"),
            ("LMPC-TD3", "Direct LMPC", "LMPC-TD3 vs Direct LMPC"),
        ]
        for controller, baseline, label in pairs:
            for metric in ["mean_rmse", "mean_abs_du"]:
                value = mode_df.loc[controller, metric]
                base = mode_df.loc[baseline, metric]
                rows.append(
                    {
                        "mode": mode,
                        "comparison": label,
                        "metric": metric,
                        "percent_gap": 100.0 * (value - base) / base,
                    }
                )
    return pd.DataFrame(rows)


def plot_loss_curves() -> Path:
    of_loss = pd.read_csv(OF_PRETRAIN_DIR / "loss_arrays.csv")
    lmpc_loss = pd.read_csv(LMPC_PRETRAIN_DIR / "loss_arrays.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for df, label, color in [
        (of_loss, "OF-MPC labels", COLORS["OF-TD3"]),
        (lmpc_loss, "LMPC labels", COLORS["LMPC-TD3"]),
    ]:
        actor = df["actor_bc_losses"].dropna().to_numpy(dtype=float)
        critic = df["critic_losses"].dropna().to_numpy(dtype=float)
        axes[0].plot(np.arange(actor.size), actor, label=label, color=color, linewidth=1.8)
        axes[1].plot(np.arange(critic.size), critic, label=label, color=color, linewidth=1.8)

    axes[0].set_title("Actor behavioral cloning")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_yscale("log")
    axes[1].set_title("Offline critic warm-up")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("TD target MSE")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    path = FIG_DIR / "loss_curves_full_scale.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_metric_bars(metrics: pd.DataFrame) -> Path:
    metric_labels = [
        ("mean_rmse", "Mean output RMSE"),
        ("reward_mean", "Mean reward"),
        ("mean_abs_du", "Mean abs. input move"),
    ]
    modes = ["nominal", "disturb"]
    controllers = ["OF-TD3", "OF-MPC", "LMPC-TD3", "Direct LMPC"]
    width = 0.18
    x = np.arange(len(modes))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metric_labels):
        for idx, controller in enumerate(controllers):
            values = [
                float(
                    metrics[
                        (metrics["mode"] == mode) & (metrics["controller"] == controller)
                    ][metric].iloc[0]
                )
                for mode in modes
            ]
            ax.bar(
                x + (idx - 1.5) * width,
                values,
                width=width,
                label=controller,
                color=COLORS[controller],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["Nominal", "Disturbance"])
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Physical output units")
    axes[2].set_ylabel("Physical input units/step")
    axes[0].legend(frameon=False, ncol=2)
    path = FIG_DIR / "rollout_metric_bars_full_scale.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_gap_bars(gaps: pd.DataFrame) -> Path:
    modes = ["nominal", "disturb"]
    comparisons = ["OF-TD3 vs OF-MPC", "LMPC-TD3 vs Direct LMPC"]
    metric_titles = {
        "mean_rmse": "Mean RMSE gap",
        "mean_abs_du": "Input movement gap",
    }
    width = 0.34
    x = np.arange(len(modes))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for ax, metric in zip(axes, ["mean_rmse", "mean_abs_du"]):
        for idx, comparison in enumerate(comparisons):
            values = [
                gaps[
                    (gaps["mode"] == mode)
                    & (gaps["comparison"] == comparison)
                    & (gaps["metric"] == metric)
                ]["percent_gap"].iloc[0]
                for mode in modes
            ]
            color = COLORS["OF-TD3"] if comparison.startswith("OF") else COLORS["LMPC-TD3"]
            ax.bar(x + (idx - 0.5) * width, values, width=width, label=comparison, color=color)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_title(metric_titles[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(["Nominal", "Disturbance"])
        ax.set_ylabel("% versus matched baseline")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    path = FIG_DIR / "matched_baseline_gap_bars_full_scale.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_outputs(mode: str) -> Path:
    rollouts = load_rollouts(mode)
    y_sp = physical_setpoint(next(iter(rollouts.values()))["y_sp_scaled"])
    n = y_sp.shape[0]
    time = np.arange(n)
    titles = [r"$\eta$", r"$T$"]
    units = ["viscosity units", "K"]

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.8), sharex=True, constrained_layout=True)
    for channel, ax in enumerate(axes):
        ax.plot(time, y_sp[:, channel], color="black", linestyle="--", linewidth=1.6, label="Setpoint")
        for label, data in rollouts.items():
            y = data["y"]
            y_aligned = y[1 : n + 1] if y.shape[0] == n + 1 else y[:n]
            ax.plot(time, y_aligned[:, channel], color=COLORS[label], linewidth=1.15, label=label)
        ax.set_title(f"{titles[channel]} tracking, {mode}")
        ax.set_ylabel(units[channel])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Step")
    axes[0].legend(frameon=False, ncol=3)
    path = FIG_DIR / f"rollout_outputs_{mode}_full_scale.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_inputs(mode: str) -> Path:
    rollouts = load_rollouts(mode)
    n = next(iter(rollouts.values()))["u"].shape[0]
    time = np.arange(n)
    labels = [r"$Q_c$", r"$Q_m$"]

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.8), sharex=True, constrained_layout=True)
    for channel, ax in enumerate(axes):
        for label, data in rollouts.items():
            ax.plot(time, data["u"][:n, channel], color=COLORS[label], linewidth=1.15, label=label)
        ax.set_title(f"{labels[channel]} input, {mode}")
        ax.set_ylabel("Physical input")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Step")
    axes[0].legend(frameon=False, ncol=2)
    path = FIG_DIR / f"rollout_inputs_{mode}_full_scale.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_lmpc_label_feasibility() -> Path:
    diagnostics = load_json(LMPC_PRETRAIN_DIR / "label_diagnostics.json")
    rates = pd.DataFrame(
        [
            {
                "kind": "Total",
                "acceptance": diagnostics["acceptance_rate"],
                "solve success": diagnostics["solve_success_rate"],
            },
            {
                "kind": "Broad",
                "acceptance": diagnostics["broad"]["acceptance_rate"],
                "solve success": diagnostics["broad"]["solve_success_rate"],
            },
            {
                "kind": "Steady",
                "acceptance": diagnostics["steady"]["acceptance_rate"],
                "solve success": diagnostics["steady"]["solve_success_rate"],
            },
        ]
    )
    failures = (
        pd.Series(diagnostics["broad"]["failure_reasons"], dtype=float)
        .sort_values(ascending=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    x = np.arange(len(rates))
    width = 0.34
    axes[0].bar(x - width / 2, 100 * rates["acceptance"], width=width, color="#0072B2", label="Accepted")
    axes[0].bar(x + width / 2, 100 * rates["solve success"], width=width, color="#009E73", label="Solved")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rates["kind"])
    axes[0].set_ylabel("% of attempted labels")
    axes[0].set_ylim(98.5, 100.1)
    axes[0].set_title("LMPC label feasibility")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].barh(failures.index, failures.values, color="#D55E00")
    axes[1].set_title("Broad-sample rejection reasons")
    axes[1].set_xlabel("Count")
    axes[1].grid(True, axis="x", alpha=0.25)
    path = FIG_DIR / "lmpc_label_feasibility_full_scale.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def build_summary(metrics: pd.DataFrame, gaps: pd.DataFrame, figure_paths: list[Path]) -> dict[str, Any]:
    of_summary = load_json(OF_PRETRAIN_DIR / "summary.json")
    lmpc_summary = load_json(LMPC_PRETRAIN_DIR / "summary.json")
    of_loss = load_json(OF_PRETRAIN_DIR / "loss_summary.json")
    lmpc_loss = load_json(LMPC_PRETRAIN_DIR / "loss_summary.json")
    lmpc_diag = load_json(LMPC_PRETRAIN_DIR / "label_diagnostics.json")

    def actor_ratio(loss: dict[str, Any]) -> float:
        series = loss["series"]["actor_bc_losses"]
        return float(series["last"] / series["first"])

    def critic_ratio(loss: dict[str, Any]) -> float:
        series = loss["series"]["critic_losses"]
        return float(series["last"] / series["first"])

    return {
        "artifact_paths": {
            "of_pretraining": repo_rel(OF_PRETRAIN_DIR),
            "lmpc_pretraining": repo_rel(LMPC_PRETRAIN_DIR),
            "of_comparison": repo_rel(OF_COMPARISON_DIR),
            "lmpc_comparison": repo_rel(LMPC_COMPARISON_DIR),
        },
        "training": {
            "of_mpc": {
                "buffer_size": of_summary["buffer_size"],
                "mpc_samples": of_summary["mpc_samples"],
                "steady_samples": of_summary["steady_samples"],
                "elapsed_hours": of_summary["elapsed_seconds"] / 3600.0,
                "actor_first": of_loss["series"]["actor_bc_losses"]["first"],
                "actor_last": of_loss["series"]["actor_bc_losses"]["last"],
                "actor_last_over_first": actor_ratio(of_loss),
                "critic_first": of_loss["series"]["critic_losses"]["first"],
                "critic_last": of_loss["series"]["critic_losses"]["last"],
                "critic_last_over_first": critic_ratio(of_loss),
            },
            "lmpc": {
                "buffer_size": lmpc_summary["buffer_size"],
                "lmpc_samples": lmpc_summary["lmpc_samples"],
                "steady_samples": lmpc_summary["steady_samples"],
                "elapsed_hours": lmpc_summary["elapsed_seconds"] / 3600.0,
                "actor_first": lmpc_loss["series"]["actor_bc_losses"]["first"],
                "actor_last": lmpc_loss["series"]["actor_bc_losses"]["last"],
                "actor_last_over_first": actor_ratio(lmpc_loss),
                "critic_first": lmpc_loss["series"]["critic_losses"]["first"],
                "critic_last": lmpc_loss["series"]["critic_losses"]["last"],
                "critic_last_over_first": critic_ratio(lmpc_loss),
                "acceptance_rate": lmpc_diag["acceptance_rate"],
                "solve_success_rate": lmpc_diag["solve_success_rate"],
                "attempted_total": lmpc_diag["attempted_total"],
                "accepted_total": lmpc_diag["accepted_total"],
                "broad_failure_reasons": lmpc_diag["broad"]["failure_reasons"],
            },
        },
        "metrics": metrics.to_dict(orient="records"),
        "matched_gaps": gaps.to_dict(orient="records"),
        "figures": [repo_rel(path) for path in figure_paths],
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = compact_metric_rows()
    gaps = matched_gap_rows(metrics)

    metrics_path = FIG_DIR / "compact_metrics.csv"
    gaps_path = FIG_DIR / "matched_gaps.csv"
    metrics.to_csv(metrics_path, index=False)
    gaps.to_csv(gaps_path, index=False)

    figure_paths = [
        plot_loss_curves(),
        plot_metric_bars(metrics),
        plot_gap_bars(gaps),
        plot_lmpc_label_feasibility(),
    ]
    for mode in ["nominal", "disturb"]:
        figure_paths.append(plot_outputs(mode))
        figure_paths.append(plot_inputs(mode))

    summary = build_summary(metrics, gaps, figure_paths)
    summary["tables"] = {
        "compact_metrics_csv": repo_rel(metrics_path),
        "matched_gaps_csv": repo_rel(gaps_path),
    }
    summary_path = FIG_DIR / "analysis_summary.json"
    write_json(summary_path, summary)

    print(f"Wrote {repo_rel(summary_path)}")
    for path in figure_paths:
        print(f"Wrote {repo_rel(path)}")


if __name__ == "__main__":
    main()
