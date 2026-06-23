from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch

from experiments.run_gart_target_selector_study import _build_context as build_gart_context
from experiments.run_gart_target_selector_study import run_closed_loop as run_gart_closed_loop
from utils.online_disturbance_runner import (
    ONLINE_TD3_PRESETS,
    build_disturbance_context,
    run_online_td3_disturbance_preset,
)
from utils.path_helpers import resolve_repo_path
from utils.two_phase_profiles import (
    TwoPhaseExperimentSpec,
    build_two_phase_profiles,
    episode_len_from_spec,
    jsonable_two_phase_profile,
    phase2_steps_from_spec,
)


N_SEEDS = 1
SEED_START = 0
SEEDS: tuple[int, ...] | None = None
METHODS = (
    "ofmpc_pretrained_safety_gate",
    "ofmpc_pretrained_no_safety_gate",
    "cold_start_safety_gate",
    "cold_start_no_safety_gate",
    "gart_lmpc",
)
OUTPUT_ROOT = Path.home() / "Desktop" / "Lyapunov_polymer_results"
STUDY_NAME = "OnlineTD3_TwoPhaseStudy"
EXPORT_PROFILE = "compact"
SAVE_PLOTS = True
DETERMINISTIC_BASELINE_METHODS = {"gart_lmpc"}

AGENT_PATH = Path("Data") / "agent_2507171027.pkl"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: _jsonable(value)
                for key, value in row.items()
                if value is None or isinstance(value, (str, bool, int, float, np.bool_, np.integer, np.floating))
            }
        )
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in scalar_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scalar_rows:
            writer.writerow(row)


def _parse_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_seed_list(value: str | None, *, n_seeds: int, seed_start: int) -> tuple[int, ...]:
    if value:
        return tuple(int(part) for part in _parse_csv_list(value))
    return tuple(range(int(seed_start), int(seed_start) + int(n_seeds)))


def _resolve_methods(value: str | None) -> tuple[str, ...]:
    methods = _parse_csv_list(value) if value else METHODS
    known = set(METHODS)
    unknown = [method for method in methods if method not in known]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Known methods: {sorted(known)}")
    return tuple(methods)


def _effective_seeds_for_methods(methods: tuple[str, ...], seeds: tuple[int, ...]) -> tuple[int, ...]:
    if len(seeds) <= 1:
        return seeds
    if methods and all(method in DETERMINISTIC_BASELINE_METHODS for method in methods):
        return (int(seeds[0]),)
    return seeds


def _pretrained_agent_path(agent_path: str | None) -> str:
    selected = Path(agent_path) if agent_path is not None else AGENT_PATH
    resolved = resolve_repo_path(selected)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Pretrained OF-MPC TD3 checkpoint not found: {resolved}. "
            "Pass --agent-path to override it."
        )
    return str(resolved)


def _profile_context():
    return build_disturbance_context("gart")


def build_profiles_for_study(spec: TwoPhaseExperimentSpec) -> dict[str, Any]:
    context = _profile_context()
    profile = build_two_phase_profiles(
        spec=spec,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        steady_outputs=context.setup.steady_states["y_ss"],
        n_inputs=context.dimensions.inputs_number,
    )
    reporting_window_steps = int(profile["reporting_window_steps"])
    scenario_len = int(np.asarray(context.y_sp_scenario).shape[0])
    if reporting_window_steps % scenario_len != 0:
        raise ValueError(
            "reporting_window_steps must be divisible by the rollout setpoint-scenario count; "
            f"got reporting_window_steps={reporting_window_steps} and scenario_len={scenario_len}."
        )
    profile["rollout_n_tests"] = int(profile["total_reporting_windows"])
    profile["rollout_set_points_len"] = int(reporting_window_steps // scenario_len)
    profile["rollout_time_in_sub_episodes"] = int(reporting_window_steps)
    return profile


def _expected_exploration_sigma(*, method: str, step_idx: int, profile: dict[str, Any]) -> float:
    start = 0.02 if method.startswith("ofmpc_pretrained") else 0.10
    end = 0.005
    decay_steps = max(1, int(profile["phase1_steps"]))
    if decay_steps <= 1:
        return end
    frac = min(max(float(step_idx) / float(decay_steps - 1), 0.0), 1.0)
    return float(start + (end - start) * frac)


def _training_phase_learning_episode_multiplier(profile: dict[str, Any]) -> int:
    reporting_window_steps = int(profile.get("reporting_window_steps", profile.get("episode_len", 1)))
    learning_episode_steps = int(profile.get("phase1_episode_len", reporting_window_steps))
    if reporting_window_steps <= 0 or learning_episode_steps <= 0:
        return 1
    if learning_episode_steps % reporting_window_steps != 0:
        raise ValueError(
            "phase1_episode_len must be an integer multiple of reporting_window_steps "
            "to preserve training-phase episode units."
        )
    return max(1, learning_episode_steps // reporting_window_steps)


def _scale_training_phase_episode_counts(
    overrides: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    scaled = dict(overrides)
    multiplier = _training_phase_learning_episode_multiplier(profile)
    if multiplier <= 1:
        return scaled
    for key in (
        "warmup_buffer_only_episodes",
        "behavior_clone_teacher_episodes",
        "handoff_episodes",
    ):
        if key in scaled and scaled[key] is not None:
            scaled[key] = int(scaled[key]) * multiplier
    scaled["configured_learning_episode_steps"] = int(profile.get("phase1_episode_len", 0))
    scaled["rollout_reporting_window_steps"] = int(profile.get("reporting_window_steps", 0))
    scaled["learning_episode_to_reporting_window_multiplier"] = int(multiplier)
    return scaled


def validate_two_phase_profile(profile: dict[str, Any], spec: TwoPhaseExperimentSpec) -> dict[str, Any]:
    phase1_episode_len = episode_len_from_spec(spec)
    phase1_steps = int(spec.phase1_episodes) * phase1_episode_len
    total_steps = int(profile["total_steps"])
    y_phys = np.asarray(profile["setpoint_profile_phys"], dtype=float)
    disturbance = profile["disturbance_profile"]
    checks = {
        "total_steps": int(total_steps),
        "phase1_episode_len": int(phase1_episode_len),
        "reporting_window_steps": int(profile["reporting_window_steps"]),
        "total_reporting_windows": int(profile["total_reporting_windows"]),
        "rollout_n_tests": int(profile["rollout_n_tests"]),
        "rollout_set_points_len": int(profile["rollout_set_points_len"]),
        "phase1_steps": int(phase1_steps),
        "phase1_learning_episodes": int(spec.phase1_episodes),
        "phase2_steps": int(phase2_steps_from_spec(spec)),
        "phase2_episodes": None if spec.phase2_episodes is None else int(spec.phase2_episodes),
        "setpoint_switch_report_window": int(profile["phase1_reporting_windows"]) + 1,
        "setpoint_switch_step": int(phase1_steps),
        "pretrained_exploration_sigma_at_phase1_end": _expected_exploration_sigma(
            method="ofmpc_pretrained_safety_gate",
            step_idx=phase1_steps - 1,
            profile=profile,
        ),
        "pretrained_exploration_sigma_after_phase1": _expected_exploration_sigma(
            method="ofmpc_pretrained_safety_gate",
            step_idx=phase1_steps,
            profile=profile,
        ),
        "cold_start_exploration_sigma_at_phase1_end": _expected_exploration_sigma(
            method="cold_start_safety_gate",
            step_idx=phase1_steps - 1,
            profile=profile,
        ),
        "cold_start_exploration_sigma_after_phase1": _expected_exploration_sigma(
            method="cold_start_safety_gate",
            step_idx=phase1_steps,
            profile=profile,
        ),
    }
    if int(profile["total_steps"]) != total_steps:
        raise AssertionError(f"profile total_steps={profile['total_steps']} != expected {total_steps}")
    if y_phys.shape[0] != total_steps:
        raise AssertionError(f"setpoint profile length={y_phys.shape[0]} != expected {total_steps}")
    if phase1_steps < y_phys.shape[0] and np.allclose(y_phys[phase1_steps - 1], y_phys[phase1_steps]):
        raise AssertionError("setpoint did not switch at the first Phase-2 step.")
    if total_steps % int(profile["reporting_window_steps"]) != 0:
        raise AssertionError("total_steps is not divisible by reporting_window_steps.")
    expected_d0 = np.array([spec.nominal_qi, spec.nominal_qs, spec.nominal_ha], dtype=float)
    expected_d1 = np.array(
        [
            spec.nominal_qi * spec.phase1_qi_multiplier,
            spec.nominal_qs * spec.phase1_qs_multiplier,
            spec.nominal_ha * spec.phase1_ha_multiplier,
        ],
        dtype=float,
    )
    expected_d2 = np.array(
        [
            spec.nominal_qi * spec.phase2_qi_multiplier,
            spec.nominal_qs * spec.phase2_qs_multiplier,
            spec.nominal_ha * spec.phase2_ha_multiplier,
        ],
        dtype=float,
    )
    observed_d0 = np.array([disturbance["qi"][0], disturbance["qs"][0], disturbance["ha"][0]], dtype=float)
    observed_d1 = np.array(
        [
            disturbance["qi"][phase1_steps - 1],
            disturbance["qs"][phase1_steps - 1],
            disturbance["ha"][phase1_steps - 1],
        ],
        dtype=float,
    )
    observed_d2 = np.array([disturbance["qi"][-1], disturbance["qs"][-1], disturbance["ha"][-1]], dtype=float)
    if not np.allclose(observed_d0, expected_d0):
        raise AssertionError(f"disturbance start mismatch: {observed_d0} != {expected_d0}")
    if not np.allclose(observed_d1, expected_d1):
        raise AssertionError(f"phase1 disturbance end mismatch: {observed_d1} != {expected_d1}")
    if not np.allclose(observed_d2, expected_d2):
        raise AssertionError(f"phase2 disturbance end mismatch: {observed_d2} != {expected_d2}")
    if not np.isclose(checks["pretrained_exploration_sigma_at_phase1_end"], 0.005):
        raise AssertionError("pretrained exploration does not reach 0.005 at Phase-1 end.")
    if not np.isclose(checks["pretrained_exploration_sigma_after_phase1"], 0.005):
        raise AssertionError("pretrained exploration is not fixed after Phase 1.")
    if not np.isclose(checks["cold_start_exploration_sigma_at_phase1_end"], 0.005):
        raise AssertionError("cold-start exploration does not reach 0.005 at Phase-1 end.")
    if not np.isclose(checks["cold_start_exploration_sigma_after_phase1"], 0.005):
        raise AssertionError("cold-start exploration is not fixed after Phase 1.")
    return checks


def _cleanup_after_method() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_td3_method(
    *,
    method: str,
    seed: int,
    method_root: Path,
    profile: dict[str, Any],
    save_plots: bool,
    export_profile: str,
    agent_path: str | None,
    reset_pretrained_critic: bool = True,
    training_phase_overrides: dict[str, Any] | None = None,
    rl_observation_mode: str = "standard",
    projection_backend: str | None = None,
    reward_fallback_penalty_enabled: bool = False,
    gamma_fallback: float = 0.0,
    fallback_event_penalty: float = 0.0,
    rho_lyap: float | None = None,
    lyap_eps: float | None = None,
    lyap_tol: float | None = None,
) -> dict[str, Any]:
    pretrained = method.startswith("ofmpc_pretrained")
    overrides = _scale_training_phase_episode_counts(dict(training_phase_overrides or {}), profile)
    overrides.update({
        "exploration_decay_end_step": int(profile["phase1_steps"]),
        "exploration_decay_mode": "linear",
        "global_exploration_schedule": True,
    })
    return run_online_td3_disturbance_preset(
        method,
        episodes=int(profile["rollout_n_tests"]),
        set_points_len=int(profile["rollout_set_points_len"]),
        seed=int(seed),
        save_plots=bool(save_plots),
        agent_path=agent_path if pretrained else None,
        reset_pretrained_critic=bool(reset_pretrained_critic),
        rl_observation_mode=str(rl_observation_mode),
        projection_backend=projection_backend,
        reward_fallback_penalty_enabled=bool(reward_fallback_penalty_enabled),
        gamma_fallback=float(gamma_fallback),
        fallback_event_penalty=float(fallback_event_penalty),
        rho_lyap=rho_lyap,
        lyap_eps=lyap_eps,
        lyap_tol=lyap_tol,
        training_phase_overrides=overrides,
        setpoint_profile=profile["setpoint_profile_scaled_dev"],
        disturbance_profile=profile["disturbance_profile"],
        profile_metadata=profile,
        study_root_override=method_root,
        export_profile=export_profile,
        mirror_large_artifacts=False,
    )


def _run_gart_method(
    *,
    seed: int,
    method_root: Path,
    profile: dict[str, Any],
    save_plots: bool,
    export_profile: str,
) -> dict[str, Any]:
    ctx = build_gart_context()
    summary = run_gart_closed_loop(
        ctx,
        method_root,
        mode="disturb",
        n_tests=int(profile["rollout_n_tests"]),
        set_points_len=int(profile["rollout_set_points_len"]),
        setpoint_profile=profile["setpoint_profile_scaled_dev"],
        disturbance_profile=profile["disturbance_profile"],
        profile_metadata=profile,
        save_plots=bool(save_plots),
        export_profile=export_profile,
        save_raw_payload=False,
    )
    record = {}
    records = summary.get("records")
    if isinstance(records, list) and records:
        record = dict(records[0])
    _write_json(method_root / "record.json", record)
    return {
        "study_name": "GART_LMPC_TwoPhase",
        "case_name": "gart_lmpc",
        "result_root": str(method_root),
        "debug_dir": summary.get("artifacts", {}).get("gartlmpc", {}).get("direct_style_debug_dir"),
        "record_json": str(method_root / "record.json"),
        "config": {
            "seed": int(seed),
            "export_profile": export_profile,
            "two_phase_profile": jsonable_two_phase_profile(profile),
        },
        "gart_summary": summary,
    }


def _method_record(
    *,
    seed: int,
    method: str,
    status: str,
    method_root: Path,
    started_at: str,
    elapsed_seconds: float,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result = result or {}
    return {
        "seed": int(seed),
        "method": method,
        "status": status,
        "started_at": started_at,
        "elapsed_seconds": float(elapsed_seconds),
        "result_root": str(method_root),
        "debug_dir": result.get("debug_dir"),
        "record_json": result.get("record_json"),
        "trained_agent_path": result.get("trained_agent_path"),
        "export_profile": result.get("export_profile") or result.get("config", {}).get("export_profile"),
        "checkpoint_path": result.get("config", {}).get("initial_agent_path"),
        "error": error,
    }


def _load_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_seed_comparison(seed_root: Path, method_records: list[dict[str, Any]]) -> None:
    rows = []
    for item in method_records:
        row = dict(item)
        record = _load_record(Path(item.get("record_json") or ""))
        for key in (
            "reward_mean",
            "reward_no_penalty_mean",
            "output_rmse_mean",
            "diagnostic_unsafe_rate",
            "actual_intervention_rate",
            "fallback_rate",
            "wall_clock_seconds",
            "wall_clock_seconds_per_step",
            "wall_clock_steps_per_second",
        ):
            if key in record:
                row[key] = record.get(key)
        rows.append(row)
    _write_csv(seed_root / "seed_comparison_table.csv", rows)
    _write_json(
        seed_root / "seed_manifest.json",
        {
            "seed_root": str(seed_root),
            "records": rows,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    _plot_seed_comparison(seed_root, rows)


def _plot_seed_comparison(seed_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = seed_root / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("output_rmse_mean", "output RMSE mean"),
        ("reward_no_penalty_mean", "reward no penalty mean"),
        ("actual_intervention_rate", "actual intervention rate"),
        ("wall_clock_seconds_per_step", "wall seconds per step"),
    ]
    methods = [row["method"] for row in rows if row.get("status") == "success"]
    if not methods:
        return
    for key, ylabel in metrics:
        values = []
        labels = []
        for row in rows:
            if row.get("status") != "success":
                continue
            value = row.get(key)
            if value is None:
                continue
            labels.append(row["method"])
            values.append(float(value))
        if not values:
            continue
        plt.figure(figsize=(9, 4))
        plt.bar(labels, values)
        plt.ylabel(ylabel)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{key}.png", dpi=220, bbox_inches="tight")
        plt.close()


def run_two_phase_study(args: argparse.Namespace) -> dict[str, Any]:
    methods = _resolve_methods(args.methods)
    requested_seeds = _parse_seed_list(args.seeds, n_seeds=args.n_seeds, seed_start=args.seed_start)
    seeds = _effective_seeds_for_methods(methods, requested_seeds)
    output_root = Path(args.output_root).expanduser()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    study_root = output_root / STUDY_NAME / timestamp
    study_root.mkdir(parents=True, exist_ok=True)

    spec_kwargs = {
        "phase1_episodes": int(args.phase1_episodes),
        "set_points_len": int(args.set_points_len),
        "reporting_window_steps": int(args.reporting_window_steps),
    }
    if getattr(args, "phase2_steps", None) is not None:
        spec_kwargs["phase2_steps"] = int(args.phase2_steps)
        spec_kwargs["phase2_episodes"] = None
    else:
        spec_kwargs["phase2_episodes"] = int(args.phase2_episodes)
    for name in (
        "phase1_setpoints_y_phys",
        "phase2_setpoints_y_phys",
        "reporting_window_steps",
        "nominal_qi",
        "nominal_qs",
        "nominal_ha",
        "phase1_qi_multiplier",
        "phase1_qs_multiplier",
        "phase1_ha_multiplier",
        "phase2_qi_multiplier",
        "phase2_qs_multiplier",
        "phase2_ha_multiplier",
    ):
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                spec_kwargs[name] = value
    spec = TwoPhaseExperimentSpec(**spec_kwargs)
    profile = build_profiles_for_study(spec)
    profile_checks = validate_two_phase_profile(profile, spec)
    pretrained_agent_path = (
        _pretrained_agent_path(args.agent_path)
        if any(method.startswith("ofmpc_pretrained") for method in methods)
        else (str(resolve_repo_path(args.agent_path)) if args.agent_path else None)
    )
    profile_export = {
        "profile": jsonable_two_phase_profile(profile),
        "checks": profile_checks,
    }
    _write_json(study_root / "two_phase_profile.json", profile_export)

    manifest: dict[str, Any] = {
        "study_name": STUDY_NAME,
        "study_root": str(study_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "methods": list(methods),
        "requested_seeds": list(requested_seeds),
        "seeds": list(seeds),
        "save_plots": bool(args.save_plots),
        "export_profile": str(args.export_profile),
        "pretrained_agent_path": pretrained_agent_path,
        "profile_checks": profile_checks,
        "runs": [],
    }
    _write_json(study_root / "batch_manifest.json", manifest)

    for seed in seeds:
        seed_root = study_root / f"seed_{int(seed):03d}"
        seed_root.mkdir(parents=True, exist_ok=True)
        seed_records: list[dict[str, Any]] = []
        for method in methods:
            method_root = seed_root / method
            method_root.mkdir(parents=True, exist_ok=True)
            started_at = datetime.now().isoformat(timespec="seconds")
            tic = time.perf_counter()
            print(f"[two-phase] seed={seed} method={method} -> {method_root}")
            try:
                if method == "gart_lmpc":
                    result = _run_gart_method(
                        seed=int(seed),
                        method_root=method_root,
                        profile=profile,
                        save_plots=bool(args.save_plots),
                        export_profile=str(args.export_profile),
                    )
                else:
                    result = _run_td3_method(
                        method=method,
                        seed=int(seed),
                        method_root=method_root,
                        profile=profile,
                        save_plots=bool(args.save_plots),
                        export_profile=str(args.export_profile),
                        agent_path=pretrained_agent_path,
                        reset_pretrained_critic=bool(getattr(args, "reset_pretrained_critic", True)),
                        training_phase_overrides=getattr(args, "training_phase_overrides", None),
                        rl_observation_mode=str(getattr(args, "rl_observation_mode", "standard")),
                        projection_backend=getattr(args, "projection_backend", None),
                        reward_fallback_penalty_enabled=bool(
                            getattr(args, "reward_fallback_penalty_enabled", False)
                        ),
                        gamma_fallback=float(getattr(args, "gamma_fallback", 0.0)),
                        fallback_event_penalty=float(getattr(args, "fallback_event_penalty", 0.0)),
                        rho_lyap=getattr(args, "rho_lyap", None),
                        lyap_eps=getattr(args, "lyap_eps", None),
                        lyap_tol=getattr(args, "lyap_tol", None),
                    )
                elapsed = time.perf_counter() - tic
                record = _method_record(
                    seed=int(seed),
                    method=method,
                    status="success",
                    method_root=method_root,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    result=result,
                )
                _write_json(method_root / "method_manifest.json", record)
                pprint(record)
            except Exception as exc:
                elapsed = time.perf_counter() - tic
                traceback_text = traceback.format_exc()
                record = _method_record(
                    seed=int(seed),
                    method=method,
                    status="failed",
                    method_root=method_root,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    error=repr(exc),
                )
                record["traceback"] = traceback_text
                _write_json(method_root / "method_manifest.json", record)
                _write_json(
                    method_root / "error.json",
                    {
                        "error": repr(exc),
                        "traceback": traceback_text,
                        "method": method,
                        "seed": int(seed),
                    },
                )
                print(f"[two-phase] FAILED seed={seed} method={method}: {exc!r}")
            finally:
                _cleanup_after_method()
            seed_records.append(record)
            manifest["runs"].append(record)
            _write_json(study_root / "batch_manifest.json", manifest)

        _write_seed_comparison(seed_root, seed_records)

    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(study_root / "batch_manifest.json", manifest)
    _write_csv(study_root / "batch_manifest.csv", list(manifest["runs"]))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-phase online TD3/GART polymer CSTR study.")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, e.g. 0,1,2.")
    parser.add_argument("--methods", default=None, help=f"Comma-separated methods. Default: {','.join(METHODS)}")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--save-plots", dest="save_plots", action="store_true", default=SAVE_PLOTS)
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    parser.add_argument("--export-profile", choices=("compact", "debug"), default=EXPORT_PROFILE)
    parser.add_argument("--agent-path", default=None)
    parser.add_argument("--phase1-episodes", type=int, default=150)
    parser.add_argument("--phase2-episodes", type=int, default=50)
    parser.add_argument(
        "--phase2-steps",
        type=int,
        default=None,
        help="Optional fixed Phase-2 duration; overrides --phase2-episodes.",
    )
    parser.add_argument("--set-points-len", type=int, default=400, help="Phase-1 hold time per setpoint.")
    parser.add_argument("--reporting-window-steps", type=int, default=800)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive.")
    if args.phase1_episodes <= 0:
        raise ValueError("phase1_episodes must be positive.")
    if args.phase2_steps is not None and args.phase2_steps <= 0:
        raise ValueError("phase2_steps must be positive when provided.")
    if args.phase2_steps is None and args.phase2_episodes <= 0:
        raise ValueError("phase2_episodes must be positive when phase2_steps is not provided.")
    if args.set_points_len <= 0:
        raise ValueError("--set-points-len must be positive.")
    if args.reporting_window_steps <= 0:
        raise ValueError("--reporting-window-steps must be positive.")
    return run_two_phase_study(args)


def _has_cli_option(argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def _strip_cli_option(argv: list[str], option: str) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == option:
            skip_next = True
            continue
        if token.startswith(f"{option}="):
            continue
        cleaned.append(token)
    return cleaned


def main_for_methods(methods: tuple[str, ...] | list[str], argv: list[str] | None = None) -> dict[str, Any]:
    method_tuple = tuple(str(method).strip() for method in methods if str(method).strip())
    if not method_tuple:
        raise ValueError("At least one method is required.")
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _strip_cli_option(argv, "--methods")
    if not _has_cli_option(argv, "--timestamp"):
        suffix = "_".join(method_tuple)
        argv.extend(["--timestamp", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"])
    argv.extend(["--methods", ",".join(method_tuple)])
    return main(argv)


def main_for_method(method: str, argv: list[str] | None = None) -> dict[str, Any]:
    return main_for_methods((method,), argv=argv)


if __name__ == "__main__":
    main()
