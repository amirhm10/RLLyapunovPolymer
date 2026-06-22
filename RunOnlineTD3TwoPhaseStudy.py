from __future__ import annotations

import argparse
import csv
import gc
import json
import time
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

AGENT_PATH = Path("results") / "PretrainOFMPC" / "20260621_203346" / "of_mpc_pretrained_td3_20260622_030149.pkl"


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
    return build_two_phase_profiles(
        spec=spec,
        data_min=context.system_data["data_min"],
        data_max=context.system_data["data_max"],
        steady_outputs=context.setup.steady_states["y_ss"],
        n_inputs=context.dimensions.inputs_number,
    )


def _expected_exploration_sigma(*, method: str, step_idx: int, profile: dict[str, Any]) -> float:
    start = 0.02 if method.startswith("ofmpc_pretrained") else 0.10
    end = 0.005
    decay_steps = max(1, int(profile["phase1_steps"]))
    if decay_steps <= 1:
        return end
    frac = min(max(float(step_idx) / float(decay_steps - 1), 0.0), 1.0)
    return float(start + (end - start) * frac)


def validate_two_phase_profile(profile: dict[str, Any], spec: TwoPhaseExperimentSpec) -> dict[str, Any]:
    episode_len = episode_len_from_spec(spec)
    phase1_steps = int(spec.phase1_episodes) * episode_len
    total_steps = int(spec.phase1_episodes + spec.phase2_episodes) * episode_len
    y_phys = np.asarray(profile["setpoint_profile_phys"], dtype=float)
    disturbance = profile["disturbance_profile"]
    checks = {
        "total_steps": int(total_steps),
        "episode_len": int(episode_len),
        "phase1_steps": int(phase1_steps),
        "setpoint_switch_episode": int(spec.phase1_episodes) + 1,
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
) -> dict[str, Any]:
    pretrained = method.startswith("ofmpc_pretrained")
    overrides = {
        "exploration_decay_end_step": int(profile["phase1_steps"]),
        "exploration_decay_mode": "linear",
        "global_exploration_schedule": True,
    }
    return run_online_td3_disturbance_preset(
        method,
        episodes=int(profile["total_episodes"]),
        set_points_len=int(profile["spec"].set_points_len),
        seed=int(seed),
        save_plots=bool(save_plots),
        agent_path=agent_path if pretrained else None,
        reset_pretrained_critic=True,
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
        n_tests=int(profile["total_episodes"]),
        set_points_len=int(profile["spec"].set_points_len),
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
    seeds = _parse_seed_list(args.seeds, n_seeds=args.n_seeds, seed_start=args.seed_start)
    output_root = Path(args.output_root).expanduser()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    study_root = output_root / STUDY_NAME / timestamp
    study_root.mkdir(parents=True, exist_ok=True)

    spec = TwoPhaseExperimentSpec(
        phase1_episodes=int(args.phase1_episodes),
        phase2_episodes=int(args.phase2_episodes),
        set_points_len=int(args.set_points_len),
    )
    profile = build_profiles_for_study(spec)
    profile_checks = validate_two_phase_profile(profile, spec)
    pretrained_agent_path = _pretrained_agent_path(args.agent_path)
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
                record = _method_record(
                    seed=int(seed),
                    method=method,
                    status="failed",
                    method_root=method_root,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    error=repr(exc),
                )
                _write_json(method_root / "method_manifest.json", record)
                _write_json(method_root / "error.json", {"error": repr(exc), "method": method, "seed": int(seed)})
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
    parser.add_argument("--phase1-episodes", type=int, default=200)
    parser.add_argument("--phase2-episodes", type=int, default=50)
    parser.add_argument("--set-points-len", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive.")
    if args.phase1_episodes <= 0 or args.phase2_episodes <= 0:
        raise ValueError("Both phase episode counts must be positive.")
    if args.set_points_len <= 0:
        raise ValueError("--set-points-len must be positive.")
    return run_two_phase_study(args)


if __name__ == "__main__":
    main()
