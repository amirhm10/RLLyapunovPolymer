from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

from utils.gart_runtime import GARTStudyLimits, ResourceGuard, set_single_thread_env

set_single_thread_env(1)

from experiments.run_gart_target_selector_study import (
    GART_RELAXED_DY2_OVERRIDES,
    GART_RELAXED_DY4_OVERRIDES,
    GART_RELAXED_TARGET_OVERRIDES,
    _build_context,
    _jsonable,
    run_closed_loop,
    run_synthetic_target_only,
)
from utils.path_helpers import repo_path


# GART-LMPC runner configuration.
#
# This root runner is intentionally editable, following the style of
# DirectLyapunovMPC.py. Change these values here for day-to-day experiments.

MODE = "disturb"  # "disturb" or "nominal"
N_TESTS = 5
SET_POINTS_LEN = 400

# Closed-loop disturbance test defaults. Keep mixed cases disabled below.
RUN_TARGET_ONLY = False
RUN_CLOSED_LOOP = True
FULL_RUN = True
CONFIRM_FULL = True
THREADS = 1
MAX_TARGET_EVALS = 10000
MAX_CLOSED_LOOP_STEPS = 10000
MAX_SOLVER_CALLS = 10000
MAX_WALL_CLOCK_SECONDS = 14400.0
MAX_MEMORY_MB = 4096.0

# Set to None for an automatic timestamp, or use a fixed string to rerun into a
# predictable folder.
TIMESTAMP = None
TARGET_ONLY_OVERRIDES = GART_RELAXED_DY2_OVERRIDES

# Toggle individual closed-loop cases here.
CASE_SPECS = [
    {
        "enabled": False,
        "case_name": "old_governed_reference",
        "objective": None,
        "lyapunov_mode": None,
        "label": "Old governed-reference Direct LMPC (disabled; retained for manual comparison only)",
    },
    {
        "enabled": True,
        "case_name": "gart_target_raw_no_dx_headroom_0p01_dy2",
        "objective": "raw",
        "lyapunov_mode": "hard",
        "target_overrides": GART_RELAXED_DY2_OVERRIDES,
        "label": "GART raw, no dx_s rate, 1% headroom, dy scale 2",
    },
    {
        "enabled": True,
        "case_name": "gart_target_raw_no_dx_headroom_0p01_dy4",
        "objective": "raw",
        "lyapunov_mode": "hard",
        "target_overrides": GART_RELAXED_DY4_OVERRIDES,
        "label": "GART raw, no dx_s rate, 1% headroom, dy scale 4",
    },
    {
        "enabled": False,
        "case_name": "gart_target_raw_no_dx_headroom_0p01_dy2_probe_log_only",
        "objective": "raw",
        "lyapunov_mode": "hard",
        "target_overrides": {
            **GART_RELAXED_TARGET_OVERRIDES,
            "dy_rate_scale": 2.0,
            "contraction_probe_log_only": True,
        },
        "label": "Diagnostic only: same relaxed target, contraction probe log-only",
    },
    {
        "enabled": False,
        "case_name": "gart_target_mixed_objective",
        "objective": "mixed",
        "lyapunov_mode": "hard",
        "label": "GART target, mixed objective",
    },
    {
        "enabled": False,
        "case_name": "gart_target_mixed_soft",
        "objective": "mixed",
        "lyapunov_mode": "soft",
        "label": "GART target, mixed objective with soft Lyapunov",
    },
]

# Set to True if you want command-line arguments to override the editable values
# above. Leaving this on keeps old commands like
# `python GARTLyapunovMPC.py --mode nominal --closed-loop` working.
ALLOW_CLI_OVERRIDES = True


def _enabled_case_specs() -> list[dict[str, object]]:
    return [
        {
            "case_name": str(case["case_name"]),
            "objective": case.get("objective"),
            "lyapunov_mode": case.get("lyapunov_mode"),
            "target_overrides": case.get("target_overrides"),
            "mpc_overrides": case.get("mpc_overrides"),
        }
        for case in CASE_SPECS
        if bool(case.get("enabled", True))
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the editable GART-LMPC Polymer CSTR study.")
    parser.add_argument("--mode", choices=["nominal", "disturb"], default=MODE)
    parser.add_argument("--n-tests", type=int, default=N_TESTS)
    parser.add_argument("--set-points-len", type=int, default=SET_POINTS_LEN)
    parser.add_argument("--target-only", action=argparse.BooleanOptionalAction, default=RUN_TARGET_ONLY)
    parser.add_argument("--closed-loop", action=argparse.BooleanOptionalAction, default=RUN_CLOSED_LOOP)
    parser.add_argument("--full", action="store_true", default=FULL_RUN)
    parser.add_argument("--confirm-full", action="store_true", default=CONFIRM_FULL)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--max-target-evals", type=int, default=MAX_TARGET_EVALS)
    parser.add_argument("--max-closed-loop-steps", type=int, default=MAX_CLOSED_LOOP_STEPS)
    parser.add_argument("--max-solver-calls", type=int, default=MAX_SOLVER_CALLS)
    parser.add_argument("--max-wall-clock-seconds", type=float, default=MAX_WALL_CLOCK_SECONDS)
    parser.add_argument("--max-memory-mb", type=float, default=MAX_MEMORY_MB)
    parser.add_argument("--timestamp", default=TIMESTAMP)
    return parser.parse_args()


def _config_from_script() -> argparse.Namespace:
    return argparse.Namespace(
        mode=MODE,
        n_tests=N_TESTS,
        set_points_len=SET_POINTS_LEN,
        target_only=RUN_TARGET_ONLY,
        closed_loop=RUN_CLOSED_LOOP,
        full=FULL_RUN,
        confirm_full=CONFIRM_FULL,
        threads=THREADS,
        max_target_evals=MAX_TARGET_EVALS,
        max_closed_loop_steps=MAX_CLOSED_LOOP_STEPS,
        max_solver_calls=MAX_SOLVER_CALLS,
        max_wall_clock_seconds=MAX_WALL_CLOCK_SECONDS,
        max_memory_mb=MAX_MEMORY_MB,
        timestamp=TIMESTAMP,
    )


def run_configured_study() -> dict:
    args = _parse_args() if ALLOW_CLI_OVERRIDES else _config_from_script()
    if not args.target_only and not args.closed_loop:
        raise ValueError("At least one of RUN_TARGET_ONLY or RUN_CLOSED_LOOP must be enabled.")
    if args.full and not args.confirm_full:
        raise RuntimeError("Full GART runs require both --full and --confirm-full.")
    if not args.full:
        full_like = args.mode == "disturb" or int(args.n_tests) > 1 or int(args.set_points_len) > 20
        if full_like and not args.confirm_full:
            raise RuntimeError("Non-smoke GART runs require --full --confirm-full.")

    set_single_thread_env(args.threads)
    guard = ResourceGuard(
        GARTStudyLimits(
            max_target_evals=int(args.max_target_evals),
            max_closed_loop_steps=int(args.max_closed_loop_steps),
            max_solver_calls=int(args.max_solver_calls),
            max_wall_clock_seconds=float(args.max_wall_clock_seconds),
            max_memory_mb=None if args.max_memory_mb is None or float(args.max_memory_mb) <= 0.0 else float(args.max_memory_mb),
        )
    )
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(repo_path())
    ctx = _build_context()

    summaries: dict = {
        "timestamp": timestamp,
        "mode": args.mode,
        "n_tests": int(args.n_tests),
        "set_points_len": int(args.set_points_len),
        "run_target_only": bool(args.target_only),
        "run_closed_loop": bool(args.closed_loop),
        "full": bool(args.full),
        "confirm_full": bool(args.confirm_full),
        "resource_limits": guard.limits.__dict__.copy(),
        "enabled_cases": [str(case["case_name"]) for case in _enabled_case_specs()],
        "target_only_overrides": TARGET_ONLY_OVERRIDES,
    }

    print("GART-LMPC editable runner configuration:")
    pprint(summaries)

    if args.target_only:
        target_dir = root / "results" / "GARTTargetSelectorStudy" / timestamp
        summaries["target_only"] = run_synthetic_target_only(
            ctx,
            target_dir,
            n_tests=int(args.n_tests),
            set_points_len=int(args.set_points_len),
            target_overrides=TARGET_ONLY_OVERRIDES,
            guard=guard,
        )
        summaries["target_only_dir"] = str(target_dir.relative_to(root))

    if args.closed_loop:
        lmpc_dir = root / "results" / "GARTLMPC" / timestamp
        summaries["closed_loop"] = run_closed_loop(
            ctx,
            lmpc_dir,
            mode=str(args.mode),
            n_tests=int(args.n_tests),
            set_points_len=int(args.set_points_len),
            case_specs=_enabled_case_specs(),
            guard=guard,
        )
        summaries["closed_loop_dir"] = str(lmpc_dir.relative_to(root))

    print(json.dumps(_jsonable(summaries), indent=2))
    return summaries


if __name__ == "__main__":
    run_configured_study()
