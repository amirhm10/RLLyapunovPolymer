from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

from experiments.run_gart_target_selector_study import (
    _build_context,
    _jsonable,
    run_closed_loop,
    run_target_only,
)
from utils.direct_lyapunov_study import DIRECT_DISTURBANCE_SETPOINT_LEN
from utils.path_helpers import repo_path


# GART-LMPC runner configuration.
#
# This root runner is intentionally editable, following the style of
# DirectLyapunovMPC.py. Change these values here for day-to-day experiments.

MODE = "disturb"  # "disturb" or "nominal"
N_TESTS = 5
SET_POINTS_LEN = DIRECT_DISTURBANCE_SETPOINT_LEN

# The target-only study is useful for debugging the selector, but it can take a
# while at SET_POINTS_LEN=400. Keep it off for normal closed-loop monitoring.
RUN_TARGET_ONLY = False
RUN_CLOSED_LOOP = True

# Set to None for an automatic timestamp, or use a fixed string to rerun into a
# predictable folder.
TIMESTAMP = None

# Toggle individual closed-loop cases here.
CASE_SPECS = [
    {
        "enabled": True,
        "case_name": "old_governed_reference",
        "objective": None,
        "lyapunov_mode": None,
        "label": "Old governed-reference Direct LMPC",
    },
    {
        "enabled": True,
        "case_name": "gart_target_raw_objective",
        "objective": "raw",
        "lyapunov_mode": "hard",
        "label": "GART target, raw y_sp objective",
    },
    {
        "enabled": True,
        "case_name": "gart_target_mixed_objective",
        "objective": "mixed",
        "lyapunov_mode": "hard",
        "label": "GART target, mixed objective",
    },
    {
        "enabled": True,
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


def _enabled_case_tuples() -> list[tuple[str, str | None, str | None]]:
    return [
        (str(case["case_name"]), case.get("objective"), case.get("lyapunov_mode"))
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
    parser.add_argument("--timestamp", default=TIMESTAMP)
    return parser.parse_args()


def _config_from_script() -> argparse.Namespace:
    return argparse.Namespace(
        mode=MODE,
        n_tests=N_TESTS,
        set_points_len=SET_POINTS_LEN,
        target_only=RUN_TARGET_ONLY,
        closed_loop=RUN_CLOSED_LOOP,
        timestamp=TIMESTAMP,
    )


def run_configured_study() -> dict:
    args = _parse_args() if ALLOW_CLI_OVERRIDES else _config_from_script()
    if not args.target_only and not args.closed_loop:
        raise ValueError("At least one of RUN_TARGET_ONLY or RUN_CLOSED_LOOP must be enabled.")

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
        "enabled_cases": [case[0] for case in _enabled_case_tuples()],
    }

    print("GART-LMPC editable runner configuration:")
    pprint(summaries)

    if args.target_only:
        target_dir = root / "results" / "GARTTargetSelectorStudy" / timestamp
        summaries["target_only"] = run_target_only(
            ctx,
            target_dir,
            n_tests=int(args.n_tests),
            set_points_len=int(args.set_points_len),
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
            case_specs=_enabled_case_tuples(),
        )
        summaries["closed_loop_dir"] = str(lmpc_dir.relative_to(root))

    print(json.dumps(_jsonable(summaries), indent=2))
    return summaries


if __name__ == "__main__":
    run_configured_study()
