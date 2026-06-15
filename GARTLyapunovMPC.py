from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

from utils.gart_runtime import GARTStudyLimits, ResourceGuard, set_single_thread_env

THREADS = 4
set_single_thread_env(THREADS)

from experiments.run_gart_target_selector_study import (
    FINAL_GART_CASE_NAME,
    FINAL_GART_TARGET_OVERRIDES,
    _build_context,
    _jsonable,
    run_closed_loop,
)
from utils.path_helpers import repo_path


# GART-LMPC runner configuration.
#
# This root runner is intentionally editable, following the style of
# DirectLyapunovMPC.py. Change these values here for day-to-day experiments.

MODE = "disturb"  # "disturb" or "nominal"
N_TESTS = 5
SET_POINTS_LEN = 400

# Set to None for an automatic timestamp, or use a fixed string to rerun into a
# predictable folder.
TIMESTAMP = None
CASE_NAME = FINAL_GART_CASE_NAME


def run_configured_study() -> dict:
    set_single_thread_env(THREADS)
    guard = ResourceGuard(
        GARTStudyLimits(
            max_target_evals=None,
            max_closed_loop_steps=None,
            max_solver_calls=None,
            max_wall_clock_seconds=None,
            max_memory_mb=None,
        )
    )
    timestamp = TIMESTAMP or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(repo_path())
    ctx = _build_context()

    summaries: dict = {
        "timestamp": timestamp,
        "mode": MODE,
        "n_tests": int(N_TESTS),
        "set_points_len": int(SET_POINTS_LEN),
        "case_name": CASE_NAME,
        "resource_limits": guard.limits.__dict__.copy(),
        "target_overrides": FINAL_GART_TARGET_OVERRIDES,
    }

    print("GART-LMPC final runner configuration:")
    pprint(summaries)

    lmpc_dir = root / "results" / "GARTLMPC" / timestamp
    summaries["closed_loop"] = run_closed_loop(
        ctx,
        lmpc_dir,
        mode=str(MODE),
        n_tests=int(N_TESTS),
        set_points_len=int(SET_POINTS_LEN),
        guard=guard,
    )
    summaries["closed_loop_dir"] = str(lmpc_dir.relative_to(root))

    print(json.dumps(_jsonable(summaries), indent=2))
    return summaries


if __name__ == "__main__":
    run_configured_study()
