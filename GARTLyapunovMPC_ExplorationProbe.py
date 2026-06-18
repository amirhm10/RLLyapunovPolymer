from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from pprint import pprint

from utils.gart_runtime import GARTStudyLimits, ResourceGuard

from experiments.run_gart_target_selector_study import (
    FINAL_GART_CASE_NAME,
    FINAL_GART_TARGET_OVERRIDES,
    _build_context,
    _jsonable,
    run_closed_loop,
)
from utils.gart_defaults import GART_FINAL_LYAP_EPS, GART_FINAL_RHO_LYAP
from utils.path_helpers import repo_path

# Diagnostic GART-LMPC runner.
#
# This is a copy-style runner for testing whether small executed-input
# excitation moves dhat enough to change the practical contraction/violation
# behavior. The main GARTLyapunovMPC.py runner is not used for this probe.

MODE = "disturb"  # "disturb" or "nominal"
N_TESTS = 2
SET_POINTS_LEN = 400

RHO_LYAP = GART_FINAL_RHO_LYAP
LYAP_EPS = GART_FINAL_LYAP_EPS

DX_S_MAX_ABS = 0.05
DU_S_MAX_ABS = [0.2, 0.2]
DY_S_MAX_ABS = 0.25
D_RATE_SCALE = 0.25
ALPHA_D = 0.05
INPUT_HEADROOM_FRAC = 0.05
PRIMARY_TOL_REL = 1.0e-4
W_U_SMOOTH_DIAG = [2.0, 2.0]
TARGET_WY_DIAG = [1.0, 1.0]

# Scaled-deviation input excitation applied after the LMPC solve and before
# plant simulation/observer update. The nominal solver action is still logged.
INPUT_EXPLORATION_STD = [0.005, 0.005]
INPUT_EXPLORATION_SEED = 20260617

TIMESTAMP = None
RESULTS_SUBDIR = "GARTLMPCExplorationProbe"
CASE_NAME = FINAL_GART_CASE_NAME


def _configured_overrides() -> tuple[dict, dict]:
    target_overrides = dict(FINAL_GART_TARGET_OVERRIDES)
    target_overrides.update(
        {
            "rho": float(RHO_LYAP),
            "eps": float(LYAP_EPS),
            "dx_s_max_abs": DX_S_MAX_ABS,
            "du_s_max_abs": list(DU_S_MAX_ABS),
            "dy_s_max_abs": DY_S_MAX_ABS,
            "d_rate_scale": D_RATE_SCALE,
            "alpha_d": ALPHA_D,
            "input_headroom_frac": INPUT_HEADROOM_FRAC,
            "primary_tol_rel": PRIMARY_TOL_REL,
            "W_u_smooth_diag": list(W_U_SMOOTH_DIAG),
            "Wy_diag": list(TARGET_WY_DIAG),
        }
    )
    mpc_overrides = {
        "rho": float(RHO_LYAP),
        "eps": float(LYAP_EPS),
    }
    return target_overrides, mpc_overrides


def run_configured_study() -> dict:
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
    target_overrides, mpc_overrides = _configured_overrides()

    summaries: dict = {
        "timestamp": timestamp,
        "mode": MODE,
        "n_tests": int(N_TESTS),
        "set_points_len": int(SET_POINTS_LEN),
        "case_name": CASE_NAME,
        "rho_lyap": float(RHO_LYAP),
        "lyap_eps": float(LYAP_EPS),
        "resource_limits": guard.limits.__dict__.copy(),
        "target_overrides": target_overrides,
        "mpc_overrides": mpc_overrides,
        "input_exploration_std": list(INPUT_EXPLORATION_STD),
        "input_exploration_seed": int(INPUT_EXPLORATION_SEED),
    }

    print("GART-LMPC exploration probe configuration:")
    pprint(summaries)

    lmpc_dir = root / "results" / RESULTS_SUBDIR / timestamp
    summaries["closed_loop"] = run_closed_loop(
        ctx,
        lmpc_dir,
        mode=str(MODE),
        n_tests=int(N_TESTS),
        set_points_len=int(SET_POINTS_LEN),
        target_overrides=target_overrides,
        mpc_overrides=mpc_overrides,
        input_exploration_std=list(INPUT_EXPLORATION_STD),
        input_exploration_seed=int(INPUT_EXPLORATION_SEED),
        guard=guard,
    )
    summaries["closed_loop_dir"] = str(lmpc_dir.relative_to(root))

    print(json.dumps(_jsonable(summaries), indent=2))
    return summaries


if __name__ == "__main__":
    run_configured_study()
