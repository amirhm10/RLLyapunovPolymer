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

# GART-LMPC runner configuration.
#
# This root runner is intentionally editable, following the style of
# DirectLyapunovMPC.py. Change these values here for day-to-day experiments.

MODE = "disturb"  # "disturb" or "nominal"
N_TESTS = 2
SET_POINTS_LEN = 400

# First-step Lyapunov contraction settings used by both the GART target
# governor and the GART-LMPC solver.
RHO_LYAP = GART_FINAL_RHO_LYAP
LYAP_EPS = GART_FINAL_LYAP_EPS

# GART target rate limits in scaled-deviation coordinates.
# Original active values: dx=0.05, du=[0.998, 0.740], dy=1.0.
DX_S_MAX_ABS = 0.05
DU_S_MAX_ABS = [0.2, 0.2]
DY_S_MAX_ABS = 0.25

# Certified-disturbance target update.
# Original active values: d_rate_scale=1.0, alpha_d=0.2.
D_RATE_SCALE = 0.25
ALPHA_D = 0.05

# Target-selector geometry. Edit these values directly for manual ablations.
# Original active values: headroom=0.01, primary_tol_rel=1.0e-6,
# W_u_smooth=[1.0, 1.0], Wy=[5.0, 1.0].
INPUT_HEADROOM_FRAC = 0.05
PRIMARY_TOL_REL = 1.0e-4
W_U_SMOOTH_DIAG = [2.0, 2.0]
TARGET_WY_DIAG = [2.0, 1.0]

# Set to None for an automatic timestamp, or use a fixed string to rerun into a
# predictable folder.
TIMESTAMP = None
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
        target_overrides=target_overrides,
        mpc_overrides=mpc_overrides,
        guard=guard,
    )
    summaries["closed_loop_dir"] = str(lmpc_dir.relative_to(root))

    print(json.dumps(_jsonable(summaries), indent=2))
    return summaries


if __name__ == "__main__":
    run_configured_study()
