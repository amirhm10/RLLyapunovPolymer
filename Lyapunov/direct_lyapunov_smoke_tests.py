from __future__ import annotations

import json

import numpy as np

from Lyapunov.direct_lyapunov_mpc import design_direct_lyapunov_mpc_solver
from Lyapunov.frozen_output_disturbance_target import (
    solve_target_bounded_output_disturbance,
    solve_target_unbounded_output_disturbance,
)


def _build_scalar_augmented_model(a=0.5, b=1.0):
    A_aug = np.array([[float(a), 0.0], [0.0, 1.0]], dtype=float)
    B_aug = np.array([[float(b)], [0.0]], dtype=float)
    C_aug = np.array([[1.0, 1.0]], dtype=float)
    return A_aug, B_aug, C_aug


def _assert_close(actual, expected, tol=1e-7, name="value"):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    err = float(np.max(np.abs(actual - expected)))
    if err > float(tol):
        raise AssertionError(f"{name} mismatch: max error {err:.3e} > {tol:.3e}")


def target_solver_smoke_tests():
    A_aug, B_aug, C_aug = _build_scalar_augmented_model(a=0.5, b=1.0)
    xhat_aug = np.array([0.2, 0.1], dtype=float)
    y_sp = np.array([1.1], dtype=float)

    exact = solve_target_unbounded_output_disturbance(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
    )
    if not exact["success"]:
        raise AssertionError("unbounded exact target solve failed")
    _assert_close(exact["x_s"], [1.0], name="exact x_s")
    _assert_close(exact["u_s"], [0.5], name="exact u_s")
    if not bool(exact["invertible_I_minus_A"]):
        raise AssertionError("expected invertible I-A diagnostic")
    if not bool(exact["reduced_exact_available"]):
        raise AssertionError("expected reduced-form diagnostic to be available")

    A_aug_ls = np.array([[0.5, 0.0], [0.0, 1.0]], dtype=float)
    B_aug_ls = np.array([[1.0, 0.5], [0.0, 0.0]], dtype=float)
    C_aug_ls = np.array([[1.0, 1.0]], dtype=float)
    lsq = solve_target_unbounded_output_disturbance(
        A_aug=A_aug_ls,
        B_aug=B_aug_ls,
        C_aug=C_aug_ls,
        xhat_aug=np.array([0.0, 0.1], dtype=float),
        y_sp=np.array([1.1], dtype=float),
    )
    if not bool(lsq["used_lstsq"]):
        raise AssertionError("expected least-squares fallback on rectangular stacked solve")

    bounded = solve_target_bounded_output_disturbance(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=np.array([0.0], dtype=float),
        u_max=np.array([0.3], dtype=float),
    )
    if not bounded["success"]:
        raise AssertionError("bounded target solve failed")
    if not bool(bounded["bounded_solution_used"]):
        raise AssertionError("expected bounded least-squares fallback to be used")
    if not bool(np.asarray(bounded["bounded_active_upper_mask"], dtype=bool).item()):
        raise AssertionError("expected active upper bound in bounded solve")

    return {
        "exact_solver_mode": exact["solver_mode_used"],
        "exact_cond_M": exact["cond_M"],
        "lsq_used_lstsq": bool(lsq["used_lstsq"]),
        "bounded_solution_used": bool(bounded["bounded_solution_used"]),
        "bounded_residual_norm": bounded["bounded_residual_norm"],
    }


def direct_controller_smoke_tests():
    A_aug, B_aug, C_aug = _build_scalar_augmented_model(a=0.5, b=1.0)
    solver = design_direct_lyapunov_mpc_solver(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        Qy_diag=np.array([1.0], dtype=float),
        NP=2,
        NC=1,
        Su_diag=np.array([1.0], dtype=float),
        u_min=np.array([-2.0], dtype=float),
        u_max=np.array([2.0], dtype=float),
        Rdu_diag=np.array([0.1], dtype=float),
    )
    hard = solver.solve_tracking_mpc_step(
        IC_opt=np.zeros(1, dtype=float),
        bnds=(( -2.0, 2.0),),
        y_target=np.array([0.0], dtype=float),
        u_prev_dev=np.array([0.0], dtype=float),
        x0_aug=np.array([1.0, 0.0], dtype=float),
        x_s=np.array([0.0], dtype=float),
        u_s=np.array([0.0], dtype=float),
        alpha_terminal=None,
        rho_lyap=0.95,
        eps_lyap=0.0,
        first_step_contraction_on=True,
        lyapunov_mode="hard",
    )
    if not hard.success:
        raise AssertionError(f"hard direct controller smoke test failed: {hard.message}")
    if not bool(hard.first_step_contraction_satisfied):
        raise AssertionError("expected hard-mode contraction to be satisfied")

    A_aug_soft, B_aug_soft, C_aug_soft = _build_scalar_augmented_model(a=1.1, b=1.0)
    soft_solver = design_direct_lyapunov_mpc_solver(
        A_aug=A_aug_soft,
        B_aug=B_aug_soft,
        C_aug=C_aug_soft,
        Qy_diag=np.array([1.0], dtype=float),
        NP=2,
        NC=1,
        Su_diag=np.array([1.0], dtype=float),
        u_min=np.array([0.0], dtype=float),
        u_max=np.array([0.0], dtype=float),
        Rdu_diag=np.array([0.1], dtype=float),
    )
    soft = soft_solver.solve_tracking_mpc_step(
        IC_opt=np.zeros(1, dtype=float),
        bnds=((0.0, 0.0),),
        y_target=np.array([0.0], dtype=float),
        u_prev_dev=np.array([0.0], dtype=float),
        x0_aug=np.array([1.0, 0.0], dtype=float),
        x_s=np.array([0.0], dtype=float),
        u_s=np.array([0.0], dtype=float),
        alpha_terminal=None,
        rho_lyap=0.05,
        eps_lyap=0.0,
        first_step_contraction_on=True,
        lyapunov_mode="soft",
        slack_penalty=1e4,
    )
    if not soft.success:
        raise AssertionError(f"soft direct controller smoke test failed: {soft.message}")
    if not (float(soft.slack_lyap) > 0.0):
        raise AssertionError("expected positive Lyapunov slack in soft mode")

    return {
        "hard_solver": hard.solver,
        "hard_contraction_margin": hard.contraction_margin,
        "soft_solver": soft.solver,
        "soft_slack_lyap": float(soft.slack_lyap),
        "soft_relaxed_contraction_satisfied": bool(soft.relaxed_contraction_satisfied),
    }


def run_all_smoke_tests():
    return {
        "target_solver": target_solver_smoke_tests(),
        "direct_controller": direct_controller_smoke_tests(),
    }


if __name__ == "__main__":
    print(json.dumps(run_all_smoke_tests(), indent=2))
