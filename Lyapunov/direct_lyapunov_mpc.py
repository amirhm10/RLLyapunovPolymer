from __future__ import annotations

import csv
import json
import os
import pickle
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

from Plotting_fns.mpc_plot_fns import plot_mpc_results_cstr
from Lyapunov.frozen_output_disturbance_target import solve_output_disturbance_target
from Lyapunov.lyapunov_core import (
    _OPTIMAL_STATUSES,
    _TRACKING_TOL_BY_STATUS,
    _bounds_to_horizon_matrices,
    _extract_num_iters,
    compute_terminal_alpha_input_only,
    design_standard_tracking_terminal_ingredients,
    first_step_contraction_metrics,
    lyapunov_bound,
    lyapunov_value,
    FirstStepContractionTrackingLyapunovMpcSolver,
)
from Simulation.run_mpc_lyapunov import (
    _reset_system_on_entry,
    _set_system_input_phys,
    _system_io_phys,
)
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.lyapunov_utils import (
    compute_du_sequence,
    get_y_sp_step,
    reshape_u_sequence,
    shift_input_guess,
    tracking_solver_sequence,
)
from utils.plot_style import PAPER_COLORS, paper_plot_context
from utils.scaling_helpers import apply_min_max, reverse_min_max


DEFAULT_DIRECT_TARGET_CONFIG: Dict[str, Any] = {}
DEFAULT_DIRECT_SOFT_SLACK_PENALTY = 1.0e6


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def _save_npz(path: str, arrays: Dict[str, Any]) -> None:
    saveable = {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            saveable[key] = value
    np.savez(path, **saveable)


def _array_or_none(info: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    value = info.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1)


def _stack_vectors(step_info_storage: Iterable[Dict[str, Any]], key: str, width: int) -> np.ndarray:
    rows = list(step_info_storage)
    out = np.full((len(rows), width), np.nan, dtype=float)
    for idx, info in enumerate(rows):
        arr = _array_or_none(info, key)
        if arr is None:
            continue
        use = min(width, arr.size)
        out[idx, :use] = arr[:use]
    return out


def _inf_norm(value: Any) -> Optional[float]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _row_inf_norms(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for idx, row in enumerate(arr):
        finite = row[np.isfinite(row)]
        if finite.size:
            out[idx] = float(np.max(np.abs(finite)))
    return out


def _as_mode(value: str, allowed: Iterable[str], name: str) -> str:
    mode = str(value).strip().lower()
    allowed_tuple = tuple(str(item).strip().lower() for item in allowed)
    if mode not in allowed_tuple:
        allowed_str = ", ".join(repr(item) for item in allowed_tuple)
        raise ValueError(f"{name} must be one of {allowed_str}.")
    return mode


def _as_scalar_float(value: Any, name: str) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar.")
    return float(arr.item())


class DirectOutputDisturbanceLyapunovMpcSolver(FirstStepContractionTrackingLyapunovMpcSolver):
    def _objective_steady_input_cost_on(self) -> bool:
        return bool(getattr(self, "objective_steady_input_cost", False))

    def _objective_terminal_cost_on(self) -> bool:
        return bool(getattr(self, "objective_terminal_cost", False))

    def _evaluate_soft_tracking_solution(
        self,
        *,
        u_sequence: np.ndarray,
        x_pred: np.ndarray,
        x0_aug: np.ndarray,
        x_s: np.ndarray,
        lower: Optional[np.ndarray],
        upper: Optional[np.ndarray],
        alpha_terminal: Optional[float],
        terminal_constraint_active: bool,
        rho_lyap: float,
        eps_lyap: float,
        slack_lyap: float,
        status: str,
    ) -> Dict[str, Any]:
        result = super()._evaluate_tracking_solution(
            u_sequence=u_sequence,
            x_pred=x_pred,
            x0_aug=x0_aug,
            x_s=x_s,
            lower=lower,
            upper=upper,
            alpha_terminal=alpha_terminal,
            terminal_constraint_active=terminal_constraint_active,
            status=status,
        )
        contraction = first_step_contraction_metrics(
            x0_aug=x0_aug,
            x_pred=x_pred,
            x_s=x_s,
            P_x=self.P_x,
            rho=rho_lyap,
            eps_lyap=eps_lyap,
            tol=_TRACKING_TOL_BY_STATUS.get(status, 1e-5),
        )
        slack_lyap = max(float(slack_lyap), 0.0)
        relaxed_violation = max(float(contraction["contraction_margin"]) - slack_lyap, 0.0)
        relaxed_ok = bool(relaxed_violation <= _TRACKING_TOL_BY_STATUS.get(status, 1e-5))

        result.update(contraction)
        result.update(
            {
                "lyapunov_mode": "soft",
                "slack_lyap": float(slack_lyap),
                "relaxed_contraction_satisfied": bool(relaxed_ok),
                "relaxed_contraction_violation": float(relaxed_violation),
            }
        )
        if result["accepted"] and not relaxed_ok:
            result["accepted"] = False
            result["reject_reason"] = "soft_contraction_constraint"
        return result

    def standard_tracking_report(
        self,
        x_opt,
        x0_aug,
        x_s,
        u_s,
        y_target,
        u_prev_dev,
        alpha_terminal,
        rho_lyap=None,
        eps_lyap=None,
        first_step_contraction_on=False,
        lyapunov_mode="hard",
        slack_lyap=0.0,
        slack_penalty=None,
    ):
        report = super().standard_tracking_report(
            x_opt=x_opt,
            x0_aug=x0_aug,
            x_s=x_s,
            u_s=u_s,
            y_target=y_target,
            u_prev_dev=u_prev_dev,
            alpha_terminal=alpha_terminal,
            rho_lyap=rho_lyap,
            eps_lyap=eps_lyap,
            first_step_contraction_on=first_step_contraction_on,
        )
        lyapunov_mode = _as_mode(lyapunov_mode, ("hard", "soft"), "lyapunov_mode")
        slack_lyap = max(float(slack_lyap), 0.0)
        report.update(
            {
                "lyapunov_mode": lyapunov_mode,
                "slack_lyap": float(slack_lyap),
                "slack_penalty": None if slack_penalty is None else float(slack_penalty),
                "relaxed_contraction_satisfied": report.get("first_step_contraction_satisfied"),
                "relaxed_contraction_violation": report.get("contraction_constraint_violation"),
            }
        )
        if lyapunov_mode == "soft" and first_step_contraction_on:
            contraction_margin = report.get("contraction_margin")
            relaxed_violation = None
            relaxed_ok = None
            if contraction_margin is not None:
                relaxed_violation = max(float(contraction_margin) - slack_lyap, 0.0)
                relaxed_ok = bool(relaxed_violation <= 1e-9)
            report["relaxed_contraction_satisfied"] = relaxed_ok
            report["relaxed_contraction_violation"] = relaxed_violation
        return report

    def solve_tracking_mpc_step(
        self,
        IC_opt,
        bnds,
        y_target,
        u_prev_dev,
        x0_aug,
        x_s,
        u_s,
        alpha_terminal,
        rho_lyap=0.99,
        eps_lyap=1e-9,
        first_step_contraction_on=True,
        lyapunov_mode="hard",
        slack_penalty=DEFAULT_DIRECT_SOFT_SLACK_PENALTY,
        options=None,
    ):
        lyapunov_mode = _as_mode(lyapunov_mode, ("hard", "soft"), "lyapunov_mode")
        if lyapunov_mode == "hard":
            original_su_mat = self.Su_mat
            original_terminal_cost_scale = self.terminal_cost_scale
            if not self._objective_steady_input_cost_on():
                self.Su_mat = np.zeros_like(self.Su_mat)
            if not self._objective_terminal_cost_on():
                self.terminal_cost_scale = 0.0
            try:
                result = super().solve_tracking_mpc_step(
                    IC_opt=IC_opt,
                    bnds=bnds,
                    y_target=y_target,
                    u_prev_dev=u_prev_dev,
                    x0_aug=x0_aug,
                    x_s=x_s,
                    u_s=u_s,
                    alpha_terminal=alpha_terminal,
                    rho_lyap=rho_lyap,
                    eps_lyap=eps_lyap,
                    first_step_contraction_on=first_step_contraction_on,
                    options=options,
                )
            finally:
                self.Su_mat = original_su_mat
                self.terminal_cost_scale = original_terminal_cost_scale
            result.lyapunov_mode = "hard"
            result.slack_lyap = 0.0
            result.slack_penalty = float(slack_penalty)
            result.relaxed_contraction_satisfied = result.first_step_contraction_satisfied
            result.relaxed_contraction_violation = result.contraction_constraint_violation
            return result

        if not HAS_CVXPY:
            raise ImportError("CVXPY is required for the direct Lyapunov MPC solver.")

        options = {} if options is None else dict(options)
        solver_pref_override = options.pop("solver_pref", None)
        warm_start = bool(options.pop("warm_start", True))
        verbose = bool(options.pop("verbose", False))
        solve_kwargs = dict(options.pop("solve_kwargs", {}))

        x0_aug = np.asarray(x0_aug, float).reshape(-1)
        x_s = np.asarray(x_s, float).reshape(self.n_x)
        u_s = np.asarray(u_s, float).reshape(self.n_u)
        y_target = np.asarray(y_target, float).reshape(self.n_y)
        u_prev_dev = np.asarray(u_prev_dev, float).reshape(self.n_u)
        lower, upper = _bounds_to_horizon_matrices(bnds, self.n_u, self.NC)

        active_terminal_constraint = (
            self.terminal_set_on
            and alpha_terminal is not None
            and np.isfinite(float(alpha_terminal))
        )
        active_first_step_contraction = bool(first_step_contraction_on)

        u_var = cp.Variable((self.NC, self.n_u))
        x_var = cp.Variable((self.n_aug, self.NP + 1))
        lyap_slack = cp.Variable(nonneg=True)

        constraints = [x_var[:, 0] == x0_aug]
        if lower is not None:
            lower_rows, lower_cols = np.where(np.isfinite(lower))
            for row_idx, col_idx in zip(lower_rows, lower_cols):
                constraints.append(u_var[row_idx, col_idx] >= float(lower[row_idx, col_idx]))
        if upper is not None:
            upper_rows, upper_cols = np.where(np.isfinite(upper))
            for row_idx, col_idx in zip(upper_rows, upper_cols):
                constraints.append(u_var[row_idx, col_idx] <= float(upper[row_idx, col_idx]))

        objective = 0.0
        for step_idx in range(self.NP):
            ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
            constraints.append(
                x_var[:, step_idx + 1] == self.A @ x_var[:, step_idx] + self.B @ u_var[ctrl_idx, :]
            )
            y_expr = self.C @ x_var[:, step_idx + 1]
            if self.D is not None:
                y_expr = y_expr + self.D @ u_var[ctrl_idx, :]
            objective += cp.quad_form(y_expr - y_target, self.Qy_mat)

        if self._objective_steady_input_cost_on():
            for ctrl_idx in range(self.NC):
                objective += cp.quad_form(u_var[ctrl_idx, :] - u_s, self.Su_mat)

        if self.Rdu_mat is not None:
            objective += cp.quad_form(u_var[0, :] - u_prev_dev, self.Rdu_mat)
            for ctrl_idx in range(1, self.NC):
                objective += cp.quad_form(u_var[ctrl_idx, :] - u_var[ctrl_idx - 1, :], self.Rdu_mat)

        terminal_error = x_var[:self.n_x, self.NP] - x_s
        terminal_value_expr = cp.quad_form(terminal_error, self.P_x)
        if self._objective_terminal_cost_on():
            objective += self.terminal_cost_scale * terminal_value_expr
        if active_terminal_constraint:
            constraints.append(terminal_value_expr <= float(alpha_terminal))

        if active_first_step_contraction:
            V_k = lyapunov_value(x0_aug[:self.n_x] - x_s, self.P_x)
            V_bound = float(lyapunov_bound(V_k, rho=rho_lyap, eps_lyap=eps_lyap))
            first_step_error = x_var[:self.n_x, 1] - x_s
            first_step_value_expr = cp.quad_form(first_step_error, self.P_x)
            constraints.append(first_step_value_expr <= V_bound + lyap_slack)

        objective += float(slack_penalty) * lyap_slack
        problem = cp.Problem(cp.Minimize(objective), constraints)

        ic_flat = np.asarray(IC_opt, float).reshape(-1)
        if ic_flat.size == self.n_u * self.NC:
            try:
                u_guess = reshape_u_sequence(ic_flat, self.n_u, self.NC)
                u_var.value = u_guess
                x_guess, _ = self._predict_from_sequence(u_guess, x0_aug)
                x_var.value = x_guess
            except Exception:
                pass

        if solver_pref_override is None:
            solver_pref = self.solver_pref_conic
        else:
            solver_pref = solver_pref_override
        solver_sequence = tracking_solver_sequence(True, solver_pref=solver_pref)

        last_status = None
        last_solver = None
        last_error = None
        last_objective = None
        last_nit = None
        last_eval = None
        last_slack = None

        for solver_name in solver_sequence:
            try:
                problem.solve(
                    solver=solver_name,
                    warm_start=warm_start,
                    verbose=verbose,
                    **solve_kwargs,
                )
                last_status = problem.status
                last_solver = solver_name
                last_nit = _extract_num_iters(problem)
                if problem.value is not None:
                    last_objective = float(problem.value)
                if u_var.value is None or x_var.value is None or lyap_slack.value is None:
                    continue

                u_value = np.asarray(u_var.value, float)
                x_value = np.asarray(x_var.value, float)
                last_slack = float(np.asarray(lyap_slack.value).item())
                last_eval = self._evaluate_soft_tracking_solution(
                    u_sequence=u_value,
                    x_pred=x_value,
                    x0_aug=x0_aug,
                    x_s=x_s,
                    lower=lower,
                    upper=upper,
                    alpha_terminal=alpha_terminal,
                    terminal_constraint_active=active_terminal_constraint,
                    rho_lyap=rho_lyap,
                    eps_lyap=eps_lyap,
                    slack_lyap=last_slack,
                    status=problem.status,
                )
                if problem.status in _OPTIMAL_STATUSES and last_eval["accepted"]:
                    return SimpleNamespace(
                        success=True,
                        x=u_value.reshape(-1),
                        status=problem.status,
                        message="optimal",
                        fun=last_objective,
                        nit=last_nit,
                        solver=solver_name,
                        error=None,
                        objective_value=last_objective,
                        dyn_residual_inf=last_eval["dyn_residual_inf"],
                        bound_violation_inf=last_eval["bound_violation_inf"],
                        terminal_value=last_eval["terminal_value"],
                        terminal_constraint_violation=last_eval["terminal_constraint_violation"],
                        V_k=last_eval["V_k"],
                        V_next_first=last_eval["V_next_first"],
                        V_bound=last_eval["V_bound"],
                        contraction_margin=last_eval["contraction_margin"],
                        first_step_contraction_satisfied=last_eval["first_step_contraction_satisfied"],
                        contraction_constraint_violation=last_eval["contraction_constraint_violation"],
                        lyapunov_mode="soft",
                        slack_lyap=last_slack,
                        slack_penalty=float(slack_penalty),
                        relaxed_contraction_satisfied=last_eval["relaxed_contraction_satisfied"],
                        relaxed_contraction_violation=last_eval["relaxed_contraction_violation"],
                    )
            except Exception as exc:
                last_error = repr(exc)

        reject_reason = None if last_eval is None else last_eval.get("reject_reason")
        if reject_reason is None and last_error is not None:
            reject_reason = "solver_error"
        if reject_reason is None:
            reject_reason = "solver_status"

        return SimpleNamespace(
            success=False,
            x=None,
            status=last_status,
            message=reject_reason,
            fun=last_objective,
            nit=last_nit,
            solver=last_solver,
            error=last_error,
            objective_value=last_objective,
            dyn_residual_inf=None if last_eval is None else last_eval["dyn_residual_inf"],
            bound_violation_inf=None if last_eval is None else last_eval["bound_violation_inf"],
            terminal_value=None if last_eval is None else last_eval["terminal_value"],
            terminal_constraint_violation=None
            if last_eval is None
            else last_eval["terminal_constraint_violation"],
            V_k=None if last_eval is None else last_eval["V_k"],
            V_next_first=None if last_eval is None else last_eval["V_next_first"],
            V_bound=None if last_eval is None else last_eval["V_bound"],
            contraction_margin=None if last_eval is None else last_eval["contraction_margin"],
            first_step_contraction_satisfied=None
            if last_eval is None
            else last_eval["first_step_contraction_satisfied"],
            contraction_constraint_violation=None
            if last_eval is None
            else last_eval["contraction_constraint_violation"],
            lyapunov_mode="soft",
            slack_lyap=0.0 if last_slack is None else float(last_slack),
            slack_penalty=float(slack_penalty),
            relaxed_contraction_satisfied=None
            if last_eval is None
            else last_eval["relaxed_contraction_satisfied"],
            relaxed_contraction_violation=None
            if last_eval is None
            else last_eval["relaxed_contraction_violation"],
        )


def design_direct_lyapunov_mpc_solver(
    A_aug,
    B_aug,
    C_aug,
    Qy_diag,
    NP,
    NC,
    *,
    Su_diag=None,
    u_min=None,
    u_max=None,
    lambda_u=1.0,
    qx_eps=1e-10,
    Rdu_diag=None,
    terminal_set_on=True,
    terminal_alpha_scale=1.0,
    terminal_cost_scale=1.0,
    objective_steady_input_cost=False,
    objective_terminal_cost=False,
    D=None,
    solver_pref_qp=None,
    solver_pref_conic=None,
    return_design=False,
):
    P_x, K_x, design_debug = design_standard_tracking_terminal_ingredients(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        Qy_diag=Qy_diag,
        Su_diag=Su_diag,
        u_min=u_min,
        u_max=u_max,
        lambda_u=lambda_u,
        qx_eps=qx_eps,
        return_debug=True,
    )
    Su_used = np.diag(np.asarray(design_debug["Su"], dtype=float)).copy()
    solver = DirectOutputDisturbanceLyapunovMpcSolver(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        Qy_diag=Qy_diag,
        Su_diag=Su_used,
        NP=NP,
        NC=NC,
        P_x=P_x,
        K_x=K_x,
        Rdu_diag=Rdu_diag,
        terminal_set_on=terminal_set_on,
        terminal_alpha_scale=terminal_alpha_scale,
        terminal_cost_scale=terminal_cost_scale,
        D=D,
        solver_pref_qp=solver_pref_qp,
        solver_pref_conic=solver_pref_conic,
    )
    solver.objective_steady_input_cost = bool(objective_steady_input_cost)
    solver.objective_terminal_cost = bool(objective_terminal_cost)
    if return_design:
        design = dict(design_debug)
        design.update({"P_x": np.asarray(P_x, float).copy(), "K_x": np.asarray(K_x, float).copy(), "Su_diag_used": Su_used})
        design.update(
            {
                "objective_steady_input_cost": bool(objective_steady_input_cost),
                "objective_terminal_cost": bool(objective_terminal_cost),
            }
        )
        return solver, design
    return solver


def _target_config_dict(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_DIRECT_TARGET_CONFIG)
    if config:
        merged.update(dict(config))
    return merged


def direct_lyapunov_evaluation_ingredients(LMPC_obj) -> Dict[str, np.ndarray]:
    A_aug = np.asarray(LMPC_obj.A, dtype=float)
    B_aug = np.asarray(LMPC_obj.B, dtype=float)
    C_aug = np.asarray(LMPC_obj.C, dtype=float)
    P_x = getattr(LMPC_obj, "P_x", None)
    if P_x is None:
        raise ValueError("LMPC_obj must expose P_x for Lyapunov candidate evaluation.")

    n_y = C_aug.shape[0]
    n_x = A_aug.shape[0] - n_y
    return {
        "A_phys": A_aug[:n_x, :n_x].copy(),
        "B_phys": B_aug[:n_x, :].copy(),
        "C_phys": C_aug[:, :n_x].copy(),
        "Cd_phys": C_aug[:, n_x:].copy(),
        "P_x": np.asarray(P_x, dtype=float).copy(),
    }


def prepare_direct_output_disturbance_step(
    *,
    LMPC_obj,
    x0_aug: np.ndarray,
    y_sp_k: np.ndarray,
    u_prev_dev: np.ndarray,
    u_dev_min: np.ndarray,
    u_dev_max: np.ndarray,
    target_mode: str = "bounded",
    target_config: Optional[Dict[str, Any]] = None,
    target_H: Optional[np.ndarray] = None,
    x_target_prev_success: Optional[np.ndarray] = None,
    step_idx: Optional[int] = None,
    y_prev_scaled: Optional[np.ndarray] = None,
    plant_mode: Optional[str] = None,
    disturbance_after_step: Optional[bool] = None,
    use_target_output_for_tracking: bool = False,
    slack_penalty: float = DEFAULT_DIRECT_SOFT_SLACK_PENALTY,
) -> Dict[str, Any]:
    n_inputs = LMPC_obj.B.shape[1]
    n_outputs = LMPC_obj.C.shape[0]
    n_aug = LMPC_obj.A.shape[0]
    n_x = n_aug - n_outputs

    x0_aug = np.asarray(x0_aug, dtype=float).reshape(-1)
    y_sp_k = np.asarray(y_sp_k, dtype=float).reshape(-1)
    u_prev_dev = np.asarray(u_prev_dev, dtype=float).reshape(-1)
    u_dev_min = np.asarray(u_dev_min, dtype=float).reshape(-1)
    u_dev_max = np.asarray(u_dev_max, dtype=float).reshape(-1)

    yhat_now = np.asarray(LMPC_obj.C @ x0_aug, float).reshape(-1)
    innovation = None
    if y_prev_scaled is not None:
        innovation = np.asarray(y_prev_scaled, dtype=float).reshape(-1) - yhat_now

    target_info = solve_output_disturbance_target(
        LMPC_obj.A,
        LMPC_obj.B,
        LMPC_obj.C,
        x0_aug,
        y_sp_k,
        target_mode=target_mode,
        u_min=u_dev_min,
        u_max=u_dev_max,
        config=_target_config_dict(target_config),
        H=target_H,
        u_ref=u_prev_dev,
        x_ref=x_target_prev_success,
    )
    target_info = {} if target_info is None else dict(target_info)
    target_info.update(
        {
            "step": int(step_idx) if step_idx is not None else -1,
            "y_sp": y_sp_k.copy(),
            "x0_aug": x0_aug.copy(),
            "yhat_now": yhat_now.copy(),
            "innovation": None if innovation is None else innovation.copy(),
            "target_mode": target_mode,
        }
    )

    x_target_next = x_target_prev_success
    if target_info.get("success", False) and target_info.get("x_s") is not None:
        x_target_next = np.asarray(target_info.get("x_s"), float).reshape(n_x).copy()

    step_info = {
        "step": int(step_idx) if step_idx is not None else -1,
        "success": False,
        "method": None,
        "target_mode": target_mode,
        "lyapunov_mode": None,
        "plant_mode": plant_mode,
        "disturbance_after_step": disturbance_after_step,
        "target_success": bool(target_info.get("success", False)),
        "target_stage": target_info.get("solve_stage"),
        "target_variant": target_info.get("target_variant"),
        "x0_aug": x0_aug.copy(),
        "y_sp": y_sp_k.copy(),
        "yhat_now": yhat_now.copy(),
        "innovation": None if innovation is None else innovation.copy(),
        "use_target_output_for_tracking": bool(use_target_output_for_tracking),
        "u_prev_dev": u_prev_dev.copy(),
        "x_s": None if target_info.get("x_s") is None else np.asarray(target_info.get("x_s"), float).copy(),
        "u_s": None if target_info.get("u_s") is None else np.asarray(target_info.get("u_s"), float).copy(),
        "d_s": None if target_info.get("d_s") is None else np.asarray(target_info.get("d_s"), float).copy(),
        "x_s_aug": None if target_info.get("x_s_aug") is None else np.asarray(target_info.get("x_s_aug"), float).copy(),
        "y_s": None if target_info.get("y_s") is None else np.asarray(target_info.get("y_s"), float).copy(),
        "y_target": None,
        "y_s_minus_y_sp": None,
        "y_target_minus_y_sp": None,
        "status": None,
        "message": None,
        "fun": None,
        "solver_nit": None,
        "tracking_solver": None,
        "tracking_error": None,
        "alpha_terminal_raw": None,
        "alpha_terminal": None,
        "alpha_terminal_used": None,
        "terminal_constraint_skipped": None,
        "u_apply": None,
        "target_rank_M": target_info.get("rank_M"),
        "target_cond_M": target_info.get("cond_M"),
        "target_cond_G": target_info.get("cond_G"),
        "target_residual_total_norm": target_info.get("residual_total_norm"),
        "target_exact_within_bounds": target_info.get("exact_within_bounds"),
        "target_bounded_solution_used": target_info.get("bounded_solution_used"),
        "target_u_ref": None
        if target_info.get("u_ref") is None
        else np.asarray(target_info.get("u_ref"), float).copy(),
        "target_u_ref_weight": None
        if target_info.get("u_ref_weight") is None
        else np.asarray(target_info.get("u_ref_weight"), float).copy(),
        "target_u_ref_active": target_info.get("u_ref_active"),
        "target_u_ref_penalty": target_info.get("u_ref_penalty"),
        "target_us_u_ref_inf": target_info.get("us_u_ref_inf"),
        "target_x_ref": None
        if target_info.get("x_ref") is None
        else np.asarray(target_info.get("x_ref"), float).copy(),
        "target_x_ref_weight": None
        if target_info.get("x_ref_weight") is None
        else np.asarray(target_info.get("x_ref_weight"), float).copy(),
        "target_x_ref_active": target_info.get("x_ref_active"),
        "target_x_ref_penalty": target_info.get("x_ref_penalty"),
        "target_xs_x_ref_inf": target_info.get("xs_x_ref_inf"),
        "target_bounded_active_lower_mask": None
        if target_info.get("bounded_active_lower_mask") is None
        else np.asarray(target_info.get("bounded_active_lower_mask"), bool).copy(),
        "target_bounded_active_upper_mask": None
        if target_info.get("bounded_active_upper_mask") is None
        else np.asarray(target_info.get("bounded_active_upper_mask"), bool).copy(),
        "target_exact_active_lower_mask": None
        if target_info.get("exact_active_lower_mask") is None
        else np.asarray(target_info.get("exact_active_lower_mask"), bool).copy(),
        "target_exact_active_upper_mask": None
        if target_info.get("exact_active_upper_mask") is None
        else np.asarray(target_info.get("exact_active_upper_mask"), bool).copy(),
        "slack_lyap": 0.0,
        "slack_penalty": float(slack_penalty),
        "objective_steady_input_cost": bool(getattr(LMPC_obj, "objective_steady_input_cost", False)),
        "objective_terminal_cost": bool(getattr(LMPC_obj, "objective_terminal_cost", False)),
    }
    return {
        "target_info": target_info,
        "step_info": step_info,
        "x_target_prev_success_next": x_target_next,
        "yhat_now": yhat_now,
        "innovation": innovation,
    }


def solve_direct_tracking_from_target(
    *,
    LMPC_obj,
    x0_aug: np.ndarray,
    y_sp_k: np.ndarray,
    u_prev_dev: np.ndarray,
    target_info: Dict[str, Any],
    step_info: Optional[Dict[str, Any]],
    IC_opt: np.ndarray,
    bnds,
    u_dev_min: np.ndarray,
    u_dev_max: np.ndarray,
    rho_lyap: float = 0.99,
    lyap_eps: float = 1e-9,
    lyapunov_mode: str = "hard",
    use_target_output_for_tracking: bool = False,
    skip_terminal_if_alpha_small: bool = True,
    alpha_terminal_min: float = 1e-8,
    use_target_on_solver_fail: bool = False,
    slack_penalty: float = DEFAULT_DIRECT_SOFT_SLACK_PENALTY,
    first_step_contraction_on: bool = True,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    step_info = {} if step_info is None else dict(step_info)
    step_info["lyapunov_mode"] = lyapunov_mode

    x0_aug = np.asarray(x0_aug, dtype=float).reshape(-1)
    y_sp_k = np.asarray(y_sp_k, dtype=float).reshape(-1)
    u_prev_dev = np.asarray(u_prev_dev, dtype=float).reshape(-1)
    u_dev_min = np.asarray(u_dev_min, dtype=float).reshape(-1)
    u_dev_max = np.asarray(u_dev_max, dtype=float).reshape(-1)
    IC_opt = np.asarray(IC_opt, dtype=float).copy()

    if target_info.get("success", False) and target_info.get("x_s") is not None and target_info.get("u_s") is not None:
        x_s = np.asarray(target_info["x_s"], float).reshape(-1)
        u_s = np.asarray(target_info["u_s"], float).reshape(-1)
        y_s = np.asarray(target_info["y_s"], float).reshape(-1)
        y_target = y_s.copy() if use_target_output_for_tracking else y_sp_k.copy()

        alpha_terminal_raw = compute_terminal_alpha_input_only(
            P_x=LMPC_obj.P_x,
            K_x=LMPC_obj.K_x,
            u_s=u_s,
            u_min=u_dev_min,
            u_max=u_dev_max,
            alpha_scale=1.0,
        )
        alpha_terminal = compute_terminal_alpha_input_only(
            P_x=LMPC_obj.P_x,
            K_x=LMPC_obj.K_x,
            u_s=u_s,
            u_min=u_dev_min,
            u_max=u_dev_max,
            alpha_scale=LMPC_obj.terminal_alpha_scale,
        )
        terminal_constraint_skipped = bool(
            skip_terminal_if_alpha_small and alpha_terminal <= float(alpha_terminal_min)
        )
        terminal_set_on_prev = LMPC_obj.terminal_set_on
        alpha_for_solver = None if terminal_constraint_skipped else float(alpha_terminal)
        if terminal_constraint_skipped:
            LMPC_obj.terminal_set_on = False

        try:
            sol = LMPC_obj.solve_tracking_mpc_step(
                IC_opt=IC_opt,
                bnds=bnds,
                y_target=y_target,
                u_prev_dev=u_prev_dev,
                x0_aug=x0_aug,
                x_s=x_s,
                u_s=u_s,
                alpha_terminal=alpha_for_solver,
                rho_lyap=rho_lyap,
                eps_lyap=lyap_eps,
                first_step_contraction_on=first_step_contraction_on,
                lyapunov_mode=lyapunov_mode,
                slack_penalty=slack_penalty,
                options=solver_options,
            )
        finally:
            LMPC_obj.terminal_set_on = terminal_set_on_prev

        step_info.update(
            {
                "status": getattr(sol, "status", None),
                "message": getattr(sol, "message", None),
                "fun": float(sol.fun) if getattr(sol, "fun", None) is not None else None,
                "solver_nit": getattr(sol, "nit", None),
                "tracking_solver": getattr(sol, "solver", None),
                "tracking_error": getattr(sol, "error", None),
                "alpha_terminal_raw": float(alpha_terminal_raw),
                "alpha_terminal": float(alpha_terminal),
                "alpha_terminal_used": None if alpha_for_solver is None else float(alpha_for_solver),
                "terminal_constraint_skipped": terminal_constraint_skipped,
                "y_target": y_target.copy(),
                "y_s_minus_y_sp": y_s - y_sp_k,
                "y_target_minus_y_sp": y_target - y_sp_k,
            }
        )

        if getattr(sol, "success", False):
            u_dev_apply = np.asarray(sol.x[: LMPC_obj.B.shape[1]], float).reshape(-1)
            u_dev_apply = np.clip(u_dev_apply, u_dev_min, u_dev_max)
            IC_opt_next = shift_input_guess(sol.x, LMPC_obj.B.shape[1], LMPC_obj.NC)
            report = LMPC_obj.standard_tracking_report(
                x_opt=sol.x,
                x0_aug=x0_aug,
                x_s=x_s,
                u_s=u_s,
                y_target=y_target,
                u_prev_dev=u_prev_dev,
                alpha_terminal=alpha_for_solver,
                rho_lyap=rho_lyap,
                eps_lyap=lyap_eps,
                first_step_contraction_on=first_step_contraction_on,
                lyapunov_mode=lyapunov_mode,
                slack_lyap=getattr(sol, "slack_lyap", 0.0),
                slack_penalty=slack_penalty,
            )
            step_info.update(
                {
                    "success": True,
                    "method": "direct_lyapunov_mpc",
                    "u_apply": u_dev_apply.copy(),
                    **report,
                }
            )
            return u_dev_apply, IC_opt_next, step_info

        if use_target_on_solver_fail:
            u_dev_apply = np.clip(u_s, u_dev_min, u_dev_max)
            fail_method = "solver_fail_use_target"
        else:
            u_dev_apply = np.clip(u_prev_dev, u_dev_min, u_dev_max)
            fail_method = "solver_fail_hold_prev"
        IC_opt_next = np.tile(u_dev_apply, LMPC_obj.NC)
        step_info.update({"method": fail_method, "u_apply": u_dev_apply.copy()})
        return u_dev_apply, IC_opt_next, step_info

    u_dev_apply = np.clip(u_prev_dev, u_dev_min, u_dev_max)
    IC_opt_next = np.tile(u_dev_apply, LMPC_obj.NC)
    step_info.update(
        {
            "method": "target_fail_hold_prev",
            "u_apply": u_dev_apply.copy(),
            "message": "target solve failed",
        }
    )
    return u_dev_apply, IC_opt_next, step_info


def run_direct_output_disturbance_lyapunov_mpc(
    system,
    LMPC_obj,
    y_sp_scenario,
    n_tests,
    set_points_len,
    steady_states,
    IC_opt,
    bnds,
    L,
    data_min,
    data_max,
    test_cycle,
    reward_fn,
    nominal_qi,
    nominal_qs,
    nominal_ha,
    qi_change,
    qs_change,
    ha_change,
    *,
    target_mode="unbounded",
    lyapunov_mode="hard",
    target_config=None,
    target_H=None,
    mode="nominal",
    disturbance_after_step=True,
    use_target_output_for_tracking=False,
    skip_terminal_if_alpha_small=True,
    alpha_terminal_min=1e-8,
    use_target_on_solver_fail=False,
    rho_lyap=0.99,
    lyap_eps=1e-9,
    slack_penalty=DEFAULT_DIRECT_SOFT_SLACK_PENALTY,
    first_step_contraction_on=True,
    reset_system_on_entry=True,
    solver_options=None,
):
    target_mode = _as_mode(target_mode, ("unbounded", "bounded"), "target_mode")
    lyapunov_mode = _as_mode(lyapunov_mode, ("hard", "soft"), "lyapunov_mode")
    mode = _as_mode(mode, ("nominal", "disturb"), "mode")
    disturbance_after_step = bool(disturbance_after_step)
    nominal_qi_value = _as_scalar_float(nominal_qi, "nominal_qi")
    nominal_qs_value = _as_scalar_float(nominal_qs, "nominal_qs")
    nominal_ha_value = _as_scalar_float(nominal_ha, "nominal_ha")

    system.Qi = nominal_qi_value
    system.Qs = nominal_qs_value
    system.hA = nominal_ha_value
    if reset_system_on_entry:
        _reset_system_on_entry(system)
        system.Qi = nominal_qi_value
        system.Qs = nominal_qs_value
        system.hA = nominal_ha_value

    (
        y_sp,
        nFE,
        sub_changes,
        time_in_sub_episodes,
        _,
        _,
        qi,
        qs,
        ha,
    ) = generate_setpoints_training_rl_gradually(
        y_sp_scenario,
        n_tests,
        set_points_len,
        0,
        test_cycle,
        nominal_qi,
        nominal_qs,
        nominal_ha,
        qi_change,
        qs_change,
        ha_change,
    )

    n_inputs = LMPC_obj.B.shape[1]
    n_outputs = LMPC_obj.C.shape[0]
    n_aug = LMPC_obj.A.shape[0]
    n_x = n_aug - n_outputs

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    u_dev_min = np.array([bnds[j][0] for j in range(n_inputs)], dtype=float)
    u_dev_max = np.array([bnds[j][1] for j in range(n_inputs)], dtype=float)

    y_mpc = np.zeros((nFE + 1, n_outputs), dtype=float)
    y_mpc[0, :] = _system_io_phys(system, steady_states)[1]
    u_applied_phys = np.zeros((nFE, n_inputs), dtype=float)
    yhat = np.zeros((n_outputs, nFE), dtype=float)
    xhatdhat = np.zeros((n_aug, nFE + 1), dtype=float)
    rewards = np.zeros(nFE, dtype=float)
    avg_rewards = []
    delta_y_storage = []
    delta_u_storage = []
    direct_info_storage = []
    target_info_storage = []
    x_target_prev_success = None

    IC_opt = np.asarray(IC_opt, float).copy()
    for step_idx in range(nFE):
        x0_aug = xhatdhat[:, step_idx].copy()
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        u_prev_dev = scaled_current_input - ss_scaled_inputs
        y_sp_k = get_y_sp_step(y_sp, step_idx, n_outputs)
        y_prev_scaled = apply_min_max(y_mpc[step_idx, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        step_context = prepare_direct_output_disturbance_step(
            LMPC_obj=LMPC_obj,
            x0_aug=x0_aug,
            y_sp_k=y_sp_k,
            u_prev_dev=u_prev_dev,
            u_dev_min=u_dev_min,
            u_dev_max=u_dev_max,
            target_mode=target_mode,
            target_config=target_config,
            target_H=target_H,
            x_target_prev_success=x_target_prev_success,
            step_idx=step_idx,
            y_prev_scaled=y_prev_scaled,
            plant_mode=mode,
            disturbance_after_step=disturbance_after_step,
            use_target_output_for_tracking=use_target_output_for_tracking,
            slack_penalty=slack_penalty,
        )
        target_info = step_context["target_info"]
        target_info_storage.append(target_info)
        x_target_prev_success = step_context["x_target_prev_success_next"]
        step_info = step_context["step_info"]
        yhat_now = step_context["yhat_now"]
        innovation = step_context["innovation"]

        u_dev_apply, IC_opt, step_info = solve_direct_tracking_from_target(
            LMPC_obj=LMPC_obj,
            x0_aug=x0_aug,
            y_sp_k=y_sp_k,
            u_prev_dev=u_prev_dev,
            target_info=target_info,
            step_info=step_info,
            IC_opt=IC_opt,
            bnds=bnds,
            u_dev_min=u_dev_min,
            u_dev_max=u_dev_max,
            rho_lyap=rho_lyap,
            lyap_eps=lyap_eps,
            lyapunov_mode=lyapunov_mode,
            use_target_output_for_tracking=use_target_output_for_tracking,
            skip_terminal_if_alpha_small=skip_terminal_if_alpha_small,
            alpha_terminal_min=alpha_terminal_min,
            use_target_on_solver_fail=use_target_on_solver_fail,
            slack_penalty=slack_penalty,
            first_step_contraction_on=first_step_contraction_on,
            solver_options=solver_options,
        )

        u_scaled = u_dev_apply + ss_scaled_inputs
        u_phys = reverse_min_max(u_scaled, data_min[:n_inputs], data_max[:n_inputs])
        u_applied_phys[step_idx, :] = u_phys.copy()
        delta_u = u_scaled - scaled_current_input

        if mode == "disturb" and not disturbance_after_step:
            system.hA = ha[step_idx]
            system.Qs = qs[step_idx]
            system.Qi = qi[step_idx]

        _set_system_input_phys(system, steady_states, u_phys)
        system.step()

        if mode == "disturb" and disturbance_after_step:
            system.hA = ha[step_idx]
            system.Qs = qs[step_idx]
            system.Qi = qi[step_idx]

        y_phys = _system_io_phys(system, steady_states)[1]
        y_mpc[step_idx + 1, :] = y_phys

        y_current_scaled = apply_min_max(y_mpc[step_idx + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        delta_y = y_current_scaled - y_sp_k
        y_target_step = step_info.get("y_target")
        delta_y_target = None if y_target_step is None else y_current_scaled - np.asarray(y_target_step, dtype=float).reshape(n_outputs)

        yhat[:, step_idx] = yhat_now
        xhat_next_openloop = LMPC_obj.A @ x0_aug + LMPC_obj.B @ u_dev_apply
        observer_correction = L @ innovation
        xhatdhat[:, step_idx + 1] = xhat_next_openloop + observer_correction

        y_sp_phys = reverse_min_max(y_sp_k + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        reward = reward_fn(delta_y, delta_u, y_sp_phys)
        rewards[step_idx] = reward
        delta_y_storage.append(delta_y.copy())
        delta_u_storage.append(delta_u.copy())

        step_info.update(
            {
                "y_current_scaled": y_current_scaled.copy(),
                "xhat_next_openloop": xhat_next_openloop.copy(),
                "observer_correction": observer_correction.copy(),
                "xhat_next": xhatdhat[:, step_idx + 1].copy(),
                "reward": float(reward),
                "delta_y": delta_y.copy(),
                "y_minus_y_sp": delta_y.copy(),
                "y_minus_y_target": None if delta_y_target is None else delta_y_target.copy(),
                "delta_u": delta_u.copy(),
                "slack_lyap": float(step_info.get("slack_lyap", 0.0) or 0.0),
            }
        )
        direct_info_storage.append(step_info)

        if step_idx in sub_changes:
            avg_rewards.append(np.mean(rewards[step_idx - time_in_sub_episodes + 1:step_idx + 1]))
            last = direct_info_storage[-1]
            print(
                "Sub_Episode:", sub_changes[step_idx],
                "| avg. reward:", avg_rewards[-1],
                "| target_mode:", target_mode,
                "| lyapunov_mode:", lyapunov_mode,
                "| plant_mode:", mode,
                "| success:", last.get("success"),
                "| target_stage:", last.get("target_stage"),
                "| contraction_margin:", last.get("contraction_margin"),
                "| slack_lyap:", last.get("slack_lyap"),
                "| nit:", last.get("solver_nit"),
            )

    return {
        "y_system": y_mpc,
        "u_applied_phys": u_applied_phys,
        "avg_rewards": avg_rewards,
        "rewards": rewards,
        "xhatdhat": xhatdhat,
        "nFE": int(nFE),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "y_sp": np.asarray(y_sp, float).copy(),
        "yhat": yhat,
        "delta_y_storage": delta_y_storage,
        "delta_u_storage": delta_u_storage,
        "direct_info_storage": direct_info_storage,
        "target_info_storage": target_info_storage,
        "qi": np.asarray(qi, float).copy(),
        "qs": np.asarray(qs, float).copy(),
        "ha": np.asarray(ha, float).copy(),
        "target_mode": target_mode,
        "lyapunov_mode": lyapunov_mode,
        "plant_mode": mode,
        "disturbance_after_step": disturbance_after_step,
        "use_target_output_for_tracking": bool(use_target_output_for_tracking),
        "nominal_qi": nominal_qi_value,
        "nominal_qs": nominal_qs_value,
        "nominal_ha": nominal_ha_value,
        "final_qi": float(system.Qi),
        "final_qs": float(system.Qs),
        "final_ha": float(system.hA),
        "rho_lyap": float(rho_lyap),
        "lyap_eps": float(lyap_eps),
        "slack_penalty": float(slack_penalty),
        "first_step_contraction_on": bool(first_step_contraction_on),
        "u_dev_min": u_dev_min.copy(),
        "u_dev_max": u_dev_max.copy(),
        "delta_t": float(getattr(system, "delta_t", 1.0)),
    }


def _normalize_step_matrix(values: np.ndarray, n_steps: int, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        if values.size != width:
            raise ValueError(f"Expected vector of width {width}, got {values.size}.")
        return np.tile(values.reshape(1, -1), (n_steps, 1))
    if values.shape == (width, n_steps):
        return values.T.copy()
    if values.shape == (n_steps, width):
        return values.copy()
    raise ValueError(f"Cannot normalize array of shape {values.shape} into ({n_steps}, {width}).")


def make_direct_lyapunov_step_records(step_info_storage):
    records = []
    for info in step_info_storage:
        row = {
            "step": int(info.get("step", len(records))),
            "success": bool(info.get("success", False)),
            "method": info.get("method"),
            "target_mode": info.get("target_mode"),
            "lyapunov_mode": info.get("lyapunov_mode"),
            "plant_mode": info.get("plant_mode"),
            "disturbance_after_step": info.get("disturbance_after_step"),
            "target_success": bool(info.get("target_success", False)),
            "target_stage": info.get("target_stage"),
            "target_variant": info.get("target_variant"),
            "target_rank_M": info.get("target_rank_M"),
            "target_cond_M": info.get("target_cond_M"),
            "target_cond_G": info.get("target_cond_G"),
            "target_residual_total_norm": info.get("target_residual_total_norm"),
            "target_exact_within_bounds": info.get("target_exact_within_bounds"),
            "target_bounded_solution_used": info.get("target_bounded_solution_used"),
            "target_u_ref": json.dumps(_jsonable(info.get("target_u_ref"))),
            "target_u_ref_weight": json.dumps(_jsonable(info.get("target_u_ref_weight"))),
            "target_u_ref_active": info.get("target_u_ref_active"),
            "target_u_ref_penalty": info.get("target_u_ref_penalty"),
            "target_us_u_ref_inf": info.get("target_us_u_ref_inf"),
            "target_x_ref": json.dumps(_jsonable(info.get("target_x_ref"))),
            "target_x_ref_weight": json.dumps(_jsonable(info.get("target_x_ref_weight"))),
            "target_x_ref_active": info.get("target_x_ref_active"),
            "target_x_ref_penalty": info.get("target_x_ref_penalty"),
            "target_xs_x_ref_inf": info.get("target_xs_x_ref_inf"),
            "target_bounded_active_lower_count": None
            if info.get("target_bounded_active_lower_mask") is None
            else int(np.sum(np.asarray(info.get("target_bounded_active_lower_mask"), dtype=bool))),
            "target_bounded_active_upper_count": None
            if info.get("target_bounded_active_upper_mask") is None
            else int(np.sum(np.asarray(info.get("target_bounded_active_upper_mask"), dtype=bool))),
            "target_exact_active_lower_count": None
            if info.get("target_exact_active_lower_mask") is None
            else int(np.sum(np.asarray(info.get("target_exact_active_lower_mask"), dtype=bool))),
            "target_exact_active_upper_count": None
            if info.get("target_exact_active_upper_mask") is None
            else int(np.sum(np.asarray(info.get("target_exact_active_upper_mask"), dtype=bool))),
            "solver_status": info.get("status"),
            "solver_name": info.get("tracking_solver"),
            "solver_message": info.get("message"),
            "solver_nit": info.get("solver_nit"),
            "objective_value": info.get("fun"),
            "terminal_value": info.get("terminal_value"),
            "terminal_margin": info.get("terminal_margin"),
            "terminal_set_violated": info.get("terminal_set_violated"),
            "V_k": info.get("V_k"),
            "V_next_first": info.get("V_next_first"),
            "V_bound": info.get("V_bound"),
            "contraction_margin": info.get("contraction_margin"),
            "first_step_contraction_satisfied": info.get("first_step_contraction_satisfied"),
            "contraction_constraint_violation": info.get("contraction_constraint_violation"),
            "slack_lyap": info.get("slack_lyap"),
            "slack_penalty": info.get("slack_penalty"),
            "objective_steady_input_cost": info.get("objective_steady_input_cost"),
            "objective_terminal_cost": info.get("objective_terminal_cost"),
            "relaxed_contraction_satisfied": info.get("relaxed_contraction_satisfied"),
            "relaxed_contraction_violation": info.get("relaxed_contraction_violation"),
            "reward": info.get("reward"),
            "use_target_output_for_tracking": info.get("use_target_output_for_tracking"),
            "u_apply": json.dumps(_jsonable(info.get("u_apply"))),
            "u_prev_dev": json.dumps(_jsonable(info.get("u_prev_dev"))),
            "u_s": json.dumps(_jsonable(info.get("u_s"))),
            "x_s": json.dumps(_jsonable(info.get("x_s"))),
            "d_s": json.dumps(_jsonable(info.get("d_s"))),
            "y_sp": json.dumps(_jsonable(info.get("y_sp"))),
            "y_s": json.dumps(_jsonable(info.get("y_s"))),
            "y_target": json.dumps(_jsonable(info.get("y_target"))),
            "y_s_minus_y_sp": json.dumps(_jsonable(info.get("y_s_minus_y_sp"))),
            "y_target_minus_y_sp": json.dumps(_jsonable(info.get("y_target_minus_y_sp"))),
            "delta_y": json.dumps(_jsonable(info.get("delta_y"))),
            "y_minus_y_sp": json.dumps(_jsonable(info.get("y_minus_y_sp"))),
            "y_minus_y_target": json.dumps(_jsonable(info.get("y_minus_y_target"))),
            "delta_u": json.dumps(_jsonable(info.get("delta_u"))),
        }
        records.append(row)
    return records


def summarize_direct_lyapunov_bundle(bundle):
    step_info_storage = list(bundle["direct_info_storage"])
    rewards = np.asarray(bundle["rewards"], dtype=float)
    slack = np.asarray(bundle["slack_lyap"], dtype=float)
    target_success = np.asarray(bundle["target_success_flags"], dtype=float)
    solver_success = np.asarray(bundle["solver_success_flags"], dtype=float)
    hard_ok = np.asarray(bundle["first_step_contraction_satisfied_flags"], dtype=float)
    relaxed_ok = np.asarray(bundle["relaxed_contraction_satisfied_flags"], dtype=float)

    methods = [str(info.get("method")) for info in step_info_storage]
    target_stages = [str(info.get("target_stage")) for info in step_info_storage]
    solver_statuses = [str(info.get("status")) for info in step_info_storage]

    summary = {
        "source": bundle.get("source"),
        "n_steps": int(bundle["nFE"]),
        "target_mode": bundle.get("target_mode"),
        "lyapunov_mode": bundle.get("lyapunov_mode"),
        "plant_mode": bundle.get("plant_mode"),
        "disturbance_after_step": bundle.get("disturbance_after_step"),
        "use_target_output_for_tracking": bundle.get("use_target_output_for_tracking"),
        "nominal_qi": bundle.get("nominal_qi"),
        "nominal_qs": bundle.get("nominal_qs"),
        "nominal_ha": bundle.get("nominal_ha"),
        "final_qi": bundle.get("final_qi"),
        "final_qs": bundle.get("final_qs"),
        "final_ha": bundle.get("final_ha"),
        "reward_mean": float(np.mean(rewards)) if rewards.size else None,
        "reward_sum": float(np.sum(rewards)) if rewards.size else None,
        "target_success_rate": float(np.mean(target_success)) if target_success.size else None,
        "solver_success_rate": float(np.mean(solver_success)) if solver_success.size else None,
        "hard_contraction_rate": float(np.mean(hard_ok)) if hard_ok.size else None,
        "relaxed_contraction_rate": float(np.mean(relaxed_ok)) if relaxed_ok.size else None,
        "slack_lyap_max": float(np.nanmax(slack)) if slack.size else None,
        "slack_lyap_mean": float(np.nanmean(slack)) if slack.size else None,
        "slack_lyap_active_steps": int(np.sum(np.nan_to_num(slack, nan=0.0) > 1.0e-9)),
        "contraction_margin_max": float(np.nanmax(bundle["contraction_margin"])) if bundle["contraction_margin"].size else None,
        "contraction_margin_min": float(np.nanmin(bundle["contraction_margin"])) if bundle["contraction_margin"].size else None,
        "target_residual_total_norm_max": float(np.nanmax(bundle["target_residual_total_norm"])) if bundle["target_residual_total_norm"].size else None,
        "target_reference_error_inf_mean": _safe_nanmean(bundle.get("target_reference_error_inf", [])),
        "target_reference_error_inf_max": _safe_nanmax(bundle.get("target_reference_error_inf", [])),
        "tracking_reference_error_inf_mean": _safe_nanmean(bundle.get("tracking_reference_error_inf", [])),
        "tracking_reference_error_inf_max": _safe_nanmax(bundle.get("tracking_reference_error_inf", [])),
        "output_reference_error_inf_mean": _safe_nanmean(bundle.get("output_reference_error_inf", [])),
        "output_reference_error_inf_max": _safe_nanmax(bundle.get("output_reference_error_inf", [])),
        "output_tracking_error_inf_mean": _safe_nanmean(bundle.get("output_tracking_error_inf", [])),
        "output_tracking_error_inf_max": _safe_nanmax(bundle.get("output_tracking_error_inf", [])),
        "target_u_ref_penalty_mean": _safe_nanmean(bundle.get("target_u_ref_penalty", [])),
        "target_u_ref_penalty_max": _safe_nanmax(bundle.get("target_u_ref_penalty", [])),
        "target_us_u_ref_inf_mean": _safe_nanmean(bundle.get("target_us_u_ref_inf", [])),
        "target_us_u_ref_inf_max": _safe_nanmax(bundle.get("target_us_u_ref_inf", [])),
        "target_u_ref_active_steps": int(np.nansum(bundle.get("target_u_ref_active_flags", []))),
        "target_x_ref_penalty_mean": _safe_nanmean(bundle.get("target_x_ref_penalty", [])),
        "target_x_ref_penalty_max": _safe_nanmax(bundle.get("target_x_ref_penalty", [])),
        "target_xs_x_ref_inf_mean": _safe_nanmean(bundle.get("target_xs_x_ref_inf", [])),
        "target_xs_x_ref_inf_max": _safe_nanmax(bundle.get("target_xs_x_ref_inf", [])),
        "target_x_ref_active_steps": int(np.nansum(bundle.get("target_x_ref_active_flags", []))),
        "target_cond_M_max": float(np.nanmax(bundle["target_cond_M"])) if bundle["target_cond_M"].size else None,
        "bounded_solution_used_steps": int(np.sum(np.nan_to_num(bundle["target_bounded_solution_used_flags"], nan=0.0) > 0.5)),
        "bounded_active_lower_count_max": float(np.nanmax(bundle["target_bounded_active_lower_count"])) if bundle["target_bounded_active_lower_count"].size else None,
        "bounded_active_upper_count_max": float(np.nanmax(bundle["target_bounded_active_upper_count"])) if bundle["target_bounded_active_upper_count"].size else None,
    }
    summary["method_counts"] = {name: methods.count(name) for name in sorted(set(methods))}
    summary["target_stage_counts"] = {name: target_stages.count(name) for name in sorted(set(target_stages))}
    summary["solver_status_counts"] = {name: solver_statuses.count(name) for name in sorted(set(solver_statuses))}
    return summary


def _safe_nanmean(values: Any) -> Optional[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _safe_nanmax(values: Any) -> Optional[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _physical_setpoint_steps(bundle) -> np.ndarray:
    y_sp = np.asarray(bundle.get("y_sp_steps", bundle.get("y_sp")), dtype=float)
    if y_sp.ndim == 1:
        y_sp = y_sp.reshape(1, -1)

    y_plot, _ = _output_deviation_steps_to_plot_units(bundle, y_sp)
    return y_plot


def _output_deviation_steps_to_plot_units(bundle, values) -> Tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    steady_states = bundle.get("steady_states")
    data_min = bundle.get("data_min")
    data_max = bundle.get("data_max")
    if steady_states is None or data_min is None or data_max is None:
        return values.copy(), "scaled deviation"

    n_u = int(np.asarray(bundle["u_applied_phys"]).shape[1])
    y_ss_scaled = apply_min_max(
        np.asarray(steady_states["y_ss"], dtype=float).reshape(-1),
        np.asarray(data_min, dtype=float)[n_u:],
        np.asarray(data_max, dtype=float)[n_u:],
    )
    return (
        reverse_min_max(
            values + y_ss_scaled.reshape(1, -1),
            np.asarray(data_min, dtype=float)[n_u:],
            np.asarray(data_max, dtype=float)[n_u:],
        ),
        "physical units",
    )


def _bundle_output_plot_array(bundle, physical_key: str, deviation_key: str) -> Tuple[np.ndarray, str]:
    if physical_key in bundle:
        arr = np.asarray(bundle[physical_key], dtype=float)
        if arr.size and np.any(np.isfinite(arr)):
            return arr.copy(), "physical units"
    return _output_deviation_steps_to_plot_units(bundle, bundle[deviation_key])


def _plot_title_prefix(bundle) -> str:
    source = bundle.get("source")
    if source is None or str(source).strip() == "":
        return "Direct Lyapunov MPC"
    return f"Direct Lyapunov MPC - {source}"


def direct_output_rmse_post_step(bundle) -> np.ndarray:
    y_sp_phys = _physical_setpoint_steps(bundle)
    y_post = np.asarray(bundle["y_system"], dtype=float)[1 : 1 + y_sp_phys.shape[0], :]
    n_rows = min(y_post.shape[0], y_sp_phys.shape[0])
    n_cols = min(y_post.shape[1], y_sp_phys.shape[1])
    if n_rows <= 0 or n_cols <= 0:
        return np.array([], dtype=float)
    err = y_post[:n_rows, :n_cols] - y_sp_phys[:n_rows, :n_cols]
    return np.sqrt(np.mean(err**2, axis=0))


def _count_bool_step_values(step_info_storage, key: str, expected: bool) -> int:
    count = 0
    for info in step_info_storage:
        value = info.get(key)
        if value is None:
            continue
        if bool(value) is bool(expected):
            count += 1
    return int(count)


def _max_mask_count(step_info_storage, key: str) -> Optional[float]:
    counts = []
    for info in step_info_storage:
        value = info.get(key)
        if value is None:
            continue
        counts.append(float(np.sum(np.asarray(value, dtype=bool))))
    if not counts:
        return None
    return float(np.max(counts))


def make_direct_lyapunov_comparison_record(case_name, bundle, debug_dir=None):
    summary = dict(bundle.get("summary", {}))
    step_info_storage = list(bundle.get("direct_info_storage", []))
    rmse = direct_output_rmse_post_step(bundle)

    record = {
        "case_name": str(case_name),
        "target_mode": summary.get("target_mode", bundle.get("target_mode")),
        "lyapunov_mode": summary.get("lyapunov_mode", bundle.get("lyapunov_mode")),
        "plant_mode": summary.get("plant_mode", bundle.get("plant_mode")),
        "disturbance_after_step": summary.get("disturbance_after_step", bundle.get("disturbance_after_step")),
        "use_target_output_for_tracking": summary.get(
            "use_target_output_for_tracking", bundle.get("use_target_output_for_tracking")
        ),
        "nominal_qi": summary.get("nominal_qi", bundle.get("nominal_qi")),
        "nominal_qs": summary.get("nominal_qs", bundle.get("nominal_qs")),
        "nominal_ha": summary.get("nominal_ha", bundle.get("nominal_ha")),
        "final_qi": summary.get("final_qi", bundle.get("final_qi")),
        "final_qs": summary.get("final_qs", bundle.get("final_qs")),
        "final_ha": summary.get("final_ha", bundle.get("final_ha")),
        "n_steps": summary.get("n_steps", bundle.get("nFE")),
        "reward_mean": summary.get("reward_mean"),
        "reward_sum": summary.get("reward_sum"),
        "solver_success_rate": summary.get("solver_success_rate"),
        "target_success_rate": summary.get("target_success_rate"),
        "hard_contraction_rate": summary.get("hard_contraction_rate"),
        "relaxed_contraction_rate": summary.get("relaxed_contraction_rate"),
        "slack_lyap_mean": summary.get("slack_lyap_mean"),
        "slack_lyap_max": summary.get("slack_lyap_max"),
        "slack_lyap_active_steps": summary.get("slack_lyap_active_steps"),
        "target_residual_total_norm_max": summary.get("target_residual_total_norm_max"),
        "target_reference_error_inf_mean": summary.get("target_reference_error_inf_mean"),
        "target_reference_error_inf_max": summary.get("target_reference_error_inf_max"),
        "tracking_reference_error_inf_mean": summary.get("tracking_reference_error_inf_mean"),
        "tracking_reference_error_inf_max": summary.get("tracking_reference_error_inf_max"),
        "output_reference_error_inf_mean": summary.get("output_reference_error_inf_mean"),
        "output_reference_error_inf_max": summary.get("output_reference_error_inf_max"),
        "output_tracking_error_inf_mean": summary.get("output_tracking_error_inf_mean"),
        "output_tracking_error_inf_max": summary.get("output_tracking_error_inf_max"),
        "target_u_ref_penalty_mean": summary.get("target_u_ref_penalty_mean"),
        "target_u_ref_penalty_max": summary.get("target_u_ref_penalty_max"),
        "target_us_u_ref_inf_mean": summary.get("target_us_u_ref_inf_mean"),
        "target_us_u_ref_inf_max": summary.get("target_us_u_ref_inf_max"),
        "target_u_ref_active_steps": summary.get("target_u_ref_active_steps"),
        "target_x_ref_penalty_mean": summary.get("target_x_ref_penalty_mean"),
        "target_x_ref_penalty_max": summary.get("target_x_ref_penalty_max"),
        "target_xs_x_ref_inf_mean": summary.get("target_xs_x_ref_inf_mean"),
        "target_xs_x_ref_inf_max": summary.get("target_xs_x_ref_inf_max"),
        "target_x_ref_active_steps": summary.get("target_x_ref_active_steps"),
        "target_cond_M_max": summary.get("target_cond_M_max"),
        "bounded_solution_used_steps": summary.get("bounded_solution_used_steps"),
        "bounded_active_lower_count_max": summary.get("bounded_active_lower_count_max"),
        "bounded_active_upper_count_max": summary.get("bounded_active_upper_count_max"),
        "exact_target_within_bounds_steps": _count_bool_step_values(
            step_info_storage, "target_exact_within_bounds", True
        ),
        "exact_target_out_of_bounds_steps": _count_bool_step_values(
            step_info_storage, "target_exact_within_bounds", False
        ),
        "exact_active_lower_count_max": _max_mask_count(
            step_info_storage, "target_exact_active_lower_mask"
        ),
        "exact_active_upper_count_max": _max_mask_count(
            step_info_storage, "target_exact_active_upper_mask"
        ),
        "state_target_error_inf_mean": _safe_nanmean(bundle.get("state_target_error_inf", [])),
        "state_target_error_inf_max": _safe_nanmax(bundle.get("state_target_error_inf", [])),
        "disturbance_target_error_inf_mean": _safe_nanmean(
            bundle.get("disturbance_target_error_inf", [])
        ),
        "disturbance_target_error_inf_max": _safe_nanmax(
            bundle.get("disturbance_target_error_inf", [])
        ),
        "method_counts": json.dumps(_jsonable(summary.get("method_counts", {}))),
        "solver_status_counts": json.dumps(_jsonable(summary.get("solver_status_counts", {}))),
        "target_stage_counts": json.dumps(_jsonable(summary.get("target_stage_counts", {}))),
        "debug_dir": None if debug_dir is None else str(debug_dir),
    }
    for idx, value in enumerate(rmse):
        record[f"output{idx}_rmse"] = float(value)
    record["output_rmse_mean"] = _safe_nanmean(rmse)
    return record


def _comparison_plot_path(output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def _record_series(records, key: str) -> np.ndarray:
    out = []
    for record in records:
        value = record.get(key)
        if value is None:
            out.append(np.nan)
        else:
            try:
                out.append(float(value))
            except Exception:
                out.append(np.nan)
    return np.asarray(out, dtype=float)


def _save_comparison_bar(records, keys, labels, ylabel, title, path):
    x = np.arange(len(records))
    case_labels = [str(record["case_name"]) for record in records]
    width = min(0.8 / max(len(keys), 1), 0.35)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for idx, key in enumerate(keys):
        offset = (idx - (len(keys) - 1) / 2.0) * width
        ax.bar(x + offset, _record_series(records, key), width=width, label=labels[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_target_diagnostics_comparison(records, path):
    x = np.arange(len(records))
    case_labels = [str(record["case_name"]) for record in records]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    residual = _record_series(records, "target_residual_total_norm_max")
    axes[0].bar(x, residual, width=0.55, color="tab:purple", label="target residual max")
    finite_positive = residual[np.isfinite(residual) & (residual > 0.0)]
    if finite_positive.size:
        axes[0].set_yscale("log")
    axes[0].set_ylabel("residual")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(loc="best")

    keys = [
        "bounded_solution_used_steps",
        "bounded_active_lower_count_max",
        "bounded_active_upper_count_max",
    ]
    labels = ["bounded used steps", "active lower max", "active upper max"]
    width = 0.24
    for idx, key in enumerate(keys):
        axes[1].bar(
            x + (idx - 1) * width,
            _record_series(records, key),
            width=width,
            label=labels[idx],
        )
    axes[1].set_ylabel("count")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(loc="best")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(case_labels, rotation=25, ha="right")
    axes[0].set_title("Direct Lyapunov Four-Scenario Target Diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_comparison_overlay_plots(bundles_by_case, output_dir):
    if not bundles_by_case:
        return {}

    first_bundle = next(iter(bundles_by_case.values()))
    n_y = int(np.asarray(first_bundle["y_system"]).shape[1])
    n_u = int(np.asarray(first_bundle["u_applied_phys"]).shape[1])
    paths = {}

    fig, axes = plt.subplots(n_y, 1, figsize=(11, 3.2 * n_y), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(
        "Direct Lyapunov Four-Scenario Output Overlay (physical units)",
        fontsize=14,
        fontweight="bold",
    )
    for case_name, bundle in bundles_by_case.items():
        y_system = np.asarray(bundle["y_system"], dtype=float)
        time_y = np.arange(y_system.shape[0])
        for idx, ax in enumerate(axes):
            ax.plot(time_y, y_system[:, idx], linewidth=1.8, label=str(case_name))
    y_sp_phys = _physical_setpoint_steps(first_bundle)
    time_sp = np.arange(y_sp_phys.shape[0])
    for idx, ax in enumerate(axes):
        if idx < y_sp_phys.shape[1]:
            ax.step(
                time_sp,
                y_sp_phys[:, idx],
                where="post",
                color="black",
                linewidth=1.2,
                linestyle="--",
                label="setpoint",
            )
        ax.set_ylabel(f"y[{idx}]")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("step")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    paths["outputs_overlay"] = _comparison_plot_path(output_dir, "comparison_outputs_overlay.png")
    fig.savefig(paths["outputs_overlay"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_u, 1, figsize=(11, 3.0 * n_u), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(
        "Direct Lyapunov Four-Scenario Input Overlay (physical units)",
        fontsize=14,
        fontweight="bold",
    )
    for case_name, bundle in bundles_by_case.items():
        u_applied = np.asarray(bundle["u_applied_phys"], dtype=float)
        time_u = np.arange(u_applied.shape[0])
        for idx, ax in enumerate(axes):
            ax.step(time_u, u_applied[:, idx], where="post", linewidth=1.8, label=str(case_name))
    u_bounds = first_bundle.get("u_bounds_phys")
    for idx, ax in enumerate(axes):
        if u_bounds is not None:
            lower, upper = u_bounds
            ax.axhline(lower[idx], color="tab:red", linewidth=1.0, linestyle=":")
            ax.axhline(upper[idx], color="tab:brown", linewidth=1.0, linestyle=":")
        ax.set_ylabel(f"u[{idx}]")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("step")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    paths["inputs_overlay"] = _comparison_plot_path(output_dir, "comparison_inputs_overlay.png")
    fig.savefig(paths["inputs_overlay"], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def save_direct_lyapunov_comparison_artifacts(
    records,
    bundles_by_case,
    study_root,
    *,
    save_plots=True,
):
    os.makedirs(study_root, exist_ok=True)
    records = [dict(record) for record in records]

    comparison_csv = os.path.join(study_root, "comparison_table.csv")
    _write_csv(comparison_csv, records)

    comparison_pkl = os.path.join(study_root, "comparison_table.pkl")
    if HAS_PANDAS:
        pd.DataFrame(records).to_pickle(comparison_pkl)
    else:
        with open(comparison_pkl, "wb") as f:
            pickle.dump(records, f)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(records),
        "case_names": [str(record.get("case_name")) for record in records],
        "comparison_table_csv": comparison_csv,
        "comparison_table_pkl": comparison_pkl,
        "case_debug_dirs": {
            str(record.get("case_name")): record.get("debug_dir")
            for record in records
        },
    }
    comparison_summary_json = os.path.join(study_root, "comparison_summary.json")
    with open(comparison_summary_json, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)

    plot_paths = {}
    if save_plots:
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required to save direct comparison plots.")
        plot_dir = os.path.join(study_root, "comparison_plots")
        plot_paths["reward_mean"] = _save_comparison_bar(
            records,
            ["reward_mean"],
            ["reward mean"],
            "reward_mean",
            "Direct Lyapunov Four-Scenario Reward",
            _comparison_plot_path(plot_dir, "comparison_reward_mean.png"),
        )
        output_rmse_keys = [
            key for key in records[0].keys()
            if key.startswith("output") and key.endswith("_rmse")
        ] if records else []
        if output_rmse_keys:
            _save_comparison_bar(
                records,
                output_rmse_keys,
                output_rmse_keys,
                "RMSE (physical units)",
                "Direct Lyapunov Four-Scenario Output RMSE",
                _comparison_plot_path(plot_dir, "comparison_output_rmse.png"),
            )
            plot_paths["output_rmse"] = os.path.join(plot_dir, "comparison_output_rmse.png")
        plot_paths["solver_contraction_rates"] = _save_comparison_bar(
            records,
            ["solver_success_rate", "hard_contraction_rate", "relaxed_contraction_rate"],
            ["solver success", "hard contraction", "relaxed contraction"],
            "rate",
            "Direct Lyapunov Four-Scenario Solver And Contraction Rates",
            _comparison_plot_path(plot_dir, "comparison_solver_contraction_rates.png"),
        )
        plot_paths["slack"] = _save_comparison_bar(
            records,
            ["slack_lyap_mean", "slack_lyap_max"],
            ["slack mean", "slack max"],
            "slack",
            "Direct Lyapunov Four-Scenario Lyapunov Slack",
            _comparison_plot_path(plot_dir, "comparison_slack.png"),
        )
        plot_paths["target_residual_bounded_activity"] = _save_target_diagnostics_comparison(
            records,
            _comparison_plot_path(plot_dir, "comparison_target_residual_bounded_activity.png"),
        )
        plot_paths["reference_errors"] = _save_comparison_bar(
            records,
            [
                "target_reference_error_inf_mean",
                "output_reference_error_inf_mean",
                "output_tracking_error_inf_mean",
            ],
            ["mean |y_s-y_sp|_inf", "mean |y-y_sp|_inf", "mean |y-y_target|_inf"],
            "scaled-deviation infinity norm",
            "Direct Lyapunov Four-Scenario Reference Errors",
            _comparison_plot_path(plot_dir, "comparison_reference_errors.png"),
        )
        plot_paths.update(_save_comparison_overlay_plots(bundles_by_case, plot_dir))

    summary["plot_paths"] = plot_paths
    with open(comparison_summary_json, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)
    figure_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "recommended_figures": [
            {"key": key, "path": path}
            for key, path in plot_paths.items()
            if path is not None and os.path.exists(path)
        ],
        "case_debug_dirs": summary["case_debug_dirs"],
    }
    with open(os.path.join(study_root, "figure_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(figure_manifest), f, indent=2)
    return {
        "comparison_table_csv": comparison_csv,
        "comparison_table_pkl": comparison_pkl,
        "comparison_summary_json": comparison_summary_json,
        "plot_paths": plot_paths,
    }


def build_direct_lyapunov_run_bundle(
    source,
    results,
    *,
    steady_states=None,
    config=None,
    data_min=None,
    data_max=None,
    extra=None,
):
    y_system = np.asarray(results["y_system"], dtype=float)
    u_applied_phys = np.asarray(results["u_applied_phys"], dtype=float)
    rewards = np.asarray(results["rewards"], dtype=float)
    xhatdhat = np.asarray(results["xhatdhat"], dtype=float)
    yhat = np.asarray(results["yhat"], dtype=float)
    y_sp_steps = _normalize_step_matrix(results["y_sp"], int(results["nFE"]), y_system.shape[1])
    direct_info_storage = list(results["direct_info_storage"])
    nFE = int(results["nFE"])
    n_u = u_applied_phys.shape[1]
    n_y = y_system.shape[1]
    n_x = xhatdhat.shape[0] - n_y

    bundle = {
        "source": str(source),
        "config": {} if config is None else dict(config),
        "steady_states": steady_states,
        "data_min": None if data_min is None else np.asarray(data_min, dtype=float).copy(),
        "data_max": None if data_max is None else np.asarray(data_max, dtype=float).copy(),
        "extra": {} if extra is None else dict(extra),
        "nFE": nFE,
        "time_in_sub_episodes": int(results["time_in_sub_episodes"]),
        "avg_rewards": list(results["avg_rewards"]),
        "rewards": rewards.copy(),
        "y_system": y_system.copy(),
        "u_applied_phys": u_applied_phys.copy(),
        "xhatdhat": xhatdhat.copy(),
        "y_sp": y_sp_steps.copy(),
        "y_sp_steps": y_sp_steps.copy(),
        "yhat": yhat.copy(),
        "qi": np.asarray(results["qi"], dtype=float).copy(),
        "qs": np.asarray(results["qs"], dtype=float).copy(),
        "ha": np.asarray(results["ha"], dtype=float).copy(),
        "direct_info_storage": direct_info_storage,
        "target_info_storage": list(results["target_info_storage"]),
        "target_mode": results.get("target_mode"),
        "lyapunov_mode": results.get("lyapunov_mode"),
        "plant_mode": results.get("plant_mode"),
        "disturbance_after_step": results.get("disturbance_after_step"),
        "use_target_output_for_tracking": results.get("use_target_output_for_tracking"),
        "nominal_qi": results.get("nominal_qi"),
        "nominal_qs": results.get("nominal_qs"),
        "nominal_ha": results.get("nominal_ha"),
        "final_qi": results.get("final_qi"),
        "final_qs": results.get("final_qs"),
        "final_ha": results.get("final_ha"),
        "rho_lyap": results.get("rho_lyap"),
        "lyap_eps": results.get("lyap_eps"),
        "delta_t": float(results.get("delta_t", 1.0)),
        "slack_penalty": results.get("slack_penalty"),
        "first_step_contraction_on": results.get("first_step_contraction_on"),
        "x_target_store": _stack_vectors(direct_info_storage, "x_s", n_x),
        "d_target_store": _stack_vectors(direct_info_storage, "d_s", n_y),
        "u_target_dev_store": _stack_vectors(direct_info_storage, "u_s", n_u),
        "target_u_ref_store": _stack_vectors(direct_info_storage, "target_u_ref", n_u),
        "target_u_ref_weight_store": _stack_vectors(direct_info_storage, "target_u_ref_weight", n_u),
        "target_x_ref_store": _stack_vectors(direct_info_storage, "target_x_ref", n_x),
        "target_x_ref_weight_store": _stack_vectors(direct_info_storage, "target_x_ref_weight", n_x),
        "y_target_store": _stack_vectors(direct_info_storage, "y_s", n_y),
        "y_tracking_store": _stack_vectors(direct_info_storage, "y_target", n_y),
        "y_s_minus_y_sp_store": _stack_vectors(direct_info_storage, "y_s_minus_y_sp", n_y),
        "y_target_minus_y_sp_store": _stack_vectors(direct_info_storage, "y_target_minus_y_sp", n_y),
        "y_minus_y_sp_store": _stack_vectors(direct_info_storage, "y_minus_y_sp", n_y),
        "y_minus_y_target_store": _stack_vectors(direct_info_storage, "y_minus_y_target", n_y),
        "u_prev_dev_store": _stack_vectors(direct_info_storage, "u_prev_dev", n_u),
        "u_apply_dev_store": _stack_vectors(direct_info_storage, "u_apply", n_u),
        "V_k": np.array([info.get("V_k", np.nan) for info in direct_info_storage], dtype=float),
        "V_next_first": np.array([info.get("V_next_first", np.nan) for info in direct_info_storage], dtype=float),
        "V_bound": np.array([info.get("V_bound", np.nan) for info in direct_info_storage], dtype=float),
        "contraction_margin": np.array([info.get("contraction_margin", np.nan) for info in direct_info_storage], dtype=float),
        "slack_lyap": np.array([info.get("slack_lyap", np.nan) for info in direct_info_storage], dtype=float),
        "target_rank_M": np.array([info.get("target_rank_M", np.nan) for info in direct_info_storage], dtype=float),
        "target_cond_M": np.array([info.get("target_cond_M", np.nan) for info in direct_info_storage], dtype=float),
        "target_cond_G": np.array([info.get("target_cond_G", np.nan) for info in direct_info_storage], dtype=float),
        "target_residual_total_norm": np.array([info.get("target_residual_total_norm", np.nan) for info in direct_info_storage], dtype=float),
        "target_u_ref_penalty": np.array([info.get("target_u_ref_penalty", np.nan) for info in direct_info_storage], dtype=float),
        "target_us_u_ref_inf": np.array([info.get("target_us_u_ref_inf", np.nan) for info in direct_info_storage], dtype=float),
        "target_u_ref_active_flags": np.array([1.0 if bool(info.get("target_u_ref_active", False)) else 0.0 for info in direct_info_storage], dtype=float),
        "target_x_ref_penalty": np.array([info.get("target_x_ref_penalty", np.nan) for info in direct_info_storage], dtype=float),
        "target_xs_x_ref_inf": np.array([info.get("target_xs_x_ref_inf", np.nan) for info in direct_info_storage], dtype=float),
        "target_x_ref_active_flags": np.array([1.0 if bool(info.get("target_x_ref_active", False)) else 0.0 for info in direct_info_storage], dtype=float),
        "target_bounded_active_lower_count": np.array(
            [
                np.nan if info.get("target_bounded_active_lower_mask") is None
                else float(np.sum(np.asarray(info.get("target_bounded_active_lower_mask"), dtype=bool)))
                for info in direct_info_storage
            ],
            dtype=float,
        ),
        "target_bounded_active_upper_count": np.array(
            [
                np.nan if info.get("target_bounded_active_upper_mask") is None
                else float(np.sum(np.asarray(info.get("target_bounded_active_upper_mask"), dtype=bool)))
                for info in direct_info_storage
            ],
            dtype=float,
        ),
        "solver_success_flags": np.array([1.0 if bool(info.get("success", False)) else 0.0 for info in direct_info_storage], dtype=float),
        "target_success_flags": np.array([1.0 if bool(info.get("target_success", False)) else 0.0 for info in direct_info_storage], dtype=float),
        "first_step_contraction_satisfied_flags": np.array([1.0 if bool(info.get("first_step_contraction_satisfied", False)) else 0.0 for info in direct_info_storage], dtype=float),
        "relaxed_contraction_satisfied_flags": np.array([1.0 if bool(info.get("relaxed_contraction_satisfied", False)) else 0.0 for info in direct_info_storage], dtype=float),
        "target_bounded_solution_used_flags": np.array([1.0 if bool(info.get("target_bounded_solution_used", False)) else 0.0 for info in direct_info_storage], dtype=float),
    }

    bundle["state_target_error_inf"] = np.array(
        [
            np.nan if info.get("x_s") is None else float(np.max(np.abs(xhatdhat[:n_x, idx] - np.asarray(info["x_s"], dtype=float).reshape(-1))))
            for idx, info in enumerate(direct_info_storage)
        ],
        dtype=float,
    )
    bundle["disturbance_target_error_inf"] = np.array(
        [
            np.nan if info.get("d_s") is None else float(np.max(np.abs(xhatdhat[n_x:, idx] - np.asarray(info["d_s"], dtype=float).reshape(-1))))
            for idx, info in enumerate(direct_info_storage)
        ],
        dtype=float,
    )
    bundle["target_reference_error_inf"] = _row_inf_norms(bundle["y_s_minus_y_sp_store"])
    bundle["tracking_reference_error_inf"] = _row_inf_norms(bundle["y_target_minus_y_sp_store"])
    bundle["output_reference_error_inf"] = _row_inf_norms(bundle["y_minus_y_sp_store"])
    bundle["output_tracking_error_inf"] = _row_inf_norms(bundle["y_minus_y_target_store"])

    if steady_states is not None and data_min is not None and data_max is not None:
        ss_inputs = np.asarray(steady_states["ss_inputs"], dtype=float).reshape(-1)
        ss_scaled_inputs = apply_min_max(ss_inputs, data_min[:n_u], data_max[:n_u])
        y_scale = np.asarray(data_max, dtype=float)[n_u:] - np.asarray(data_min, dtype=float)[n_u:]
        y_ss_scaled = apply_min_max(
            np.asarray(steady_states["y_ss"], dtype=float).reshape(-1),
            data_min[n_u:],
            data_max[n_u:],
        )
        bundle["u_target_phys_store"] = reverse_min_max(
            bundle["u_target_dev_store"] + ss_scaled_inputs.reshape(1, -1),
            data_min[:n_u],
            data_max[:n_u],
        )
        bundle["target_u_ref_phys_store"] = reverse_min_max(
            bundle["target_u_ref_store"] + ss_scaled_inputs.reshape(1, -1),
            data_min[:n_u],
            data_max[:n_u],
        )
        bundle["y_target_phys_store"] = reverse_min_max(
            bundle["y_target_store"] + y_ss_scaled.reshape(1, -1),
            data_min[n_u:],
            data_max[n_u:],
        )
        bundle["y_tracking_phys_store"] = reverse_min_max(
            bundle["y_tracking_store"] + y_ss_scaled.reshape(1, -1),
            data_min[n_u:],
            data_max[n_u:],
        )
        bundle["y_s_minus_y_sp_phys_store"] = bundle["y_s_minus_y_sp_store"] * y_scale.reshape(1, -1)
        bundle["y_target_minus_y_sp_phys_store"] = bundle["y_target_minus_y_sp_store"] * y_scale.reshape(1, -1)
        bundle["y_minus_y_sp_phys_store"] = bundle["y_minus_y_sp_store"] * y_scale.reshape(1, -1)
        bundle["y_minus_y_target_phys_store"] = bundle["y_minus_y_target_store"] * y_scale.reshape(1, -1)
        if results.get("u_dev_min") is not None and results.get("u_dev_max") is not None:
            u_lower_dev = np.asarray(results["u_dev_min"], dtype=float).reshape(-1)
            u_upper_dev = np.asarray(results["u_dev_max"], dtype=float).reshape(-1)
            bundle["u_bounds_phys"] = (
                reverse_min_max(
                    u_lower_dev + ss_scaled_inputs,
                    data_min[:n_u],
                    data_max[:n_u],
                ),
                reverse_min_max(
                    u_upper_dev + ss_scaled_inputs,
                    data_min[:n_u],
                    data_max[:n_u],
                ),
            )
        else:
            bundle["u_bounds_phys"] = None
    else:
        bundle["u_target_phys_store"] = np.full_like(bundle["u_target_dev_store"], np.nan)
        bundle["y_target_phys_store"] = np.full_like(bundle["y_target_store"], np.nan)
        bundle["y_tracking_phys_store"] = np.full_like(bundle["y_tracking_store"], np.nan)
        bundle["y_s_minus_y_sp_phys_store"] = np.full_like(bundle["y_s_minus_y_sp_store"], np.nan)
        bundle["y_target_minus_y_sp_phys_store"] = np.full_like(bundle["y_target_minus_y_sp_store"], np.nan)
        bundle["y_minus_y_sp_phys_store"] = np.full_like(bundle["y_minus_y_sp_store"], np.nan)
        bundle["y_minus_y_target_phys_store"] = np.full_like(bundle["y_minus_y_target_store"], np.nan)
        bundle["u_bounds_phys"] = None

    bundle["summary"] = summarize_direct_lyapunov_bundle(bundle)
    return bundle


def _plot_ctx(paper_style: bool):
    return paper_plot_context() if paper_style else nullcontext()


def _save_existing_cstr_plot_views(bundle, output_dir, *, paper_style=False):
    steady_states = bundle.get("steady_states")
    data_min = bundle.get("data_min")
    data_max = bundle.get("data_max")
    if steady_states is None or data_min is None or data_max is None:
        return None
    return plot_mpc_results_cstr(
        y_sp=bundle["y_sp"],
        steady_states=steady_states,
        nFE=int(bundle["nFE"]),
        delta_t=float(bundle.get("delta_t", 1.0)),
        time_in_sub_episodes=int(bundle["time_in_sub_episodes"]),
        y_mpc=bundle["y_system"],
        u_mpc=bundle["u_applied_phys"],
        data_min=data_min,
        data_max=data_max,
        directory=output_dir,
        prefix_name="",
        y_target=bundle.get("y_target_store"),
        y_tracking_target=bundle.get("y_tracking_store"),
        u_target=bundle.get("u_target_phys_store"),
        u_bounds=bundle.get("u_bounds_phys"),
        timestamp_subdir=False,
        paper_style=paper_style,
    )


def plot_direct_lyapunov_bundle(bundle, output_dir, *, paper_style=False):
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required to plot direct Lyapunov diagnostics.")

    os.makedirs(output_dir, exist_ok=True)
    _save_existing_cstr_plot_views(bundle, output_dir, paper_style=paper_style)
    title_prefix = _plot_title_prefix(bundle)
    y_system = np.asarray(bundle["y_system"], dtype=float)
    y_sp_steps = np.asarray(bundle["y_sp_steps"], dtype=float)
    y_sp_plot, y_unit_label = _output_deviation_steps_to_plot_units(bundle, y_sp_steps)
    y_target_plot, _ = _bundle_output_plot_array(bundle, "y_target_phys_store", "y_target_store")
    y_tracking_plot, _ = _bundle_output_plot_array(bundle, "y_tracking_phys_store", "y_tracking_store")
    u_applied_phys = np.asarray(bundle["u_applied_phys"], dtype=float)
    u_target_phys_store = np.asarray(bundle["u_target_phys_store"], dtype=float)
    xhatdhat = np.asarray(bundle["xhatdhat"], dtype=float)
    x_target_store = np.asarray(bundle["x_target_store"], dtype=float)
    d_target_store = np.asarray(bundle["d_target_store"], dtype=float)
    V_k = np.asarray(bundle["V_k"], dtype=float)
    V_next_first = np.asarray(bundle["V_next_first"], dtype=float)
    V_bound = np.asarray(bundle["V_bound"], dtype=float)
    contraction_margin = np.asarray(bundle["contraction_margin"], dtype=float)
    slack_lyap = np.asarray(bundle["slack_lyap"], dtype=float)
    target_residual = np.asarray(bundle["target_residual_total_norm"], dtype=float)
    target_cond_M = np.asarray(bundle["target_cond_M"], dtype=float)
    target_rank_M = np.asarray(bundle["target_rank_M"], dtype=float)
    target_bounded_active_lower = np.asarray(bundle["target_bounded_active_lower_count"], dtype=float)
    target_bounded_active_upper = np.asarray(bundle["target_bounded_active_upper_count"], dtype=float)

    nFE = int(bundle["nFE"])
    n_y = y_system.shape[1]
    n_u = u_applied_phys.shape[1]
    n_x = x_target_store.shape[1]
    t_y = np.arange(nFE + 1)
    t_u = np.arange(nFE)

    with _plot_ctx(paper_style):
        fig, axes = plt.subplots(n_y, 1, figsize=(11, 3.2 * n_y), sharex=True)
        axes = np.atleast_1d(axes)
        fig.suptitle(
            f"{title_prefix}: Output Trajectories And Targets ({y_unit_label})",
            fontsize=14,
            fontweight="bold",
        )
        for idx, ax in enumerate(axes):
            ax.plot(t_y, y_system[:, idx], linewidth=2.0, color=PAPER_COLORS["output"], label="y")
            ax.step(t_u, y_sp_plot[:, idx], where="post", linewidth=1.8, linestyle="--", color=PAPER_COLORS["setpoint"], label="y_sp")
            ax.step(t_u, y_target_plot[:, idx], where="post", linewidth=1.5, linestyle="-.", color=PAPER_COLORS["target"], label="y_s")
            ax.step(t_u, y_tracking_plot[:, idx], where="post", linewidth=1.2, linestyle=":", label="stage target")
            ax.set_ylabel(f"y[{idx}]\n{y_unit_label}")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
        axes[-1].set_xlabel("step")
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        plt.savefig(os.path.join(output_dir, "01_outputs_vs_targets.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(n_u, 1, figsize=(11, 3.0 * n_u), sharex=True)
        axes = np.atleast_1d(axes)
        fig.suptitle(
            f"{title_prefix}: Applied Inputs And Steady Input Targets (physical units)",
            fontsize=14,
            fontweight="bold",
        )
        for idx, ax in enumerate(axes):
            ax.step(t_u, u_applied_phys[:, idx], where="post", linewidth=2.0, color=PAPER_COLORS.get(f"input_{idx}", "tab:blue"), label="u_applied")
            ax.step(t_u, u_target_phys_store[:, idx], where="post", linewidth=1.5, linestyle="--", color=PAPER_COLORS["target"], label="u_s")
            if bundle.get("u_bounds_phys") is not None:
                lower_bounds, upper_bounds = bundle["u_bounds_phys"]
                ax.axhline(lower_bounds[idx], color="tab:red", linewidth=1.0, linestyle=":")
                ax.axhline(upper_bounds[idx], color="tab:brown", linewidth=1.0, linestyle=":")
            ax.set_ylabel(f"u[{idx}]")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
        axes[-1].set_xlabel("step")
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        plt.savefig(os.path.join(output_dir, "02_inputs_vs_targets.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(max(n_x, 1), 1, figsize=(11, 2.6 * max(n_x, 1)), sharex=True)
        axes = np.atleast_1d(axes)
        fig.suptitle(
            f"{title_prefix}: State Estimate Minus Steady Target",
            fontsize=14,
            fontweight="bold",
        )
        xhat_phys = xhatdhat[:n_x, :-1]
        for idx, ax in enumerate(axes[:n_x]):
            err = xhat_phys[idx, :] - x_target_store[:, idx]
            ax.plot(t_u, err, linewidth=1.8, label="xhat - x_s")
            ax.set_ylabel(f"dx[{idx}]")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")
        axes[-1].set_xlabel("step")
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        plt.savefig(os.path.join(output_dir, "03_state_target_error.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        fig.suptitle(
            f"{title_prefix}: Lyapunov Contraction Diagnostics",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].plot(t_u, V_k, linewidth=1.8, label="V_k")
        axes[0].plot(t_u, V_next_first, linewidth=1.8, label="V_next_first")
        axes[0].plot(t_u, V_bound, linewidth=1.8, linestyle="--", label="V_bound")
        axes[0].legend(loc="best")
        axes[0].set_ylabel("V")
        axes[1].plot(t_u, contraction_margin, linewidth=1.8, label="contraction_margin")
        axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axes[1].legend(loc="best")
        axes[1].set_ylabel("margin")
        axes[2].plot(t_u, slack_lyap, linewidth=1.8, color="tab:red", label="slack_lyap")
        axes[2].legend(loc="best")
        axes[2].set_ylabel("slack")
        axes[3].step(t_u, bundle["first_step_contraction_satisfied_flags"], where="post", linewidth=1.8, label="hard_ok")
        axes[3].step(t_u, bundle["relaxed_contraction_satisfied_flags"], where="post", linewidth=1.4, linestyle="--", label="relaxed_ok")
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].set_ylabel("flags")
        axes[3].set_xlabel("step")
        axes[3].legend(loc="best")
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        plt.savefig(os.path.join(output_dir, "04_lyapunov_diagnostics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
        fig.suptitle(
            f"{title_prefix}: Target Selector Diagnostics",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].plot(t_u, target_residual, linewidth=1.8, label="target_residual_total_norm")
        axes[0].legend(loc="best")
        axes[0].set_ylabel("residual")
        axes[1].plot(t_u, target_cond_M, linewidth=1.8, label="cond_M")
        axes[1].legend(loc="best")
        axes[1].set_ylabel("cond")
        axes[2].plot(t_u, target_rank_M, linewidth=1.8, label="rank_M")
        axes[2].legend(loc="best")
        axes[2].set_ylabel("rank")
        axes[3].step(t_u, np.nan_to_num(target_bounded_active_lower, nan=0.0), where="post", linewidth=1.6, color="tab:red", label="active_lower_count")
        axes[3].step(t_u, np.nan_to_num(target_bounded_active_upper, nan=0.0), where="post", linewidth=1.6, color="tab:brown", linestyle="--", label="active_upper_count")
        axes[3].legend(loc="best")
        axes[3].set_ylabel("active")
        d_hat = xhatdhat[n_x:, :-1]
        for idx in range(d_hat.shape[0]):
            axes[4].plot(t_u, d_hat[idx, :], linewidth=1.4, color=PAPER_COLORS["disturbance"], label=f"d_hat[{idx}]")
            axes[4].plot(t_u, d_target_store[:, idx], linewidth=1.2, linestyle="--", color=PAPER_COLORS["target"], label=f"d_s[{idx}]")
        axes[4].legend(loc="best")
        axes[4].set_ylabel("disturbance")
        axes[4].set_xlabel("step")
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        plt.savefig(os.path.join(output_dir, "05_target_diagnostics.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        tail = min(max(25, nFE // 5), nFE)
        if tail > 0:
            start = nFE - tail
            fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
            fig.suptitle(
                f"{title_prefix}: Tail Window Summary",
                fontsize=14,
                fontweight="bold",
            )
            local_t = np.arange(start, nFE)
            for idx in range(n_y):
                axes[0].plot(np.arange(start, nFE + 1), y_system[start:nFE + 1, idx], linewidth=1.8, label=f"y[{idx}]")
                axes[0].step(local_t, y_sp_plot[start:nFE, idx], where="post", linewidth=1.2, linestyle="--", label=f"y_sp[{idx}]")
            axes[0].legend(loc="best")
            axes[0].set_ylabel("outputs")
            for idx in range(n_u):
                axes[1].step(local_t, u_applied_phys[start:nFE, idx], where="post", linewidth=1.8, label=f"u[{idx}]")
            axes[1].legend(loc="best")
            axes[1].set_ylabel("inputs")
            axes[2].plot(local_t, contraction_margin[start:nFE], linewidth=1.8, label="contraction_margin")
            axes[2].plot(local_t, slack_lyap[start:nFE], linewidth=1.6, linestyle="--", label="slack_lyap")
            axes[2].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
            axes[2].legend(loc="best")
            axes[2].set_ylabel("Lyapunov")
            axes[2].set_xlabel("step")
            for ax in axes:
                ax.grid(True, linestyle="--", alpha=0.35)
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            plt.savefig(os.path.join(output_dir, "06_tail_window_summary.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def save_direct_lyapunov_debug_artifacts(
    bundle,
    *,
    directory=None,
    prefix_name="direct_lyapunov_mpc",
    save_plots=True,
    save_paper_plots=True,
    timestamp_subdir=True,
):
    if directory is None:
        directory = os.getcwd()
    if timestamp_subdir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(directory, prefix_name, timestamp)
    else:
        out_dir = os.path.join(directory, prefix_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(bundle["summary"]), f, indent=2)

    summary_rows = [{"key": key, "value": json.dumps(_jsonable(value))} for key, value in bundle["summary"].items()]
    _write_csv(os.path.join(out_dir, "summary.csv"), summary_rows)
    step_records = make_direct_lyapunov_step_records(bundle["direct_info_storage"])
    _write_csv(os.path.join(out_dir, "step_table.csv"), step_records)
    _save_npz(os.path.join(out_dir, "arrays.npz"), bundle)

    step_table_pkl = os.path.join(out_dir, "step_table.pkl")
    if HAS_PANDAS:
        pd.DataFrame(step_records).to_pickle(step_table_pkl)
    else:
        with open(step_table_pkl, "wb") as f:
            pickle.dump(step_records, f)

    if save_plots:
        plot_direct_lyapunov_bundle(bundle, os.path.join(out_dir, "plots"), paper_style=False)
        if save_paper_plots:
            plot_direct_lyapunov_bundle(bundle, os.path.join(out_dir, "paper_plots"), paper_style=True)
    figure_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": bundle.get("source"),
        "recommended_figures": [
            {
                "key": "plots",
                "path": os.path.join(out_dir, "plots"),
                "description": "Scenario-level diagnostic plots.",
            },
            {
                "key": "paper_plots",
                "path": os.path.join(out_dir, "paper_plots"),
                "description": "Paper-style scenario plots.",
            },
        ],
    }
    with open(os.path.join(out_dir, "figure_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(figure_manifest), f, indent=2)
    return out_dir


def load_direct_lyapunov_debug_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
