from dataclasses import asdict, dataclass

import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

from utils.lyapunov_utils import DEFAULT_CVXPY_SOLVERS, diag_psd_from_vector


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
_DYN_TOL_BY_STATUS = {
    "optimal": 1e-6,
    "optimal_inaccurate": 1e-5,
}

REFINED_STEP_A_SELECTOR_NAME = "refined_step_a"
TARGET_SELECTOR_MODES = (REFINED_STEP_A_SELECTOR_NAME,)
_SELECTOR_TERM_NAMES = (
    "target_tracking",
    "u_applied_anchor",
    "u_prev_smoothing",
    "x_prev_smoothing",
    "xhat_anchor",
)


@dataclass
class TargetSelectorConfig:
    Qr_diag: object = None
    Rdu_diag: object = None
    R_u_ref_diag: object = None
    R_delta_u_sel_diag: object = None
    Q_delta_x_diag: object = None
    Q_x_ref_diag: object = None
    term_activation: object = None
    alpha_u_ref: float = 0.5
    alpha_du_sel: float = 0.5
    alpha_dx_sel: float = 0.05
    alpha_x_ref: float = 0.01
    x_weight_base: str = "CtQC"
    u_nom: object = None
    use_output_bounds_in_selector: bool = True
    u_tight: object = None
    y_tight: object = None
    solver_pref: object = None
    accept_statuses: tuple = ("optimal", "optimal_inaccurate")
    tol_optimal: float = 1e-6
    tol_optimal_inaccurate: float = 1e-5


def _solver_sequence(solver_pref):
    if solver_pref is None:
        return DEFAULT_CVXPY_SOLVERS
    if isinstance(solver_pref, str):
        return (solver_pref,)
    return tuple(solver_pref)


def _resolve_term_activation(cfg):
    spec = cfg.get("term_activation")
    if spec is None:
        return {name: True for name in _SELECTOR_TERM_NAMES}
    if not isinstance(spec, dict):
        raise ValueError("term_activation must be a dict mapping selector term names to booleans.")
    unknown = sorted(set(spec) - set(_SELECTOR_TERM_NAMES))
    if unknown:
        raise ValueError(
            "term_activation contains unsupported selector term names: "
            + ", ".join(unknown)
        )
    activation = {name: True for name in _SELECTOR_TERM_NAMES}
    for name, active in spec.items():
        activation[name] = bool(active)
    return activation


def _reset_variable_values(variables):
    for var in variables:
        if var is not None:
            var.value = None


def _apply_initial_values(variables, initial_values):
    if initial_values is None:
        return
    for var, init in zip(variables, initial_values):
        if var is None or init is None:
            continue
        var.value = np.asarray(init, float).copy()


def _solve_problem_with_preferences(problem, variables, solver_pref, warm_start=False, initial_values=None):
    last_status = None
    last_solver = None
    last_err = None

    for solver_name in _solver_sequence(solver_pref):
        try:
            _reset_variable_values(variables)
            _apply_initial_values(variables, initial_values)
            problem.solve(solver=solver_name, warm_start=bool(warm_start), verbose=False)
            last_status = problem.status
            last_solver = solver_name

            if any(var is not None and var.value is None for var in variables):
                continue
            if problem.status in _OPTIMAL_STATUSES:
                return {
                    "accepted_by_status": True,
                    "status": problem.status,
                    "solver": solver_name,
                    "error": None,
                    "objective_value": float(problem.value) if problem.value is not None else None,
                }
        except Exception as exc:
            last_err = repr(exc)

    return {
        "accepted_by_status": False,
        "status": last_status,
        "solver": last_solver,
        "error": last_err,
        "objective_value": float(problem.value) if problem.value is not None else None,
    }


def _bound_violation_inf(value, lower=None, upper=None):
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(np.max(np.maximum(lower - value, 0.0))))
    if upper is not None:
        violation = max(violation, float(np.max(np.maximum(value - upper, 0.0))))
    return violation


def _as_config_dict(config):
    if config is None:
        return {}
    if isinstance(config, TargetSelectorConfig):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("config must be a TargetSelectorConfig, dict, or None.")


def _diag_like(source, size, scale=1.0, default=1.0):
    if source is None:
        return np.full(size, float(scale * default), dtype=float)
    arr = np.asarray(source, float).reshape(-1)
    if arr.size == 1:
        arr = np.full(size, float(arr.item()), dtype=float)
    if arr.size != size:
        raise ValueError(f"Expected vector of size {size}, got {arr.size}.")
    return float(scale) * arr.copy()


def _coerce_selector_mode(selector_mode):
    if selector_mode is None:
        return None
    mode = str(selector_mode).strip()
    if mode in ("", REFINED_STEP_A_SELECTOR_NAME):
        return REFINED_STEP_A_SELECTOR_NAME
    if mode in {
        "current_exact_fallback_frozen_d",
        "free_disturbance_prior",
        "compromised_reference",
        "single_stage_robust_sstp",
    }:
        return REFINED_STEP_A_SELECTOR_NAME
    raise ValueError(
        f"Unsupported selector_mode '{selector_mode}'. Only '{REFINED_STEP_A_SELECTOR_NAME}' is active."
    )


def build_target_selector_config(
    selector_mode=None,
    user_overrides=None,
    n_x=None,
    n_u=None,
    n_y=None,
    n_d=None,
    Q_out=None,
    Rmove_diag=None,
):
    _coerce_selector_mode(selector_mode)
    if None in (n_x, n_u, n_y):
        raise ValueError("n_x, n_u, and n_y must be provided to build_target_selector_config.")
    if n_d is None:
        n_d = n_y

    defaults = {
        "Qr_diag": _diag_like(Q_out, n_y, scale=1.0, default=1.0),
        "Rdu_diag": _diag_like(Rmove_diag, n_u, scale=1.0, default=1.0),
        "R_u_ref_diag": None,
        "R_delta_u_sel_diag": None,
        "Q_delta_x_diag": None,
        "Q_x_ref_diag": None,
        "alpha_u_ref": 0.5,
        "alpha_du_sel": 0.5,
        "alpha_dx_sel": 0.05,
        "alpha_x_ref": 0.01,
        "x_weight_base": "CtQC",
        "u_nom": np.zeros(n_u, dtype=float),
        "use_output_bounds_in_selector": True,
        "u_tight": np.zeros(n_u, dtype=float),
        "y_tight": np.zeros(n_y, dtype=float),
        "solver_pref": DEFAULT_CVXPY_SOLVERS,
        "accept_statuses": ("optimal", "optimal_inaccurate"),
        "tol_optimal": 1e-6,
        "tol_optimal_inaccurate": 1e-5,
    }

    overrides = {} if user_overrides is None else dict(user_overrides)

    if "Ty_diag" in overrides and "Qr_diag" not in overrides:
        overrides["Qr_diag"] = overrides.pop("Ty_diag")
    if "Ru_diag" in overrides and "R_u_ref_diag" not in overrides:
        overrides["R_u_ref_diag"] = overrides.pop("Ru_diag")
    if "Qdx_diag" in overrides and "Q_delta_x_diag" not in overrides:
        overrides["Q_delta_x_diag"] = overrides.pop("Qdx_diag")
    if "Qx_diag" in overrides and "Q_x_ref_diag" not in overrides:
        overrides["Q_x_ref_diag"] = overrides.pop("Qx_diag")
    if "soft_output_bounds" in overrides and "use_output_bounds_in_selector" not in overrides:
        overrides["use_output_bounds_in_selector"] = overrides.pop("soft_output_bounds")
    overrides.pop("selector_mode", None)
    overrides.pop("Qd_diag", None)
    overrides.pop("rho_x", None)
    overrides.pop("rho_y", None)
    overrides.pop("delta_d_inf", None)
    overrides.pop("freeze_d_at_estimate", None)
    overrides.pop("Wy_low_diag", None)
    overrides.pop("Wy_high_diag", None)
    overrides.pop("w_x", None)

    merged = defaults
    merged.update(overrides)
    return TargetSelectorConfig(**merged)


def _extract_previous_target(prev_target=None, x_s_prev=None, u_s_prev=None):
    if x_s_prev is None and isinstance(prev_target, dict):
        x_s_prev = prev_target.get("x_s")
    if u_s_prev is None and isinstance(prev_target, dict):
        u_s_prev = prev_target.get("u_s")

    x_prev = None if x_s_prev is None else np.asarray(x_s_prev, float).reshape(-1)
    u_prev = None if u_s_prev is None else np.asarray(u_s_prev, float).reshape(-1)
    return x_prev, u_prev


def _resolve_tracking_weight(cfg, H_arr, n_y, n_r):
    qr_spec = cfg.get("Qr_diag")
    if qr_spec is None:
        qr_vals = np.ones(n_r, dtype=float)
        return np.diag(qr_vals), qr_vals

    qr_vals = np.asarray(qr_spec, float).reshape(-1)
    if qr_vals.size == 1:
        qr_vals = np.full(n_r, float(qr_vals.item()), dtype=float)
        return np.diag(np.maximum(qr_vals, 1e-12)), np.maximum(qr_vals, 1e-12)
    if qr_vals.size == n_r:
        return np.diag(np.maximum(qr_vals, 1e-12)), np.maximum(qr_vals, 1e-12)
    if H_arr is not None and qr_vals.size == n_y:
        q_full = np.diag(np.maximum(qr_vals, 1e-12))
        q_r = H_arr @ q_full @ H_arr.T
        q_r = 0.5 * (q_r + q_r.T)
        q_r += 1e-12 * np.eye(n_r)
        return q_r, np.diag(q_r).copy()
    raise ValueError(f"Qr_diag must have size 1, {n_r}, or {n_y} when H is provided.")


def _resolve_weight_matrices(C, H_arr, cfg, n_x, n_u, n_y, n_r):
    Qr, Qr_diag_used = _resolve_tracking_weight(cfg, H_arr, n_y, n_r)
    Rdu_base, Rdu_diag_used = diag_psd_from_vector(cfg.get("Rdu_diag"), n_u, eps=1e-12, default=1.0)

    if str(cfg.get("x_weight_base", "CtQC")).lower() == "identity":
        Qx_base = np.eye(n_x, dtype=float)
        x_weight_base_used = "identity"
    else:
        Cr = C if H_arr is None else H_arr @ C
        Qx_base = Cr.T @ Qr @ Cr
        Qx_base = 0.5 * (Qx_base + Qx_base.T)
        Qx_base += 1e-12 * np.eye(n_x)
        x_weight_base_used = "CtQC"

    if cfg.get("R_u_ref_diag") is not None:
        R_u_ref, R_u_ref_diag_used = diag_psd_from_vector(cfg.get("R_u_ref_diag"), n_u, eps=1e-12, default=1.0)
    else:
        R_u_ref = float(cfg.get("alpha_u_ref", 0.5)) * Rdu_base
        R_u_ref_diag_used = np.diag(R_u_ref).copy()

    if cfg.get("R_delta_u_sel_diag") is not None:
        R_delta_u_sel, R_delta_u_sel_diag_used = diag_psd_from_vector(cfg.get("R_delta_u_sel_diag"), n_u, eps=1e-12, default=1.0)
    else:
        R_delta_u_sel = float(cfg.get("alpha_du_sel", 0.5)) * Rdu_base
        R_delta_u_sel_diag_used = np.diag(R_delta_u_sel).copy()

    if cfg.get("Q_delta_x_diag") is not None:
        Q_delta_x, Q_delta_x_diag_used = diag_psd_from_vector(cfg.get("Q_delta_x_diag"), n_x, eps=1e-12, default=1.0)
    else:
        Q_delta_x = float(cfg.get("alpha_dx_sel", 0.05)) * Qx_base
        Q_delta_x_diag_used = np.diag(Q_delta_x).copy()

    if cfg.get("Q_x_ref_diag") is not None:
        Q_x_ref, Q_x_ref_diag_used = diag_psd_from_vector(cfg.get("Q_x_ref_diag"), n_x, eps=1e-12, default=1.0)
    else:
        Q_x_ref = float(cfg.get("alpha_x_ref", 0.01)) * Qx_base
        Q_x_ref_diag_used = np.diag(Q_x_ref).copy()

    return {
        "Qr": Qr,
        "Qr_diag_used": Qr_diag_used,
        "R_u_ref": R_u_ref,
        "R_u_ref_diag_used": R_u_ref_diag_used,
        "R_delta_u_sel": R_delta_u_sel,
        "R_delta_u_sel_diag_used": R_delta_u_sel_diag_used,
        "Qx_base": Qx_base,
        "Qx_base_diag_used": np.diag(Qx_base).copy(),
        "Q_delta_x": Q_delta_x,
        "Q_delta_x_diag_used": Q_delta_x_diag_used,
        "Q_x_ref": Q_x_ref,
        "Q_x_ref_diag_used": Q_x_ref_diag_used,
        "Rdu_diag_used": Rdu_diag_used,
        "x_weight_base_used": x_weight_base_used,
    }


def _finalize_target_info(
    *,
    success,
    status,
    solver_name,
    solver_error,
    objective_value,
    x_s,
    u_s,
    d_s,
    y_s,
    r_s,
    y_sp,
    A,
    B,
    Bd,
    C,
    Cd,
    xhat_k,
    x_s_prev,
    u_s_prev,
    u_applied_k,
    u_lo,
    u_hi,
    y_lo,
    y_hi,
    weights,
    cfg,
    term_activation,
    warm_start_enabled,
    warm_start_available,
    warm_start_used,
    objective_terms,
    status_tol,
):
    selector_debug = {
        "status": status,
        "solver": solver_name,
        "solver_error": solver_error,
        "objective_value": objective_value,
        "objective_terms": objective_terms,
        "warm_start_enabled": bool(warm_start_enabled),
        "warm_start_available": bool(warm_start_available),
        "warm_start_used": bool(warm_start_used),
        "prev_input_term_active": bool(u_s_prev is not None),
        "prev_state_term_active": bool(x_s_prev is not None),
        "use_output_bounds_in_selector": bool(cfg.get("use_output_bounds_in_selector", True)),
        "alpha_u_ref": float(cfg.get("alpha_u_ref", 0.5)),
        "alpha_du_sel": float(cfg.get("alpha_du_sel", 0.5)),
        "alpha_dx_sel": float(cfg.get("alpha_dx_sel", 0.05)),
        "alpha_x_ref": float(cfg.get("alpha_x_ref", 0.01)),
        "term_activation": dict(term_activation),
        "x_weight_base": str(cfg.get("x_weight_base", "CtQC")),
        "Qr_diag_used": weights["Qr_diag_used"].copy(),
        "R_u_ref_diag_used": weights["R_u_ref_diag_used"].copy(),
        "R_delta_u_sel_diag_used": weights["R_delta_u_sel_diag_used"].copy(),
        "Q_delta_x_diag_used": weights["Q_delta_x_diag_used"].copy(),
        "Q_x_ref_diag_used": weights["Q_x_ref_diag_used"].copy(),
        "Qx_base_diag_used": weights["Qx_base_diag_used"].copy(),
        "Rdu_diag_used": weights["Rdu_diag_used"].copy(),
        "status_tol": float(status_tol),
    }

    if not success:
        return {
            "success": False,
            "selector_mode": REFINED_STEP_A_SELECTOR_NAME,
            "selector_name": REFINED_STEP_A_SELECTOR_NAME,
            "solve_stage": "failed",
            "x_s": None,
            "u_s": None,
            "d_s": None,
            "x_s_aug": None,
            "y_s": None,
            "yc_s": None,
            "r_s": None,
            "requested_y_sp": y_sp.copy(),
            "objective_value": objective_value,
            "objective": objective_value,
            "target_error": None,
            "target_error_inf": None,
            "target_error_norm": None,
            "target_slack": None,
            "target_slack_inf": None,
            "target_slack_2": None,
            "target_eq_residual_inf": None,
            "dyn_residual_inf": None,
            "bound_violation_inf": None,
            "input_bound_violation_inf": None,
            "output_bound_violation_inf": None,
            "d_s_minus_dhat_inf": None,
            "d_s_frozen": True,
            "d_s_optimized": False,
            "objective_terms": objective_terms,
            "warm_start": {
                "enabled": bool(warm_start_enabled),
                "available": bool(warm_start_available),
                "used": bool(warm_start_used),
            },
            "status": status,
            "solver": solver_name,
            "selector_debug": selector_debug,
        }

    dyn_residual = x_s - (A @ x_s + B @ u_s + Bd @ d_s)
    dyn_residual_inf = float(np.max(np.abs(dyn_residual)))
    output_residual = y_s - (C @ x_s + Cd @ d_s)
    output_residual_inf = float(np.max(np.abs(output_residual)))
    input_bound_violation_inf = _bound_violation_inf(u_s, lower=u_lo, upper=u_hi)
    output_bound_violation_inf = 0.0
    if y_lo is not None or y_hi is not None:
        output_bound_violation_inf = _bound_violation_inf(y_s, lower=y_lo, upper=y_hi)
    bound_violation_inf = max(input_bound_violation_inf, output_bound_violation_inf)

    target_error = r_s - y_sp
    target_error_inf = float(np.max(np.abs(target_error)))
    target_error_norm = float(np.linalg.norm(target_error))

    target_info = {
        "success": True,
        "selector_mode": REFINED_STEP_A_SELECTOR_NAME,
        "selector_name": REFINED_STEP_A_SELECTOR_NAME,
        "solve_stage": REFINED_STEP_A_SELECTOR_NAME,
        "x_s": x_s.copy(),
        "u_s": u_s.copy(),
        "d_s": d_s.copy(),
        "x_s_aug": np.concatenate([x_s, d_s]).copy(),
        "y_s": y_s.copy(),
        "yc_s": r_s.copy(),
        "r_s": r_s.copy(),
        "requested_y_sp": y_sp.copy(),
        "objective_value": objective_value,
        "objective": objective_value,
        "target_error": target_error.copy(),
        "target_error_inf": target_error_inf,
        "target_error_norm": target_error_norm,
        "target_slack": target_error.copy(),
        "target_slack_inf": target_error_inf,
        "target_slack_2": target_error_norm,
        "target_eq_residual_inf": output_residual_inf,
        "dyn_residual_inf": dyn_residual_inf,
        "bound_violation_inf": bound_violation_inf,
        "input_bound_violation_inf": input_bound_violation_inf,
        "output_bound_violation_inf": output_bound_violation_inf,
        "d_s_minus_dhat_inf": 0.0,
        "d_s_frozen": True,
        "d_s_optimized": False,
        "objective_terms": objective_terms,
        "warm_start": {
            "enabled": bool(warm_start_enabled),
            "available": bool(warm_start_available),
            "used": bool(warm_start_used),
        },
        "status": status,
        "solver": solver_name,
        "selector_debug": selector_debug,
        "margin_to_u_min": (u_s - u_lo).copy(),
        "margin_to_u_max": (u_hi - u_s).copy(),
        "y_s_minus_y_sp": (y_s - y_sp[: y_s.size]).copy() if y_sp.size == y_s.size else None,
        "r_s_minus_y_sp": target_error.copy(),
        "u_s_minus_u_applied": (u_s - u_applied_k).copy(),
        "u_s_minus_u_prev": None if u_s_prev is None else (u_s - u_s_prev).copy(),
        "x_s_minus_x_prev": None if x_s_prev is None else (x_s - x_s_prev).copy(),
        "x_s_minus_xhat": (x_s - xhat_k).copy(),
    }
    if y_lo is not None:
        target_info["margin_to_y_min"] = (y_s - y_lo).copy()
    if y_hi is not None:
        target_info["margin_to_y_max"] = (y_hi - y_s).copy()
    return target_info


def compute_refined_step_a_target(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_applied_k=None,
    config=None,
    prev_target=None,
    x_s_prev=None,
    u_s_prev=None,
    y_min=None,
    y_max=None,
    warm_start=True,
    return_debug=False,
    H=None,
):
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for the refined Step A target selector.")

    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)
    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)

    if A_aug.ndim != 2 or A_aug.shape[0] != A_aug.shape[1]:
        raise ValueError("A_aug must be square.")
    if B_aug.ndim != 2 or B_aug.shape[0] != A_aug.shape[0]:
        raise ValueError("B_aug has incompatible shape.")
    if C_aug.ndim != 2 or C_aug.shape[1] != A_aug.shape[0]:
        raise ValueError("C_aug has incompatible shape.")
    if xhat_aug.size != A_aug.shape[0]:
        raise ValueError("xhat_aug has incorrect size.")

    n_aug = A_aug.shape[0]
    n_y = C_aug.shape[0]
    n_x = n_aug - n_y
    n_u = B_aug.shape[1]
    if n_x <= 0:
        raise ValueError("Invalid augmented model: inferred physical-state dimension must be positive.")
    if u_min.size != n_u or u_max.size != n_u:
        raise ValueError("u_min and u_max must have size n_u.")

    A = A_aug[:n_x, :n_x]
    Bd = A_aug[:n_x, n_x:]
    B = B_aug[:n_x, :]
    C = C_aug[:, :n_x]
    Cd = C_aug[:, n_x:]
    xhat_k = xhat_aug[:n_x].copy()
    d_hat = xhat_aug[n_x:].copy()

    if d_hat.size != n_y:
        raise ValueError("This selector assumes the augmented state is ordered as [x; d] with len(d) == n_y.")

    H_arr = None if H is None else np.asarray(H, float)
    if H_arr is not None and (H_arr.ndim != 2 or H_arr.shape[1] != n_y):
        raise ValueError("H must have shape (n_r, n_y).")
    n_r = n_y if H_arr is None else H_arr.shape[0]
    y_sp = np.asarray(y_sp, float).reshape(-1)
    if y_sp.size != n_r:
        raise ValueError(f"y_sp has incorrect size. Expected {n_r}, got {y_sp.size}.")

    cfg = build_target_selector_config(
        user_overrides=_as_config_dict(config),
        n_x=n_x,
        n_u=n_u,
        n_y=n_y,
        n_d=n_y,
        Q_out=np.ones(n_y, dtype=float),
        Rmove_diag=np.ones(n_u, dtype=float),
    )
    cfg_dict = asdict(cfg)

    x_s_prev_arr, u_s_prev_arr = _extract_previous_target(prev_target=prev_target, x_s_prev=x_s_prev, u_s_prev=u_s_prev)
    if x_s_prev_arr is not None and x_s_prev_arr.size != n_x:
        raise ValueError("x_s_prev has incorrect size.")
    if u_s_prev_arr is not None and u_s_prev_arr.size != n_u:
        raise ValueError("u_s_prev has incorrect size.")

    if u_applied_k is None:
        u_nom = cfg_dict.get("u_nom")
        if u_nom is None:
            u_applied_k = np.zeros(n_u, dtype=float)
        else:
            u_applied_k = np.asarray(u_nom, float).reshape(-1)
    else:
        u_applied_k = np.asarray(u_applied_k, float).reshape(-1)
    if u_applied_k.size != n_u:
        raise ValueError("u_applied_k has incorrect size.")

    u_tight = np.maximum(np.asarray(cfg_dict.get("u_tight"), float).reshape(-1), 0.0)
    if u_tight.size == 1 and n_u > 1:
        u_tight = np.full(n_u, float(u_tight.item()), dtype=float)
    if u_tight.size != n_u:
        raise ValueError("u_tight has incorrect size.")
    u_lo = u_min + u_tight
    u_hi = u_max - u_tight
    if np.any(u_lo > u_hi):
        raise ValueError("Input tightening is too large. Tightened bounds are infeasible.")

    use_output_bounds = bool(cfg_dict.get("use_output_bounds_in_selector", True))
    y_tight = np.asarray(cfg_dict.get("y_tight"), float).reshape(-1)
    if y_tight.size == 1 and n_y > 1:
        y_tight = np.full(n_y, float(y_tight.item()), dtype=float)
    if y_tight.size != n_y:
        raise ValueError("y_tight has incorrect size.")
    y_tight = np.maximum(y_tight, 0.0)

    y_lo = None
    y_hi = None
    if use_output_bounds and y_min is not None:
        y_min = np.asarray(y_min, float).reshape(-1)
        if y_min.size != n_y:
            raise ValueError("y_min has incorrect size.")
        y_lo = y_min + y_tight
    if use_output_bounds and y_max is not None:
        y_max = np.asarray(y_max, float).reshape(-1)
        if y_max.size != n_y:
            raise ValueError("y_max has incorrect size.")
        y_hi = y_max - y_tight
    if y_lo is not None and y_hi is not None and np.any(y_lo > y_hi):
        raise ValueError("Output tightening is too large. Tightened output bounds are infeasible.")

    weights = _resolve_weight_matrices(C, H_arr, cfg_dict, n_x=n_x, n_u=n_u, n_y=n_y, n_r=n_r)
    term_activation = _resolve_term_activation(cfg_dict)

    x_var = cp.Variable(n_x)
    u_var = cp.Variable(n_u)
    y_expr = C @ x_var + Cd @ d_hat
    r_expr = y_expr if H_arr is None else H_arr @ y_expr

    objective = 0
    if term_activation["target_tracking"]:
        objective += cp.quad_form(r_expr - y_sp, weights["Qr"])
    if term_activation["u_applied_anchor"]:
        objective += cp.quad_form(u_var - u_applied_k, weights["R_u_ref"])
    if u_s_prev_arr is not None:
        if term_activation["u_prev_smoothing"]:
            objective += cp.quad_form(u_var - u_s_prev_arr, weights["R_delta_u_sel"])
    if x_s_prev_arr is not None:
        if term_activation["x_prev_smoothing"]:
            objective += cp.quad_form(x_var - x_s_prev_arr, weights["Q_delta_x"])
    if term_activation["xhat_anchor"]:
        objective += cp.quad_form(x_var - xhat_k, weights["Q_x_ref"])

    constraints = [
        x_var == A @ x_var + B @ u_var + Bd @ d_hat,
        u_var >= u_lo,
        u_var <= u_hi,
    ]
    if y_lo is not None:
        constraints.append(y_expr >= y_lo)
    if y_hi is not None:
        constraints.append(y_expr <= y_hi)

    problem = cp.Problem(cp.Minimize(objective), constraints)

    warm_start_enabled = bool(warm_start)
    warm_start_available = bool(x_s_prev_arr is not None or u_s_prev_arr is not None)
    warm_start_used = bool(warm_start_enabled and warm_start_available)
    solve_info = _solve_problem_with_preferences(
        problem=problem,
        variables=[x_var, u_var],
        solver_pref=cfg_dict.get("solver_pref"),
        warm_start=warm_start_used,
        initial_values=[x_s_prev_arr, u_s_prev_arr],
    )

    status = solve_info.get("status")
    status_tol = _DYN_TOL_BY_STATUS.get(status, 1e-5)
    success = bool(solve_info.get("accepted_by_status", False) and x_var.value is not None and u_var.value is not None)

    x_s = None
    u_s = None
    y_s = None
    r_s = None
    if success:
        x_s = np.asarray(x_var.value, float).reshape(-1)
        u_s = np.asarray(u_var.value, float).reshape(-1)
        y_s = np.asarray(C @ x_s + Cd @ d_hat, float).reshape(-1)
        r_s = y_s.copy() if H_arr is None else np.asarray(H_arr @ y_s, float).reshape(-1)

        dyn_residual = x_s - (A @ x_s + B @ u_s + Bd @ d_hat)
        dyn_residual_inf = float(np.max(np.abs(dyn_residual)))
        bound_violation_inf = _bound_violation_inf(u_s, lower=u_lo, upper=u_hi)
        if y_lo is not None or y_hi is not None:
            bound_violation_inf = max(bound_violation_inf, _bound_violation_inf(y_s, lower=y_lo, upper=y_hi))
        if dyn_residual_inf > status_tol or bound_violation_inf > status_tol:
            success = False

    if success:
        objective_terms = {
            "target_tracking": 0.0 if not term_activation["target_tracking"] else float((r_s - y_sp).T @ weights["Qr"] @ (r_s - y_sp)),
            "u_applied_anchor": 0.0 if not term_activation["u_applied_anchor"] else float((u_s - u_applied_k).T @ weights["R_u_ref"] @ (u_s - u_applied_k)),
            "u_prev_smoothing": 0.0 if (u_s_prev_arr is None or not term_activation["u_prev_smoothing"]) else float((u_s - u_s_prev_arr).T @ weights["R_delta_u_sel"] @ (u_s - u_s_prev_arr)),
            "x_prev_smoothing": 0.0 if (x_s_prev_arr is None or not term_activation["x_prev_smoothing"]) else float((x_s - x_s_prev_arr).T @ weights["Q_delta_x"] @ (x_s - x_s_prev_arr)),
            "xhat_anchor": 0.0 if not term_activation["xhat_anchor"] else float((x_s - xhat_k).T @ weights["Q_x_ref"] @ (x_s - xhat_k)),
        }
        objective_terms["total"] = float(sum(objective_terms.values()))
    else:
        objective_terms = {
            "target_tracking": 0.0 if not term_activation["target_tracking"] else None,
            "u_applied_anchor": 0.0 if not term_activation["u_applied_anchor"] else None,
            "u_prev_smoothing": 0.0 if (u_s_prev_arr is None or not term_activation["u_prev_smoothing"]) else None,
            "x_prev_smoothing": 0.0 if (x_s_prev_arr is None or not term_activation["x_prev_smoothing"]) else None,
            "xhat_anchor": 0.0 if not term_activation["xhat_anchor"] else None,
            "total": solve_info.get("objective_value"),
        }

    target_info = _finalize_target_info(
        success=success,
        status=status,
        solver_name=solve_info.get("solver"),
        solver_error=solve_info.get("error"),
        objective_value=solve_info.get("objective_value"),
        x_s=x_s,
        u_s=u_s,
        d_s=d_hat,
        y_s=y_s,
        r_s=r_s,
        y_sp=y_sp,
        A=A,
        B=B,
        Bd=Bd,
        C=C,
        Cd=Cd,
        xhat_k=xhat_k,
        x_s_prev=x_s_prev_arr,
        u_s_prev=u_s_prev_arr,
        u_applied_k=u_applied_k,
        u_lo=u_lo,
        u_hi=u_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        weights=weights,
        cfg=cfg_dict,
        term_activation=term_activation,
        warm_start_enabled=warm_start_enabled,
        warm_start_available=warm_start_available,
        warm_start_used=warm_start_used,
        objective_terms=objective_terms,
        status_tol=status_tol,
    )

    if return_debug:
        return target_info, dict(target_info.get("selector_debug", {}))
    return target_info


def compute_ss_target_refined_rawlings(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    Ty_diag=None,
    Ru_diag=None,
    Qx_diag=None,
    w_x=1e-6,
    x_s_prev=None,
    u_s_prev=None,
    Qdx_diag=None,
    Rdu_diag=None,
    y_min=None,
    y_max=None,
    u_tight=None,
    y_tight=None,
    soft_output_bounds=True,
    Wy_low_diag=None,
    Wy_high_diag=None,
    solver_pref=DEFAULT_CVXPY_SOLVERS,
    warm_start=True,
    return_debug=False,
    H=None,
    u_applied_k=None,
):
    config = build_target_selector_config(
        user_overrides={
            "Qr_diag": Ty_diag,
            "R_u_ref_diag": Ru_diag,
            "Q_x_ref_diag": Qx_diag if Qx_diag is not None else w_x,
            "Q_delta_x_diag": Qdx_diag,
            "Rdu_diag": Rdu_diag,
            "u_nom": u_nom,
            "use_output_bounds_in_selector": bool(soft_output_bounds),
            "u_tight": u_tight,
            "y_tight": y_tight,
            "solver_pref": solver_pref,
        },
        n_x=A_aug.shape[0] - C_aug.shape[0],
        n_u=B_aug.shape[1],
        n_y=C_aug.shape[0],
        n_d=C_aug.shape[0],
        Q_out=Ty_diag if Ty_diag is not None else np.ones(C_aug.shape[0], dtype=float),
        Rmove_diag=Rdu_diag if Rdu_diag is not None else np.ones(B_aug.shape[1], dtype=float),
    )
    target_info = prepare_filter_target(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        config=config,
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
        y_min=y_min,
        y_max=y_max,
        warm_start=warm_start,
        H=H,
        u_applied_k=u_applied_k,
    )
    x_s = None if target_info.get("x_s") is None else np.asarray(target_info["x_s"], float).reshape(-1).copy()
    u_s = None if target_info.get("u_s") is None else np.asarray(target_info["u_s"], float).reshape(-1).copy()
    d_s = None if target_info.get("d_s") is None else np.asarray(target_info["d_s"], float).reshape(-1).copy()
    dbg = dict(target_info.get("selector_debug", {}))
    dbg.update({
        "success": bool(target_info.get("success", False)),
        "solve_stage": target_info.get("solve_stage"),
        "x_s": x_s,
        "u_s": u_s,
        "d_s": d_s,
        "y_s": None if target_info.get("y_s") is None else np.asarray(target_info["y_s"], float).reshape(-1).copy(),
        "r_s": None if target_info.get("r_s") is None else np.asarray(target_info["r_s"], float).reshape(-1).copy(),
        "target_error": None if target_info.get("target_error") is None else np.asarray(target_info["target_error"], float).reshape(-1).copy(),
        "target_error_inf": target_info.get("target_error_inf"),
        "target_error_norm": target_info.get("target_error_norm"),
        "target_slack_inf": target_info.get("target_slack_inf"),
        "target_eq_residual_inf": target_info.get("target_eq_residual_inf"),
        "dyn_residual_inf": target_info.get("dyn_residual_inf"),
        "bound_violation_inf": target_info.get("bound_violation_inf"),
        "objective_value": target_info.get("objective_value"),
        "objective_terms": target_info.get("objective_terms"),
        "status": target_info.get("status"),
        "solver": target_info.get("solver"),
    })
    if return_debug:
        return x_s, u_s, d_s, dbg
    return x_s, u_s, d_s


def prepare_filter_target_from_refined_selector(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    Ty_diag=None,
    Ru_diag=None,
    Qx_diag=None,
    w_x=1e-6,
    prev_target=None,
    x_s_prev=None,
    u_s_prev=None,
    Qdx_diag=None,
    Rdu_diag=None,
    y_min=None,
    y_max=None,
    u_tight=None,
    y_tight=None,
    soft_output_bounds=True,
    Wy_low_diag=None,
    Wy_high_diag=None,
    solver_pref=DEFAULT_CVXPY_SOLVERS,
    warm_start=True,
    return_debug=False,
    H=None,
    u_applied_k=None,
):
    target_info = prepare_filter_target(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        config=build_target_selector_config(
            user_overrides={
                "Qr_diag": Ty_diag,
                "R_u_ref_diag": Ru_diag,
                "Q_x_ref_diag": Qx_diag if Qx_diag is not None else w_x,
                "Q_delta_x_diag": Qdx_diag,
                "Rdu_diag": Rdu_diag,
                "u_nom": u_nom,
                "use_output_bounds_in_selector": bool(soft_output_bounds),
                "u_tight": u_tight,
                "y_tight": y_tight,
                "solver_pref": solver_pref,
            },
            n_x=A_aug.shape[0] - C_aug.shape[0],
            n_u=B_aug.shape[1],
            n_y=C_aug.shape[0],
            n_d=C_aug.shape[0],
            Q_out=Ty_diag if Ty_diag is not None else np.ones(C_aug.shape[0], dtype=float),
            Rmove_diag=Rdu_diag if Rdu_diag is not None else np.ones(B_aug.shape[1], dtype=float),
        ),
        prev_target=prev_target,
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
        y_min=y_min,
        y_max=y_max,
        warm_start=warm_start,
        H=H,
        u_applied_k=u_applied_k,
        selector_mode=REFINED_STEP_A_SELECTOR_NAME,
    )
    if return_debug:
        return target_info, dict(target_info.get("selector_debug", {}))
    return target_info


def prepare_filter_target(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    config=None,
    prev_target=None,
    x_s_prev=None,
    u_s_prev=None,
    y_min=None,
    y_max=None,
    warm_start=True,
    return_debug=False,
    H=None,
    selector_mode=None,
    u_applied_k=None,
):
    _coerce_selector_mode(selector_mode)
    return compute_refined_step_a_target(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        u_applied_k=u_applied_k,
        config=config,
        prev_target=prev_target,
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
        y_min=y_min,
        y_max=y_max,
        warm_start=warm_start,
        return_debug=return_debug,
        H=H,
    )
