from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    cp = None
    HAS_CVXPY = False

try:
    from utils.lyapunov_utils import DEFAULT_CVXPY_SOLVERS
except Exception:
    DEFAULT_CVXPY_SOLVERS = ("CLARABEL", "ECOS", "OSQP", "SCS")


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


@dataclass
class CertifiedDisturbanceConfig:
    alpha_d: float
    alpha_d_slow: float
    d_rate_max: np.ndarray
    d_min: np.ndarray
    d_max: np.ndarray
    innovation_gate: float | None = None
    innovation_norm: str = "inf"
    freeze_on_bad_innovation: bool = False
    adaptive_rate_enabled: bool = False
    adaptive_rate_trust_radius: Any = None
    adaptive_rate_min_scale: float = 0.10
    adaptive_rate_eps: float = 1.0e-8


@dataclass
class GARTTargetState:
    d_cert: np.ndarray | None = None
    x_s: np.ndarray | None = None
    u_s: np.ndarray | None = None
    y_s: np.ndarray | None = None
    r_cmd: np.ndarray | None = None
    valid: bool = False
    last_success_step: int = -1


@dataclass
class GARTTargetConfig:
    disturbance: CertifiedDisturbanceConfig

    input_headroom_frac: float = 0.03
    output_headroom_frac: float = 0.0
    alpha_terminal_min: float = 1.0e-8

    dy_s_max: np.ndarray | None = None
    du_s_max: np.ndarray | None = None
    dx_s_max: np.ndarray | None = None
    y_min: np.ndarray | None = None
    y_max: np.ndarray | None = None

    primary_tol_abs: float = 1.0e-8
    primary_tol_rel: float = 1.0e-6

    Wy_diag: np.ndarray | None = None
    W_u_smooth_diag: np.ndarray | None = None
    W_x_smooth_diag: np.ndarray | None = None
    W_y_smooth_diag: np.ndarray | None = None
    W_u_mid_diag: np.ndarray | None = None

    rho: float = 0.99
    eps: float = 1.0e-4
    contraction_margin_tol: float = 1.0e-8
    require_contraction_probe: bool = True
    contraction_probe_log_only: bool = False

    governor_enabled: bool = True
    governor_grid: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    governor_bisect_iters: int = 8

    target_exact_tol: float = 1.0e-6
    target_good_tol: float = 0.1
    target_acceptable_tol: float = 0.5

    margin_candidate_search_enabled: bool = False
    margin_candidate_search_step_frac: float = 0.01

    solver_pref: tuple[str, ...] | None = None


@dataclass
class GARTTargetResult:
    # Backward-compatible high-level result. After the correctness patch,
    # success means the target is accepted and usable for LMPC.
    success: bool
    solve_success: bool
    accepted: bool
    usable_for_lmpc: bool
    rejection_reason: str | None

    status: str
    stage: str

    x_s: np.ndarray | None
    u_s: np.ndarray | None
    y_s: np.ndarray | None
    d_cert: np.ndarray | None
    d_raw: np.ndarray | None
    r_cmd: np.ndarray | None

    target_error: np.ndarray | None
    target_error_inf: float | None
    primary_cost: float | None
    tiebreak_cost: float | None

    governor_alpha: float | None
    governor_active: bool
    hold_previous: bool

    terminal_alpha_min_feasible: bool
    contraction_probe_success: bool | None
    contraction_probe_margin_good: float | None
    # Compatibility alias for older logging paths. Positive is also good here.
    contraction_probe_margin: float | None
    contraction_probe_min_value: float | None
    contraction_probe_bound: float | None

    target_rate_y_inf: float | None
    target_rate_u_inf: float | None
    target_rate_x_inf: float | None

    input_headroom_min: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self.__dict__)


@dataclass
class GARTModel:
    A_aug: np.ndarray
    B_aug: np.ndarray
    C_aug: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Cd: np.ndarray
    n_x: int
    n_y: int
    n_u: int


@dataclass
class GARTStageResult:
    success: bool
    stage: str
    status: str | None
    solver: str | None
    x_s: np.ndarray | None
    u_s: np.ndarray | None
    y_s: np.ndarray | None
    primary_cost: float | None
    tiebreak_cost: float | None = None
    objective_value: float | None = None
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "__dict__") and value.__class__.__module__ == __name__:
        return jsonable(value.__dict__)
    return value


def _as_float_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}.")
    return arr


def _as_vector(value: Any, name: str, size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have length {size}, got {arr.size}.")
    return arr


def _optional_vector(value: Any, size: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _as_vector(value, name, size)


def _diag_vector(value: Any, size: int, *, default: float) -> np.ndarray:
    if value is None:
        return np.full(size, float(default), dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.full(size, float(default), dtype=float)
    if arr.size == 1:
        return np.full(size, float(arr.item()), dtype=float)
    if arr.size != size:
        raise ValueError(f"Expected scalar or vector length {size}, got {arr.size}.")
    return arr.copy()


def _inf_norm(value: Any) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _solver_sequence(solver_pref: Any) -> tuple[Any, ...]:
    if solver_pref is None:
        return tuple(DEFAULT_CVXPY_SOLVERS)
    if isinstance(solver_pref, str):
        return (solver_pref,)
    return tuple(solver_pref)


def _solve_problem(problem: Any, variables: Iterable[Any], solver_pref: Any, *, warm_start: bool = True) -> dict[str, Any]:
    last_status = None
    last_solver = None
    last_error = None
    last_value = None
    for solver_name in _solver_sequence(solver_pref):
        try:
            for var in variables:
                var.value = None
            problem.solve(solver=solver_name, warm_start=warm_start, verbose=False)
            last_status = problem.status
            last_solver = solver_name
            last_value = problem.value
            if problem.status in _OPTIMAL_STATUSES and all(var.value is not None for var in variables):
                return {
                    "success": True,
                    "status": problem.status,
                    "solver": solver_name,
                    "error": None,
                    "objective_value": None if problem.value is None else float(problem.value),
                }
        except Exception as exc:
            last_error = repr(exc)
    return {
        "success": False,
        "status": last_status,
        "solver": last_solver,
        "error": last_error,
        "objective_value": None if last_value is None else float(last_value),
    }


def split_output_disturbance_model(A_aug: Any, B_aug: Any, C_aug: Any, n_y: int) -> GARTModel:
    """Return A, B, C, Cd and dimensions for an [x; d] output-disturbance model."""
    A_aug = _as_float_array(A_aug, "A_aug", ndim=2)
    B_aug = _as_float_array(B_aug, "B_aug", ndim=2)
    C_aug = _as_float_array(C_aug, "C_aug", ndim=2)
    n_y = int(n_y)
    if A_aug.shape[0] != A_aug.shape[1]:
        raise ValueError("A_aug must be square.")
    if B_aug.shape[0] != A_aug.shape[0]:
        raise ValueError("B_aug row dimension must match A_aug.")
    if C_aug.shape[1] != A_aug.shape[0]:
        raise ValueError("C_aug column dimension must match A_aug.")
    if C_aug.shape[0] != n_y:
        raise ValueError("n_y must match C_aug row dimension.")

    n_aug = int(A_aug.shape[0])
    n_x = n_aug - n_y
    if n_x <= 0:
        raise ValueError("Expected augmented state ordered as [x; d] with n_aug > n_y.")

    A = np.asarray(A_aug[:n_x, :n_x], dtype=float)
    B = np.asarray(B_aug[:n_x, :], dtype=float)
    C = np.asarray(C_aug[:, :n_x], dtype=float)
    Cd = np.asarray(C_aug[:, n_x:], dtype=float)

    if Cd.shape != (n_y, n_y) or not np.allclose(Cd, np.eye(n_y), atol=1.0e-8, rtol=1.0e-8):
        raise ValueError("GART v0 supports only output-disturbance models with C_d approximately I.")

    A_xd = np.asarray(A_aug[:n_x, n_x:], dtype=float)
    A_dx = np.asarray(A_aug[n_x:, :n_x], dtype=float)
    A_dd = np.asarray(A_aug[n_x:, n_x:], dtype=float)
    B_d = np.asarray(B_aug[n_x:, :], dtype=float)
    if not np.allclose(A_xd, 0.0, atol=1.0e-8, rtol=1.0e-8):
        raise ValueError("GART v0 expects no disturbance-to-state dynamics in A_aug[:n_x,n_x:].")
    if not np.allclose(A_dx, 0.0, atol=1.0e-8, rtol=1.0e-8):
        raise ValueError("GART v0 expects no state-to-disturbance dynamics in A_aug[n_x:,:n_x].")
    if not np.allclose(A_dd, np.eye(n_y), atol=1.0e-8, rtol=1.0e-8):
        raise ValueError("GART v0 expects integrator disturbance dynamics in A_aug[n_x:,n_x:].")
    if not np.allclose(B_d, 0.0, atol=1.0e-8, rtol=1.0e-8):
        raise ValueError("GART v0 expects B_d = 0 for output-disturbance states.")

    return GARTModel(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        A=A,
        B=B,
        C=C,
        Cd=Cd,
        n_x=n_x,
        n_y=n_y,
        n_u=int(B_aug.shape[1]),
    )


def _innovation_norm(innovation: np.ndarray | None, kind: str) -> float | None:
    if innovation is None:
        return None
    arr = np.asarray(innovation, dtype=float).reshape(-1)
    kind = str(kind or "inf").strip().lower()
    if kind in {"inf", "linf", "max"}:
        return _inf_norm(arr)
    if kind in {"2", "l2", "euclidean"}:
        return float(np.linalg.norm(arr))
    raise ValueError(f"Unsupported innovation_norm={kind!r}.")


def update_certified_disturbance(
    d_cert_prev: np.ndarray | None,
    d_raw: np.ndarray,
    *,
    config: CertifiedDisturbanceConfig,
    innovation: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    d_raw = _as_vector(d_raw, "d_raw")
    d_rate_max = _as_vector(config.d_rate_max, "d_rate_max", d_raw.size)
    d_min = _as_vector(config.d_min, "d_min", d_raw.size)
    d_max = _as_vector(config.d_max, "d_max", d_raw.size)
    if np.any(d_rate_max < 0.0):
        raise ValueError("d_rate_max must be nonnegative.")
    if np.any(d_min > d_max):
        raise ValueError("d_min must be <= d_max.")

    d_raw_clipped = np.clip(d_raw, d_min, d_max)
    if d_cert_prev is None:
        d_prev = d_raw_clipped.copy()
    else:
        d_prev = np.clip(_as_vector(d_cert_prev, "d_cert_prev", d_raw.size), d_min, d_max)
    d_raw_gap = d_raw - d_prev

    norm_value = _innovation_norm(innovation, config.innovation_norm)
    gate_active = bool(
        config.innovation_gate is not None
        and norm_value is not None
        and norm_value > float(config.innovation_gate)
    )
    alpha = float(config.alpha_d_slow if gate_active else config.alpha_d)
    if gate_active and config.freeze_on_bad_innovation:
        alpha = 0.0

    adaptive_rate_enabled = bool(config.adaptive_rate_enabled)
    adaptive_rate_scale = np.ones_like(d_rate_max, dtype=float)
    if adaptive_rate_enabled:
        min_scale = float(config.adaptive_rate_min_scale)
        eps = float(config.adaptive_rate_eps)
        if not 0.0 <= min_scale <= 1.0:
            raise ValueError("adaptive_rate_min_scale must be in [0, 1].")
        if eps < 0.0:
            raise ValueError("adaptive_rate_eps must be nonnegative.")
        if config.adaptive_rate_trust_radius is None:
            trust_radius = d_rate_max.copy()
        else:
            trust_radius = _diag_vector(config.adaptive_rate_trust_radius, d_raw.size, default=0.0)
        if np.any(trust_radius < 0.0):
            raise ValueError("adaptive_rate_trust_radius must be nonnegative.")
        adaptive_rate_scale = np.clip(trust_radius / (np.abs(d_raw_gap) + eps), min_scale, 1.0)
    d_rate_max_effective = d_rate_max * adaptive_rate_scale
    delta_raw = alpha * (d_raw_clipped - d_prev)
    delta = np.clip(delta_raw, -d_rate_max_effective, d_rate_max_effective)
    d_cert = np.clip(d_prev + delta, d_min, d_max)
    actual_delta = d_cert - d_prev
    return d_cert, {
        "d_raw": d_raw.copy(),
        "d_raw_gap": d_raw_gap.copy(),
        "d_raw_gap_inf": _inf_norm(d_raw_gap),
        "d_cert_prev": None if d_cert_prev is None else np.asarray(d_cert_prev, dtype=float).reshape(-1).copy(),
        "d_cert": d_cert.copy(),
        "d_cert_delta": actual_delta.copy(),
        "d_cert_delta_inf": _inf_norm(actual_delta),
        "innovation_norm": norm_value,
        "innovation_gate_active": gate_active,
        "alpha_d_used": alpha,
        "d_raw_clipped": d_raw_clipped.copy(),
        "adaptive_rate_enabled": adaptive_rate_enabled,
        "adaptive_rate_scale": adaptive_rate_scale.copy(),
        "adaptive_rate_scale_min": None if adaptive_rate_scale.size == 0 else float(np.min(adaptive_rate_scale)),
        "adaptive_rate_scale_mean": None if adaptive_rate_scale.size == 0 else float(np.mean(adaptive_rate_scale)),
        "adaptive_rate_scale_max": None if adaptive_rate_scale.size == 0 else float(np.max(adaptive_rate_scale)),
        "d_rate_max_base": d_rate_max.copy(),
        "d_rate_max_base_inf": _inf_norm(d_rate_max),
        "d_rate_max_effective": d_rate_max_effective.copy(),
        "d_rate_max_effective_inf": _inf_norm(d_rate_max_effective),
    }


def terminal_input_tightening(P_x: Any, K_x: Any, alpha_min: float) -> np.ndarray:
    P_x = _as_float_array(P_x, "P_x", ndim=2)
    K_x = _as_float_array(K_x, "K_x", ndim=2)
    if P_x.shape[0] != P_x.shape[1] or K_x.shape[1] != P_x.shape[0]:
        raise ValueError("P_x must be square and K_x must have matching state columns.")
    alpha_min = max(float(alpha_min), 0.0)
    P_inv = np.linalg.pinv(P_x)
    g = np.array([K_x[j, :] @ P_inv @ K_x[j, :].T for j in range(K_x.shape[0])], dtype=float)
    return np.sqrt(np.maximum(alpha_min * g, 0.0))


def _headroom_and_tight_bounds(
    u_min: np.ndarray,
    u_max: np.ndarray,
    config: GARTTargetConfig,
    P_x: np.ndarray | None,
    K_x: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = u_max - u_min
    if np.any(width <= 0.0):
        raise ValueError("Input bounds must satisfy u_min < u_max.")
    h_u = max(float(config.input_headroom_frac), 0.0) * width
    if P_x is not None and K_x is not None and float(config.alpha_terminal_min) > 0.0:
        terminal = terminal_input_tightening(P_x, K_x, config.alpha_terminal_min)
    else:
        terminal = np.zeros_like(u_min)
    lo = u_min + h_u + terminal
    hi = u_max - h_u - terminal
    return lo, hi, h_u, terminal


def _output_bounds(config: GARTTargetConfig, n_y: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    y_min = _optional_vector(config.y_min, n_y, "y_min")
    y_max = _optional_vector(config.y_max, n_y, "y_max")
    if y_min is None or y_max is None:
        return y_min, y_max
    if np.any(y_min > y_max):
        raise ValueError("y_min must be <= y_max.")
    width = y_max - y_min
    h_y = max(float(config.output_headroom_frac), 0.0) * width
    return y_min + h_y, y_max - h_y


def _rate_constraints(
    x_var: Any,
    u_var: Any,
    y_expr: Any,
    prev_target: GARTTargetState | None,
    config: GARTTargetConfig,
) -> list[Any]:
    constraints: list[Any] = []
    if prev_target is None or not bool(prev_target.valid):
        return constraints
    if prev_target.y_s is not None and config.dy_s_max is not None:
        y_prev = _as_vector(prev_target.y_s, "prev_target.y_s")
        dy = _as_vector(config.dy_s_max, "dy_s_max", y_prev.size)
        constraints.extend([y_expr - y_prev <= dy, y_prev - y_expr <= dy])
    if prev_target.u_s is not None and config.du_s_max is not None:
        u_prev = _as_vector(prev_target.u_s, "prev_target.u_s")
        du = _as_vector(config.du_s_max, "du_s_max", u_prev.size)
        constraints.extend([u_var - u_prev <= du, u_prev - u_var <= du])
    if prev_target.x_s is not None and config.dx_s_max is not None:
        x_prev = _as_vector(prev_target.x_s, "prev_target.x_s")
        dx = _as_vector(config.dx_s_max, "dx_s_max", x_prev.size)
        constraints.extend([x_var - x_prev <= dx, x_prev - x_var <= dx])
    return constraints


def _target_rates(
    x_s: np.ndarray | None,
    u_s: np.ndarray | None,
    y_s: np.ndarray | None,
    prev_target: GARTTargetState | None,
) -> tuple[float | None, float | None, float | None]:
    if prev_target is None or not bool(prev_target.valid):
        return None, None, None
    rate_y = None
    rate_u = None
    rate_x = None
    if y_s is not None and prev_target.y_s is not None:
        rate_y = _inf_norm(np.asarray(y_s, dtype=float).reshape(-1) - np.asarray(prev_target.y_s, dtype=float).reshape(-1))
    if u_s is not None and prev_target.u_s is not None:
        rate_u = _inf_norm(np.asarray(u_s, dtype=float).reshape(-1) - np.asarray(prev_target.u_s, dtype=float).reshape(-1))
    if x_s is not None and prev_target.x_s is not None:
        rate_x = _inf_norm(np.asarray(x_s, dtype=float).reshape(-1) - np.asarray(prev_target.x_s, dtype=float).reshape(-1))
    return rate_y, rate_u, rate_x


def solve_stage1_closest_reachable(
    model: GARTModel,
    xhat: np.ndarray,
    d_cert: np.ndarray,
    reference: np.ndarray,
    bounds: dict[str, Any],
    prev_target: GARTTargetState | None,
    config: GARTTargetConfig,
) -> GARTStageResult:
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for GART target selection.")
    reference = _as_vector(reference, "reference", model.n_y)
    d_cert = _as_vector(d_cert, "d_cert", model.n_y)
    u_lo = _as_vector(bounds["u_lo"], "u_lo", model.n_u)
    u_hi = _as_vector(bounds["u_hi"], "u_hi", model.n_u)
    if np.any(u_lo > u_hi):
        return GARTStageResult(False, "stage1", "infeasible_tight_input_bounds", None, None, None, None, None)

    Wy = _diag_vector(config.Wy_diag, model.n_y, default=1.0)
    x_var = cp.Variable(model.n_x)
    u_var = cp.Variable(model.n_u)
    y_expr = model.C @ x_var + model.Cd @ d_cert
    constraints = [
        (np.eye(model.n_x) - model.A) @ x_var - model.B @ u_var == 0,
        u_var >= u_lo,
        u_var <= u_hi,
    ]
    y_lo, y_hi = _output_bounds(config, model.n_y)
    if y_lo is not None:
        constraints.append(y_expr >= y_lo)
    if y_hi is not None:
        constraints.append(y_expr <= y_hi)
    constraints.extend(_rate_constraints(x_var, u_var, y_expr, prev_target, config))
    residual = cp.multiply(Wy, y_expr - reference)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(residual)), constraints)
    solve_info = _solve_problem(problem, [x_var, u_var], config.solver_pref)
    if not solve_info["success"]:
        return GARTStageResult(
            False,
            "stage1",
            solve_info.get("status"),
            solve_info.get("solver"),
            None,
            None,
            None,
            None,
            error=solve_info.get("error"),
            diagnostics={"solve_info": solve_info},
        )
    x_s = np.asarray(x_var.value, dtype=float).reshape(model.n_x)
    u_s = np.asarray(u_var.value, dtype=float).reshape(model.n_u)
    y_s = np.asarray(model.C @ x_s + model.Cd @ d_cert, dtype=float).reshape(model.n_y)
    primary = float(np.sum((Wy * (y_s - reference)) ** 2))
    return GARTStageResult(
        True,
        "stage1",
        solve_info.get("status"),
        solve_info.get("solver"),
        x_s,
        u_s,
        y_s,
        primary,
        objective_value=solve_info.get("objective_value"),
        diagnostics={"solve_info": solve_info, "xhat_inf": _inf_norm(xhat)},
    )


def solve_stage2_tiebreak(
    model: GARTModel,
    d_cert: np.ndarray,
    reference: np.ndarray,
    stage1_result: GARTStageResult,
    bounds: dict[str, Any],
    prev_target: GARTTargetState | None,
    config: GARTTargetConfig,
    u_smooth_ref: np.ndarray | None = None,
) -> GARTStageResult:
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for GART target selection.")
    if not stage1_result.success or stage1_result.primary_cost is None:
        return GARTStageResult(False, "stage2", "stage1_failed", None, None, None, None, None)

    reference = _as_vector(reference, "reference", model.n_y)
    d_cert = _as_vector(d_cert, "d_cert", model.n_y)
    u_lo = _as_vector(bounds["u_lo"], "u_lo", model.n_u)
    u_hi = _as_vector(bounds["u_hi"], "u_hi", model.n_u)
    Wy = _diag_vector(config.Wy_diag, model.n_y, default=1.0)
    W_u = _diag_vector(config.W_u_smooth_diag, model.n_u, default=0.0)
    W_x = _diag_vector(config.W_x_smooth_diag, model.n_x, default=0.0)
    W_y = _diag_vector(config.W_y_smooth_diag, model.n_y, default=0.0)
    W_mid = _diag_vector(config.W_u_mid_diag, model.n_u, default=1.0e-6)

    x_var = cp.Variable(model.n_x)
    u_var = cp.Variable(model.n_u)
    y_expr = model.C @ x_var + model.Cd @ d_cert
    primary_expr = cp.sum_squares(cp.multiply(Wy, y_expr - reference))
    primary_shell = float(stage1_result.primary_cost) + float(config.primary_tol_abs) + float(config.primary_tol_rel) * max(
        1.0, float(stage1_result.primary_cost)
    )
    constraints = [
        (np.eye(model.n_x) - model.A) @ x_var - model.B @ u_var == 0,
        u_var >= u_lo,
        u_var <= u_hi,
        primary_expr <= primary_shell,
    ]
    y_lo, y_hi = _output_bounds(config, model.n_y)
    if y_lo is not None:
        constraints.append(y_expr >= y_lo)
    if y_hi is not None:
        constraints.append(y_expr <= y_hi)
    constraints.extend(_rate_constraints(x_var, u_var, y_expr, prev_target, config))

    u_mid = 0.5 * (u_lo + u_hi)
    objective = cp.sum_squares(cp.multiply(W_mid, u_var - u_mid))
    u_smooth_source = None
    if u_smooth_ref is not None:
        u_smooth_target = _as_vector(u_smooth_ref, "u_smooth_ref", model.n_u)
        u_smooth_source = "previous_applied_input"
        objective += cp.sum_squares(cp.multiply(W_u, u_var - u_smooth_target))
    else:
        u_smooth_target = None
    if prev_target is not None and bool(prev_target.valid):
        if u_smooth_target is None and prev_target.u_s is not None:
            objective += cp.sum_squares(cp.multiply(W_u, u_var - _as_vector(prev_target.u_s, "prev_target.u_s", model.n_u)))
            u_smooth_source = "previous_target_u_s"
        if prev_target.x_s is not None:
            objective += cp.sum_squares(cp.multiply(W_x, x_var - _as_vector(prev_target.x_s, "prev_target.x_s", model.n_x)))
        if prev_target.y_s is not None:
            objective += cp.sum_squares(cp.multiply(W_y, y_expr - _as_vector(prev_target.y_s, "prev_target.y_s", model.n_y)))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    solve_info = _solve_problem(problem, [x_var, u_var], config.solver_pref)
    if not solve_info["success"]:
        return GARTStageResult(
            False,
            "stage2",
            solve_info.get("status"),
            solve_info.get("solver"),
            None,
            None,
            None,
            None,
            error=solve_info.get("error"),
            diagnostics={"solve_info": solve_info, "primary_shell": primary_shell},
        )
    x_s = np.asarray(x_var.value, dtype=float).reshape(model.n_x)
    u_s = np.asarray(u_var.value, dtype=float).reshape(model.n_u)
    y_s = np.asarray(model.C @ x_s + model.Cd @ d_cert, dtype=float).reshape(model.n_y)
    primary = float(np.sum((Wy * (y_s - reference)) ** 2))
    return GARTStageResult(
        True,
        "stage2",
        solve_info.get("status"),
        solve_info.get("solver"),
        x_s,
        u_s,
        y_s,
        primary,
        tiebreak_cost=solve_info.get("objective_value"),
        objective_value=solve_info.get("objective_value"),
        diagnostics={"solve_info": solve_info, "primary_shell": primary_shell, "u_smooth_source": u_smooth_source},
    )


def contraction_probe(
    A: Any,
    B: Any,
    P_x: Any,
    xhat: Any,
    x_s: Any,
    u_min: Any,
    u_max: Any,
    *,
    rho: float,
    eps: float,
    solver_pref: Any = None,
) -> dict[str, Any]:
    if not HAS_CVXPY:
        return {
            "probe_success": False,
            "probe_margin_good": None,
            "probe_margin": None,
            "probe_min_value": None,
            "probe_bound": None,
            "probe_u": None,
            "V_k": None,
            "probe_status": "cvxpy_unavailable",
        }
    A = _as_float_array(A, "A", ndim=2)
    B = _as_float_array(B, "B", ndim=2)
    P_x = _as_float_array(P_x, "P_x", ndim=2)
    xhat = _as_vector(xhat, "xhat", A.shape[0])
    x_s = _as_vector(x_s, "x_s", A.shape[0])
    u_min = _as_vector(u_min, "u_min", B.shape[1])
    u_max = _as_vector(u_max, "u_max", B.shape[1])

    e_k = xhat - x_s
    V_k = float(e_k.T @ P_x @ e_k)
    V_bound = float(rho) * V_k + float(eps)
    u_var = cp.Variable(B.shape[1])
    e_next = A @ xhat + B @ u_var - x_s
    problem = cp.Problem(cp.Minimize(cp.quad_form(e_next, cp.psd_wrap(P_x))), [u_var >= u_min, u_var <= u_max])
    solve_info = _solve_problem(problem, [u_var], solver_pref)
    if not solve_info["success"]:
        return {
            "probe_success": False,
            "probe_margin_good": None,
            "probe_margin": None,
            "probe_min_value": None,
            "probe_bound": V_bound,
            "probe_u": None,
            "V_k": V_k,
            "probe_status": solve_info.get("status"),
            "probe_solver": solve_info.get("solver"),
            "probe_error": solve_info.get("error"),
        }
    u_star = np.asarray(u_var.value, dtype=float).reshape(B.shape[1])
    e_next_value = A @ xhat + B @ u_star - x_s
    V_min = float(e_next_value.T @ P_x @ e_next_value)
    margin_good = float(V_bound - V_min)
    return {
        "probe_success": True,
        "probe_margin_good": margin_good,
        "probe_margin": margin_good,
        "probe_min_value": V_min,
        "probe_bound": V_bound,
        "probe_u": u_star.copy(),
        "V_k": V_k,
        "probe_status": solve_info.get("status"),
        "probe_solver": solve_info.get("solver"),
    }


def _input_headroom_min(u_s: np.ndarray | None, u_min: np.ndarray, u_max: np.ndarray) -> float | None:
    if u_s is None:
        return None
    return float(min(np.min(u_s - u_min), np.min(u_max - u_s)))


def evaluate_target_acceptance(
    *,
    solve_success: bool,
    terminal_alpha_min_feasible: bool,
    contraction_probe_success: bool | None,
    config: GARTTargetConfig,
) -> tuple[bool, str | None]:
    if not solve_success:
        return False, "target_solve_failed"
    if not terminal_alpha_min_feasible:
        return False, "terminal_alpha_infeasible"
    if (
        config.require_contraction_probe
        and not bool(config.contraction_probe_log_only)
        and not bool(contraction_probe_success)
    ):
        return False, "contraction_probe_failed"
    return True, None


def _target_quality_flags(target_error_inf: float | None, config: GARTTargetConfig) -> dict[str, bool | None]:
    if target_error_inf is None:
        return {
            "target_exact": None,
            "target_good": None,
            "target_acceptable": None,
            "target_unreachable": None,
        }
    value = float(target_error_inf)
    return {
        "target_exact": bool(value <= float(config.target_exact_tol)),
        "target_good": bool(value <= float(config.target_good_tol)),
        "target_acceptable": bool(value <= float(config.target_acceptable_tol)),
        "target_unreachable": bool(value > float(config.target_acceptable_tol)),
    }


def _stage_probe(
    *,
    stage: GARTStageResult,
    model: GARTModel,
    xhat: np.ndarray,
    P_x: np.ndarray | None,
    u_min: np.ndarray,
    u_max: np.ndarray,
    config: GARTTargetConfig,
) -> dict[str, Any]:
    if not stage.success or stage.x_s is None or P_x is None:
        return {
            "probe_success": None,
            "probe_margin_good": None,
            "probe_min_value": None,
            "probe_bound": None,
        }
    probe = contraction_probe(
        model.A,
        model.B,
        P_x,
        xhat,
        stage.x_s,
        u_min,
        u_max,
        rho=config.rho,
        eps=config.eps,
        solver_pref=config.solver_pref,
    )
    margin_good = probe.get("probe_margin_good", probe.get("probe_margin"))
    success = probe.get("probe_success")
    if success is not None and margin_good is not None:
        success = bool(float(margin_good) >= -float(config.contraction_margin_tol))
    return {
        "probe_success": success,
        "probe_margin_good": margin_good,
        "probe_min_value": probe.get("probe_min_value"),
        "probe_bound": probe.get("probe_bound"),
        "probe_status": probe.get("probe_status"),
        "probe_solver": probe.get("probe_solver"),
    }


def _primary_shell(stage1: GARTStageResult, config: GARTTargetConfig) -> float | None:
    if stage1.primary_cost is None:
        return None
    return float(stage1.primary_cost) + float(config.primary_tol_abs) + float(config.primary_tol_rel) * max(
        1.0, float(stage1.primary_cost)
    )


def _numeric_rate_ok(
    *,
    x_s: np.ndarray,
    u_s: np.ndarray,
    y_s: np.ndarray,
    prev_target: GARTTargetState | None,
    config: GARTTargetConfig,
) -> bool:
    if prev_target is None or not bool(prev_target.valid):
        return True
    if prev_target.y_s is not None and config.dy_s_max is not None:
        dy = _as_vector(config.dy_s_max, "dy_s_max", y_s.size)
        if np.any(np.abs(y_s - _as_vector(prev_target.y_s, "prev_target.y_s", y_s.size)) > dy + 1.0e-10):
            return False
    if prev_target.u_s is not None and config.du_s_max is not None:
        du = _as_vector(config.du_s_max, "du_s_max", u_s.size)
        if np.any(np.abs(u_s - _as_vector(prev_target.u_s, "prev_target.u_s", u_s.size)) > du + 1.0e-10):
            return False
    if prev_target.x_s is not None and config.dx_s_max is not None:
        dx = _as_vector(config.dx_s_max, "dx_s_max", x_s.size)
        if np.any(np.abs(x_s - _as_vector(prev_target.x_s, "prev_target.x_s", x_s.size)) > dx + 1.0e-10):
            return False
    return True


def _margin_candidate_search(
    *,
    model: GARTModel,
    d_cert: np.ndarray,
    reference: np.ndarray,
    stage1: GARTStageResult,
    stage2: GARTStageResult,
    bounds: dict[str, Any],
    prev_target: GARTTargetState | None,
    config: GARTTargetConfig,
    xhat: np.ndarray,
    P_x: np.ndarray | None,
    u_min: np.ndarray,
    u_max: np.ndarray,
) -> GARTStageResult:
    if not bool(config.margin_candidate_search_enabled) or not stage2.success or stage2.u_s is None or P_x is None:
        return stage2
    shell = _primary_shell(stage1, config)
    if shell is None:
        return stage2
    try:
        u_lo = _as_vector(bounds["u_lo"], "u_lo", model.n_u)
        u_hi = _as_vector(bounds["u_hi"], "u_hi", model.n_u)
        reference = _as_vector(reference, "reference", model.n_y)
        d_cert = _as_vector(d_cert, "d_cert", model.n_y)
        Wy = _diag_vector(config.Wy_diag, model.n_y, default=1.0)
        step = max(float(config.margin_candidate_search_step_frac), 0.0) * np.maximum(u_hi - u_lo, 1.0e-8)
        if np.all(step <= 0.0):
            return stage2
        steady_matrix = np.eye(model.n_x) - model.A
        best = stage2
        best_probe = _stage_probe(stage=stage2, model=model, xhat=xhat, P_x=P_x, u_min=u_min, u_max=u_max, config=config)
        best_margin = best_probe.get("probe_margin_good")
        best_margin_value = -np.inf if best_margin is None else float(best_margin)
        deltas = np.array(np.meshgrid(*[[-s, 0.0, s] for s in step], indexing="ij"), dtype=float).reshape(model.n_u, -1).T
        for delta in deltas:
            u_candidate = np.clip(np.asarray(stage2.u_s, dtype=float).reshape(model.n_u) + delta, u_lo, u_hi)
            x_candidate = np.linalg.lstsq(steady_matrix, model.B @ u_candidate, rcond=None)[0].reshape(model.n_x)
            y_candidate = np.asarray(model.C @ x_candidate + model.Cd @ d_cert, dtype=float).reshape(model.n_y)
            primary = float(np.sum((Wy * (y_candidate - reference)) ** 2))
            if primary > shell + 1.0e-10:
                continue
            if not _numeric_rate_ok(x_s=x_candidate, u_s=u_candidate, y_s=y_candidate, prev_target=prev_target, config=config):
                continue
            probe = _stage_probe(
                stage=GARTStageResult(True, "margin_search", "candidate", None, x_candidate, u_candidate, y_candidate, primary),
                model=model,
                xhat=xhat,
                P_x=P_x,
                u_min=u_min,
                u_max=u_max,
                config=config,
            )
            margin = probe.get("probe_margin_good")
            margin_value = -np.inf if margin is None else float(margin)
            if margin_value > best_margin_value:
                best_margin_value = margin_value
                best = GARTStageResult(
                    True,
                    "margin_search",
                    "selected",
                    None,
                    x_candidate,
                    u_candidate,
                    y_candidate,
                    primary,
                    tiebreak_cost=stage2.tiebreak_cost,
                    objective_value=stage2.objective_value,
                    diagnostics={
                        "source_stage": stage2.stage,
                        "primary_shell": shell,
                        "probe": probe,
                    },
                )
        if best is not stage2:
            best.diagnostics["margin_search_improved_probe_margin_good"] = best_margin_value
            best.diagnostics["stage2_probe_margin_good"] = best_probe.get("probe_margin_good")
        return best
    except Exception as exc:
        stage2.diagnostics["margin_candidate_search_error"] = repr(exc)
        return stage2


def _result_from_candidate(
    *,
    candidate: GARTStageResult,
    model: GARTModel,
    xhat: np.ndarray,
    d_cert: np.ndarray,
    d_raw: np.ndarray,
    reference: np.ndarray,
    y_sp: np.ndarray,
    prev_target: GARTTargetState | None,
    config: GARTTargetConfig,
    P_x: np.ndarray | None,
    u_min: np.ndarray,
    u_max: np.ndarray,
    terminal_feasible: bool,
    governor_alpha: float | None,
    governor_active: bool,
    hold_previous: bool,
    status: str,
    extra_diagnostics: dict[str, Any] | None = None,
) -> GARTTargetResult:
    y_s = None if candidate.y_s is None else np.asarray(candidate.y_s, dtype=float).reshape(model.n_y)
    target_error = None if y_s is None else y_s - y_sp
    rate_y, rate_u, rate_x = _target_rates(candidate.x_s, candidate.u_s, y_s, prev_target)
    probe = {
        "probe_success": None,
        "probe_margin_good": None,
        "probe_margin": None,
        "probe_min_value": None,
        "probe_bound": None,
        "probe_u": None,
        "V_k": None,
    }
    if candidate.success and candidate.x_s is not None and P_x is not None:
        probe = contraction_probe(
            model.A,
            model.B,
            P_x,
            xhat,
            candidate.x_s,
            u_min,
            u_max,
            rho=config.rho,
            eps=config.eps,
            solver_pref=config.solver_pref,
        )
    probe_success = probe.get("probe_success")
    probe_margin_good = probe.get("probe_margin_good", probe.get("probe_margin"))
    if probe_success is not None and probe_margin_good is not None:
        probe_success = bool(float(probe_margin_good) >= -float(config.contraction_margin_tol))
        probe["probe_success"] = probe_success
    solve_success = bool(candidate.success)
    accepted, rejection_reason = evaluate_target_acceptance(
        solve_success=solve_success,
        terminal_alpha_min_feasible=bool(terminal_feasible),
        contraction_probe_success=probe_success,
        config=config,
    )
    target_error_inf = None if target_error is None else _inf_norm(target_error)
    target_flags = _target_quality_flags(target_error_inf, config)
    diagnostics = {
        "stage_status": candidate.status,
        "stage_solver": candidate.solver,
        "stage_error": candidate.error,
        "reference": reference.copy(),
        "target_stage_diagnostics": candidate.diagnostics,
        "contraction_probe": probe,
        "solve_success": solve_success,
        "accepted": accepted,
        "usable_for_lmpc": accepted,
        "rejection_reason": rejection_reason,
        **target_flags,
    }
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    return GARTTargetResult(
        success=accepted,
        solve_success=solve_success,
        accepted=accepted,
        usable_for_lmpc=accepted,
        rejection_reason=rejection_reason,
        status=status,
        stage=candidate.stage,
        x_s=None if candidate.x_s is None else candidate.x_s.copy(),
        u_s=None if candidate.u_s is None else candidate.u_s.copy(),
        y_s=None if y_s is None else y_s.copy(),
        d_cert=d_cert.copy(),
        d_raw=d_raw.copy(),
        r_cmd=reference.copy(),
        target_error=None if target_error is None else target_error.copy(),
        target_error_inf=target_error_inf,
        primary_cost=candidate.primary_cost,
        tiebreak_cost=candidate.tiebreak_cost,
        governor_alpha=governor_alpha,
        governor_active=bool(governor_active),
        hold_previous=bool(hold_previous),
        terminal_alpha_min_feasible=bool(terminal_feasible),
        contraction_probe_success=probe_success,
        contraction_probe_margin_good=probe_margin_good,
        contraction_probe_margin=probe_margin_good,
        contraction_probe_min_value=probe.get("probe_min_value"),
        contraction_probe_bound=probe.get("probe_bound"),
        target_rate_y_inf=rate_y,
        target_rate_u_inf=rate_u,
        target_rate_x_inf=rate_x,
        input_headroom_min=_input_headroom_min(candidate.u_s, u_min, u_max),
        diagnostics=diagnostics,
    )


def _candidate_accepted(result: GARTTargetResult, config: GARTTargetConfig) -> bool:
    return bool(result.accepted and result.usable_for_lmpc)


def _reference_motion_ok(y_sp: np.ndarray, prev_target: GARTTargetState | None, config: GARTTargetConfig) -> bool:
    if prev_target is None or not bool(prev_target.valid) or config.dy_s_max is None:
        return True
    r_prev = prev_target.r_cmd if prev_target.r_cmd is not None else prev_target.y_s
    if r_prev is None:
        return True
    dy = _as_vector(config.dy_s_max, "dy_s_max", y_sp.size)
    return bool(np.all(np.abs(y_sp - np.asarray(r_prev, dtype=float).reshape(y_sp.size)) <= dy + 1.0e-12))


def _solve_candidate(
    *,
    model: GARTModel,
    xhat: np.ndarray,
    d_cert: np.ndarray,
    d_raw: np.ndarray,
    reference: np.ndarray,
    y_sp: np.ndarray,
    prev_target: GARTTargetState | None,
    config: GARTTargetConfig,
    P_x: np.ndarray | None,
    K_x: np.ndarray | None,
    u_min: np.ndarray,
    u_max: np.ndarray,
    u_smooth_ref: np.ndarray | None,
    governor_alpha: float | None,
    governor_active: bool,
    hold_previous: bool,
    status: str,
    extra_diagnostics: dict[str, Any] | None = None,
) -> GARTTargetResult:
    try:
        u_lo, u_hi, h_u, terminal_tightening = _headroom_and_tight_bounds(u_min, u_max, config, P_x, K_x)
        terminal_feasible = bool(np.all(u_lo <= u_hi))
        bounds = {"u_lo": u_lo, "u_hi": u_hi}
        stage1 = solve_stage1_closest_reachable(model, xhat, d_cert, reference, bounds, prev_target, config)
        stage2 = (
            solve_stage2_tiebreak(model, d_cert, reference, stage1, bounds, prev_target, config, u_smooth_ref=u_smooth_ref)
            if stage1.success
            else stage1
        )
        stage1_probe = _stage_probe(stage=stage1, model=model, xhat=xhat, P_x=P_x, u_min=u_min, u_max=u_max, config=config)
        stage2_probe = _stage_probe(stage=stage2, model=model, xhat=xhat, P_x=P_x, u_min=u_min, u_max=u_max, config=config)
        stage2_minus_stage1 = None
        if stage1_probe.get("probe_margin_good") is not None and stage2_probe.get("probe_margin_good") is not None:
            stage2_minus_stage1 = float(stage2_probe["probe_margin_good"]) - float(stage1_probe["probe_margin_good"])
        candidate = stage2 if stage2.success else stage1
        if stage2.success:
            candidate = _margin_candidate_search(
                model=model,
                d_cert=d_cert,
                reference=reference,
                stage1=stage1,
                stage2=stage2,
                bounds=bounds,
                prev_target=prev_target,
                config=config,
                xhat=xhat,
                P_x=P_x,
                u_min=u_min,
                u_max=u_max,
            )
        diagnostics = {
            "u_tight_lower": u_lo.copy(),
            "u_tight_upper": u_hi.copy(),
            "input_headroom": h_u.copy(),
            "terminal_tightening": terminal_tightening.copy(),
            "stage1": stage1,
            "stage2": stage2,
            "stage1_probe": stage1_probe,
            "stage2_probe": stage2_probe,
            "stage1_probe_margin_good": stage1_probe.get("probe_margin_good"),
            "stage2_probe_margin_good": stage2_probe.get("probe_margin_good"),
            "stage2_minus_stage1_probe_margin_good": stage2_minus_stage1,
            "stage1_primary_cost": stage1.primary_cost,
            "stage2_primary_cost": stage2.primary_cost,
            "stage2_tiebreak_cost": stage2.tiebreak_cost,
            "stage2_u_smooth_source": stage2.diagnostics.get("u_smooth_source") if isinstance(stage2.diagnostics, dict) else None,
        }
        if extra_diagnostics:
            diagnostics.update(extra_diagnostics)
        return _result_from_candidate(
            candidate=candidate,
            model=model,
            xhat=xhat,
            d_cert=d_cert,
            d_raw=d_raw,
            reference=reference,
            y_sp=y_sp,
            prev_target=prev_target,
            config=config,
            P_x=P_x,
            u_min=u_min,
            u_max=u_max,
            terminal_feasible=terminal_feasible,
            governor_alpha=governor_alpha,
            governor_active=governor_active,
            hold_previous=hold_previous,
            status=status,
            extra_diagnostics=diagnostics,
        )
    except Exception as exc:
        return GARTTargetResult(
            success=False,
            solve_success=False,
            accepted=False,
            usable_for_lmpc=False,
            rejection_reason="target_solve_failed",
            status=f"{status}_failed",
            stage="exception",
            x_s=None,
            u_s=None,
            y_s=None,
            d_cert=d_cert.copy(),
            d_raw=d_raw.copy(),
            r_cmd=reference.copy(),
            target_error=None,
            target_error_inf=None,
            primary_cost=None,
            tiebreak_cost=None,
            governor_alpha=governor_alpha,
            governor_active=bool(governor_active),
            hold_previous=bool(hold_previous),
            terminal_alpha_min_feasible=False,
            contraction_probe_success=None,
            contraction_probe_margin_good=None,
            contraction_probe_margin=None,
            contraction_probe_min_value=None,
            contraction_probe_bound=None,
            target_rate_y_inf=None,
            target_rate_u_inf=None,
            target_rate_x_inf=None,
            input_headroom_min=None,
            diagnostics={"error": repr(exc)},
        )


def _hold_previous_result(
    *,
    model: GARTModel,
    xhat: np.ndarray,
    d_cert: np.ndarray,
    d_raw: np.ndarray,
    y_sp: np.ndarray,
    state: GARTTargetState,
    config: GARTTargetConfig,
    P_x: np.ndarray | None,
    u_min: np.ndarray,
    u_max: np.ndarray,
    disturbance_info: dict[str, Any],
) -> GARTTargetResult:
    candidate = GARTStageResult(
        success=bool(state.valid),
        stage="hold_previous",
        status="hold_previous",
        solver=None,
        x_s=None if state.x_s is None else np.asarray(state.x_s, dtype=float).reshape(model.n_x),
        u_s=None if state.u_s is None else np.asarray(state.u_s, dtype=float).reshape(model.n_u),
        y_s=None if state.y_s is None else np.asarray(state.y_s, dtype=float).reshape(model.n_y),
        primary_cost=None,
        tiebreak_cost=None,
        diagnostics={"held_previous_target": True},
    )
    return _result_from_candidate(
        candidate=candidate,
        model=model,
        xhat=xhat,
        d_cert=d_cert,
        d_raw=d_raw,
        reference=np.asarray(state.r_cmd if state.r_cmd is not None else state.y_s, dtype=float).reshape(model.n_y),
        y_sp=y_sp,
        prev_target=state,
        config=config,
        P_x=P_x,
        u_min=u_min,
        u_max=u_max,
        terminal_feasible=True,
        governor_alpha=0.0,
        governor_active=True,
        hold_previous=True,
        status="hold_previous",
        extra_diagnostics={"disturbance": disturbance_info},
    )


def _state_from_result(result: GARTTargetResult, prior: GARTTargetState | None) -> GARTTargetState:
    if result.accepted and result.x_s is not None and result.u_s is not None and result.y_s is not None:
        last_step = 0 if prior is None else int(prior.last_success_step) + 1
        return GARTTargetState(
            d_cert=None if result.d_cert is None else result.d_cert.copy(),
            x_s=result.x_s.copy(),
            u_s=result.u_s.copy(),
            y_s=result.y_s.copy(),
            r_cmd=None if result.r_cmd is None else result.r_cmd.copy(),
            valid=True,
            last_success_step=last_step,
        )
    if prior is not None:
        out = GARTTargetState(
            d_cert=None if result.d_cert is None else result.d_cert.copy(),
            x_s=None if prior.x_s is None else np.asarray(prior.x_s, dtype=float).copy(),
            u_s=None if prior.u_s is None else np.asarray(prior.u_s, dtype=float).copy(),
            y_s=None if prior.y_s is None else np.asarray(prior.y_s, dtype=float).copy(),
            r_cmd=None if prior.r_cmd is None else np.asarray(prior.r_cmd, dtype=float).copy(),
            valid=bool(prior.valid),
            last_success_step=int(prior.last_success_step),
        )
        return out
    return GARTTargetState(d_cert=None if result.d_cert is None else result.d_cert.copy(), valid=False)


def select_gart_target(
    A_aug: Any,
    B_aug: Any,
    C_aug: Any,
    xhat_aug_raw: Any,
    y_sp: Any,
    u_min: Any,
    u_max: Any,
    *,
    state: GARTTargetState | None,
    config: GARTTargetConfig,
    P_x: Any,
    K_x: Any,
    innovation: np.ndarray | None = None,
    u_smooth_ref: Any | None = None,
) -> tuple[GARTTargetResult, GARTTargetState]:
    xhat_aug_raw = _as_vector(xhat_aug_raw, "xhat_aug_raw")
    y_sp = _as_vector(y_sp, "y_sp")
    model = split_output_disturbance_model(A_aug, B_aug, C_aug, y_sp.size)
    if xhat_aug_raw.size != model.A_aug.shape[0]:
        raise ValueError("xhat_aug_raw has incorrect size.")
    u_min = _as_vector(u_min, "u_min", model.n_u)
    u_max = _as_vector(u_max, "u_max", model.n_u)
    u_smooth_arr = None if u_smooth_ref is None else _as_vector(u_smooth_ref, "u_smooth_ref", model.n_u)
    P_x_arr = None if P_x is None else _as_float_array(P_x, "P_x", ndim=2)
    K_x_arr = None if K_x is None else _as_float_array(K_x, "K_x", ndim=2)
    xhat = xhat_aug_raw[: model.n_x].copy()
    d_raw = xhat_aug_raw[model.n_x :].copy()
    d_cert, disturbance_info = update_certified_disturbance(
        None if state is None else state.d_cert,
        d_raw,
        config=config.disturbance,
        innovation=innovation,
    )

    prev_target = state if state is not None and bool(state.valid) else None
    raw_result = _solve_candidate(
        model=model,
        xhat=xhat,
        d_cert=d_cert,
        d_raw=d_raw,
        reference=y_sp,
        y_sp=y_sp,
        prev_target=prev_target,
        config=config,
        P_x=P_x_arr,
        K_x=K_x_arr,
        u_min=u_min,
        u_max=u_max,
        u_smooth_ref=u_smooth_arr,
        governor_alpha=1.0,
        governor_active=False,
        hold_previous=False,
        status="accepted_raw_reference",
        extra_diagnostics={"disturbance": disturbance_info},
    )
    if _candidate_accepted(raw_result, config) and _reference_motion_ok(y_sp, prev_target, config):
        return raw_result, _state_from_result(raw_result, state)

    if prev_target is None:
        if raw_result.accepted:
            raw_result.status = "initial_target_accepted"
            return raw_result, _state_from_result(raw_result, state)
        raw_result.status = "initial_target_rejected"
        raw_result.usable_for_lmpc = False
        raw_result.success = False
        raw_result.accepted = False
        raw_result.rejection_reason = raw_result.rejection_reason or "initial_target_rejected"
        raw_result.diagnostics["initial_raw_acceptance_failed"] = raw_result.diagnostics.get("contraction_probe")
        return raw_result, _state_from_result(raw_result, state)

    if not config.governor_enabled:
        return raw_result, _state_from_result(raw_result, state)

    r_prev = np.asarray(prev_target.r_cmd if prev_target.r_cmd is not None else prev_target.y_s, dtype=float).reshape(model.n_y)
    best: GARTTargetResult | None = None
    low_alpha: float | None = None
    high_alpha: float | None = None
    for alpha in sorted({float(a) for a in config.governor_grid if 0.0 <= float(a) <= 1.0}, reverse=True):
        reference = r_prev + alpha * (y_sp - r_prev)
        result = _solve_candidate(
            model=model,
            xhat=xhat,
            d_cert=d_cert,
            d_raw=d_raw,
            reference=reference,
            y_sp=y_sp,
            prev_target=prev_target,
            config=config,
            P_x=P_x_arr,
            K_x=K_x_arr,
            u_min=u_min,
            u_max=u_max,
            u_smooth_ref=u_smooth_arr,
            governor_alpha=alpha,
            governor_active=True,
            hold_previous=False,
            status="accepted_governed_reference",
            extra_diagnostics={"disturbance": disturbance_info, "raw_result": raw_result},
        )
        if _candidate_accepted(result, config) and _reference_motion_ok(reference, prev_target, config):
            best = result
            low_alpha = alpha
            break
        high_alpha = alpha if high_alpha is None else min(high_alpha, alpha)

    if best is not None and low_alpha is not None and high_alpha is not None and high_alpha > low_alpha:
        lo = low_alpha
        hi = high_alpha
        for _ in range(max(int(config.governor_bisect_iters), 0)):
            mid = 0.5 * (lo + hi)
            reference = r_prev + mid * (y_sp - r_prev)
            result = _solve_candidate(
                model=model,
                xhat=xhat,
                d_cert=d_cert,
                d_raw=d_raw,
                reference=reference,
                y_sp=y_sp,
                prev_target=prev_target,
                config=config,
                P_x=P_x_arr,
                K_x=K_x_arr,
                u_min=u_min,
                u_max=u_max,
                u_smooth_ref=u_smooth_arr,
                governor_alpha=mid,
                governor_active=True,
                hold_previous=False,
                status="accepted_governed_reference_bisect",
                extra_diagnostics={"disturbance": disturbance_info, "raw_result": raw_result},
            )
            if _candidate_accepted(result, config) and _reference_motion_ok(reference, prev_target, config):
                lo = mid
                best = result
            else:
                hi = mid

    if best is not None and best.governor_alpha is not None and best.governor_alpha > 0.0:
        return best, _state_from_result(best, state)

    held = _hold_previous_result(
        model=model,
        xhat=xhat,
        d_cert=d_cert,
        d_raw=d_raw,
        y_sp=y_sp,
        state=prev_target,
        config=config,
        P_x=P_x_arr,
        u_min=u_min,
        u_max=u_max,
        disturbance_info=disturbance_info,
    )
    held.diagnostics["raw_result"] = raw_result
    return held, _state_from_result(held, state)
