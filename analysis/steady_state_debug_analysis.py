from __future__ import annotations

import csv
import os
import pickle
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:
    pd = None
    HAS_PANDAS = False

try:
    from scipy.optimize import lsq_linear

    HAS_SCIPY = True
except Exception:
    lsq_linear = None
    HAS_SCIPY = False

from utils.scaling_helpers import apply_min_max, reverse_min_max


DEFAULT_ANALYSIS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "solver_mode": "auto",
    "cond_warn_threshold": 1.0e8,
    "residual_warn_threshold": 1.0e-8,
    "rank_tol": None,
    "enable_box_analysis": True,
    "box_bound_tol": 1.0e-9,
    "box_use_reduced_first": True,
    "box_event_window_radius": 5,
    "box_dhat_event_threshold": 5.0e-2,
    "box_max_event_plots": 6,
    "tail_window_samples": 20,
    "save_csv": True,
    "save_plots": True,
    "sample_table_stride": 10,
    "output_dir": None,
    "case_name": "disturbance",
}


def _as_float_array(value: Any, name: str, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}.")
    return array


def _safe_matrix_rank(matrix: np.ndarray, rank_tol: Optional[float] = None) -> int:
    try:
        return int(np.linalg.matrix_rank(matrix, tol=rank_tol))
    except TypeError:
        return int(np.linalg.matrix_rank(matrix))


def _safe_cond(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(matrix))
    except Exception:
        return float("inf")


def _safe_singular_values(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.svd(matrix, compute_uv=False)
    except Exception:
        return np.full((min(matrix.shape),), np.nan, dtype=float)


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float).reshape(-1)))


def _inf_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _is_well_conditioned(cond_value: float, threshold: float) -> bool:
    return np.isfinite(cond_value) and cond_value <= float(threshold)


def _model_classification(n_states: int, n_outputs: int, n_inputs: int) -> str:
    rows = n_states + n_outputs
    cols = n_states + n_inputs
    if rows == cols:
        return "square"
    if rows < cols:
        return "underdetermined"
    return "overdetermined"


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


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_ANALYSIS_CONFIG)
    if config:
        merged.update(config)
    return merged


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as file:
            file.write("")
        return

    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                headers.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def _format_scalar(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if np.isnan(numeric):
        return "nan"
    if abs(numeric) >= 1.0e4 or (abs(numeric) > 0.0 and abs(numeric) < 1.0e-3):
        return f"{numeric:.4e}"
    return f"{numeric:.6f}"


def _rows_to_markdown(rows: List[Dict[str, Any]], columns: Iterable[str]) -> str:
    columns = list(columns)
    if not rows:
        return "_No rows available._"

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [_format_scalar(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _summary_stat_block(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "max": float(np.max(finite)),
    }


def _step_axis(n_steps: int, delta_t: float) -> np.ndarray:
    return np.arange(n_steps, dtype=float) * float(delta_t)


def _setpoint_change_indices(y_sp: np.ndarray) -> np.ndarray:
    if y_sp.shape[0] <= 1:
        return np.array([], dtype=int)
    diff = np.abs(np.diff(y_sp, axis=0))
    return np.where(np.any(diff > 0.0, axis=1))[0] + 1


def _append_vertical_lines(ax: Any, x_positions: np.ndarray, delta_t: float) -> None:
    for step_idx in x_positions:
        ax.axvline(float(step_idx) * float(delta_t), color="0.75", linestyle="--", linewidth=1.0)


def _validate_state_space(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    n_states = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n_states:
        raise ValueError(f"A must be square, got shape {A.shape}.")
    if B.ndim != 2 or B.shape[0] != n_states:
        raise ValueError(f"B must have {n_states} rows, got shape {B.shape}.")
    if C.ndim != 2 or C.shape[1] != n_states:
        raise ValueError(f"C must have {n_states} columns, got shape {C.shape}.")


def build_legacy_ss_system(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    rank_tol: Optional[float] = None,
) -> Dict[str, Any]:
    A = _as_float_array(A, "A", ndim=2)
    B = _as_float_array(B, "B", ndim=2)
    C = _as_float_array(C, "C", ndim=2)
    _validate_state_space(A, B, C)

    n_states = A.shape[0]
    n_outputs = C.shape[0]
    n_inputs = B.shape[1]
    I_minus_A = np.eye(n_states, dtype=float) - A
    zeros_block = np.zeros((n_outputs, n_inputs), dtype=float)
    M = np.block([[I_minus_A, -B], [C, zeros_block]])
    singular_values = _safe_singular_values(M)

    return {
        "M": M,
        "I_minus_A": I_minus_A,
        "n_states": int(n_states),
        "n_outputs": int(n_outputs),
        "n_inputs": int(n_inputs),
        "M_shape": tuple(int(v) for v in M.shape),
        "rank_M": _safe_matrix_rank(M, rank_tol=rank_tol),
        "cond_M": _safe_cond(M),
        "smallest_sv_M": float(singular_values[-1]) if singular_values.size else float("nan"),
        "singular_values_M": singular_values,
        "is_M_square": bool(M.shape[0] == M.shape[1]),
        "classification": _model_classification(n_states, n_outputs, n_inputs),
    }


def build_legacy_ss_rhs(y_sp_k: np.ndarray, d_hat_k: np.ndarray, n_states: int) -> Dict[str, Any]:
    y_sp_k = _as_float_array(y_sp_k, "y_sp_k", ndim=1)
    d_hat_k = _as_float_array(d_hat_k, "d_hat_k", ndim=1)
    if y_sp_k.shape != d_hat_k.shape:
        raise ValueError(
            f"y_sp_k and d_hat_k must have the same shape, got {y_sp_k.shape} and {d_hat_k.shape}."
        )
    rhs_output = y_sp_k - d_hat_k
    rhs = np.concatenate([np.zeros(int(n_states), dtype=float), rhs_output])
    return {
        "rhs": rhs,
        "rhs_output": rhs_output,
        "n_states": int(n_states),
    }


def compute_reduced_gain(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    cond_warn_threshold: float = 1.0e8,
    rank_tol: Optional[float] = None,
) -> Dict[str, Any]:
    A = _as_float_array(A, "A", ndim=2)
    B = _as_float_array(B, "B", ndim=2)
    C = _as_float_array(C, "C", ndim=2)
    _validate_state_space(A, B, C)

    n_states = A.shape[0]
    I_minus_A = np.eye(n_states, dtype=float) - A
    rank_I_minus_A = _safe_matrix_rank(I_minus_A, rank_tol=rank_tol)
    cond_I_minus_A = _safe_cond(I_minus_A)
    invertible_I_minus_A = bool(
        I_minus_A.shape[0] == I_minus_A.shape[1]
        and rank_I_minus_A == I_minus_A.shape[0]
        and _is_well_conditioned(cond_I_minus_A, cond_warn_threshold)
    )

    G = None
    state_to_input_gain = None
    cond_G = float("inf")
    rank_G = 0
    singular_values_G = np.array([], dtype=float)
    smallest_sv_G = float("nan")
    is_G_square = False
    G_shape = None

    if invertible_I_minus_A:
        state_to_input_gain = np.linalg.solve(I_minus_A, B)
        G = C @ state_to_input_gain
        singular_values_G = _safe_singular_values(G)
        cond_G = _safe_cond(G)
        rank_G = _safe_matrix_rank(G, rank_tol=rank_tol)
        smallest_sv_G = float(singular_values_G[-1]) if singular_values_G.size else float("nan")
        is_G_square = bool(G.shape[0] == G.shape[1])
        G_shape = tuple(int(v) for v in G.shape)

    reduced_exact_available = bool(
        invertible_I_minus_A
        and G is not None
        and is_G_square
        and rank_G == G.shape[0]
        and _is_well_conditioned(cond_G, cond_warn_threshold)
    )
    reduced_lstsq_available = bool(invertible_I_minus_A and G is not None)

    return {
        "I_minus_A": I_minus_A,
        "rank_I_minus_A": rank_I_minus_A,
        "cond_I_minus_A": cond_I_minus_A,
        "invertible_I_minus_A": invertible_I_minus_A,
        "state_to_input_gain": state_to_input_gain,
        "G": G,
        "G_shape": G_shape,
        "rank_G": rank_G,
        "cond_G": cond_G,
        "smallest_sv_G": smallest_sv_G,
        "singular_values_G": singular_values_G,
        "is_G_square": is_G_square,
        "reduced_exact_available": reduced_exact_available,
        "reduced_lstsq_available": reduced_lstsq_available,
    }


def _fallback_mode(
    structure: Dict[str, Any],
    reduced_info: Dict[str, Any],
    requested_mode: str,
    cond_warn_threshold: float,
) -> Dict[str, Any]:
    rank_M = int(structure["rank_M"])
    M = structure["M"]
    cond_M = float(structure["cond_M"])

    stacked_exact_ok = bool(
        structure["is_M_square"]
        and rank_M == M.shape[0]
        and _is_well_conditioned(cond_M, cond_warn_threshold)
    )
    reduced_exact_ok = bool(reduced_info["reduced_exact_available"])
    reduced_lstsq_ok = bool(reduced_info["reduced_lstsq_available"])

    if requested_mode == "stacked_exact":
        if stacked_exact_ok:
            return {"mode": "stacked_exact", "fallback": False, "reason": None}
        if reduced_exact_ok:
            return {"mode": "reduced_exact", "fallback": True, "reason": "stacked_exact_unavailable"}
        if reduced_lstsq_ok:
            return {"mode": "reduced_lstsq", "fallback": True, "reason": "stacked_exact_unavailable"}
        return {"mode": "stacked_lstsq", "fallback": True, "reason": "stacked_exact_unavailable"}

    if requested_mode == "reduced_exact":
        if reduced_exact_ok:
            return {"mode": "reduced_exact", "fallback": False, "reason": None}
        if stacked_exact_ok:
            return {"mode": "stacked_exact", "fallback": True, "reason": "reduced_exact_unavailable"}
        if reduced_lstsq_ok:
            return {"mode": "reduced_lstsq", "fallback": True, "reason": "reduced_exact_unavailable"}
        return {"mode": "stacked_lstsq", "fallback": True, "reason": "reduced_exact_unavailable"}

    if requested_mode == "reduced_lstsq":
        if reduced_lstsq_ok:
            return {"mode": "reduced_lstsq", "fallback": False, "reason": None}
        return {"mode": "stacked_lstsq", "fallback": True, "reason": "reduced_lstsq_unavailable"}

    if requested_mode == "stacked_lstsq":
        return {"mode": "stacked_lstsq", "fallback": False, "reason": None}

    if stacked_exact_ok:
        return {"mode": "stacked_exact", "fallback": False, "reason": None}
    if reduced_exact_ok:
        return {"mode": "reduced_exact", "fallback": False, "reason": None}
    if reduced_lstsq_ok:
        return {"mode": "reduced_lstsq", "fallback": False, "reason": None}
    return {"mode": "stacked_lstsq", "fallback": False, "reason": None}


def solve_legacy_ss_exact(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    y_sp_k: np.ndarray,
    d_hat_k: np.ndarray,
    solver_mode: str = "auto",
    cond_warn_threshold: float = 1.0e8,
    residual_warn_threshold: float = 1.0e-8,
    rank_tol: Optional[float] = None,
) -> Dict[str, Any]:
    structure = build_legacy_ss_system(A, B, C, rank_tol=rank_tol)
    reduced_info = compute_reduced_gain(
        A,
        B,
        C,
        cond_warn_threshold=cond_warn_threshold,
        rank_tol=rank_tol,
    )
    rhs_info = build_legacy_ss_rhs(y_sp_k, d_hat_k, n_states=structure["n_states"])
    M = structure["M"]
    rhs = rhs_info["rhs"]
    rhs_output = rhs_info["rhs_output"]
    n_states = int(structure["n_states"])
    n_inputs = int(structure["n_inputs"])
    requested_mode = str(solver_mode).lower()

    mode_info = _fallback_mode(structure, reduced_info, requested_mode, cond_warn_threshold)
    solver_mode_used = mode_info["mode"]
    fallback_reason = mode_info["reason"]
    requested_mode_fallback = bool(mode_info["fallback"])

    singular_values_solver = None
    x_s = None
    u_s = None
    rank_used = None

    if solver_mode_used == "stacked_exact":
        solution = np.linalg.solve(M, rhs)
        x_s = solution[:n_states]
        u_s = solution[n_states:]
        singular_values_solver = structure["singular_values_M"]
        rank_used = structure["rank_M"]
    elif solver_mode_used == "stacked_lstsq":
        solution, _, rank_used, singular_values_solver = np.linalg.lstsq(M, rhs, rcond=rank_tol)
        x_s = solution[:n_states]
        u_s = solution[n_states:]
    elif solver_mode_used == "reduced_exact":
        G = reduced_info["G"]
        state_to_input_gain = reduced_info["state_to_input_gain"]
        u_s = np.linalg.solve(G, rhs_output)
        x_s = state_to_input_gain @ u_s
        singular_values_solver = reduced_info["singular_values_G"]
        rank_used = reduced_info["rank_G"]
    elif solver_mode_used == "reduced_lstsq":
        G = reduced_info["G"]
        state_to_input_gain = reduced_info["state_to_input_gain"]
        u_s, _, rank_used, singular_values_solver = np.linalg.lstsq(G, rhs_output, rcond=rank_tol)
        x_s = state_to_input_gain @ u_s
    else:
        raise ValueError(f"Unsupported solver_mode '{solver_mode_used}'.")

    x_s = np.asarray(x_s, dtype=float).reshape(n_states)
    u_s = np.asarray(u_s, dtype=float).reshape(n_inputs)
    d_s = _as_float_array(d_hat_k, "d_hat_k", ndim=1).copy()
    y_s = np.asarray(C, dtype=float) @ x_s + d_s

    I_minus_A = structure["I_minus_A"]
    residual_dyn = I_minus_A @ x_s - np.asarray(B, dtype=float) @ u_s
    residual_out = np.asarray(C, dtype=float) @ x_s - rhs_output
    residual_total = M @ np.concatenate([x_s, u_s]) - rhs

    residual_dyn_norm = _norm(residual_dyn)
    residual_out_norm = _norm(residual_out)
    residual_total_norm = _norm(residual_total)
    is_exact_solution = bool(residual_total_norm <= float(residual_warn_threshold))
    used_lstsq = bool(solver_mode_used.endswith("lstsq"))

    return {
        "x_s": x_s,
        "u_s": u_s,
        "d_s": d_s,
        "y_s": y_s,
        "residual_dyn": residual_dyn,
        "residual_out": residual_out,
        "residual_total": residual_total,
        "residual_dyn_norm": residual_dyn_norm,
        "residual_out_norm": residual_out_norm,
        "residual_total_norm": residual_total_norm,
        "rank_M": int(structure["rank_M"]),
        "rank_G": int(reduced_info["rank_G"]),
        "rank_solver_matrix": None if rank_used is None else int(rank_used),
        "cond_M": float(structure["cond_M"]),
        "cond_G": float(reduced_info["cond_G"]),
        "cond_I_minus_A": float(reduced_info["cond_I_minus_A"]),
        "smallest_sv_M": float(structure["smallest_sv_M"]),
        "smallest_sv_G": float(reduced_info["smallest_sv_G"]),
        "singular_values_solver": np.asarray(singular_values_solver, dtype=float),
        "solver_mode_requested": requested_mode,
        "solver_mode_used": solver_mode_used,
        "requested_mode_fallback": requested_mode_fallback,
        "fallback_reason": fallback_reason,
        "used_lstsq": used_lstsq,
        "is_exact_solution": is_exact_solution,
        "M_shape": tuple(int(v) for v in structure["M_shape"]),
        "G_shape": reduced_info["G_shape"],
        "rhs_output": rhs_output,
        "invertible_I_minus_A": bool(reduced_info["invertible_I_minus_A"]),
        "reduced_exact_available": bool(reduced_info["reduced_exact_available"]),
        "reduced_lstsq_available": bool(reduced_info["reduced_lstsq_available"]),
    }


def solve_exact_steady_state_unbounded(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility wrapper for the exact unbounded steady-state solve."""
    return solve_legacy_ss_exact(*args, **kwargs)


def check_box_bounds(
    u: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    tol: float = 1.0e-9,
) -> Dict[str, Any]:
    u = _as_float_array(u, "u", ndim=1)
    u_min = _as_float_array(u_min, "u_min", ndim=1)
    u_max = _as_float_array(u_max, "u_max", ndim=1)
    if u.shape != u_min.shape or u.shape != u_max.shape:
        raise ValueError("u, u_min, and u_max must have the same shape.")

    lower_violation = np.maximum(u_min - u, 0.0)
    upper_violation = np.maximum(u - u_max, 0.0)
    within_bounds = bool(
        np.all(u >= (u_min - float(tol))) and np.all(u <= (u_max + float(tol)))
    )
    active_lower_mask = u <= (u_min + float(tol))
    active_upper_mask = u >= (u_max - float(tol))

    return {
        "within_bounds": within_bounds,
        "lower_violation": lower_violation,
        "upper_violation": upper_violation,
        "lower_violation_inf": _inf_norm(lower_violation),
        "upper_violation_inf": _inf_norm(upper_violation),
        "violation_inf": max(_inf_norm(lower_violation), _inf_norm(upper_violation)),
        "active_lower_mask": active_lower_mask,
        "active_upper_mask": active_upper_mask,
    }


def solve_bounded_steady_state_least_squares(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    y_sp_k: np.ndarray,
    d_hat_k: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    cond_warn_threshold: float = 1.0e8,
    rank_tol: Optional[float] = None,
    box_bound_tol: float = 1.0e-9,
    use_reduced_first: bool = True,
) -> Dict[str, Any]:
    A = _as_float_array(A, "A", ndim=2)
    B = _as_float_array(B, "B", ndim=2)
    C = _as_float_array(C, "C", ndim=2)
    u_min = _as_float_array(u_min, "u_min", ndim=1)
    u_max = _as_float_array(u_max, "u_max", ndim=1)

    structure = build_legacy_ss_system(A, B, C, rank_tol=rank_tol)
    reduced_info = compute_reduced_gain(
        A,
        B,
        C,
        cond_warn_threshold=cond_warn_threshold,
        rank_tol=rank_tol,
    )
    rhs_info = build_legacy_ss_rhs(y_sp_k, d_hat_k, n_states=structure["n_states"])
    rhs = rhs_info["rhs"]
    rhs_output = rhs_info["rhs_output"]
    n_states = int(structure["n_states"])
    n_inputs = int(structure["n_inputs"])

    if u_min.shape != (n_inputs,) or u_max.shape != (n_inputs,):
        raise ValueError("u_min and u_max must match the number of inputs.")

    if not HAS_SCIPY:
        return {
            "solve_success": False,
            "solver_name": "unavailable",
            "solve_form": "none",
            "status": "scipy_unavailable",
            "message": "scipy.optimize.lsq_linear is required for bounded steady-state analysis.",
            "x_s": np.full(n_states, np.nan, dtype=float),
            "u_s": np.full(n_inputs, np.nan, dtype=float),
            "d_s": _as_float_array(d_hat_k, "d_hat_k", ndim=1).copy(),
            "y_s": np.full(C.shape[0], np.nan, dtype=float),
            "residual_dyn": np.full(n_states, np.nan, dtype=float),
            "residual_out": np.full(C.shape[0], np.nan, dtype=float),
            "residual_total": np.full(n_states + n_inputs, np.nan, dtype=float),
            "residual_norm": float("nan"),
            "state_residual_inf": float("nan"),
            "output_residual_inf": float("nan"),
            "active_lower_mask": np.zeros(n_inputs, dtype=bool),
            "active_upper_mask": np.zeros(n_inputs, dtype=bool),
        }

    solve_attempts: List[str] = []
    last_result = None

    if bool(use_reduced_first) and reduced_info["reduced_lstsq_available"]:
        G = np.asarray(reduced_info["G"], dtype=float)
        solve_attempts.append("reduced_lsq_linear")
        try:
            reduced_result = lsq_linear(G, rhs_output, bounds=(u_min, u_max))
            last_result = reduced_result
            if reduced_result.success:
                u_s = np.asarray(reduced_result.x, dtype=float).reshape(n_inputs)
                x_s = np.asarray(reduced_info["state_to_input_gain"], dtype=float) @ u_s
                d_s = _as_float_array(d_hat_k, "d_hat_k", ndim=1).copy()
                y_s = np.asarray(C, dtype=float) @ x_s + d_s
                residual_dyn = structure["I_minus_A"] @ x_s - np.asarray(B, dtype=float) @ u_s
                residual_out = np.asarray(C, dtype=float) @ x_s - rhs_output
                residual_total = structure["M"] @ np.concatenate([x_s, u_s]) - rhs
                bounds_info = check_box_bounds(u_s, u_min, u_max, tol=box_bound_tol)
                return {
                    "solve_success": True,
                    "solver_name": "scipy.optimize.lsq_linear",
                    "solve_form": "reduced",
                    "status": str(reduced_result.status),
                    "message": getattr(reduced_result, "message", ""),
                    "x_s": x_s,
                    "u_s": u_s,
                    "d_s": d_s,
                    "y_s": y_s,
                    "residual_dyn": residual_dyn,
                    "residual_out": residual_out,
                    "residual_total": residual_total,
                    "residual_norm": _norm(residual_total),
                    "state_residual_inf": _inf_norm(residual_dyn),
                    "output_residual_inf": _inf_norm(residual_out),
                    "active_lower_mask": bounds_info["active_lower_mask"],
                    "active_upper_mask": bounds_info["active_upper_mask"],
                    "cost": float(reduced_result.cost),
                    "nit": int(getattr(reduced_result, "nit", 0)),
                    "optimality": float(getattr(reduced_result, "optimality", np.nan)),
                }
        except Exception:
            last_result = None

    solve_attempts.append("full_lsq_linear")
    M = np.asarray(structure["M"], dtype=float)
    lower = np.concatenate([np.full(n_states, -np.inf, dtype=float), u_min])
    upper = np.concatenate([np.full(n_states, np.inf, dtype=float), u_max])
    try:
        full_result = lsq_linear(M, rhs, bounds=(lower, upper))
        last_result = full_result
        if full_result.success:
            solution = np.asarray(full_result.x, dtype=float)
            x_s = solution[:n_states]
            u_s = solution[n_states:]
            d_s = _as_float_array(d_hat_k, "d_hat_k", ndim=1).copy()
            y_s = np.asarray(C, dtype=float) @ x_s + d_s
            residual_dyn = structure["I_minus_A"] @ x_s - np.asarray(B, dtype=float) @ u_s
            residual_out = np.asarray(C, dtype=float) @ x_s - rhs_output
            residual_total = M @ np.concatenate([x_s, u_s]) - rhs
            bounds_info = check_box_bounds(u_s, u_min, u_max, tol=box_bound_tol)
            return {
                "solve_success": True,
                "solver_name": "scipy.optimize.lsq_linear",
                "solve_form": "full",
                "status": str(full_result.status),
                "message": getattr(full_result, "message", ""),
                "x_s": x_s,
                "u_s": u_s,
                "d_s": d_s,
                "y_s": y_s,
                "residual_dyn": residual_dyn,
                "residual_out": residual_out,
                "residual_total": residual_total,
                "residual_norm": _norm(residual_total),
                "state_residual_inf": _inf_norm(residual_dyn),
                "output_residual_inf": _inf_norm(residual_out),
                "active_lower_mask": bounds_info["active_lower_mask"],
                "active_upper_mask": bounds_info["active_upper_mask"],
                "cost": float(full_result.cost),
                "nit": int(getattr(full_result, "nit", 0)),
                "optimality": float(getattr(full_result, "optimality", np.nan)),
            }
    except Exception:
        last_result = None

    message = "bounded least-squares solve failed."
    status = "failed"
    if last_result is not None:
        message = getattr(last_result, "message", message)
        status = str(getattr(last_result, "status", status))

    return {
        "solve_success": False,
        "solver_name": "scipy.optimize.lsq_linear",
        "solve_form": "none",
        "status": status,
        "message": message,
        "solve_attempts": solve_attempts,
        "x_s": np.full(n_states, np.nan, dtype=float),
        "u_s": np.full(n_inputs, np.nan, dtype=float),
        "d_s": _as_float_array(d_hat_k, "d_hat_k", ndim=1).copy(),
        "y_s": np.full(C.shape[0], np.nan, dtype=float),
        "residual_dyn": np.full(n_states, np.nan, dtype=float),
        "residual_out": np.full(C.shape[0], np.nan, dtype=float),
        "residual_total": np.full(n_states + n_inputs, np.nan, dtype=float),
        "residual_norm": float("nan"),
        "state_residual_inf": float("nan"),
        "output_residual_inf": float("nan"),
        "active_lower_mask": np.zeros(n_inputs, dtype=bool),
        "active_upper_mask": np.zeros(n_inputs, dtype=bool),
    }


def _build_box_event_rows(
    y_sp: np.ndarray,
    dhat_current: np.ndarray,
    u_s_exact: np.ndarray,
    u_s_bounded: np.ndarray,
    solve_mode: List[str],
    exact_eq_residual_state_inf: np.ndarray,
    exact_eq_residual_output_inf: np.ndarray,
    bounded_residual_norm: np.ndarray,
    setpoint_change_indices: np.ndarray,
    event_window_radius: int,
    dhat_event_threshold: float,
) -> List[Dict[str, Any]]:
    dhat_delta = np.zeros(dhat_current.shape[0], dtype=float)
    if dhat_current.shape[0] > 1:
        dhat_delta[1:] = np.max(np.abs(np.diff(dhat_current, axis=0)), axis=1)
    dhat_event_indices = np.where(dhat_delta >= float(dhat_event_threshold))[0]

    rows: List[Dict[str, Any]] = []
    event_meta: List[tuple[int, str]] = []
    event_meta.extend((int(idx), "setpoint_change") for idx in np.asarray(setpoint_change_indices, dtype=int))
    event_meta.extend((int(idx), "dhat_jump") for idx in np.asarray(dhat_event_indices, dtype=int))
    if not event_meta:
        return rows

    n_steps = y_sp.shape[0]
    for event_idx, event_kind in event_meta:
        start = max(event_idx - int(event_window_radius), 0)
        stop = min(event_idx + int(event_window_radius) + 1, n_steps)
        for step_idx in range(start, stop):
            rows.append(
                {
                    "event_kind": event_kind,
                    "event_anchor": int(event_idx),
                    "k": int(step_idx),
                    "y_sp": np.asarray(y_sp[step_idx, :], dtype=float),
                    "dhat_k": np.asarray(dhat_current[step_idx, :], dtype=float),
                    "us_exact": np.asarray(u_s_exact[step_idx, :], dtype=float),
                    "us_bounded": np.asarray(u_s_bounded[step_idx, :], dtype=float),
                    "solve_mode": solve_mode[step_idx],
                    "exact_eq_residual_state_inf": float(exact_eq_residual_state_inf[step_idx]),
                    "exact_eq_residual_output_inf": float(exact_eq_residual_output_inf[step_idx]),
                    "bounded_residual_norm": float(bounded_residual_norm[step_idx]),
                    "dhat_delta_inf": float(dhat_delta[step_idx]),
                }
            )
    return rows


def run_parallel_steady_state_box_analysis(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    y_sp: np.ndarray,
    dhat_current: np.ndarray,
    x_s_exact: np.ndarray,
    u_s_exact: np.ndarray,
    d_s_exact: np.ndarray,
    y_s_exact: np.ndarray,
    exact_solution_flags: np.ndarray,
    exact_eq_residual_state_inf: np.ndarray,
    exact_eq_residual_output_inf: np.ndarray,
    u_box_min: np.ndarray,
    u_box_max: np.ndarray,
    setpoint_change_indices: np.ndarray,
    analysis_config: Dict[str, Any],
) -> Dict[str, Any]:
    n_steps = y_sp.shape[0]
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_outputs = C.shape[0]

    x_s_bounded = np.full((n_steps, n_states), np.nan, dtype=float)
    u_s_bounded = np.full((n_steps, n_inputs), np.nan, dtype=float)
    d_s_bounded = np.full((n_steps, n_outputs), np.nan, dtype=float)
    y_s_bounded = np.full((n_steps, n_outputs), np.nan, dtype=float)
    bounded_success = np.zeros(n_steps, dtype=bool)
    exact_solve_success = np.asarray(exact_solution_flags, dtype=bool).copy()
    exact_within_bounds = np.zeros(n_steps, dtype=bool)
    exact_bound_violation_inf = np.zeros(n_steps, dtype=float)
    exact_bound_violation_lower_inf = np.zeros(n_steps, dtype=float)
    exact_bound_violation_upper_inf = np.zeros(n_steps, dtype=float)
    bounded_residual_norm = np.full(n_steps, np.nan, dtype=float)
    bounded_state_residual_inf = np.full(n_steps, np.nan, dtype=float)
    bounded_output_residual_inf = np.full(n_steps, np.nan, dtype=float)
    bounded_active_lower_mask = np.zeros((n_steps, n_inputs), dtype=bool)
    bounded_active_upper_mask = np.zeros((n_steps, n_inputs), dtype=bool)
    solve_mode: List[str] = []
    box_solver_name: List[str] = []
    box_solver_form: List[str] = []
    us_exact_minus_us_bounded_inf = np.full(n_steps, np.nan, dtype=float)
    xs_exact_minus_xs_bounded_inf = np.full(n_steps, np.nan, dtype=float)

    per_step_rows: List[Dict[str, Any]] = []
    for step_idx in range(n_steps):
        exact_bounds = check_box_bounds(
            u_s_exact[step_idx, :],
            u_box_min,
            u_box_max,
            tol=float(analysis_config["box_bound_tol"]),
        )
        exact_within_bounds[step_idx] = bool(exact_bounds["within_bounds"])
        exact_bound_violation_inf[step_idx] = float(exact_bounds["violation_inf"])
        exact_bound_violation_lower_inf[step_idx] = float(exact_bounds["lower_violation_inf"])
        exact_bound_violation_upper_inf[step_idx] = float(exact_bounds["upper_violation_inf"])

        if exact_solve_success[step_idx] and exact_within_bounds[step_idx]:
            mode = "exact_bounded"
            bounded_success[step_idx] = True
            x_s_bounded[step_idx, :] = x_s_exact[step_idx, :]
            u_s_bounded[step_idx, :] = u_s_exact[step_idx, :]
            d_s_bounded[step_idx, :] = d_s_exact[step_idx, :]
            y_s_bounded[step_idx, :] = y_s_exact[step_idx, :]
            bounded_residual_norm[step_idx] = 0.0
            bounded_state_residual_inf[step_idx] = float(exact_eq_residual_state_inf[step_idx])
            bounded_output_residual_inf[step_idx] = float(exact_eq_residual_output_inf[step_idx])
            bounded_active_lower_mask[step_idx, :] = exact_bounds["active_lower_mask"]
            bounded_active_upper_mask[step_idx, :] = exact_bounds["active_upper_mask"]
            box_solver_name.append("exact_unbounded")
            box_solver_form.append("none")
        else:
            bounded_info = solve_bounded_steady_state_least_squares(
                A,
                B,
                C,
                y_sp[step_idx, :],
                dhat_current[step_idx, :],
                u_box_min,
                u_box_max,
                cond_warn_threshold=float(analysis_config["cond_warn_threshold"]),
                rank_tol=analysis_config["rank_tol"],
                box_bound_tol=float(analysis_config["box_bound_tol"]),
                use_reduced_first=bool(analysis_config["box_use_reduced_first"]),
            )
            box_solver_name.append(str(bounded_info["solver_name"]))
            box_solver_form.append(str(bounded_info["solve_form"]))
            if bounded_info["solve_success"]:
                mode = (
                    "exact_unbounded_fallback_bounded_ls"
                    if exact_solve_success[step_idx]
                    else "exact_unsolved_fallback_bounded_ls"
                )
                bounded_success[step_idx] = True
                x_s_bounded[step_idx, :] = bounded_info["x_s"]
                u_s_bounded[step_idx, :] = bounded_info["u_s"]
                d_s_bounded[step_idx, :] = bounded_info["d_s"]
                y_s_bounded[step_idx, :] = bounded_info["y_s"]
                bounded_residual_norm[step_idx] = float(bounded_info["residual_norm"])
                bounded_state_residual_inf[step_idx] = float(bounded_info["state_residual_inf"])
                bounded_output_residual_inf[step_idx] = float(bounded_info["output_residual_inf"])
                bounded_active_lower_mask[step_idx, :] = np.asarray(
                    bounded_info["active_lower_mask"], dtype=bool
                )
                bounded_active_upper_mask[step_idx, :] = np.asarray(
                    bounded_info["active_upper_mask"], dtype=bool
                )
            else:
                mode = "failed"

        solve_mode.append(mode)
        if bounded_success[step_idx]:
            us_exact_minus_us_bounded_inf[step_idx] = _inf_norm(
                u_s_exact[step_idx, :] - u_s_bounded[step_idx, :]
            )
            xs_exact_minus_xs_bounded_inf[step_idx] = _inf_norm(
                x_s_exact[step_idx, :] - x_s_bounded[step_idx, :]
            )

        per_step_rows.append(
            {
                "k": int(step_idx),
                "r_k": np.asarray(y_sp[step_idx, :] - dhat_current[step_idx, :], dtype=float),
                "xs_exact": np.asarray(x_s_exact[step_idx, :], dtype=float),
                "us_exact": np.asarray(u_s_exact[step_idx, :], dtype=float),
                "exact_solve_success": bool(exact_solve_success[step_idx]),
                "exact_within_bounds": bool(exact_within_bounds[step_idx]),
                "exact_bound_violation_inf": float(exact_bound_violation_inf[step_idx]),
                "exact_eq_residual_state_inf": float(exact_eq_residual_state_inf[step_idx]),
                "exact_eq_residual_output_inf": float(exact_eq_residual_output_inf[step_idx]),
                "xs_bounded": np.asarray(x_s_bounded[step_idx, :], dtype=float),
                "us_bounded": np.asarray(u_s_bounded[step_idx, :], dtype=float),
                "bounded_solve_success": bool(bounded_success[step_idx]),
                "bounded_residual_norm": float(bounded_residual_norm[step_idx]),
                "bounded_active_lower_mask": np.asarray(bounded_active_lower_mask[step_idx, :], dtype=bool),
                "bounded_active_upper_mask": np.asarray(bounded_active_upper_mask[step_idx, :], dtype=bool),
                "solve_mode": mode,
            }
        )

    solve_mode_counts: Dict[str, int] = {}
    for mode_name in solve_mode:
        solve_mode_counts[mode_name] = solve_mode_counts.get(mode_name, 0) + 1

    overall_summary = {
        "pct_exact_bounded": 100.0 * solve_mode_counts.get("exact_bounded", 0) / max(n_steps, 1),
        "pct_exact_unbounded_fallback_bounded_ls": 100.0
        * solve_mode_counts.get("exact_unbounded_fallback_bounded_ls", 0)
        / max(n_steps, 1),
        "pct_exact_unsolved_fallback_bounded_ls": 100.0
        * solve_mode_counts.get("exact_unsolved_fallback_bounded_ls", 0)
        / max(n_steps, 1),
        "pct_failed": 100.0 * solve_mode_counts.get("failed", 0) / max(n_steps, 1),
        "pct_exact_solutions_inside_bounds": 100.0
        * float(np.mean(np.logical_and(exact_solve_success, exact_within_bounds))),
        "avg_exact_bound_violation_inf": float(np.nanmean(exact_bound_violation_inf)),
        "max_exact_bound_violation_inf": float(np.nanmax(exact_bound_violation_inf)),
        "avg_bounded_residual_norm": float(np.nanmean(bounded_residual_norm)),
        "max_bounded_residual_norm": float(np.nanmax(bounded_residual_norm)),
        "avg_us_exact_minus_us_bounded_inf": float(np.nanmean(us_exact_minus_us_bounded_inf)),
        "max_us_exact_minus_us_bounded_inf": float(np.nanmax(us_exact_minus_us_bounded_inf)),
        "avg_xs_exact_minus_xs_bounded_inf": float(np.nanmean(xs_exact_minus_xs_bounded_inf)),
        "max_xs_exact_minus_xs_bounded_inf": float(np.nanmax(xs_exact_minus_xs_bounded_inf)),
    }

    per_input_rows: List[Dict[str, Any]] = []
    for input_idx in range(n_inputs):
        lower_violation = np.maximum(u_box_min[input_idx] - u_s_exact[:, input_idx], 0.0)
        upper_violation = np.maximum(u_s_exact[:, input_idx] - u_box_max[input_idx], 0.0)
        per_input_rows.append(
            {
                "input_index": int(input_idx),
                "fraction_lower_bound_active": float(np.mean(bounded_active_lower_mask[:, input_idx])),
                "fraction_upper_bound_active": float(np.mean(bounded_active_upper_mask[:, input_idx])),
                "average_exact_violation_below_lower": float(np.mean(lower_violation)),
                "average_exact_violation_above_upper": float(np.mean(upper_violation)),
            }
        )

    event_rows = _build_box_event_rows(
        y_sp=y_sp,
        dhat_current=dhat_current,
        u_s_exact=u_s_exact,
        u_s_bounded=u_s_bounded,
        solve_mode=solve_mode,
        exact_eq_residual_state_inf=exact_eq_residual_state_inf,
        exact_eq_residual_output_inf=exact_eq_residual_output_inf,
        bounded_residual_norm=bounded_residual_norm,
        setpoint_change_indices=setpoint_change_indices,
        event_window_radius=int(analysis_config["box_event_window_radius"]),
        dhat_event_threshold=float(analysis_config["box_dhat_event_threshold"]),
    )

    return {
        "enabled": True,
        "u_box_min": u_box_min,
        "u_box_max": u_box_max,
        "x_s_bounded": x_s_bounded,
        "u_s_bounded": u_s_bounded,
        "d_s_bounded": d_s_bounded,
        "y_s_bounded": y_s_bounded,
        "exact_solve_success": exact_solve_success,
        "exact_within_bounds": exact_within_bounds,
        "exact_bound_violation_inf": exact_bound_violation_inf,
        "exact_bound_violation_lower_inf": exact_bound_violation_lower_inf,
        "exact_bound_violation_upper_inf": exact_bound_violation_upper_inf,
        "bounded_solve_success": bounded_success,
        "bounded_residual_norm": bounded_residual_norm,
        "bounded_state_residual_inf": bounded_state_residual_inf,
        "bounded_output_residual_inf": bounded_output_residual_inf,
        "bounded_active_lower_mask": bounded_active_lower_mask,
        "bounded_active_upper_mask": bounded_active_upper_mask,
        "solve_mode": solve_mode,
        "solve_mode_counts": solve_mode_counts,
        "solver_name": box_solver_name,
        "solver_form": box_solver_form,
        "us_exact_minus_us_bounded_inf": us_exact_minus_us_bounded_inf,
        "xs_exact_minus_xs_bounded_inf": xs_exact_minus_xs_bounded_inf,
        "overall_summary": overall_summary,
        "per_input_rows": per_input_rows,
        "event_rows": event_rows,
        "per_step_rows": per_step_rows,
    }


def _physical_output_from_dev(
    y_dev: np.ndarray,
    y_ss_scaled: np.ndarray,
    data_min_y: np.ndarray,
    data_max_y: np.ndarray,
) -> np.ndarray:
    y_dev = np.asarray(y_dev, dtype=float)
    y_abs_scaled = y_dev + y_ss_scaled
    return reverse_min_max(y_abs_scaled, data_min_y, data_max_y)


def _physical_input_from_dev(
    u_dev: np.ndarray,
    u_ss_scaled: np.ndarray,
    data_min_u: np.ndarray,
    data_max_u: np.ndarray,
) -> np.ndarray:
    u_dev = np.asarray(u_dev, dtype=float)
    u_abs_scaled = u_dev + u_ss_scaled
    return reverse_min_max(u_abs_scaled, data_min_u, data_max_u)


def analyze_offsetfree_rollout(
    rollout: Dict[str, Any],
    system_data: Dict[str, Any],
    steady_states: Dict[str, Any],
    analysis_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = _merge_config(analysis_config)
    if not bool(config["enabled"]):
        raise ValueError("Steady-state debug analysis is disabled by configuration.")

    A = _as_float_array(system_data["A"], "system_data['A']", ndim=2)
    B = _as_float_array(system_data["B"], "system_data['B']", ndim=2)
    C = _as_float_array(system_data["C"], "system_data['C']", ndim=2)
    data_min = _as_float_array(system_data["data_min"], "system_data['data_min']", ndim=1)
    data_max = _as_float_array(system_data["data_max"], "system_data['data_max']", ndim=1)
    _validate_state_space(A, B, C)

    y_mpc = _as_float_array(rollout["y_mpc"], "rollout['y_mpc']", ndim=2)
    u_mpc = _as_float_array(rollout["u_mpc"], "rollout['u_mpc']", ndim=2)
    xhatdhat = _as_float_array(rollout["xhatdhat"], "rollout['xhatdhat']", ndim=2)
    y_sp = _as_float_array(rollout["y_sp"], "rollout['y_sp']", ndim=2)
    nFE = int(rollout["nFE"])
    delta_t = float(rollout.get("delta_t", 1.0))

    n_states = A.shape[0]
    n_outputs = C.shape[0]
    n_inputs = B.shape[1]

    if y_mpc.shape[0] != nFE + 1:
        raise ValueError(f"y_mpc must have nFE+1 rows, got {y_mpc.shape[0]} for nFE={nFE}.")
    if u_mpc.shape[0] != nFE:
        raise ValueError(f"u_mpc must have nFE rows, got {u_mpc.shape[0]} for nFE={nFE}.")
    if y_sp.shape[0] != nFE:
        raise ValueError(f"y_sp must have nFE rows, got {y_sp.shape[0]} for nFE={nFE}.")
    if xhatdhat.shape[1] < nFE:
        raise ValueError(f"xhatdhat must have at least nFE columns, got {xhatdhat.shape[1]} for nFE={nFE}.")
    if xhatdhat.shape[0] < n_states + n_outputs:
        raise ValueError(
            "xhatdhat does not contain enough augmented states for the legacy offset-free split."
        )

    u_scale_min = data_min[:n_inputs]
    u_scale_max = data_max[:n_inputs]
    y_min = data_min[n_inputs:]
    y_max = data_max[n_inputs:]

    u_ss_scaled = apply_min_max(steady_states["ss_inputs"], u_scale_min, u_scale_max)
    y_ss_scaled = apply_min_max(steady_states["y_ss"], y_min, y_max)
    if "b_min" not in system_data or "b_max" not in system_data:
        raise ValueError("system_data must include 'b_min' and 'b_max' for the box-constrained analysis.")
    u_box_min = _as_float_array(system_data["b_min"], "system_data['b_min']", ndim=1)
    u_box_max = _as_float_array(system_data["b_max"], "system_data['b_max']", ndim=1)

    y_current_phys = y_mpc[:-1].copy()
    y_after_step_phys = y_mpc[1:].copy()
    y_current_scaled_dev = apply_min_max(y_current_phys, y_min, y_max) - y_ss_scaled
    y_after_step_scaled_dev = apply_min_max(y_after_step_phys, y_min, y_max) - y_ss_scaled

    u_applied_phys = u_mpc.copy()
    u_applied_scaled_dev = apply_min_max(u_applied_phys, u_scale_min, u_scale_max) - u_ss_scaled

    xhat_current = xhatdhat[:n_states, :nFE].T.copy()
    dhat_current = xhatdhat[n_states : n_states + n_outputs, :nFE].T.copy()

    base_structure = build_legacy_ss_system(A, B, C, rank_tol=config["rank_tol"])
    reduced_info = compute_reduced_gain(
        A,
        B,
        C,
        cond_warn_threshold=config["cond_warn_threshold"],
        rank_tol=config["rank_tol"],
    )

    model_structure_row = {
        "n": int(n_states),
        "p": int(n_outputs),
        "q": int(n_inputs),
        "M_shape": str(base_structure["M_shape"]),
        "G_shape": "n/a" if reduced_info["G_shape"] is None else str(reduced_info["G_shape"]),
        "I_minus_A_invertible": bool(reduced_info["invertible_I_minus_A"]),
        "M_square": bool(base_structure["is_M_square"]),
        "G_square": bool(reduced_info["is_G_square"]),
        "rank_M": int(base_structure["rank_M"]),
        "rank_G": int(reduced_info["rank_G"]),
        "classification": base_structure["classification"],
    }
    linear_algebra_row = {
        "cond_I_minus_A": float(reduced_info["cond_I_minus_A"]),
        "cond_M": float(base_structure["cond_M"]),
        "cond_G": float(reduced_info["cond_G"]),
        "smallest_sv_M": float(base_structure["smallest_sv_M"]),
        "smallest_sv_G": float(reduced_info["smallest_sv_G"]),
        "configured_solver_mode": str(config["solver_mode"]),
    }
    reduced_gain = None
    if reduced_info["G"] is not None:
        reduced_gain = np.asarray(reduced_info["G"], dtype=float)

    x_s_store = np.zeros((nFE, n_states), dtype=float)
    u_s_store = np.zeros((nFE, n_inputs), dtype=float)
    d_s_store = np.zeros((nFE, n_outputs), dtype=float)
    y_s_store = np.zeros((nFE, n_outputs), dtype=float)
    u_s_phys = np.zeros((nFE, n_inputs), dtype=float)
    y_s_phys = np.zeros((nFE, n_outputs), dtype=float)
    rhs_output_store = np.zeros((nFE, n_outputs), dtype=float)
    residual_dyn_norm = np.zeros(nFE, dtype=float)
    residual_out_norm = np.zeros(nFE, dtype=float)
    residual_total_norm = np.zeros(nFE, dtype=float)
    exact_solution_flags = np.zeros(nFE, dtype=bool)
    used_lstsq_flags = np.zeros(nFE, dtype=bool)
    requested_fallback_flags = np.zeros(nFE, dtype=bool)
    solver_mode_used: List[str] = []
    rank_M_store = np.full(nFE, int(base_structure["rank_M"]), dtype=int)
    rank_G_store = np.full(nFE, int(reduced_info["rank_G"]), dtype=int)
    cond_M_store = np.full(nFE, float(base_structure["cond_M"]), dtype=float)
    cond_G_store = np.full(nFE, float(reduced_info["cond_G"]), dtype=float)
    cond_I_minus_A_store = np.full(nFE, float(reduced_info["cond_I_minus_A"]), dtype=float)
    residual_dyn_store = np.zeros((nFE, n_states), dtype=float)
    residual_out_store = np.zeros((nFE, n_outputs), dtype=float)
    residual_total_store = np.zeros((nFE, n_states + n_inputs), dtype=float)
    u_applied_minus_u_s = np.zeros((nFE, n_inputs), dtype=float)
    y_current_minus_y_s = np.zeros((nFE, n_outputs), dtype=float)
    y_s_minus_y_sp = np.zeros((nFE, n_outputs), dtype=float)
    xhat_minus_x_s = np.zeros((nFE, n_states), dtype=float)
    dhat_minus_d_s = np.zeros((nFE, n_outputs), dtype=float)
    u_applied_minus_u_s_phys = np.zeros((nFE, n_inputs), dtype=float)
    y_current_minus_y_s_phys = np.zeros((nFE, n_outputs), dtype=float)
    G_u_s_exact_store = np.full((nFE, n_outputs), np.nan, dtype=float)
    G_u_s_exact_minus_rhs_store = np.full((nFE, n_outputs), np.nan, dtype=float)

    step_rows: List[Dict[str, Any]] = []
    for step_idx in range(nFE):
        solve_info = solve_legacy_ss_exact(
            A,
            B,
            C,
            y_sp[step_idx, :],
            dhat_current[step_idx, :],
            solver_mode=config["solver_mode"],
            cond_warn_threshold=config["cond_warn_threshold"],
            residual_warn_threshold=config["residual_warn_threshold"],
            rank_tol=config["rank_tol"],
        )

        x_s_store[step_idx, :] = solve_info["x_s"]
        u_s_store[step_idx, :] = solve_info["u_s"]
        d_s_store[step_idx, :] = solve_info["d_s"]
        y_s_store[step_idx, :] = solve_info["y_s"]
        rhs_output_store[step_idx, :] = solve_info["rhs_output"]
        residual_dyn_norm[step_idx] = solve_info["residual_dyn_norm"]
        residual_out_norm[step_idx] = solve_info["residual_out_norm"]
        residual_total_norm[step_idx] = solve_info["residual_total_norm"]
        exact_solution_flags[step_idx] = solve_info["is_exact_solution"]
        used_lstsq_flags[step_idx] = solve_info["used_lstsq"]
        requested_fallback_flags[step_idx] = solve_info["requested_mode_fallback"]
        solver_mode_used.append(solve_info["solver_mode_used"])
        residual_dyn_store[step_idx, :] = solve_info["residual_dyn"]
        residual_out_store[step_idx, :] = solve_info["residual_out"]
        residual_total_store[step_idx, :] = solve_info["residual_total"]
        if reduced_gain is not None:
            G_u_s_exact_store[step_idx, :] = reduced_gain @ solve_info["u_s"]
            G_u_s_exact_minus_rhs_store[step_idx, :] = (
                G_u_s_exact_store[step_idx, :] - solve_info["rhs_output"]
            )

        u_s_phys[step_idx, :] = _physical_input_from_dev(
            solve_info["u_s"],
            u_ss_scaled,
            u_scale_min,
            u_scale_max,
        )
        y_s_phys[step_idx, :] = _physical_output_from_dev(
            solve_info["y_s"],
            y_ss_scaled,
            y_min,
            y_max,
        )

        u_applied_minus_u_s[step_idx, :] = u_applied_scaled_dev[step_idx, :] - solve_info["u_s"]
        y_current_minus_y_s[step_idx, :] = y_current_scaled_dev[step_idx, :] - solve_info["y_s"]
        y_s_minus_y_sp[step_idx, :] = solve_info["y_s"] - y_sp[step_idx, :]
        xhat_minus_x_s[step_idx, :] = xhat_current[step_idx, :] - solve_info["x_s"]
        dhat_minus_d_s[step_idx, :] = dhat_current[step_idx, :] - solve_info["d_s"]
        u_applied_minus_u_s_phys[step_idx, :] = u_applied_phys[step_idx, :] - u_s_phys[step_idx, :]
        y_current_minus_y_s_phys[step_idx, :] = y_current_phys[step_idx, :] - y_s_phys[step_idx, :]

        step_rows.append(
            {
                "k": int(step_idx),
                "residual_dyn_norm": float(solve_info["residual_dyn_norm"]),
                "residual_out_norm": float(solve_info["residual_out_norm"]),
                "residual_total_norm": float(solve_info["residual_total_norm"]),
                "exact_solution": bool(solve_info["is_exact_solution"]),
                "used_lstsq": bool(solve_info["used_lstsq"]),
                "solver_mode": solve_info["solver_mode_used"],
                "requested_mode_fallback": bool(solve_info["requested_mode_fallback"]),
                "u_applied_minus_u_s_norm": _norm(u_applied_minus_u_s[step_idx, :]),
                "u_applied_minus_u_s_phys_norm": _norm(u_applied_minus_u_s_phys[step_idx, :]),
                "y_current_minus_y_s_norm": _norm(y_current_minus_y_s[step_idx, :]),
                "y_current_minus_y_s_phys_norm": _norm(y_current_minus_y_s_phys[step_idx, :]),
                "y_s_minus_y_sp_norm": _norm(y_s_minus_y_sp[step_idx, :]),
                "xhat_minus_x_s_norm": _norm(xhat_minus_x_s[step_idx, :]),
                "dhat_minus_d_s_norm": _norm(dhat_minus_d_s[step_idx, :]),
                "rhs_output_norm": _norm(solve_info["rhs_output"]),
                "u_s_dev_norm": _norm(solve_info["u_s"]),
                "x_s_norm": _norm(solve_info["x_s"]),
                "reduced_rhs_exact_residual_norm": _norm(G_u_s_exact_minus_rhs_store[step_idx, :]),
            }
        )

    summary_stats = {
        "residual_dyn_norm": _summary_stat_block(residual_dyn_norm),
        "residual_out_norm": _summary_stat_block(residual_out_norm),
        "residual_total_norm": _summary_stat_block(residual_total_norm),
        "u_applied_minus_u_s_norm": _summary_stat_block(np.linalg.norm(u_applied_minus_u_s, axis=1)),
        "y_current_minus_y_s_norm": _summary_stat_block(np.linalg.norm(y_current_minus_y_s, axis=1)),
        "xhat_minus_x_s_norm": _summary_stat_block(np.linalg.norm(xhat_minus_x_s, axis=1)),
        "rhs_output_norm": _summary_stat_block(np.linalg.norm(rhs_output_store, axis=1)),
        "u_s_dev_norm": _summary_stat_block(np.linalg.norm(u_s_store, axis=1)),
        "x_s_norm": _summary_stat_block(np.linalg.norm(x_s_store, axis=1)),
        "reduced_rhs_exact_residual_norm": _summary_stat_block(
            np.linalg.norm(G_u_s_exact_minus_rhs_store, axis=1)
        ),
    }

    solver_mode_counts: Dict[str, int] = {}
    for mode_name in solver_mode_used:
        solver_mode_counts[mode_name] = solver_mode_counts.get(mode_name, 0) + 1

    change_indices = _setpoint_change_indices(y_sp)
    exact_eq_residual_state_inf = np.max(np.abs(residual_dyn_store), axis=1)
    exact_eq_residual_output_inf = np.max(np.abs(residual_out_store), axis=1)
    box_analysis = None
    if bool(config["enable_box_analysis"]):
        box_analysis = run_parallel_steady_state_box_analysis(
            A=A,
            B=B,
            C=C,
            y_sp=y_sp,
            dhat_current=dhat_current,
            x_s_exact=x_s_store,
            u_s_exact=u_s_store,
            d_s_exact=d_s_store,
            y_s_exact=y_s_store,
            exact_solution_flags=exact_solution_flags,
            exact_eq_residual_state_inf=exact_eq_residual_state_inf,
            exact_eq_residual_output_inf=exact_eq_residual_output_inf,
            u_box_min=u_box_min,
            u_box_max=u_box_max,
            setpoint_change_indices=change_indices,
            analysis_config=config,
        )
        u_box_min_phys = _physical_input_from_dev(u_box_min, u_ss_scaled, u_scale_min, u_scale_max)
        u_box_max_phys = _physical_input_from_dev(u_box_max, u_ss_scaled, u_scale_min, u_scale_max)
        box_analysis["u_box_min_phys"] = u_box_min_phys
        box_analysis["u_box_max_phys"] = u_box_max_phys
        box_analysis["u_s_bounded_phys"] = _physical_input_from_dev(
            box_analysis["u_s_bounded"], u_ss_scaled, u_scale_min, u_scale_max
        )
        box_analysis["y_s_bounded_phys"] = _physical_output_from_dev(
            box_analysis["y_s_bounded"], y_ss_scaled, y_min, y_max
        )
        box_analysis["u_s_exact_phys"] = _physical_input_from_dev(
            u_s_store, u_ss_scaled, u_scale_min, u_scale_max
        )
        box_analysis["y_s_exact_phys"] = _physical_output_from_dev(
            y_s_store, y_ss_scaled, y_min, y_max
        )
        box_analysis["u_applied_phys"] = u_applied_phys
        box_analysis["u_applied_dev"] = u_applied_scaled_dev
        box_analysis["dhat_current"] = dhat_current
        box_analysis["y_sp_phys"] = _physical_output_from_dev(y_sp, y_ss_scaled, y_min, y_max)
        if reduced_gain is not None:
            box_analysis["G_u_s_exact"] = G_u_s_exact_store.copy()
            box_analysis["G_u_s_exact_minus_rhs"] = G_u_s_exact_minus_rhs_store.copy()
            box_analysis["G_u_s_bounded"] = box_analysis["u_s_bounded"] @ reduced_gain.T
            box_analysis["G_u_s_bounded_minus_rhs"] = (
                box_analysis["G_u_s_bounded"] - rhs_output_store
            )
            summary_stats["reduced_rhs_bounded_residual_norm"] = _summary_stat_block(
                np.linalg.norm(box_analysis["G_u_s_bounded_minus_rhs"], axis=1)
            )
        for row, box_row in zip(step_rows, box_analysis["per_step_rows"]):
            row.update(
                {
                    "exact_solve_success": box_row["exact_solve_success"],
                    "exact_within_bounds": box_row["exact_within_bounds"],
                    "exact_bound_violation_inf": box_row["exact_bound_violation_inf"],
                    "exact_eq_residual_state_inf": box_row["exact_eq_residual_state_inf"],
                    "exact_eq_residual_output_inf": box_row["exact_eq_residual_output_inf"],
                    "bounded_solve_success": box_row["bounded_solve_success"],
                    "bounded_residual_norm": box_row["bounded_residual_norm"],
                    "box_solve_mode": box_row["solve_mode"],
                    "us_exact_minus_us_bounded_inf": float(
                        box_analysis["us_exact_minus_us_bounded_inf"][row["k"]]
                    ),
                    "xs_exact_minus_xs_bounded_inf": float(
                        box_analysis["xs_exact_minus_xs_bounded_inf"][row["k"]]
                    ),
                    "reduced_rhs_bounded_residual_norm": float(
                        _norm(box_analysis["G_u_s_bounded_minus_rhs"][row["k"], :])
                    )
                    if reduced_gain is not None
                    else float("nan"),
                }
            )

    summary_stats_row = {}
    for metric_name, stats in summary_stats.items():
        for stat_name, stat_value in stats.items():
            summary_stats_row[f"{metric_name}_{stat_name}"] = stat_value
    sampled_rows = step_rows[:: max(int(config["sample_table_stride"]), 1)]

    bundle: Dict[str, Any] = {
        "config": config,
        "case_name": config["case_name"],
        "nFE": int(nFE),
        "delta_t": float(delta_t),
        "steady_states": deepcopy(steady_states),
        "xhatdhat_full": xhatdhat.copy(),
        "model_structure": model_structure_row,
        "linear_algebra_summary": linear_algebra_row,
        "summary_stats": summary_stats,
        "summary_stats_row": summary_stats_row,
        "solver_mode_counts": solver_mode_counts,
        "setpoint_change_indices": change_indices,
        "sampled_step_rows": sampled_rows,
        "step_rows": step_rows,
        "y_sp_dev": y_sp.copy(),
        "y_sp_phys": _physical_output_from_dev(y_sp, y_ss_scaled, y_min, y_max),
        "y_current_phys": y_current_phys,
        "y_after_step_phys": y_after_step_phys,
        "y_current_dev": y_current_scaled_dev,
        "y_after_step_dev": y_after_step_scaled_dev,
        "u_applied_phys": u_applied_phys,
        "u_applied_dev": u_applied_scaled_dev,
        "xhat_current": xhat_current,
        "dhat_current": dhat_current,
        "x_s": x_s_store,
        "u_s_dev": u_s_store,
        "u_s_phys": u_s_phys,
        "G_matrix": None if reduced_gain is None else reduced_gain.copy(),
        "G_u_s_exact": G_u_s_exact_store,
        "G_u_s_exact_minus_rhs": G_u_s_exact_minus_rhs_store,
        "d_s": d_s_store,
        "y_s_dev": y_s_store,
        "y_s_phys": y_s_phys,
        "rhs_output": rhs_output_store,
        "residual_dyn": residual_dyn_store,
        "residual_out": residual_out_store,
        "residual_total": residual_total_store,
        "residual_dyn_norm": residual_dyn_norm,
        "residual_out_norm": residual_out_norm,
        "residual_total_norm": residual_total_norm,
        "exact_solution_flags": exact_solution_flags,
        "used_lstsq_flags": used_lstsq_flags,
        "requested_fallback_flags": requested_fallback_flags,
        "solver_mode_used": solver_mode_used,
        "rank_M": rank_M_store,
        "rank_G": rank_G_store,
        "cond_M": cond_M_store,
        "cond_G": cond_G_store,
        "cond_I_minus_A": cond_I_minus_A_store,
        "u_box_min": u_box_min,
        "u_box_max": u_box_max,
        "u_applied_minus_u_s": u_applied_minus_u_s,
        "u_applied_minus_u_s_phys": u_applied_minus_u_s_phys,
        "y_current_minus_y_s": y_current_minus_y_s,
        "y_current_minus_y_s_phys": y_current_minus_y_s_phys,
        "y_s_minus_y_sp": y_s_minus_y_sp,
        "xhat_minus_x_s": xhat_minus_x_s,
        "dhat_minus_d_s": dhat_minus_d_s,
        "exact_eq_residual_state_inf": exact_eq_residual_state_inf,
        "exact_eq_residual_output_inf": exact_eq_residual_output_inf,
        "box_analysis_enabled": bool(box_analysis is not None),
        "box_analysis": box_analysis,
        "time_step_axis": _step_axis(nFE, delta_t),
    }

    return bundle


def _save_outputs_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    n_outputs = bundle["y_current_phys"].shape[1]
    time_step = bundle["time_step_axis"]
    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.8 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(time_step, bundle["y_current_phys"][:, idx], linewidth=2.0, label="y")
        ax.step(time_step, bundle["y_sp_phys"][:, idx], where="post", linewidth=2.0, linestyle="--", label="y_sp")
        ax.plot(time_step, bundle["y_s_phys"][:, idx], linewidth=2.0, label="y_s")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.set_ylabel(f"output_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outputs_vs_targets.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.2 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(time_step, bundle["y_s_minus_y_sp"][:, idx], linewidth=2.0, label="y_s - y_sp")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"output_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "output_target_mismatch.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.2 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(time_step, bundle["y_current_minus_y_s_phys"][:, idx], linewidth=2.0, label="y - y_s")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"output_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "output_tracking_vs_ss.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_inputs_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    n_inputs = bundle["u_applied_phys"].shape[1]
    time_step = bundle["time_step_axis"]
    target_label = "u_s"
    target_phys = bundle["u_s_phys"]
    applied_minus_target_phys = bundle["u_applied_minus_u_s_phys"]
    target_dev = bundle["u_s_dev"]
    applied_minus_target_dev = bundle["u_applied_dev"] - target_dev
    if bundle.get("box_analysis_enabled"):
        box = bundle["box_analysis"]
        target_label = "u_s_bounded"
        target_phys = box["u_s_bounded_phys"]
        applied_minus_target_phys = bundle["u_applied_phys"] - target_phys
        target_dev = box["u_s_bounded"]
        applied_minus_target_dev = bundle["u_applied_dev"] - target_dev

    fig, axes = plt.subplots(n_inputs, 1, figsize=(10, 3.8 * n_inputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.step(time_step, bundle["u_applied_phys"][:, idx], where="post", linewidth=2.0, label="u_applied")
        ax.step(
            time_step,
            target_phys[:, idx],
            where="post",
            linewidth=2.0,
            linestyle="--",
            label=target_label,
        )
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"input_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inputs_vs_targets.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_inputs, 1, figsize=(10, 3.2 * n_inputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.step(
            time_step,
            applied_minus_target_phys[:, idx],
            where="post",
            linewidth=2.0,
            label=f"u_applied - {target_label}",
        )
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"input_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_target_mismatch.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_inputs, 1, figsize=(10, 3.8 * n_inputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.step(time_step, bundle["u_applied_dev"][:, idx], where="post", linewidth=2.0, label="u_applied_dev")
        ax.step(
            time_step,
            target_dev[:, idx],
            where="post",
            linewidth=2.0,
            linestyle="--",
            label=f"{target_label}_dev",
        )
        ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0, label="zero dev")
        if bundle.get("box_analysis_enabled"):
            ax.axhline(float(bundle["u_box_min"][idx]), color="tab:red", linestyle=":", linewidth=1.0, label="u_min_dev")
            ax.axhline(float(bundle["u_box_max"][idx]), color="tab:green", linestyle=":", linewidth=1.0, label="u_max_dev")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"input_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inputs_vs_targets_dev.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_inputs, 1, figsize=(10, 3.2 * n_inputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.step(
            time_step,
            applied_minus_target_dev[:, idx],
            where="post",
            linewidth=2.0,
            label=f"u_applied_dev - {target_label}_dev",
        )
        ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"input_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_target_mismatch_dev.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_states_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    n_states = bundle["xhat_current"].shape[1]
    time_step = bundle["time_step_axis"]
    fig, axes = plt.subplots(n_states, 1, figsize=(10, max(3.0 * n_states, 6.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(time_step, bundle["xhat_current"][:, idx], linewidth=2.0, label="xhat")
        ax.plot(time_step, bundle["x_s"][:, idx], linewidth=2.0, linestyle="--", label="x_s")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"x_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "states_vs_targets.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_states, 1, figsize=(10, max(3.0 * n_states, 6.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(time_step, bundle["xhat_minus_x_s"][:, idx], linewidth=2.0, label="xhat - x_s")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"x_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "state_target_mismatch.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_disturbance_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    n_outputs = bundle["dhat_current"].shape[1]
    time_step = bundle["time_step_axis"]
    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.8 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(time_step, bundle["dhat_current"][:, idx], linewidth=2.0, label="dhat")
        ax.plot(time_step, bundle["d_s"][:, idx], linewidth=2.0, linestyle="--", label="d_s")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"d_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disturbance_vs_targets.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.2 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(time_step, bundle["rhs_output"][:, idx], linewidth=2.0, label="y_sp - dhat")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"rhs_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disturbance_rhs.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_reduced_consistency_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    if bundle.get("G_matrix") is None:
        return

    n_outputs = bundle["rhs_output"].shape[1]
    time_step = bundle["time_step_axis"]
    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.6 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(time_step, bundle["rhs_output"][:, idx], linewidth=2.0, label="rhs_output")
        ax.plot(time_step, bundle["G_u_s_exact"][:, idx], linewidth=2.0, linestyle="--", label="G u_s_exact")
        if bundle.get("box_analysis_enabled") and "G_u_s_bounded" in bundle["box_analysis"]:
            ax.plot(
                time_step,
                bundle["box_analysis"]["G_u_s_bounded"][:, idx],
                linewidth=1.8,
                linestyle=":",
                label="G u_s_bounded",
            )
        ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"rhs_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reduced_rhs_vs_Gu.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.2 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(
            time_step,
            bundle["G_u_s_exact_minus_rhs"][:, idx],
            linewidth=2.0,
            label="G u_s_exact - rhs",
        )
        if bundle.get("box_analysis_enabled") and "G_u_s_bounded_minus_rhs" in bundle["box_analysis"]:
            ax.plot(
                time_step,
                bundle["box_analysis"]["G_u_s_bounded_minus_rhs"][:, idx],
                linewidth=1.8,
                linestyle="--",
                label="G u_s_bounded - rhs",
            )
        ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"rhs_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reduced_rhs_mismatch.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_tail_window_overview(bundle: Dict[str, Any], output_dir: str) -> None:
    tail_count = max(int(bundle["config"].get("tail_window_samples", 20)), 1)
    start_idx = max(int(bundle["nFE"]) - tail_count, 0)
    stop_idx = int(bundle["nFE"])
    if start_idx >= stop_idx:
        return

    k_axis = np.arange(start_idx, stop_idx, dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes = np.atleast_1d(axes)

    for idx in range(bundle["rhs_output"].shape[1]):
        axes[0].plot(k_axis, bundle["rhs_output"][start_idx:stop_idx, idx], marker="o", linewidth=1.8, label=f"rhs[{idx}]")
        if bundle.get("G_matrix") is not None:
            axes[0].plot(
                k_axis,
                bundle["G_u_s_exact"][start_idx:stop_idx, idx],
                marker="x",
                linewidth=1.6,
                linestyle="--",
                label=f"G u_s_exact[{idx}]",
            )
            if bundle.get("box_analysis_enabled") and "G_u_s_bounded" in bundle["box_analysis"]:
                axes[0].plot(
                    k_axis,
                    bundle["box_analysis"]["G_u_s_bounded"][start_idx:stop_idx, idx],
                    marker=".",
                    linewidth=1.4,
                    linestyle=":",
                    label=f"G u_s_bounded[{idx}]",
                )
    axes[0].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("rhs / Gu")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best")

    for idx in range(bundle["u_s_dev"].shape[1]):
        axes[1].plot(k_axis, bundle["u_applied_dev"][start_idx:stop_idx, idx], marker="o", linewidth=1.8, label=f"u_applied_dev[{idx}]")
        axes[1].plot(
            k_axis,
            bundle["u_s_dev"][start_idx:stop_idx, idx],
            marker="x",
            linewidth=1.6,
            linestyle="--",
            label=f"u_s_exact[{idx}]",
        )
        if bundle.get("box_analysis_enabled"):
            axes[1].plot(
                k_axis,
                bundle["box_analysis"]["u_s_bounded"][start_idx:stop_idx, idx],
                marker=".",
                linewidth=1.4,
                linestyle=":",
                label=f"u_s_bounded[{idx}]",
            )
            axes[1].axhline(float(bundle["u_box_min"][idx]), color="tab:red", linestyle=":", linewidth=1.0)
            axes[1].axhline(float(bundle["u_box_max"][idx]), color="tab:green", linestyle=":", linewidth=1.0)
    axes[1].axhline(0.0, color="0.4", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("u dev")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="best")

    axes[2].plot(
        k_axis,
        np.linalg.norm(bundle["rhs_output"][start_idx:stop_idx, :], axis=1),
        marker="o",
        linewidth=1.8,
        label="||rhs_output||",
    )
    axes[2].plot(
        k_axis,
        np.linalg.norm(bundle["u_s_dev"][start_idx:stop_idx, :], axis=1),
        marker="x",
        linewidth=1.6,
        linestyle="--",
        label="||u_s_exact||",
    )
    if bundle.get("box_analysis_enabled"):
        axes[2].plot(
            k_axis,
            np.linalg.norm(bundle["box_analysis"]["u_s_bounded"][start_idx:stop_idx, :], axis=1),
            marker=".",
            linewidth=1.4,
            linestyle=":",
            label="||u_s_bounded||",
        )
    axes[2].plot(
        k_axis,
        np.linalg.norm(bundle["x_s"][start_idx:stop_idx, :], axis=1),
        linewidth=1.8,
        label="||x_s||",
    )
    axes[2].plot(
        k_axis,
        np.linalg.norm(bundle["xhat_minus_x_s"][start_idx:stop_idx, :], axis=1),
        linewidth=1.8,
        linestyle="--",
        label="||xhat - x_s||",
    )
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend(loc="best")
    axes[2].set_ylabel("norm")
    axes[2].set_xlabel("sample k")

    fig.suptitle(f"Tail Window Overview (last {stop_idx - start_idx} samples)", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"tail_last_{stop_idx - start_idx}_samples_overview.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _save_residual_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    time_step = bundle["time_step_axis"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.plot(time_step, bundle["residual_dyn_norm"], linewidth=2.0, label="||residual_dyn||")
    ax.plot(time_step, bundle["residual_out_norm"], linewidth=2.0, label="||residual_out||")
    ax.plot(time_step, bundle["residual_total_norm"], linewidth=2.0, label="||residual_total||")
    fallback_steps = np.where(bundle["used_lstsq_flags"])[0]
    if fallback_steps.size:
        ax.scatter(
            fallback_steps.astype(float) * float(bundle["delta_t"]),
            bundle["residual_total_norm"][fallback_steps],
            s=30,
            color="tab:red",
            label="least-squares used",
            zorder=3,
        )
    _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
    ax.set_xlabel("time (h)")
    ax.set_ylabel("norm")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_norms.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_box_inputs_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    box = bundle["box_analysis"]
    n_inputs = bundle["u_applied_phys"].shape[1]
    time_step = bundle["time_step_axis"]
    fig, axes = plt.subplots(n_inputs, 1, figsize=(10, 3.8 * n_inputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.step(time_step, box["u_s_exact_phys"][:, idx], where="post", linewidth=2.0, label="u_s_exact")
        ax.step(time_step, box["u_s_bounded_phys"][:, idx], where="post", linewidth=2.0, linestyle="--", label="u_s_bounded")
        ax.step(time_step, bundle["u_applied_phys"][:, idx], where="post", linewidth=1.5, alpha=0.8, label="u_applied")
        ax.axhline(float(box["u_box_min_phys"][idx]), color="tab:red", linestyle=":", linewidth=1.5, label="u_min")
        ax.axhline(float(box["u_box_max_phys"][idx]), color="tab:green", linestyle=":", linewidth=1.5, label="u_max")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.set_ylabel(f"input_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "box_inputs_vs_bounds.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_box_outputs_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    box = bundle["box_analysis"]
    n_outputs = bundle["y_sp_phys"].shape[1]
    time_step = bundle["time_step_axis"]
    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.8 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.step(time_step, bundle["y_sp_phys"][:, idx], where="post", linewidth=2.0, label="y_sp")
        ax.plot(time_step, box["y_s_exact_phys"][:, idx], linewidth=2.0, label="y_s_exact")
        ax.plot(time_step, box["y_s_bounded_phys"][:, idx], linewidth=2.0, linestyle="--", label="y_s_bounded")
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.set_ylabel(f"output_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "box_outputs_exact_vs_bounded.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_outputs, 1, figsize=(10, 3.2 * n_outputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(
            time_step,
            box["y_s_bounded_phys"][:, idx] - bundle["y_sp_phys"][:, idx],
            linewidth=2.0,
            label="y_s_bounded - y_sp",
        )
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.set_ylabel(f"output_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "box_output_mismatch.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_box_residual_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    box = bundle["box_analysis"]
    time_step = bundle["time_step_axis"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.plot(time_step, bundle["exact_eq_residual_state_inf"], linewidth=2.0, label="exact state residual inf")
    ax.plot(time_step, bundle["exact_eq_residual_output_inf"], linewidth=2.0, label="exact output residual inf")
    ax.plot(time_step, box["bounded_residual_norm"], linewidth=2.0, label="bounded residual norm")
    _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
    ax.set_xlabel("time (h)")
    ax.set_ylabel("residual")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "box_residuals.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_box_constraint_activity_plot(bundle: Dict[str, Any], output_dir: str) -> None:
    box = bundle["box_analysis"]
    n_inputs = box["bounded_active_lower_mask"].shape[1]
    time_step = bundle["time_step_axis"]
    fig, axes = plt.subplots(n_inputs, 1, figsize=(10, 3.2 * n_inputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.step(
            time_step,
            box["bounded_active_lower_mask"][:, idx].astype(float),
            where="post",
            linewidth=2.0,
            label="lower active",
        )
        ax.step(
            time_step,
            box["bounded_active_upper_mask"][:, idx].astype(float),
            where="post",
            linewidth=2.0,
            linestyle="--",
            label="upper active",
        )
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"input_{idx}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "box_constraint_activity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_box_event_plots(bundle: Dict[str, Any], output_dir: str) -> None:
    box = bundle["box_analysis"]
    event_rows = box["event_rows"]
    if not event_rows:
        return

    event_dir = os.path.join(output_dir, "box_event_windows")
    os.makedirs(event_dir, exist_ok=True)
    unique_events = []
    seen = set()
    for row in event_rows:
        key = (int(row["event_anchor"]), str(row["event_kind"]))
        if key not in seen:
            seen.add(key)
            unique_events.append(key)
    unique_events = unique_events[: int(bundle["config"]["box_max_event_plots"])]

    y_sp_phys = box["y_sp_phys"]
    dhat_current = box["dhat_current"]
    for event_anchor, event_kind in unique_events:
        start = max(event_anchor - int(bundle["config"]["box_event_window_radius"]), 0)
        stop = min(event_anchor + int(bundle["config"]["box_event_window_radius"]) + 1, bundle["nFE"])
        local_time = bundle["time_step_axis"][start:stop]
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axes = np.atleast_1d(axes)
        for input_idx in range(box["u_s_exact_phys"].shape[1]):
            axes[0].step(
                local_time,
                box["u_s_exact_phys"][start:stop, input_idx],
                where="post",
                linewidth=2.0,
                label=f"u_s_exact[{input_idx}]",
            )
            axes[0].step(
                local_time,
                box["u_s_bounded_phys"][start:stop, input_idx],
                where="post",
                linewidth=2.0,
                linestyle="--",
                label=f"u_s_bounded[{input_idx}]",
            )
        axes[0].grid(True, linestyle="--", alpha=0.35)
        axes[0].legend(loc="best")
        for output_idx in range(y_sp_phys.shape[1]):
            axes[1].step(
                local_time,
                y_sp_phys[start:stop, output_idx],
                where="post",
                linewidth=2.0,
                label=f"y_sp[{output_idx}]",
            )
            axes[1].plot(
                local_time,
                box["y_s_bounded_phys"][start:stop, output_idx],
                linewidth=2.0,
                label=f"y_s_bounded[{output_idx}]",
            )
        axes[1].grid(True, linestyle="--", alpha=0.35)
        axes[1].legend(loc="best")
        axes[2].plot(local_time, box["bounded_residual_norm"][start:stop], linewidth=2.0, label="bounded residual")
        axes[2].plot(
            local_time,
            np.linalg.norm(box["y_s_bounded_phys"][start:stop, :] - y_sp_phys[start:stop, :], axis=1),
            linewidth=2.0,
            linestyle="--",
            label="bounded output error norm",
        )
        axes[2].grid(True, linestyle="--", alpha=0.35)
        axes[2].legend(loc="best")
        for output_idx in range(dhat_current.shape[1]):
            axes[3].plot(
                local_time,
                dhat_current[start:stop, output_idx],
                linewidth=2.0,
                label=f"dhat[{output_idx}]",
            )
        axes[3].grid(True, linestyle="--", alpha=0.35)
        axes[3].legend(loc="best")
        axes[3].set_xlabel("time (h)")
        fig.suptitle(f"Event {event_anchor} ({event_kind})", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(event_dir, f"event_{event_anchor:04d}_{event_kind}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_box_analysis_plots(bundle: Dict[str, Any], output_dir: str) -> None:
    if not bundle.get("box_analysis_enabled"):
        return
    _save_box_inputs_plot(bundle, output_dir)
    _save_box_outputs_plot(bundle, output_dir)
    _save_box_residual_plot(bundle, output_dir)
    _save_box_constraint_activity_plot(bundle, output_dir)
    _save_box_event_plots(bundle, output_dir)


def _save_plots(bundle: Dict[str, Any], output_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    _save_outputs_plot(bundle, output_dir)
    _save_inputs_plot(bundle, output_dir)
    _save_states_plot(bundle, output_dir)
    _save_disturbance_plot(bundle, output_dir)
    _save_reduced_consistency_plot(bundle, output_dir)
    _save_residual_plot(bundle, output_dir)
    _save_box_analysis_plots(bundle, output_dir)
    _save_tail_window_overview(bundle, output_dir)


def _build_summary_markdown(bundle: Dict[str, Any]) -> str:
    solver_rows = [
        {"solver_mode": key, "count": value}
        for key, value in sorted(bundle["solver_mode_counts"].items())
    ]
    markdown = [
        "# Offset-Free MPC Steady-State Debug Summary",
        "",
        f"- Case: `{bundle['case_name']}`",
        f"- Steps analyzed: `{bundle['nFE']}`",
        f"- Configured solver mode: `{bundle['config']['solver_mode']}`",
        f"- Least-squares used on `{int(np.sum(bundle['used_lstsq_flags']))}` steps",
        f"- Requested-mode fallbacks: `{int(np.sum(bundle['requested_fallback_flags']))}`",
        f"- Box analysis enabled: `{bool(bundle.get('box_analysis_enabled'))}`",
        "",
        "## Model Structure",
        "",
        _rows_to_markdown([bundle["model_structure"]], bundle["model_structure"].keys()),
        "",
        "## Global Linear Algebra Diagnostics",
        "",
        _rows_to_markdown([bundle["linear_algebra_summary"]], bundle["linear_algebra_summary"].keys()),
        "",
        "## Solver Mode Counts",
        "",
        _rows_to_markdown(solver_rows, ["solver_mode", "count"]),
        "",
        "## Summary Statistics",
        "",
        _rows_to_markdown([bundle["summary_stats_row"]], bundle["summary_stats_row"].keys()),
        "",
        "## Sampled Per-Step Diagnostics",
        "",
        _rows_to_markdown(
            bundle["sampled_step_rows"],
            [
                "k",
                "residual_dyn_norm",
                "residual_out_norm",
                "residual_total_norm",
                "exact_solution",
                "used_lstsq",
                "solver_mode",
                "u_applied_minus_u_s_norm",
                "y_current_minus_y_s_norm",
                "xhat_minus_x_s_norm",
            ],
        ),
        "",
    ]
    if bundle.get("box_analysis_enabled"):
        box = bundle["box_analysis"]
        box_mode_rows = [
            {"solve_mode": key, "count": value}
            for key, value in sorted(box["solve_mode_counts"].items())
        ]
        markdown.extend(
            [
                "## Box-Constrained Analysis Summary",
                "",
                _rows_to_markdown([box["overall_summary"]], box["overall_summary"].keys()),
                "",
                "## Box Solve Mode Counts",
                "",
                _rows_to_markdown(box_mode_rows, ["solve_mode", "count"]),
                "",
                "## Per-Input Bound Activity",
                "",
                _rows_to_markdown(
                    box["per_input_rows"],
                    [
                        "input_index",
                        "fraction_lower_bound_active",
                        "fraction_upper_bound_active",
                        "average_exact_violation_below_lower",
                        "average_exact_violation_above_upper",
                    ],
                ),
                "",
                "## Box Event Table",
                "",
                _rows_to_markdown(
                    box["event_rows"][: max(int(bundle["config"]["sample_table_stride"]), 1)],
                    [
                        "event_kind",
                        "event_anchor",
                        "k",
                        "solve_mode",
                        "exact_eq_residual_state_inf",
                        "exact_eq_residual_output_inf",
                        "bounded_residual_norm",
                        "dhat_delta_inf",
                    ],
                ),
                "",
            ]
        )
    return "\n".join(markdown)


def save_offsetfree_ss_debug_artifacts(
    bundle: Dict[str, Any],
    directory: Optional[str] = None,
    prefix_name: str = "mpc_offsetfree_steady_state_debug",
    save_plots: Optional[bool] = None,
    save_csv: Optional[bool] = None,
) -> str:
    config = bundle.get("config", {})
    if directory is None:
        directory = os.path.join(os.getcwd(), "Data")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    if save_csv is None:
        save_csv = bool(config.get("save_csv", True))
    if save_plots is None:
        save_plots = bool(config.get("save_plots", True))

    with open(os.path.join(out_dir, "bundle.pkl"), "wb") as file:
        pickle.dump(bundle, file)

    if save_csv:
        _write_csv(os.path.join(out_dir, "model_structure.csv"), [bundle["model_structure"]])
        _write_csv(os.path.join(out_dir, "linear_algebra_summary.csv"), [bundle["linear_algebra_summary"]])
        _write_csv(os.path.join(out_dir, "step_table.csv"), bundle["step_rows"])
        _write_csv(os.path.join(out_dir, "summary_stats.csv"), [bundle["summary_stats_row"]])
        if bundle.get("box_analysis_enabled"):
            box = bundle["box_analysis"]
            _write_csv(os.path.join(out_dir, "box_overall_summary.csv"), [box["overall_summary"]])
            _write_csv(os.path.join(out_dir, "box_per_input_activity.csv"), box["per_input_rows"])
            _write_csv(os.path.join(out_dir, "box_event_table.csv"), box["event_rows"])

    summary_md = _build_summary_markdown(bundle)
    with open(os.path.join(out_dir, "analysis_summary.md"), "w", encoding="utf-8") as file:
        file.write(summary_md)

    if HAS_PANDAS and save_csv:
        pd.DataFrame(bundle["step_rows"]).to_pickle(os.path.join(out_dir, "step_table.pkl"))
        if bundle.get("box_analysis_enabled"):
            box = bundle["box_analysis"]
            pd.DataFrame(box["event_rows"]).to_pickle(os.path.join(out_dir, "box_event_table.pkl"))

    if save_plots:
        _save_plots(bundle, out_dir)

    return out_dir


def run_synthetic_smoke_checks() -> Dict[str, Dict[str, Any]]:
    exact_case = solve_legacy_ss_exact(
        A=np.array([[0.5]], dtype=float),
        B=np.array([[1.0]], dtype=float),
        C=np.array([[1.0]], dtype=float),
        y_sp_k=np.array([0.2], dtype=float),
        d_hat_k=np.array([0.1], dtype=float),
        solver_mode="auto",
    )
    fallback_case = solve_legacy_ss_exact(
        A=np.array([[1.0]], dtype=float),
        B=np.array([[0.0]], dtype=float),
        C=np.array([[1.0]], dtype=float),
        y_sp_k=np.array([0.2], dtype=float),
        d_hat_k=np.array([0.0], dtype=float),
        solver_mode="auto",
    )
    results = {
        "exact_case": {
            "solver_mode_used": exact_case["solver_mode_used"],
            "used_lstsq": bool(exact_case["used_lstsq"]),
            "is_exact_solution": bool(exact_case["is_exact_solution"]),
            "residual_total_norm": float(exact_case["residual_total_norm"]),
        },
        "fallback_case": {
            "solver_mode_used": fallback_case["solver_mode_used"],
            "used_lstsq": bool(fallback_case["used_lstsq"]),
            "is_exact_solution": bool(fallback_case["is_exact_solution"]),
            "residual_total_norm": float(fallback_case["residual_total_norm"]),
        },
    }
    if HAS_SCIPY:
        box_case = solve_bounded_steady_state_least_squares(
            A=np.array([[0.5]], dtype=float),
            B=np.array([[1.0]], dtype=float),
            C=np.array([[1.0]], dtype=float),
            y_sp_k=np.array([0.2], dtype=float),
            d_hat_k=np.array([0.0], dtype=float),
            u_min=np.array([-0.05], dtype=float),
            u_max=np.array([0.05], dtype=float),
        )
        results["box_case"] = {
            "solve_success": bool(box_case["solve_success"]),
            "solve_form": str(box_case["solve_form"]),
            "residual_norm": float(box_case["residual_norm"]),
        }
    else:
        results["box_case"] = {
            "solve_success": False,
            "solve_form": "unavailable",
            "residual_norm": float("nan"),
        }
    return results
