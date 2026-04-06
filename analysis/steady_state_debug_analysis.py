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

from utils.scaling_helpers import apply_min_max, reverse_min_max


DEFAULT_ANALYSIS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "solver_mode": "auto",
    "cond_warn_threshold": 1.0e8,
    "residual_warn_threshold": 1.0e-8,
    "rank_tol": None,
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

    u_min = data_min[:n_inputs]
    u_max = data_max[:n_inputs]
    y_min = data_min[n_inputs:]
    y_max = data_max[n_inputs:]

    u_ss_scaled = apply_min_max(steady_states["ss_inputs"], u_min, u_max)
    y_ss_scaled = apply_min_max(steady_states["y_ss"], y_min, y_max)

    y_current_phys = y_mpc[:-1].copy()
    y_after_step_phys = y_mpc[1:].copy()
    y_current_scaled_dev = apply_min_max(y_current_phys, y_min, y_max) - y_ss_scaled
    y_after_step_scaled_dev = apply_min_max(y_after_step_phys, y_min, y_max) - y_ss_scaled

    u_applied_phys = u_mpc.copy()
    u_applied_scaled_dev = apply_min_max(u_applied_phys, u_min, u_max) - u_ss_scaled

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

        u_s_phys[step_idx, :] = _physical_input_from_dev(
            solve_info["u_s"],
            u_ss_scaled,
            u_min,
            u_max,
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
            }
        )

    summary_stats = {
        "residual_dyn_norm": _summary_stat_block(residual_dyn_norm),
        "residual_out_norm": _summary_stat_block(residual_out_norm),
        "residual_total_norm": _summary_stat_block(residual_total_norm),
        "u_applied_minus_u_s_norm": _summary_stat_block(np.linalg.norm(u_applied_minus_u_s, axis=1)),
        "y_current_minus_y_s_norm": _summary_stat_block(np.linalg.norm(y_current_minus_y_s, axis=1)),
        "xhat_minus_x_s_norm": _summary_stat_block(np.linalg.norm(xhat_minus_x_s, axis=1)),
    }

    summary_stats_row = {}
    for metric_name, stats in summary_stats.items():
        for stat_name, stat_value in stats.items():
            summary_stats_row[f"{metric_name}_{stat_name}"] = stat_value

    solver_mode_counts: Dict[str, int] = {}
    for mode_name in solver_mode_used:
        solver_mode_counts[mode_name] = solver_mode_counts.get(mode_name, 0) + 1

    change_indices = _setpoint_change_indices(y_sp)
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
        "u_applied_minus_u_s": u_applied_minus_u_s,
        "u_applied_minus_u_s_phys": u_applied_minus_u_s_phys,
        "y_current_minus_y_s": y_current_minus_y_s,
        "y_current_minus_y_s_phys": y_current_minus_y_s_phys,
        "y_s_minus_y_sp": y_s_minus_y_sp,
        "xhat_minus_x_s": xhat_minus_x_s,
        "dhat_minus_d_s": dhat_minus_d_s,
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
    fig, axes = plt.subplots(n_inputs, 1, figsize=(10, 3.8 * n_inputs), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.step(time_step, bundle["u_applied_phys"][:, idx], where="post", linewidth=2.0, label="u_applied")
        ax.step(time_step, bundle["u_s_phys"][:, idx], where="post", linewidth=2.0, linestyle="--", label="u_s")
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
            bundle["u_applied_minus_u_s_phys"][:, idx],
            where="post",
            linewidth=2.0,
            label="u_applied - u_s",
        )
        _append_vertical_lines(ax, bundle["setpoint_change_indices"], bundle["delta_t"])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        ax.set_ylabel(f"input_{idx}")
    axes[-1].set_xlabel("time (h)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_target_mismatch.png"), dpi=300, bbox_inches="tight")
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


def _save_plots(bundle: Dict[str, Any], output_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    _save_outputs_plot(bundle, output_dir)
    _save_inputs_plot(bundle, output_dir)
    _save_states_plot(bundle, output_dir)
    _save_disturbance_plot(bundle, output_dir)
    _save_residual_plot(bundle, output_dir)


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

    summary_md = _build_summary_markdown(bundle)
    with open(os.path.join(out_dir, "analysis_summary.md"), "w", encoding="utf-8") as file:
        file.write(summary_md)

    if HAS_PANDAS and save_csv:
        pd.DataFrame(bundle["step_rows"]).to_pickle(os.path.join(out_dir, "step_table.pkl"))

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
    return {
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
