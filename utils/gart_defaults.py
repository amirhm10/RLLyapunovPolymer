from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from Lyapunov.gart_lmpc import GARTMPCConfig
from Lyapunov.gart_target import CertifiedDisturbanceConfig, GARTTargetConfig, GARTTargetResult, jsonable
from utils.polymer_td3_defaults import DEFAULT_U_MAX_PHYS, DEFAULT_U_MIN_PHYS


GART_FINAL_RHO_LYAP = 0.99
GART_FINAL_LYAP_EPS = 1.0e-3
GART_FINAL_SLACK_PENALTY = 1.0e6
GART_FINAL_MPC_OBJECTIVE = "raw"
GART_FINAL_LYAPUNOV_MODE = "hard"

GART_INITIAL_DEFAULTS: dict[str, Any] = {
    "input_headroom_frac": 0.03,
    "alpha_terminal_min": 1.0e-8,
    "rho": GART_FINAL_RHO_LYAP,
    "eps": GART_FINAL_LYAP_EPS,
    "primary_tol_abs": 1.0e-8,
    "primary_tol_rel": 1.0e-6,
    "contraction_margin_tol": 1.0e-8,
    "alpha_d": 0.2,
    "alpha_d_slow": 0.02,
    "d_rate_scale": 1.0,
    "freeze_on_bad_innovation": False,
    "adaptive_rate_enabled": False,
    "adaptive_rate_trust_radius": None,
    "adaptive_rate_min_scale": 0.10,
    "adaptive_rate_eps": 1.0e-8,
    "eta_y": 0.0,
    "eta_u": 0.0,
    "target_term_gate_enabled": True,
    "target_term_gate_delta_y": 0.5,
    "target_term_gate_min_alpha": 0.5,
    "target_term_gate_disable_on_hold": True,
    "eta_y_when_gated": None,
    "eta_u_when_gated": None,
    "slack_penalty": GART_FINAL_SLACK_PENALTY,
}

GART_FINAL_TARGET_OVERRIDES: dict[str, Any] = {
    "disable_u_mid_tiebreak": True,
    "disable_x_smoothing": True,
    "disable_y_smoothing": True,
    "input_headroom_frac": 0.05,
    "primary_tol_rel": 1.0e-4,
    "dx_s_max_abs": 0.05,
    "du_s_max_abs": [0.2, 0.2],
    "dy_s_max_abs": 0.25,
    "d_rate_scale": 0.25,
    "alpha_d": 0.05,
    "W_u_smooth_diag": [2.0, 2.0],
    "Wy_diag": [1.0, 1.0],
    "adaptive_rate_enabled": False,
}

GART_FINAL_TARGET_CONFIG_OVERRIDES: dict[str, Any] = {
    **GART_FINAL_TARGET_OVERRIDES,
    "rho": GART_FINAL_RHO_LYAP,
    "eps": GART_FINAL_LYAP_EPS,
}


def _as_vector(value: Any, *, size: int | None = None, default: float | None = None) -> np.ndarray:
    if value is None:
        if size is None or default is None:
            raise ValueError("value is None and no default size/value was provided.")
        return np.full(size, float(default), dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"Expected vector of length {size}, got {arr.size}.")
    return arr.copy()


def _finite_quantile(values: list[np.ndarray], q: float, size: int) -> np.ndarray | None:
    if not values:
        return None
    try:
        arr = np.vstack([np.asarray(v, dtype=float).reshape(-1, size) for v in values])
    except Exception:
        return None
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.size == 0:
        return None
    return np.quantile(arr, q, axis=0)


def _empty_quantiles() -> dict[str, np.ndarray | None]:
    return {
        "d_q005": None,
        "d_q995": None,
        "dd_abs_q95": None,
        "dy_abs_q95": None,
        "du_abs_q95": None,
        "dx_abs_q95": None,
    }


def _npz_arrays(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return list(root.rglob("*.npz"))


def _subsample_rows(arr: np.ndarray, max_rows: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    max_rows = max(int(max_rows), 1)
    if arr.ndim == 0 or arr.shape[0] <= max_rows:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_rows).astype(int)
    return arr[idx]


def _collect_result_quantiles(
    results_roots: list[str] | None,
    n_x: int,
    n_y: int,
    n_u: int,
    *,
    max_result_files: int = 3,
    max_npz_bytes: int = 100_000_000,
    max_rows_per_array: int = 2000,
) -> dict[str, np.ndarray | None]:
    if not results_roots:
        return _empty_quantiles()
    roots = [Path(path) for path in results_roots]
    d_samples: list[np.ndarray] = []
    dd_samples: list[np.ndarray] = []
    dy_samples: list[np.ndarray] = []
    du_samples: list[np.ndarray] = []
    dx_samples: list[np.ndarray] = []
    files_read = 0

    for root in roots:
        for npz_path in _npz_arrays(root):
            if files_read >= int(max_result_files):
                break
            try:
                if npz_path.stat().st_size > int(max_npz_bytes):
                    continue
            except OSError:
                continue
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    if "xhatdhat" in data:
                        xhatdhat = np.asarray(data["xhatdhat"], dtype=float)
                        if xhatdhat.ndim == 2 and n_y <= min(xhatdhat.shape):
                            d_trace = xhatdhat[-n_y:, :].T if xhatdhat.shape[0] >= n_x + n_y else xhatdhat[:, -n_y:]
                            d_trace = _subsample_rows(d_trace, max_rows_per_array)
                            d_samples.append(d_trace)
                            if d_trace.shape[0] > 1:
                                dd_samples.append(np.diff(d_trace, axis=0))
                    if "y_target_store" in data:
                        y_trace = np.asarray(data["y_target_store"], dtype=float).reshape(-1, n_y)
                        y_trace = _subsample_rows(y_trace, max_rows_per_array)
                        if y_trace.shape[0] > 1:
                            dy_samples.append(np.diff(y_trace, axis=0))
                    if "u_target_dev_store" in data:
                        u_trace = np.asarray(data["u_target_dev_store"], dtype=float).reshape(-1, n_u)
                        u_trace = _subsample_rows(u_trace, max_rows_per_array)
                        if u_trace.shape[0] > 1:
                            du_samples.append(np.diff(u_trace, axis=0))
                    if "x_target_store" in data:
                        x_trace = np.asarray(data["x_target_store"], dtype=float).reshape(-1, n_x)
                        x_trace = _subsample_rows(x_trace, max_rows_per_array)
                        if x_trace.shape[0] > 1:
                            dx_samples.append(np.diff(x_trace, axis=0))
                    files_read += 1
            except Exception:
                continue

    return {
        "d_q005": _finite_quantile(d_samples, 0.005, n_y),
        "d_q995": _finite_quantile(d_samples, 0.995, n_y),
        "dd_abs_q95": _finite_quantile([np.abs(v) for v in dd_samples], 0.95, n_y),
        "dy_abs_q95": _finite_quantile([np.abs(v) for v in dy_samples], 0.95, n_y),
        "du_abs_q95": _finite_quantile([np.abs(v) for v in du_samples], 0.95, n_u),
        "dx_abs_q95": _finite_quantile([np.abs(v) for v in dx_samples], 0.95, n_x),
    }


def discover_gart_case_values(
    system_data: dict[str, Any],
    setup: Any,
    *,
    results_roots: list[str] | None = None,
    max_result_files: int = 3,
    max_npz_bytes: int = 100_000_000,
    max_rows_per_array: int = 2000,
) -> dict[str, Any]:
    A_aug = np.asarray(system_data["A_aug"], dtype=float)
    B_aug = np.asarray(system_data["B_aug"], dtype=float)
    C_aug = np.asarray(system_data["C_aug"], dtype=float)
    n_y = int(C_aug.shape[0])
    n_u = int(B_aug.shape[1])
    n_x = int(A_aug.shape[0] - n_y)

    u_dev_min = _as_vector(system_data.get("b_min"), size=n_u)
    u_dev_max = _as_vector(system_data.get("b_max"), size=n_u)
    min_max_dict = dict(system_data.get("min_max_dict", {}))

    x_min = _as_vector(min_max_dict.get("x_min"), size=A_aug.shape[0], default=-1.0)
    x_max = _as_vector(min_max_dict.get("x_max"), size=A_aug.shape[0], default=1.0)
    d_range_min = x_min[-n_y:]
    d_range_max = x_max[-n_y:]
    if np.any(d_range_min >= d_range_max):
        d_range_min = -0.1 * np.ones(n_y)
        d_range_max = 0.1 * np.ones(n_y)

    y_sp_min = _as_vector(min_max_dict.get("y_sp_min"), size=n_y, default=-1.0)
    y_sp_max = _as_vector(min_max_dict.get("y_sp_max"), size=n_y, default=1.0)
    if np.any(y_sp_min >= y_sp_max):
        y_sp_min = -np.ones(n_y)
        y_sp_max = np.ones(n_y)

    quant = _collect_result_quantiles(
        results_roots,
        n_x,
        n_y,
        n_u,
        max_result_files=max_result_files,
        max_npz_bytes=max_npz_bytes,
        max_rows_per_array=max_rows_per_array,
    )
    d_min = d_range_min.copy()
    d_max = d_range_max.copy()
    if quant["d_q005"] is not None and quant["d_q995"] is not None:
        pad = 0.05 * np.maximum(d_range_max - d_range_min, 1.0e-8)
        empirical_min = quant["d_q005"] - pad
        empirical_max = quant["d_q995"] + pad
        candidate_min = np.maximum(d_range_min, empirical_min)
        candidate_max = np.minimum(d_range_max, empirical_max)
        valid = candidate_min < candidate_max
        d_min[valid] = candidate_min[valid]
        d_max[valid] = candidate_max[valid]

    d_width = np.maximum(d_max - d_min, 1.0e-8)
    d_rate_default = 0.02 * d_width
    if quant["dd_abs_q95"] is not None:
        d_rate_max = np.maximum(quant["dd_abs_q95"], 0.01 * d_width)
    else:
        d_rate_max = d_rate_default

    y_width = np.maximum(y_sp_max - y_sp_min, 1.0e-8)
    u_width = np.maximum(u_dev_max - u_dev_min, 1.0e-8)
    dy_s_max = quant["dy_abs_q95"] if quant["dy_abs_q95"] is not None else 0.05 * y_width
    du_s_max = quant["du_abs_q95"] if quant["du_abs_q95"] is not None else 0.05 * u_width
    dx_s_max = quant["dx_abs_q95"] if quant["dx_abs_q95"] is not None else 0.05 * np.maximum(x_max[:n_x] - x_min[:n_x], 1.0e-8)
    dy_s_max = np.maximum(dy_s_max, 0.02 * y_width)
    du_s_max = np.maximum(du_s_max, 0.02 * u_width)

    steady_states = getattr(setup, "steady_states", None)
    if steady_states is None and isinstance(setup, dict):
        steady_states = setup.get("steady_states")

    return {
        "n_x": n_x,
        "n_y": n_y,
        "n_u": n_u,
        "u_dev_min": u_dev_min,
        "u_dev_max": u_dev_max,
        "u_min_phys": DEFAULT_U_MIN_PHYS.copy(),
        "u_max_phys": DEFAULT_U_MAX_PHYS.copy(),
        "y_sp_min": y_sp_min,
        "y_sp_max": y_sp_max,
        "d_min": d_min,
        "d_max": d_max,
        "d_rate_max": d_rate_max,
        "dy_s_max": dy_s_max,
        "du_s_max": du_s_max,
        "dx_s_max": dx_s_max,
        "Wy_diag": np.array([5.0, 1.0], dtype=float)[:n_y] if n_y <= 2 else np.ones(n_y),
        "Q_raw_diag": np.array([5.0, 1.0], dtype=float)[:n_y] if n_y <= 2 else np.ones(n_y),
        "Q_target_diag": np.array([5.0, 1.0], dtype=float)[:n_y] if n_y <= 2 else np.ones(n_y),
        "R_us_diag": np.ones(n_u, dtype=float),
        "Rdu_diag": np.ones(n_u, dtype=float),
        "steady_states": steady_states,
        "quantiles": quant,
    }


def make_gart_target_config(values: dict[str, Any], **overrides: Any) -> GARTTargetConfig:
    cfg = dict(GART_INITIAL_DEFAULTS)
    cfg.update(overrides)
    dy_s_max = np.asarray(values["dy_s_max"], dtype=float).copy()
    du_s_max = np.asarray(values["du_s_max"], dtype=float).copy()
    dx_s_max = np.asarray(values["dx_s_max"], dtype=float).copy()
    du_template = np.asarray(values["du_s_max"], dtype=float)
    dx_template = np.asarray(values["dx_s_max"], dtype=float)
    dy_template = np.asarray(values["dy_s_max"], dtype=float)

    def _diag_override(key: str, *, size: int, default: float, disable_key: str | None = None) -> np.ndarray:
        raw = cfg.get(key)
        if raw is None:
            arr = np.full(size, float(default), dtype=float)
        else:
            vec = np.asarray(raw, dtype=float).reshape(-1)
            if vec.size == 1:
                arr = np.full(size, float(vec.item()), dtype=float)
            elif vec.size == size:
                arr = vec.copy()
            else:
                raise ValueError(f"{key} must be scalar or length {size}, got length {vec.size}.")
        if disable_key is not None and bool(cfg.get(disable_key, False)):
            arr = np.zeros(size, dtype=float)
        return arr

    def _diag_config_or_base(key: str, base: np.ndarray) -> np.ndarray:
        base = np.asarray(base, dtype=float).reshape(-1)
        raw = cfg.get(key)
        if raw is None:
            return base.copy()
        vec = np.asarray(raw, dtype=float).reshape(-1)
        if vec.size == 1:
            return np.full(base.size, float(vec.item()), dtype=float)
        if vec.size == base.size:
            return vec.copy()
        raise ValueError(f"{key} must be scalar or length {base.size}, got length {vec.size}.")

    if bool(cfg.get("disable_dy_rate", False)):
        dy_s_max = None
    else:
        dy_override = cfg.get("dy_s_max_abs", cfg.get("dy_s_max_override"))
        if dy_override is not None:
            dy_vec = np.asarray(dy_override, dtype=float).reshape(-1)
            if dy_vec.size == 1:
                dy_s_max = np.full(dy_template.size, float(dy_vec.item()), dtype=float)
            elif dy_vec.size == dy_template.size:
                dy_s_max = dy_vec.copy()
            else:
                raise ValueError(f"dy_s_max_abs must be scalar or length {dy_template.size}, got length {dy_vec.size}.")
        else:
            dy_s_max = float(cfg.get("dy_rate_scale", 1.0)) * dy_s_max
    if bool(cfg.get("disable_du_rate", False)):
        du_s_max = None
    else:
        du_override = cfg.get("du_s_max_abs", cfg.get("du_s_max_override"))
        if du_override is not None:
            du_vec = np.asarray(du_override, dtype=float).reshape(-1)
            if du_vec.size == 1:
                du_s_max = np.full(du_template.size, float(du_vec.item()), dtype=float)
            elif du_vec.size == du_template.size:
                du_s_max = du_vec.copy()
            else:
                raise ValueError(f"du_s_max_abs must be scalar or length {du_template.size}, got length {du_vec.size}.")
        else:
            du_s_max = float(cfg.get("du_rate_scale", 1.0)) * du_s_max
    if bool(cfg.get("disable_dx_rate", False)):
        dx_s_max = None
    else:
        dx_override = cfg.get("dx_s_max_abs", cfg.get("dx_s_max_override"))
        if dx_override is not None:
            dx_vec = np.asarray(dx_override, dtype=float).reshape(-1)
            if dx_vec.size == 1:
                dx_s_max = np.full(dx_template.size, float(dx_vec.item()), dtype=float)
            elif dx_vec.size == dx_template.size:
                dx_s_max = dx_vec.copy()
            else:
                raise ValueError(f"dx_s_max_abs must be scalar or length {dx_template.size}, got length {dx_vec.size}.")
        else:
            dx_s_max = float(cfg.get("dx_rate_scale", 1.0)) * dx_s_max
    input_headroom_frac = cfg.get("input_headroom_frac")
    d_rate_base = np.asarray(values["d_rate_max"], dtype=float).copy()
    d_rate_scale = np.asarray(cfg.get("d_rate_scale", 1.0), dtype=float).reshape(-1)
    if d_rate_scale.size == 1:
        d_rate_max = float(d_rate_scale.item()) * d_rate_base
    elif d_rate_scale.size == d_rate_base.size:
        d_rate_max = d_rate_scale * d_rate_base
    else:
        raise ValueError(f"d_rate_scale must be scalar or length {d_rate_base.size}, got length {d_rate_scale.size}.")
    disturbance = CertifiedDisturbanceConfig(
        alpha_d=float(cfg["alpha_d"]),
        alpha_d_slow=float(cfg["alpha_d_slow"]),
        d_rate_max=d_rate_max,
        d_min=np.asarray(values["d_min"], dtype=float).copy(),
        d_max=np.asarray(values["d_max"], dtype=float).copy(),
        innovation_gate=cfg.get("innovation_gate"),
        innovation_norm=str(cfg.get("innovation_norm", "inf")),
        freeze_on_bad_innovation=bool(cfg.get("freeze_on_bad_innovation", False)),
        adaptive_rate_enabled=bool(cfg.get("adaptive_rate_enabled", False)),
        adaptive_rate_trust_radius=cfg.get("adaptive_rate_trust_radius"),
        adaptive_rate_min_scale=float(cfg.get("adaptive_rate_min_scale", 0.10)),
        adaptive_rate_eps=float(cfg.get("adaptive_rate_eps", 1.0e-8)),
    )
    return GARTTargetConfig(
        disturbance=disturbance,
        input_headroom_frac=float(0.03 if input_headroom_frac is None else input_headroom_frac),
        output_headroom_frac=float(cfg.get("output_headroom_frac", 0.0)),
        alpha_terminal_min=float(cfg["alpha_terminal_min"]),
        dy_s_max=None if dy_s_max is None else dy_s_max.copy(),
        du_s_max=None if du_s_max is None else du_s_max.copy(),
        dx_s_max=None if dx_s_max is None else dx_s_max.copy(),
        primary_tol_abs=float(cfg["primary_tol_abs"]),
        primary_tol_rel=float(cfg["primary_tol_rel"]),
        Wy_diag=_diag_config_or_base("Wy_diag", np.asarray(values["Wy_diag"], dtype=float)),
        W_u_smooth_diag=_diag_override("W_u_smooth_diag", size=du_template.size, default=1.0, disable_key="disable_u_smoothing"),
        W_x_smooth_diag=_diag_override("W_x_smooth_diag", size=dx_template.size, default=0.01, disable_key="disable_x_smoothing"),
        W_y_smooth_diag=_diag_override("W_y_smooth_diag", size=dy_template.size, default=1.0, disable_key="disable_y_smoothing"),
        W_u_mid_diag=_diag_override("W_u_mid_diag", size=du_template.size, default=0.01, disable_key="disable_u_mid_tiebreak"),
        rho=float(cfg["rho"]),
        eps=float(cfg["eps"]),
        contraction_margin_tol=float(cfg["contraction_margin_tol"]),
        require_contraction_probe=bool(cfg.get("require_contraction_probe", True)),
        contraction_probe_log_only=bool(cfg.get("contraction_probe_log_only", False)),
        governor_enabled=bool(cfg.get("governor_enabled", True)),
        governor_grid=tuple(cfg.get("governor_grid", (1.0, 0.75, 0.5, 0.25, 0.0))),
        governor_bisect_iters=int(cfg.get("governor_bisect_iters", 8)),
        target_exact_tol=float(cfg.get("target_exact_tol", 1.0e-6)),
        target_good_tol=float(cfg.get("target_good_tol", 0.1)),
        target_acceptable_tol=float(cfg.get("target_acceptable_tol", 0.5)),
        margin_candidate_search_enabled=bool(cfg.get("margin_candidate_search_enabled", False)),
        margin_candidate_search_step_frac=float(cfg.get("margin_candidate_search_step_frac", 0.01)),
        solver_pref=cfg.get("solver_pref"),
    )


def make_gart_mpc_config(values: dict[str, Any], *, objective: str = "raw", lyapunov_mode: str = "hard", **overrides: Any) -> GARTMPCConfig:
    cfg = dict(GART_INITIAL_DEFAULTS)
    cfg.update(overrides)
    objective = str(objective).strip().lower()
    if objective == "raw":
        eta_y = 0.0
        eta_u = 0.0
    elif objective == "mixed":
        eta_y = float(cfg["eta_y"])
        eta_u = float(cfg["eta_u"])
    else:
        raise ValueError("objective must be 'raw' or 'mixed'.")
    return GARTMPCConfig(
        Q_raw_diag=np.asarray(values["Q_raw_diag"], dtype=float).copy(),
        Q_target_diag=np.asarray(values["Q_target_diag"], dtype=float).copy(),
        R_us_diag=np.asarray(values["R_us_diag"], dtype=float).copy(),
        Rdu_diag=np.asarray(values["Rdu_diag"], dtype=float).copy(),
        eta_y=eta_y,
        eta_u=eta_u,
        slack_penalty=float(cfg["slack_penalty"]),
        terminal_set_on=bool(cfg.get("terminal_set_on", True)),
        first_step_contraction_on=bool(cfg.get("first_step_contraction_on", True)),
        lyapunov_mode=str(lyapunov_mode),
        rho=float(cfg["rho"]),
        eps=float(cfg["eps"]),
        alpha_terminal_min=float(cfg["alpha_terminal_min"]),
        target_term_gate_enabled=bool(cfg.get("target_term_gate_enabled", True)),
        target_term_gate_delta_y=float(cfg.get("target_term_gate_delta_y", 0.5)),
        target_term_gate_min_alpha=float(cfg.get("target_term_gate_min_alpha", 0.5)),
        target_term_gate_disable_on_hold=bool(cfg.get("target_term_gate_disable_on_hold", True)),
        eta_y_when_gated=cfg.get("eta_y_when_gated"),
        eta_u_when_gated=cfg.get("eta_u_when_gated"),
        solver_options=cfg.get("solver_options"),
    )


def _scale_pm1(value: np.ndarray, v_min: np.ndarray | None, v_max: np.ndarray | None) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(-1)
    if v_min is None or v_max is None:
        return value
    v_min = np.asarray(v_min, dtype=float).reshape(-1)
    v_max = np.asarray(v_max, dtype=float).reshape(-1)
    if v_min.size != value.size or v_max.size != value.size or np.any(v_max <= v_min):
        return value
    return 2.0 * (value - v_min) / (v_max - v_min) - 1.0


def gart_rl_observation(
    min_max_dict: dict[str, Any],
    x_aug_raw: np.ndarray,
    d_cert: np.ndarray,
    y_sp: np.ndarray,
    u_prev: np.ndarray,
    target_result: GARTTargetResult | dict[str, Any],
) -> np.ndarray:
    x_aug_raw = np.asarray(x_aug_raw, dtype=float).reshape(-1)
    d_cert = np.asarray(d_cert, dtype=float).reshape(-1)
    y_sp = np.asarray(y_sp, dtype=float).reshape(-1)
    u_prev = np.asarray(u_prev, dtype=float).reshape(-1)
    n_y = y_sp.size
    x_min = min_max_dict.get("x_min")
    x_max = min_max_dict.get("x_max")
    y_min = min_max_dict.get("y_sp_min")
    y_max = min_max_dict.get("y_sp_max")
    u_min = min_max_dict.get("u_min")
    u_max = min_max_dict.get("u_max")
    d_min = None if x_min is None else np.asarray(x_min, dtype=float).reshape(-1)[-n_y:]
    d_max = None if x_max is None else np.asarray(x_max, dtype=float).reshape(-1)[-n_y:]

    if isinstance(target_result, dict):
        r_cmd = target_result.get("r_cmd")
        if r_cmd is None:
            r_cmd = target_result.get("r_s")
        y_s = target_result.get("y_s")
        u_s = target_result.get("u_s")
    else:
        r_cmd = target_result.r_cmd
        y_s = target_result.y_s
        u_s = target_result.u_s

    r_cmd_arr = np.zeros(n_y) if r_cmd is None else np.asarray(r_cmd, dtype=float).reshape(n_y)
    y_s_arr = np.zeros(n_y) if y_s is None else np.asarray(y_s, dtype=float).reshape(n_y)
    u_s_arr = np.zeros_like(u_prev) if u_s is None else np.asarray(u_s, dtype=float).reshape(u_prev.size)

    return np.concatenate(
        [
            _scale_pm1(x_aug_raw, x_min, x_max),
            _scale_pm1(d_cert, d_min, d_max),
            _scale_pm1(y_sp, y_min, y_max),
            _scale_pm1(u_prev, u_min, u_max),
            _scale_pm1(r_cmd_arr, y_min, y_max),
            _scale_pm1(y_s_arr, y_min, y_max),
            _scale_pm1(u_s_arr, u_min, u_max),
        ]
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


__all__ = [
    "GART_FINAL_LYAPUNOV_MODE",
    "GART_FINAL_LYAP_EPS",
    "GART_FINAL_MPC_OBJECTIVE",
    "GART_FINAL_RHO_LYAP",
    "GART_FINAL_SLACK_PENALTY",
    "GART_FINAL_TARGET_CONFIG_OVERRIDES",
    "GART_FINAL_TARGET_OVERRIDES",
    "GART_INITIAL_DEFAULTS",
    "discover_gart_case_values",
    "make_gart_target_config",
    "make_gart_mpc_config",
    "gart_rl_observation",
    "write_json",
]
