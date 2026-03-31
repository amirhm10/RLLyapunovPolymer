import numpy as np

from Lyapunov.legacy_rl_projection import (
    design_riccati_P_aug_physical,
    factor_psd_left as legacy_factor_psd_left,
    lyapunov_project_layer_augstate,
)
from Lyapunov.lyapunov_core import design_lyapunov_filter_ingredients
from Lyapunov.safety_filter import apply_lyapunov_safety_filter
from Lyapunov.target_selector import (
    TargetSelectorConfig,
    build_target_selector_config,
    prepare_filter_target,
)
from Lyapunov.upstream_controllers import (
    apply_first_step_contraction_replacement,
    build_repeated_input_bounds,
    default_mpc_initial_guess,
)
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.scaling_helpers import apply_min_max, apply_rl_scaled, reverse_min_max


def _system_io_phys(system, steady_states):
    u_phys = np.asarray(system.current_input, float).reshape(-1)
    y_phys = np.asarray(system.current_output, float).reshape(-1)

    if bool(getattr(system, "deviation_form", False)):
        u_phys = u_phys + np.asarray(steady_states["ss_inputs"], float).reshape(-1)
        y_phys = y_phys + np.asarray(steady_states["y_ss"], float).reshape(-1)

    return u_phys, y_phys


def _set_system_input_phys(system, steady_states, u_phys):
    u_phys = np.asarray(u_phys, float).reshape(-1)
    if bool(getattr(system, "deviation_form", False)):
        system.current_input = u_phys - np.asarray(steady_states["ss_inputs"], float).reshape(-1)
    else:
        system.current_input = u_phys.copy()


def _capture_system_snapshot(system):
    snapshot = {}
    for name in ("current_state", "current_input", "current_output"):
        if hasattr(system, name):
            snapshot[name] = np.asarray(getattr(system, name), float).copy()
    if hasattr(system, "current_viscosity"):
        snapshot["current_viscosity"] = float(getattr(system, "current_viscosity"))
    for name in ("Qi", "Qs", "hA"):
        if hasattr(system, name):
            snapshot[name] = float(getattr(system, name))
    return snapshot


def _restore_system_snapshot(system, snapshot):
    for name, value in snapshot.items():
        if isinstance(value, np.ndarray):
            setattr(system, name, value.copy())
        else:
            setattr(system, name, float(value))


def _reset_system_on_entry(system):
    snapshot = getattr(system, "_lyap_entry_snapshot", None)
    if snapshot is None:
        snapshot = _capture_system_snapshot(system)
        try:
            system._lyap_entry_snapshot = snapshot
        except Exception:
            pass
    _restore_system_snapshot(system, snapshot)


def _selector_target_reference(y_sp_raw, target_info):
    y_sp_raw = np.asarray(y_sp_raw, float).reshape(-1)
    target_info = {} if target_info is None else dict(target_info)
    for key in ("r_s", "yc_s", "y_s"):
        value = target_info.get(key)
        if value is None:
            continue
        value = np.asarray(value, float).reshape(-1)
        if value.size == y_sp_raw.size:
            return value.copy()
    return None


def _select_mpc_tracking_target(y_sp_raw, target_info, policy="raw_setpoint"):
    y_sp_raw = np.asarray(y_sp_raw, float).reshape(-1)
    target_info = {} if target_info is None else dict(target_info)
    y_s = target_info.get("y_s")
    stage = target_info.get("solve_stage")
    selector_ref = _selector_target_reference(y_sp_raw, target_info)

    if y_s is not None:
        y_s = np.asarray(y_s, float).reshape(-1)
    if selector_ref is not None:
        selector_ref = np.asarray(selector_ref, float).reshape(-1)

    if policy == "raw_setpoint":
        return y_sp_raw.copy(), "raw_setpoint"
    if policy == "selector_reference":
        if selector_ref is not None and bool(target_info.get("success", False)):
            return selector_ref.copy(), "selector_reference"
        return y_sp_raw.copy(), "raw_setpoint"
    if policy == "admissible_if_available":
        if selector_ref is not None and bool(target_info.get("success", False)):
            return selector_ref.copy(), "admissible_target"
        if y_s is not None and bool(target_info.get("success", False)) and y_s.size == y_sp_raw.size:
            return y_s.copy(), "admissible_target_full_output"
        return y_sp_raw.copy(), "raw_setpoint"
    if policy == "admissible_on_fallback":
        if selector_ref is not None and bool(target_info.get("success", False)) and stage == "fallback":
            return selector_ref.copy(), "admissible_target_fallback"
        if y_s is not None and bool(target_info.get("success", False)) and stage == "fallback" and y_s.size == y_sp_raw.size:
            return y_s.copy(), "admissible_target_fallback_full_output"
        return y_sp_raw.copy(), "raw_setpoint"
    raise ValueError(
        "policy must be one of 'raw_setpoint', 'selector_reference', 'admissible_if_available', or 'admissible_on_fallback'."
    )


def _resolve_effective_target(current_target, prev_target, backup_policy="last_valid"):
    if isinstance(current_target, dict) and current_target.get("success", False):
        return current_target, "current_target"
    if str(backup_policy) == "last_valid" and isinstance(prev_target, dict) and prev_target.get("success", False):
        return prev_target, "last_valid_target"
    return None, None


def _normalize_tracking_target_policy(mpc_target_policy, tracking_target_policy):
    policy = mpc_target_policy if tracking_target_policy is None else tracking_target_policy
    return str(policy)


def _normalize_rl_projection_backend(projection_backend):
    if projection_backend is None:
        return "legacy_augstate"
    backend = str(projection_backend).strip().lower()
    aliases = {
        "legacy": "legacy_augstate",
        "legacy_augmented": "legacy_augstate",
        "legacy_augmented_projection": "legacy_augstate",
        "legacy_augstate": "legacy_augstate",
        "augstate": "legacy_augstate",
        "safety_filter": "safety_filter",
        "current": "safety_filter",
        "refined": "safety_filter",
        "first_step_contraction_mpc": "first_step_contraction_mpc",
        "first_step_contraction": "first_step_contraction_mpc",
        "first_step": "first_step_contraction_mpc",
    }
    if backend not in aliases:
        raise ValueError(
            "projection_backend must be 'legacy_augstate', 'safety_filter', or 'first_step_contraction_mpc'."
        )
    return aliases[backend]


def _as_selector_config_dict(config):
    if config is None:
        return {}
    if isinstance(config, TargetSelectorConfig):
        return dict(config.__dict__)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("target_selector_config must be a dict, TargetSelectorConfig, or None.")


def _target_selector_overrides(target_selector_config, u_nom_tgt, Qs_tgt_diag, Ru_tgt_diag, w_x_tgt, Qdx_tgt_diag, Rmove_diag, target_solver_pref):
    cfg_overrides = _as_selector_config_dict(target_selector_config)
    overrides = {
        "Qr_diag": Qs_tgt_diag,
        "Rdu_diag": Rmove_diag,
        "u_nom": u_nom_tgt,
        "solver_pref": target_solver_pref,
    }
    if Ru_tgt_diag is not None:
        overrides["R_u_ref_diag"] = Ru_tgt_diag
    if "Q_delta_x_diag" not in cfg_overrides and "alpha_dx_sel" not in cfg_overrides and Qdx_tgt_diag is not None:
        overrides["Q_delta_x_diag"] = Qdx_tgt_diag
    if "Q_x_ref_diag" not in cfg_overrides and "alpha_x_ref" not in cfg_overrides and w_x_tgt is not None:
        overrides["Q_x_ref_diag"] = w_x_tgt
    overrides.update(cfg_overrides)
    return overrides


def _selector_decomposition(C_aug, n_x, target_info):
    if target_info is None or not target_info.get("success", False):
        return None, None
    x_s = target_info.get("x_s")
    d_s = target_info.get("d_s")
    if x_s is None or d_s is None:
        return None, None
    C_aug = np.asarray(C_aug, float)
    C = C_aug[:, :n_x]
    Cd = C_aug[:, n_x:]
    x_s = np.asarray(x_s, float).reshape(-1)
    d_s = np.asarray(d_s, float).reshape(-1)
    return np.asarray(C @ x_s, float).reshape(-1), np.asarray(Cd @ d_s, float).reshape(-1)


def _normalize_legacy_projection_info(legacy_info, u_dev_safe, action, mpc_tracking_target, mpc_tracking_target_source, target_mismatch_inf, cx_s, cd_d_s):
    legacy_info = {} if legacy_info is None else dict(legacy_info)
    target_info = legacy_info.get("target_info", {})
    if not isinstance(target_info, dict):
        target_info = {}

    method = str(legacy_info.get("method", ""))
    success = bool(legacy_info.get("success", False))
    filtered = bool(legacy_info.get("filtered", False))

    if method == "accept":
        correction_mode = "accepted_candidate"
        accept_reason = "candidate_ok"
        reject_reason = None
        qcqp_attempted = False
        qcqp_solved = False
        qcqp_hard_accepted = False
        qcqp_status = "not_attempted"
    elif method == "qp":
        correction_mode = "optimized_correction" if filtered else "accepted_candidate"
        accept_reason = "optimized_correction" if filtered else "candidate_ok"
        reject_reason = "lyapunov"
        qcqp_attempted = True
        qcqp_solved = True
        qcqp_hard_accepted = True
        qcqp_status = "hard_accepted"
    else:
        correction_mode = "legacy_passthrough_on_fail"
        accept_reason = None
        reject_reason = "target_unavailable" if method == "target_fail" else method
        qcqp_attempted = method == "qp_fail"
        qcqp_solved = False
        qcqp_hard_accepted = False
        qcqp_status = method if method else "failed"

    V_next = legacy_info.get("V_next")
    V_bound = legacy_info.get("c")
    final_margin = None if V_next is None or V_bound is None else float(V_bound) - float(V_next)

    info = {
        "source": "rl",
        "accepted": bool(success),
        "verified": bool(success),
        "accept_reason": accept_reason,
        "reject_reason": reject_reason,
        "candidate_bounds_ok": True,
        "candidate_move_ok": True,
        "candidate_lyap_ok": True if method == "accept" else False,
        "u_cand": None if legacy_info.get("u_rl") is None else np.asarray(legacy_info["u_rl"], float).reshape(-1).copy(),
        "u_safe": np.asarray(u_dev_safe, float).reshape(-1).copy(),
        "u_prev": None if legacy_info.get("u_prev_dev") is None else np.asarray(legacy_info["u_prev_dev"], float).reshape(-1).copy(),
        "u_s": None if legacy_info.get("u_s") is None else np.asarray(legacy_info["u_s"], float).reshape(-1).copy(),
        "x_s": None if target_info.get("x_s") is None else np.asarray(target_info["x_s"], float).reshape(-1).copy(),
        "d_s": None if target_info.get("d_s") is None else np.asarray(target_info["d_s"], float).reshape(-1).copy(),
        "y_s": None if target_info.get("y_s") is None else np.asarray(target_info["y_s"], float).reshape(-1).copy(),
        "r_s": None if target_info.get("r_s") is None else np.asarray(target_info["r_s"], float).reshape(-1).copy(),
        "V_k": legacy_info.get("V_k"),
        "V_next_cand": V_next,
        "V_bound": V_bound,
        "final_lyap_value": V_next,
        "final_lyap_bound": V_bound,
        "final_lyap_margin": final_margin,
        "final_lyap_ok": bool(success),
        "final_lyap_target_source": "current_target" if target_info.get("success", False) else None,
        "rho": legacy_info.get("rho"),
        "eps_lyap": legacy_info.get("eps_v"),
        "solver_status": legacy_info.get("status"),
        "solver_name": legacy_info.get("solver"),
        "solver_residuals": {
            "legacy_margin": legacy_info.get("margin_star", legacy_info.get("margin_rl")),
            "legacy_box_violation": legacy_info.get("box_violation_star", legacy_info.get("qp_box_violation")),
        },
        "trust_region_violation": 0.0,
        "slack_v": 0.0,
        "slack_u": 0.0,
        "correction_mode": correction_mode,
        "qcqp_attempted": bool(qcqp_attempted),
        "qcqp_solved": bool(qcqp_solved),
        "qcqp_hard_accepted": bool(qcqp_hard_accepted),
        "qcqp_status": qcqp_status,
        "fallback_mode": None,
        "fallback_verified": False,
        "fallback_solver_status": None,
        "fallback_objective_value": None,
        "fallback_bounds_ok": None,
        "fallback_move_ok": None,
        "fallback_lyap_ok": None,
        "target_success": bool(target_info.get("success", False)),
        "current_target_success": bool(target_info.get("success", False)),
        "current_target_stage": target_info.get("solve_stage"),
        "effective_target_success": bool(target_info.get("success", False)),
        "effective_target_stage": target_info.get("solve_stage"),
        "effective_target_source": "current_target" if target_info.get("success", False) else None,
        "effective_target_reused": False,
        "target_source": "legacy_augstate_recomputed",
        "target_stage": target_info.get("solve_stage"),
        "selector_mode": target_info.get("selector_mode"),
        "effective_selector_mode": target_info.get("selector_mode"),
        "selector_name": target_info.get("selector_name"),
        "selector_objective_terms": target_info.get("objective_terms") or {},
        "selector_objective_value": target_info.get("objective_value"),
        "d_s_minus_dhat_inf": target_info.get("d_s_minus_dhat_inf"),
        "d_s_frozen": target_info.get("d_s_frozen"),
        "d_s_optimized": target_info.get("d_s_optimized"),
        "selector_warm_start_enabled": False,
        "selector_warm_start_available": False,
        "selector_warm_start_used": False,
        "selector_prev_input_term_active": False,
        "selector_prev_state_term_active": False,
        "selector_Qr_diag_used": target_info.get("selector_debug", {}).get("Qr_diag_used"),
        "selector_R_u_ref_diag_used": target_info.get("selector_debug", {}).get("R_u_ref_diag_used"),
        "selector_R_delta_u_sel_diag_used": target_info.get("selector_debug", {}).get("R_delta_u_sel_diag_used"),
        "selector_Q_delta_x_diag_used": target_info.get("selector_debug", {}).get("Q_delta_x_diag_used"),
        "selector_Q_x_ref_diag_used": target_info.get("selector_debug", {}).get("Q_x_ref_diag_used"),
        "selector_Qx_base_diag_used": target_info.get("selector_debug", {}).get("Qx_base_diag_used"),
        "selector_Rdu_diag_used": target_info.get("selector_debug", {}).get("Rdu_diag_used"),
        "target_info": target_info,
        "cx_s": None if cx_s is None else np.asarray(cx_s, float).reshape(-1).copy(),
        "cd_d_s": None if cd_d_s is None else np.asarray(cd_d_s, float).reshape(-1).copy(),
        "upstream_candidate_info": {
            "source": "rl_policy",
            "action_raw": np.asarray(action, float).reshape(-1).copy(),
            "mpc_tracking_target": None if mpc_tracking_target is None else np.asarray(mpc_tracking_target, float).reshape(-1).copy(),
            "mpc_tracking_target_source": mpc_tracking_target_source,
            "target_mismatch_inf": target_mismatch_inf,
        },
        "mpc_tracking_target": None if mpc_tracking_target is None else np.asarray(mpc_tracking_target, float).reshape(-1).copy(),
        "mpc_tracking_target_source": mpc_tracking_target_source,
        "target_mismatch_inf": target_mismatch_inf,
        "qcqp_tracking_target": None if mpc_tracking_target is None else np.asarray(mpc_tracking_target, float).reshape(-1).copy(),
        "qcqp_tracking_target_source": mpc_tracking_target_source,
    }
    return info


def map_to_bounds(a, low, high):
    a = np.asarray(a, float).reshape(-1)
    low = np.asarray(low, float).reshape(-1)
    high = np.asarray(high, float).reshape(-1)
    return low + 0.5 * (a + 1.0) * (high - low)


def inv_map_from_bounds(u, low, high, eps=1e-12):
    u = np.asarray(u, float).reshape(-1)
    low = np.asarray(low, float).reshape(-1)
    high = np.asarray(high, float).reshape(-1)
    denom = np.maximum(high - low, eps)
    a = 2.0 * (u - low) / denom - 1.0
    return np.clip(a, -1.0, 1.0)


def _coerce_supplied_lyapunov_matrix(P_lyap, n_x, n_aug):
    P_lyap = np.asarray(P_lyap, float)
    P_lyap = 0.5 * (P_lyap + P_lyap.T)

    if P_lyap.shape == (n_x, n_x):
        return P_lyap.copy()
    if P_lyap.shape == (n_aug, n_aug):
        return P_lyap[:n_x, :n_x].copy()
    raise ValueError(
        f"P_lyap must have shape {(n_x, n_x)} or {(n_aug, n_aug)}, got {P_lyap.shape}."
    )


def _normalize_mpc_fallback_setup(MPC_obj, u_min, u_max, IC_opt, bnds, cons):
    n_u = int(MPC_obj.B.shape[1])
    horizon_control = int(getattr(MPC_obj, "NC", 1))

    if IC_opt is None:
        IC_opt = default_mpc_initial_guess(n_u, horizon_control)
    else:
        IC_opt = np.asarray(IC_opt, float).reshape(-1)
    if IC_opt.size != n_u * horizon_control:
        raise ValueError(
            f"IC_opt has size {IC_opt.size}, expected {n_u * horizon_control}."
        )

    if bnds is None:
        bnds = build_repeated_input_bounds(u_min, u_max, horizon_control)
    if cons is None:
        cons = ()
    else:
        cons = tuple(cons)

    return IC_opt.copy(), bnds, cons


def run_rl_train(
    system,
    y_sp_scenario,
    n_tests,
    set_points_len,
    steady_states,
    min_max_dict,
    agent,
    MPC_obj,
    L,
    data_min,
    data_max,
    warm_start,
    test_cycle,
    nominal_qi,
    nominal_qs,
    nominal_ha,
    qi_change,
    qs_change,
    ha_change,
    reward_fn,
    mode="disturb",
    P_lyap=None,
    rho_lyap=0.99,
    lyap_eps=1e-9,
    lyap_tol=1e-10,
    w_rl=1.0,
    w_track=1.0,
    w_move=1.0,
    w_ss=1.0,
    Qy_track_diag=None,
    Rmove_diag=None,
    Qs_tgt_diag=None,
    Ru_tgt_diag=None,
    u_nom_tgt=None,
    w_x_tgt=1e-6,
    lambda_u_ric=1.0,
    pd_eps_ric=0.0,
    seed=0,
    use_lyap=False,
    du_min=None,
    du_max=None,
    trust_region_delta=None,
    allow_lyap_slack=False,
    target_solver_pref=None,
    filter_solver_pref=None,
    IC_opt=None,
    bnds=None,
    cons=None,
    fallback_policy="offset_free_mpc",
    mpc_target_policy="raw_setpoint",
    tracking_target_policy=None,
    selector_mode=None,
    target_selector_config=None,
    selector_H=None,
    target_backup_policy="last_valid",
    selector_warm_start=True,
    Qdx_tgt_diag=None,
    lyap_acceptance_mode="hard_only",
    allow_trust_region_slack=False,
    reuse_mpc_solution_as_ic=False,
    reset_system_on_entry=True,
    projection_backend="legacy_augstate",
    first_step_contraction_on=True,
):
    # warm_start only controls when online TD3 parameter updates begin through
    # the generated train/test schedule. It is not an MPC takeover or control
    # warm-start flag.
    if reset_system_on_entry:
        _reset_system_on_entry(system)

    (
        y_sp,
        nFE,
        sub_changes,
        time_in_sub_episodes,
        test_train_dict,
        WARM_START,
        qi,
        qs,
        ha,
    ) = generate_setpoints_training_rl_gradually(
        y_sp_scenario,
        n_tests,
        set_points_len,
        warm_start,
        test_cycle,
        nominal_qi,
        nominal_qs,
        nominal_ha,
        qi_change,
        qs_change,
        ha_change,
    )

    n_u = MPC_obj.B.shape[1]
    n_y = MPC_obj.C.shape[0]
    n_aug = MPC_obj.A.shape[0]
    n_x = n_aug - n_y
    projection_backend = _normalize_rl_projection_backend(projection_backend)
    tracking_target_policy = _normalize_tracking_target_policy(
        mpc_target_policy=mpc_target_policy,
        tracking_target_policy=tracking_target_policy,
    )

    ss_scaled_u = apply_min_max(steady_states["ss_inputs"], data_min[:n_u], data_max[:n_u])
    ss_scaled_y = apply_min_max(steady_states["y_ss"], data_min[n_u:], data_max[n_u:])

    u_min = np.asarray(min_max_dict["u_min"], float).reshape(-1)
    u_max = np.asarray(min_max_dict["u_max"], float).reshape(-1)
    if np.any(u_min > u_max):
        raise ValueError("u_min must be <= u_max elementwise.")

    fallback_ic, mpc_bnds, mpc_cons = _normalize_mpc_fallback_setup(
        MPC_obj=MPC_obj,
        u_min=u_min,
        u_max=u_max,
        IC_opt=IC_opt,
        bnds=bnds,
        cons=cons,
    )

    if Qy_track_diag is None:
        Qy_track_diag = np.asarray(MPC_obj.Q_out, float).reshape(-1)
    else:
        Qy_track_diag = np.asarray(Qy_track_diag, float).reshape(-1)

    if Rmove_diag is None:
        Rmove_diag = np.asarray(MPC_obj.R_in, float).reshape(-1)
    else:
        Rmove_diag = np.asarray(Rmove_diag, float).reshape(-1)

    if Qs_tgt_diag is None:
        Qs_tgt_diag = np.asarray(MPC_obj.Q_out, float).reshape(-1)
    else:
        Qs_tgt_diag = np.asarray(Qs_tgt_diag, float).reshape(-1)

    if Ru_tgt_diag is not None:
        Ru_tgt_diag = np.asarray(Ru_tgt_diag, float).reshape(-1)

    if Qdx_tgt_diag is None:
        Qdx_tgt_diag = np.full(n_x, float(max(w_x_tgt, 1e-6)), dtype=float)
    else:
        Qdx_tgt_diag = np.asarray(Qdx_tgt_diag, float).reshape(-1)

    selector_cfg = None
    lyap_model = None
    legacy_P_lyap = None
    legacy_S_lyap = None
    if projection_backend in {"safety_filter", "first_step_contraction_mpc"}:
        selector_cfg = build_target_selector_config(
            user_overrides=_target_selector_overrides(
                target_selector_config=target_selector_config,
                u_nom_tgt=u_nom_tgt,
                Qs_tgt_diag=Qs_tgt_diag,
                Ru_tgt_diag=Ru_tgt_diag,
                w_x_tgt=w_x_tgt,
                Qdx_tgt_diag=Qdx_tgt_diag,
                Rmove_diag=Rmove_diag,
                target_solver_pref=target_solver_pref,
            ),
            n_x=n_x,
            n_u=n_u,
            n_y=n_y,
            n_d=n_y,
            Q_out=Qs_tgt_diag,
            Rmove_diag=Rmove_diag,
        )

        lyap_model = design_lyapunov_filter_ingredients(
            A_aug=MPC_obj.A,
            B_aug=MPC_obj.B,
            C_aug=MPC_obj.C,
            Qy_diag=Qy_track_diag,
            Ru_diag=None,
            u_min=u_min,
            u_max=u_max,
            u_nom=u_nom_tgt,
            lambda_u=lambda_u_ric,
            qx_eps=pd_eps_ric,
            return_debug=False,
        )
        if P_lyap is not None:
            lyap_model["P_x"] = _coerce_supplied_lyapunov_matrix(P_lyap, n_x=n_x, n_aug=n_aug)
    elif use_lyap:
        if P_lyap is None:
            legacy_P_lyap = design_riccati_P_aug_physical(
                A_aug=MPC_obj.A,
                B_aug=MPC_obj.B,
                C_aug=MPC_obj.C,
                Qy_diag=Qy_track_diag,
                Ru_diag=None,
                u_min=u_min,
                u_max=u_max,
                u_nom=None,
                lambda_u=lambda_u_ric,
                pd_eps=pd_eps_ric,
                return_debug=False,
            )
        else:
            legacy_P_lyap = np.asarray(P_lyap, float)
            legacy_P_lyap = 0.5 * (legacy_P_lyap + legacy_P_lyap.T)
            if legacy_P_lyap.shape != (n_aug, n_aug):
                raise ValueError(
                    f"Legacy RL projection expects augmented P_lyap with shape {(n_aug, n_aug)}, got {legacy_P_lyap.shape}."
                )
        legacy_S_lyap = legacy_factor_psd_left(legacy_P_lyap)

    y_system = np.zeros((nFE + 1, n_y), dtype=float)
    _u_phys_0, y_phys_0 = _system_io_phys(system, steady_states)
    y_system[0, :] = y_phys_0

    u_scaled_applied = np.zeros((nFE, n_u), dtype=float)
    u_safe_dev_store = np.zeros((nFE, n_u), dtype=float)

    yhat = np.zeros((n_y, nFE), dtype=float)
    xhat_aug_store = np.zeros((n_aug, nFE + 1), dtype=float)

    e_store = np.zeros((nFE + 1, n_y), dtype=float)
    rewards = np.zeros(nFE, dtype=float)
    avg_rewards = []
    lyap_info_storage = []

    total_checked = 0
    total_filtered = 0
    total_fallback_mpc = 0
    checked_in_block = 0
    filtered_in_block = 0
    fallback_in_block = 0

    test = False
    np.random.seed(seed)

    prev_target_info = None
    last_verified_safe_dev = None

    for k in range(nFE):
        if k in test_train_dict:
            test = bool(test_train_dict[k])

        u_prev_phys, y_prev_phys = _system_io_phys(system, steady_states)

        u_prev_scaled = apply_min_max(u_prev_phys, data_min[:n_u], data_max[:n_u])
        u_prev_dev = u_prev_scaled - ss_scaled_u

        y_prev_dev = apply_min_max(y_prev_phys, data_min[n_u:], data_max[n_u:]) - ss_scaled_y
        y_hat_k = MPC_obj.C @ xhat_aug_store[:, k]
        yhat[:, k] = y_hat_k

        y_sp_k = np.asarray(y_sp[k, :], float).reshape(-1)
        setpoint_changed = True if k == 0 else not np.array_equal(y_sp_k, np.asarray(y_sp[k - 1, :], float).reshape(-1))

        e_k = y_prev_dev - y_sp_k
        e_store[k, :] = e_k

        rl_state = apply_rl_scaled(min_max_dict, xhat_aug_store[:, k], y_sp_k, u_prev_dev)
        if test:
            action = agent.act_eval(rl_state)
        else:
            action = agent.take_action(rl_state, explore=(not test))

        action = np.asarray(action, float).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        u_rl_dev = np.clip(map_to_bounds(action, u_min, u_max), u_min, u_max)

        if projection_backend == "legacy_augstate":
            if (k + 1) < y_sp.shape[0]:
                y_sp_kp1 = np.asarray(y_sp[k + 1, :], float).reshape(-1)
            else:
                y_sp_kp1 = y_sp_k.copy()

            mpc_tracking_target = y_sp_k.copy()
            mpc_tracking_target_source = "raw_setpoint"
            target_mismatch_inf = None

            if use_lyap:
                u_dev_safe, legacy_info = lyapunov_project_layer_augstate(
                    xhat_aug=xhat_aug_store[:, k],
                    y_sp=y_sp_k,
                    u_rl_dev=u_rl_dev,
                    u_prev_dev=u_prev_dev,
                    u_min=u_min,
                    u_max=u_max,
                    A_aug=MPC_obj.A,
                    B_aug=MPC_obj.B,
                    C_aug=MPC_obj.C,
                    P_lyap=legacy_P_lyap,
                    S_lyap=legacy_S_lyap,
                    rho=rho_lyap,
                    eps_v=lyap_eps,
                    w_rl=w_rl,
                    w_track=w_track,
                    w_move=w_move,
                    w_ss=w_ss,
                    Qy_track_diag=Qy_track_diag,
                    Rmove_diag=Rmove_diag,
                    Qs_tgt_diag=Qs_tgt_diag,
                    Ru_tgt_diag=Ru_tgt_diag,
                    u_nom_tgt=u_nom_tgt,
                    w_x_tgt=w_x_tgt,
                    solver_pref_target=(
                        ("OSQP", "CLARABEL", "SCS")
                        if target_solver_pref is None
                        else ((target_solver_pref,) if isinstance(target_solver_pref, str) else tuple(target_solver_pref))
                    ),
                    solver_pref_qp=(
                        ("CLARABEL", "SCS", "ECOS")
                        if filter_solver_pref is None
                        else ((filter_solver_pref,) if isinstance(filter_solver_pref, str) else tuple(filter_solver_pref))
                    ),
                    tol=lyap_tol,
                    box_tol=1e-9,
                )
                target_info = legacy_info.get("target_info", {})
                cx_s, cd_d_s = _selector_decomposition(MPC_obj.C, n_x, target_info)
                info = _normalize_legacy_projection_info(
                    legacy_info=legacy_info,
                    u_dev_safe=u_dev_safe,
                    action=action,
                    mpc_tracking_target=mpc_tracking_target,
                    mpc_tracking_target_source=mpc_tracking_target_source,
                    target_mismatch_inf=target_mismatch_inf,
                    cx_s=cx_s,
                    cd_d_s=cd_d_s,
                )
                info["setpoint_changed"] = bool(setpoint_changed)
                info["lyap_acceptance_mode"] = "hard_only"
                if info.get("verified", False):
                    last_verified_safe_dev = u_dev_safe.copy()
            else:
                u_dev_safe = np.clip(u_rl_dev, u_min, u_max)
                info = {
                    "source": "rl",
                    "accepted": True,
                    "verified": True,
                    "accept_reason": "bypass",
                    "reject_reason": None,
                    "candidate_bounds_ok": True,
                    "candidate_move_ok": True,
                    "candidate_lyap_ok": None,
                    "u_cand": u_rl_dev.copy(),
                    "u_safe": u_dev_safe.copy(),
                    "u_prev": u_prev_dev.copy(),
                    "u_s": None,
                    "x_s": None,
                    "d_s": None,
                    "y_s": None,
                    "r_s": None,
                    "V_k": None,
                    "V_next_cand": None,
                    "V_bound": None,
                    "final_lyap_value": None,
                    "final_lyap_bound": None,
                    "final_lyap_margin": None,
                    "final_lyap_ok": None,
                    "rho": rho_lyap,
                    "eps_lyap": lyap_eps,
                    "solver_status": None,
                    "solver_name": None,
                    "solver_residuals": {},
                    "trust_region_violation": 0.0,
                    "slack_v": 0.0,
                    "slack_u": 0.0,
                    "correction_mode": "bypass",
                    "qcqp_attempted": False,
                    "qcqp_solved": False,
                    "qcqp_hard_accepted": False,
                    "qcqp_status": "not_attempted",
                    "target_success": False,
                    "current_target_success": False,
                    "current_target_stage": None,
                    "effective_target_success": False,
                    "effective_target_stage": None,
                    "effective_target_source": None,
                    "effective_target_reused": False,
                    "target_source": "legacy_bypass",
                    "target_stage": None,
                    "selector_mode": "legacy_augstate_rl",
                    "effective_selector_mode": None,
                    "selector_name": "legacy_augmented_slack_target",
                    "selector_objective_terms": {},
                    "d_s_minus_dhat_inf": None,
                    "d_s_frozen": True,
                    "d_s_optimized": False,
                    "selector_warm_start_enabled": False,
                    "selector_warm_start_available": False,
                    "selector_warm_start_used": False,
                    "selector_prev_input_term_active": False,
                    "selector_prev_state_term_active": False,
                    "selector_Qr_diag_used": None,
                    "selector_R_u_ref_diag_used": None,
                    "selector_R_delta_u_sel_diag_used": None,
                    "selector_Q_delta_x_diag_used": None,
                    "selector_Q_x_ref_diag_used": None,
                    "selector_Qx_base_diag_used": None,
                    "selector_Rdu_diag_used": None,
                    "selector_objective_value": None,
                    "fallback_mode": None,
                    "fallback_verified": False,
                    "fallback_solver_status": None,
                    "fallback_objective_value": None,
                    "fallback_bounds_ok": None,
                    "fallback_lyap_ok": None,
                    "target_info": {
                        "success": False,
                        "selector_mode": "legacy_augstate_rl",
                        "selector_name": "legacy_augmented_slack_target",
                        "solve_stage": None,
                        "requested_y_sp": y_sp_k.copy(),
                        "selector_debug": {},
                    },
                    "setpoint_changed": bool(setpoint_changed),
                    "upstream_candidate_info": {
                        "source": "rl_policy",
                        "action_raw": action.copy(),
                        "mpc_tracking_target": mpc_tracking_target.copy(),
                        "mpc_tracking_target_source": mpc_tracking_target_source,
                        "target_mismatch_inf": None,
                    },
                    "mpc_tracking_target": mpc_tracking_target.copy(),
                    "mpc_tracking_target_source": mpc_tracking_target_source,
                    "target_mismatch_inf": None,
                    "qcqp_tracking_target": mpc_tracking_target.copy(),
                    "qcqp_tracking_target_source": mpc_tracking_target_source,
                    "cx_s": None,
                    "cd_d_s": None,
                }
                last_verified_safe_dev = u_dev_safe.copy()

            lyap_info_storage.append(info)

            if use_lyap:
                total_checked += 1
                checked_in_block += 1
                if info.get("correction_mode") == "optimized_correction":
                    total_filtered += 1
                    filtered_in_block += 1
                if str(info.get("correction_mode", "")).startswith("fallback_mpc"):
                    total_fallback_mpc += 1
                    fallback_in_block += 1

            u_safe_dev_store[k, :] = u_dev_safe

            a_used = inv_map_from_bounds(u_dev_safe, u_min, u_max).astype(np.float32)
            u_scaled_applied[k, :] = u_dev_safe + ss_scaled_u
            u_plant = reverse_min_max(u_scaled_applied[k, :], data_min[:n_u], data_max[:n_u])
            delta_u = u_scaled_applied[k, :] - u_prev_scaled

            if mode == "disturb":
                system.hA = ha[k]
                system.Qs = qs[k]
                system.Qi = qi[k]

            _set_system_input_phys(system, steady_states, u_plant)
            system.step()

            _u_phys_next, y_phys_next = _system_io_phys(system, steady_states)
            y_system[k + 1, :] = y_phys_next

            y_next_dev = apply_min_max(y_phys_next, data_min[n_u:], data_max[n_u:]) - ss_scaled_y
            e_next = y_next_dev - y_sp_kp1
            e_store[k + 1, :] = e_next

            innov = y_prev_dev - y_hat_k
            xhat_aug_store[:, k + 1] = (
                (MPC_obj.A @ xhat_aug_store[:, k])
                + (MPC_obj.B @ u_dev_safe)
                + (L @ innov)
            )

            delta_y = y_next_dev - y_sp_k
            y_sp_phys = reverse_min_max(y_sp_k + ss_scaled_y, data_min[n_u:], data_max[n_u:])
            r = reward_fn(delta_y, delta_u, y_sp_phys)
            rewards[k] = float(r)

            next_u_dev = u_scaled_applied[k, :] - ss_scaled_u
            # Keep the TD3 transition tied to the setpoint active when the
            # action was chosen and rewarded. Using y_sp_kp1 here would mix
            # two different tasks at a setpoint-change boundary.
            next_state = apply_rl_scaled(min_max_dict, xhat_aug_store[:, k + 1], y_sp_k, next_u_dev)

            done = 0.0
            if not test:
                agent.push(rl_state, a_used, float(r), next_state, float(done))
                if k >= WARM_START:
                    _ = agent.train_step(actor_update=True)

            if k in sub_changes:
                start = max(0, k - time_in_sub_episodes + 1)
                avg_rewards.append(float(np.mean(rewards[start:k + 1])))
                print("Sub_Episode:", sub_changes[k], "| avg. reward:", avg_rewards[-1])

                block_ratio = filtered_in_block / checked_in_block if checked_in_block > 0 else 0.0
                total_ratio = total_filtered / total_checked if total_checked > 0 else 0.0
                fallback_ratio = fallback_in_block / checked_in_block if checked_in_block > 0 else 0.0
                print(
                    "Lyap corrected in block:",
                    filtered_in_block, "/", checked_in_block,
                    "(ratio:", block_ratio, ")",
                    "| fallback MPC in block:",
                    fallback_in_block, "/", checked_in_block,
                    "(ratio:", fallback_ratio, ")",
                    "| total corrected:",
                    total_filtered, "/", total_checked,
                    "(ratio:", total_ratio, ")",
                )

                last = lyap_info_storage[-1]
                last_target = last.get("target_info", {})
                last_selector = {} if last_target is None else last_target.get("selector_debug", {})
                print(
                    "Last Lyap mode:", last.get("correction_mode"),
                    "| verified:", last.get("verified"),
                    "| V_next:", last.get("V_next_cand"),
                    "| V_bound:", last.get("V_bound"),
                    "| fallback_status:", last.get("fallback_solver_status"),
                    "| fallback_verified:", last.get("fallback_verified"),
                    "| target_stage:", last_target.get("solve_stage") if last_target else None,
                    "| target_slack_inf:", last_target.get("target_slack_inf") if last_target else None,
                    "| selector_status:", last_selector.get("status"),
                )

                filtered_in_block = 0
                checked_in_block = 0
                fallback_in_block = 0

            continue

        target_info = prepare_filter_target(
            A_aug=MPC_obj.A,
            B_aug=MPC_obj.B,
            C_aug=MPC_obj.C,
            xhat_aug=xhat_aug_store[:, k],
            y_sp=y_sp_k,
            u_min=u_min,
            u_max=u_max,
            config=selector_cfg,
            prev_target=prev_target_info,
            H=selector_H,
            return_debug=False,
            warm_start=selector_warm_start,
            u_applied_k=u_prev_dev,
        )
        if target_info.get("success", False):
            prev_target_info = target_info
        effective_target_info, effective_target_source = _resolve_effective_target(
            current_target=target_info,
            prev_target=prev_target_info,
            backup_policy=target_backup_policy,
        )

        mpc_tracking_target, mpc_tracking_target_source = _select_mpc_tracking_target(
            y_sp_raw=y_sp_k,
            target_info=effective_target_info,
            policy=tracking_target_policy,
        )
        target_mismatch_inf = None
        if effective_target_info is not None and effective_target_info.get("y_s") is not None:
            target_mismatch_inf = float(
                np.max(np.abs(np.asarray(effective_target_info["y_s"], float).reshape(-1) - y_sp_k))
            )
        cx_s, cd_d_s = _selector_decomposition(MPC_obj.C, n_x, effective_target_info)

        if (k + 1) < y_sp.shape[0]:
            y_sp_kp1 = np.asarray(y_sp[k + 1, :], float).reshape(-1)
        else:
            y_sp_kp1 = y_sp_k.copy()

        if use_lyap and projection_backend == "safety_filter":
            safe_filter_prev = last_verified_safe_dev if last_verified_safe_dev is not None else u_prev_dev
            u_dev_safe, info = apply_lyapunov_safety_filter(
                u_cand=u_rl_dev,
                xhat_aug=xhat_aug_store[:, k],
                target_info=target_info,
                model_info=lyap_model,
                lyap_config={
                    "source": "rl",
                    "rho": rho_lyap,
                    "eps_lyap": lyap_eps,
                    "tol": lyap_tol,
                    "selector_warm_start": bool(selector_warm_start),
                    "target_backup_policy": str(target_backup_policy),
                    "backup_target_info": prev_target_info,
                    "backup_target_source": "last_valid_target" if prev_target_info is not None else None,
                    "lyap_acceptance_mode": str(lyap_acceptance_mode),
                    "candidate_weight_diag": float(w_rl) * np.ones(n_u, dtype=float),
                    "move_weight_diag": float(w_move) * np.maximum(Rmove_diag, 1e-12),
                    "steady_weight_diag": (
                        float(w_ss) * np.ones(n_u, dtype=float)
                        if Ru_tgt_diag is None
                        else float(w_ss) * np.maximum(Ru_tgt_diag, 1e-12)
                    ),
                    "output_weight_diag": float(w_track) * np.maximum(Qy_track_diag, 1e-12),
                    "trust_region_delta": trust_region_delta,
                    "trust_region_weight": 1e4,
                    "allow_trust_region_slack": bool(allow_trust_region_slack),
                    "allow_lyap_slack": bool(allow_lyap_slack),
                    "lyap_slack_weight": 1e6,
                    "solver_pref": filter_solver_pref,
                    "use_output_tracking_term": True,
                    "tracking_output_target": mpc_tracking_target.copy(),
                    "tracking_output_target_source": mpc_tracking_target_source,
                    "final_lyap_target_info": prev_target_info,
                    "final_lyap_target_source": "last_valid_target" if prev_target_info is not None else None,
                },
                u_prev=u_prev_dev,
                bounds_info={
                    "u_min": u_min,
                    "u_max": u_max,
                    "du_min": du_min,
                    "du_max": du_max,
                    "fallback_safe_input": safe_filter_prev,
                },
                fallback_config={
                    "mode": fallback_policy,
                    "MPC_obj": MPC_obj,
                    "IC_opt": fallback_ic,
                    "bnds": mpc_bnds,
                    "cons": mpc_cons,
                    "y_sp": mpc_tracking_target,
                    "x0_model": xhat_aug_store[:, k],
                    "u_prev_dev": u_prev_dev,
                    "allow_unverified": True,
                    "tracking_target_source": mpc_tracking_target_source,
                    "target_mismatch_inf": target_mismatch_inf,
                },
                return_debug=True,
            )
            if reuse_mpc_solution_as_ic and info.get("fallback_ic_next") is not None:
                fallback_ic = np.asarray(info["fallback_ic_next"], float).reshape(-1).copy()
            info["setpoint_changed"] = bool(setpoint_changed)
            info["target_source"] = "recomputed"
            info["target_stage"] = target_info.get("solve_stage")
            info["current_target_success"] = bool(target_info.get("success", False))
            info["current_target_stage"] = target_info.get("solve_stage")
            info["effective_target_success"] = bool(effective_target_info is not None and effective_target_info.get("success", False))
            info["effective_target_stage"] = None if effective_target_info is None else effective_target_info.get("solve_stage")
            info["effective_target_source"] = effective_target_source
            info["effective_target_reused"] = bool(effective_target_source == "last_valid_target")
            info["selector_mode"] = target_info.get("selector_mode")
            info["effective_selector_mode"] = None if effective_target_info is None else effective_target_info.get("selector_mode")
            info["selector_name"] = target_info.get("selector_name")
            info["selector_objective_terms"] = target_info.get("objective_terms")
            info["r_s"] = None if effective_target_info is None or effective_target_info.get("r_s") is None else np.asarray(effective_target_info["r_s"], float).reshape(-1).copy()
            info["d_s_minus_dhat_inf"] = None if effective_target_info is None else effective_target_info.get("d_s_minus_dhat_inf")
            info["d_s_frozen"] = None if effective_target_info is None else effective_target_info.get("d_s_frozen")
            info["d_s_optimized"] = None if effective_target_info is None else effective_target_info.get("d_s_optimized")
            info["selector_objective_value"] = target_info.get("objective_value")
            selector_warm = target_info.get("warm_start", {})
            selector_dbg = target_info.get("selector_debug", {})
            info["selector_warm_start_enabled"] = bool(selector_warm.get("enabled", selector_warm_start))
            info["selector_warm_start_available"] = bool(selector_warm.get("available", False))
            info["selector_warm_start_used"] = bool(selector_warm.get("used", False))
            info["selector_prev_input_term_active"] = bool(selector_dbg.get("prev_input_term_active", False))
            info["selector_prev_state_term_active"] = bool(selector_dbg.get("prev_state_term_active", False))
            info["selector_Qr_diag_used"] = selector_dbg.get("Qr_diag_used")
            info["selector_R_u_ref_diag_used"] = selector_dbg.get("R_u_ref_diag_used")
            info["selector_R_delta_u_sel_diag_used"] = selector_dbg.get("R_delta_u_sel_diag_used")
            info["selector_Q_delta_x_diag_used"] = selector_dbg.get("Q_delta_x_diag_used")
            info["selector_Q_x_ref_diag_used"] = selector_dbg.get("Q_x_ref_diag_used")
            info["selector_Qx_base_diag_used"] = selector_dbg.get("Qx_base_diag_used")
            info["selector_Rdu_diag_used"] = selector_dbg.get("Rdu_diag_used")
            info["cx_s"] = None if cx_s is None else cx_s.copy()
            info["cd_d_s"] = None if cd_d_s is None else cd_d_s.copy()
            info["upstream_candidate_info"] = {
                "source": "rl_policy",
                "action_raw": action.copy(),
                "mpc_tracking_target": mpc_tracking_target.copy(),
                "mpc_tracking_target_source": mpc_tracking_target_source,
                "target_mismatch_inf": target_mismatch_inf,
            }
            info["mpc_tracking_target"] = mpc_tracking_target.copy()
            info["mpc_tracking_target_source"] = mpc_tracking_target_source
            info["target_mismatch_inf"] = target_mismatch_inf
            info["qcqp_tracking_target"] = mpc_tracking_target.copy()
            info["qcqp_tracking_target_source"] = mpc_tracking_target_source
            if info.get("verified", False):
                last_verified_safe_dev = u_dev_safe.copy()
        elif use_lyap and projection_backend == "first_step_contraction_mpc":
            u_dev_safe, replacement_info = apply_first_step_contraction_replacement(
                u_candidate=u_rl_dev,
                MPC_obj=MPC_obj,
                y_sp=mpc_tracking_target,
                u_prev_dev=u_prev_dev,
                x0_model=xhat_aug_store[:, k],
                effective_target_info=effective_target_info,
                ingredients=lyap_model,
                rho_lyap=rho_lyap,
                eps_lyap=lyap_eps,
                lyap_tol=lyap_tol,
                IC_opt=fallback_ic,
                bnds=mpc_bnds,
                cons=mpc_cons,
                first_step_contraction_on=first_step_contraction_on,
                return_debug=True,
            )
            constrained_info = replacement_info["constrained_info"]
            if (
                reuse_mpc_solution_as_ic
                and replacement_info.get("constrained_mpc_applied", False)
                and constrained_info.get("IC_opt_next") is not None
            ):
                fallback_ic = np.asarray(constrained_info["IC_opt_next"], float).reshape(-1).copy()

            candidate_eval = replacement_info["candidate_eval"]
            applied_eval = replacement_info["applied_eval"]
            selector_warm = target_info.get("warm_start", {})
            selector_dbg = target_info.get("selector_debug", {})
            info = {
                "source": "rl_first_step_contraction",
                "accepted": bool(replacement_info.get("accepted", False)),
                "verified": bool(replacement_info.get("verified", False)),
                "accept_reason": replacement_info.get("accept_reason"),
                "reject_reason": replacement_info.get("reject_reason"),
                "candidate_bounds_ok": candidate_eval.get("candidate_bounds_ok"),
                "candidate_move_ok": candidate_eval.get("candidate_move_ok"),
                "candidate_lyap_ok": candidate_eval.get("candidate_lyap_ok"),
                "candidate_first_step_lyap_ok": replacement_info.get("candidate_first_step_lyap_ok"),
                "first_step_contraction_triggered": bool(replacement_info.get("first_step_contraction_triggered", False)),
                "constrained_mpc_attempted": bool(replacement_info.get("constrained_mpc_attempted", False)),
                "constrained_mpc_solved": bool(replacement_info.get("constrained_mpc_solved", False)),
                "constrained_mpc_applied": bool(replacement_info.get("constrained_mpc_applied", False)),
                "constrained_mpc_failed_applied_candidate": bool(
                    replacement_info.get("constrained_mpc_failed_applied_candidate", False)
                ),
                "u_cand": u_rl_dev.copy(),
                "u_safe": np.asarray(u_dev_safe, float).reshape(-1).copy(),
                "u_constrained_mpc": None if replacement_info.get("constrained_candidate") is None else np.asarray(replacement_info["constrained_candidate"], float).reshape(-1).copy(),
                "u_prev": u_prev_dev.copy(),
                "u_s": None if effective_target_info is None or effective_target_info.get("u_s") is None else np.asarray(effective_target_info["u_s"], float).reshape(-1).copy(),
                "x_s": None if effective_target_info is None or effective_target_info.get("x_s") is None else np.asarray(effective_target_info["x_s"], float).reshape(-1).copy(),
                "d_s": None if effective_target_info is None or effective_target_info.get("d_s") is None else np.asarray(effective_target_info["d_s"], float).reshape(-1).copy(),
                "y_s": None if effective_target_info is None or effective_target_info.get("y_s") is None else np.asarray(effective_target_info["y_s"], float).reshape(-1).copy(),
                "r_s": None if effective_target_info is None or effective_target_info.get("r_s") is None else np.asarray(effective_target_info["r_s"], float).reshape(-1).copy(),
                "V_k": replacement_info.get("V_k"),
                "V_next_first": replacement_info.get("V_next_first_applied"),
                "V_next_first_candidate": replacement_info.get("V_next_first_candidate"),
                "V_next_first_applied": replacement_info.get("V_next_first_applied"),
                "V_next_cand": replacement_info.get("V_next_first_candidate"),
                "V_bound": replacement_info.get("V_bound"),
                "contraction_margin": replacement_info.get("contraction_margin_applied"),
                "contraction_margin_candidate": replacement_info.get("contraction_margin_candidate"),
                "contraction_margin_applied": replacement_info.get("contraction_margin_applied"),
                "first_step_contraction_satisfied": replacement_info.get("first_step_contraction_satisfied_applied"),
                "first_step_contraction_satisfied_applied": replacement_info.get("first_step_contraction_satisfied_applied"),
                "contraction_constraint_violation": None if replacement_info.get("contraction_margin_applied") is None else float(max(replacement_info["contraction_margin_applied"], 0.0)),
                "rho": rho_lyap,
                "eps_lyap": lyap_eps,
                "solver_status": constrained_info.get("status") if replacement_info.get("constrained_mpc_attempted", False) else None,
                "solver_name": constrained_info.get("solver_name"),
                "solver_residuals": {},
                "trust_region_violation": None,
                "slack_v": 0.0,
                "slack_u": 0.0,
                "correction_mode": replacement_info.get("correction_mode"),
                "qcqp_attempted": False,
                "qcqp_solved": False,
                "qcqp_hard_accepted": False,
                "qcqp_status": "not_attempted",
                "target_success": bool(target_info.get("success", False)),
                "current_target_success": bool(target_info.get("success", False)),
                "current_target_stage": target_info.get("solve_stage"),
                "effective_target_success": bool(effective_target_info is not None and effective_target_info.get("success", False)),
                "effective_target_stage": None if effective_target_info is None else effective_target_info.get("solve_stage"),
                "effective_target_source": effective_target_source,
                "effective_target_reused": bool(effective_target_source == "last_valid_target"),
                "selector_mode": target_info.get("selector_mode"),
                "effective_selector_mode": None if effective_target_info is None else effective_target_info.get("selector_mode"),
                "selector_name": target_info.get("selector_name"),
                "selector_objective_terms": target_info.get("objective_terms"),
                "selector_objective_value": target_info.get("objective_value"),
                "d_s_minus_dhat_inf": None if effective_target_info is None else effective_target_info.get("d_s_minus_dhat_inf"),
                "d_s_frozen": None if effective_target_info is None else effective_target_info.get("d_s_frozen"),
                "d_s_optimized": None if effective_target_info is None else effective_target_info.get("d_s_optimized"),
                "target_info": target_info,
                "effective_target_info": effective_target_info,
                "setpoint_changed": bool(setpoint_changed),
                "target_source": "recomputed",
                "target_stage": target_info.get("solve_stage"),
                "selector_warm_start_enabled": bool(selector_warm.get("enabled", selector_warm_start)),
                "selector_warm_start_available": bool(selector_warm.get("available", False)),
                "selector_warm_start_used": bool(selector_warm.get("used", False)),
                "selector_prev_input_term_active": bool(selector_dbg.get("prev_input_term_active", False)),
                "selector_prev_state_term_active": bool(selector_dbg.get("prev_state_term_active", False)),
                "selector_Qr_diag_used": selector_dbg.get("Qr_diag_used"),
                "selector_R_u_ref_diag_used": selector_dbg.get("R_u_ref_diag_used"),
                "selector_R_delta_u_sel_diag_used": selector_dbg.get("R_delta_u_sel_diag_used"),
                "selector_Q_delta_x_diag_used": selector_dbg.get("Q_delta_x_diag_used"),
                "selector_Q_x_ref_diag_used": selector_dbg.get("Q_x_ref_diag_used"),
                "selector_Qx_base_diag_used": selector_dbg.get("Qx_base_diag_used"),
                "selector_Rdu_diag_used": selector_dbg.get("Rdu_diag_used"),
                "fallback_mode": None,
                "fallback_verified": False,
                "fallback_solver_status": None,
                "fallback_objective_value": None,
                "fallback_bounds_ok": None,
                "fallback_lyap_ok": None,
                "upstream_candidate_info": {
                    "source": "rl_policy",
                    "action_raw": action.copy(),
                    "mpc_tracking_target": mpc_tracking_target.copy(),
                    "mpc_tracking_target_source": mpc_tracking_target_source,
                    "target_mismatch_inf": target_mismatch_inf,
                    "constrained_info": constrained_info,
                },
                "mpc_tracking_target": mpc_tracking_target.copy(),
                "mpc_tracking_target_source": mpc_tracking_target_source,
                "target_mismatch_inf": target_mismatch_inf,
                "qcqp_tracking_target": mpc_tracking_target.copy(),
                "qcqp_tracking_target_source": mpc_tracking_target_source,
                "cx_s": None if cx_s is None else cx_s.copy(),
                "cd_d_s": None if cd_d_s is None else cd_d_s.copy(),
                "allow_trust_region_slack": False,
                "backup_target_available": bool(prev_target_info is not None and prev_target_info.get("success", False)),
                "final_lyap_value": replacement_info.get("V_next_first_applied"),
                "final_lyap_bound": replacement_info.get("V_bound"),
                "final_lyap_margin": None if replacement_info.get("contraction_margin_applied") is None else float(-replacement_info["contraction_margin_applied"]),
                "final_lyap_ok": replacement_info.get("first_step_contraction_satisfied_applied"),
                "final_y_next_pred": None if applied_eval is None or applied_eval.get("y_next_pred") is None else np.asarray(applied_eval["y_next_pred"], float).reshape(-1).copy(),
                "final_lyap_target_source": effective_target_source,
                "lyap_acceptance_mode": "hard_only",
                "first_step_contraction_on": bool(first_step_contraction_on),
            }
            if info.get("verified", False):
                last_verified_safe_dev = u_dev_safe.copy()
        else:
            u_dev_safe = np.clip(u_rl_dev, u_min, u_max)
            info = {
                "source": "rl",
                "accepted": True,
                "accept_reason": "bypass",
                "reject_reason": None,
                "candidate_bounds_ok": True,
                "candidate_move_ok": True,
                "candidate_lyap_ok": None,
                "u_cand": u_rl_dev.copy(),
                "u_safe": u_dev_safe.copy(),
                "u_prev": u_prev_dev.copy(),
                "u_s": None if effective_target_info is None else effective_target_info["u_s"].copy(),
                "x_s": None if effective_target_info is None else effective_target_info["x_s"].copy(),
                "d_s": None if effective_target_info is None else effective_target_info["d_s"].copy(),
                "y_s": None if effective_target_info is None else effective_target_info["y_s"].copy(),
                "r_s": None if effective_target_info is None or effective_target_info.get("r_s") is None else effective_target_info["r_s"].copy(),
                "V_k": None,
                "V_next_cand": None,
                "V_bound": None,
                "rho": rho_lyap,
                "eps_lyap": lyap_eps,
                "solver_status": None,
                "solver_name": None,
                "solver_residuals": {},
                "trust_region_violation": None,
                "slack_v": 0.0,
                "slack_u": 0.0,
                "correction_mode": "bypass",
                "qcqp_attempted": False,
                "qcqp_solved": False,
                "qcqp_hard_accepted": False,
                "qcqp_status": "not_attempted",
                "verified": True,
                "target_success": bool(target_info.get("success", False)),
                "current_target_success": bool(target_info.get("success", False)),
                "current_target_stage": target_info.get("solve_stage"),
                "effective_target_success": bool(effective_target_info is not None and effective_target_info.get("success", False)),
                "effective_target_stage": None if effective_target_info is None else effective_target_info.get("solve_stage"),
                "effective_target_source": effective_target_source,
                "effective_target_reused": bool(effective_target_source == "last_valid_target"),
                "selector_mode": target_info.get("selector_mode"),
                "effective_selector_mode": None if effective_target_info is None else effective_target_info.get("selector_mode"),
                "selector_name": target_info.get("selector_name"),
                "selector_objective_terms": target_info.get("objective_terms"),
                "d_s_minus_dhat_inf": None if effective_target_info is None else effective_target_info.get("d_s_minus_dhat_inf"),
                "d_s_frozen": None if effective_target_info is None else effective_target_info.get("d_s_frozen"),
                "d_s_optimized": None if effective_target_info is None else effective_target_info.get("d_s_optimized"),
                "target_info": target_info,
                "setpoint_changed": bool(setpoint_changed),
                "target_source": "recomputed",
                "target_stage": target_info.get("solve_stage"),
                "selector_warm_start_enabled": bool(selector_warm_start),
                "selector_warm_start_available": bool(target_info.get("warm_start", {}).get("available", False)),
                "selector_warm_start_used": bool(target_info.get("warm_start", {}).get("used", False)),
                "selector_prev_input_term_active": bool(target_info.get("selector_debug", {}).get("prev_input_term_active", False)),
                "selector_prev_state_term_active": bool(target_info.get("selector_debug", {}).get("prev_state_term_active", False)),
                "selector_Qr_diag_used": target_info.get("selector_debug", {}).get("Qr_diag_used"),
                "selector_R_u_ref_diag_used": target_info.get("selector_debug", {}).get("R_u_ref_diag_used"),
                "selector_R_delta_u_sel_diag_used": target_info.get("selector_debug", {}).get("R_delta_u_sel_diag_used"),
                "selector_Q_delta_x_diag_used": target_info.get("selector_debug", {}).get("Q_delta_x_diag_used"),
                "selector_Q_x_ref_diag_used": target_info.get("selector_debug", {}).get("Q_x_ref_diag_used"),
                "selector_Qx_base_diag_used": target_info.get("selector_debug", {}).get("Qx_base_diag_used"),
                "selector_Rdu_diag_used": target_info.get("selector_debug", {}).get("Rdu_diag_used"),
                "selector_objective_value": target_info.get("objective_value"),
                "fallback_mode": None,
                "fallback_verified": False,
                "fallback_solver_status": None,
                "fallback_objective_value": None,
                "fallback_bounds_ok": None,
                "fallback_lyap_ok": None,
                "upstream_candidate_info": {
                    "source": "rl_policy",
                    "action_raw": action.copy(),
                    "mpc_tracking_target": mpc_tracking_target.copy(),
                    "mpc_tracking_target_source": mpc_tracking_target_source,
                    "target_mismatch_inf": target_mismatch_inf,
                },
                "mpc_tracking_target": mpc_tracking_target.copy(),
                "mpc_tracking_target_source": mpc_tracking_target_source,
                "target_mismatch_inf": target_mismatch_inf,
                "qcqp_tracking_target": mpc_tracking_target.copy(),
                "qcqp_tracking_target_source": mpc_tracking_target_source,
                "cx_s": None if cx_s is None else cx_s.copy(),
                "cd_d_s": None if cd_d_s is None else cd_d_s.copy(),
            }
            last_verified_safe_dev = u_dev_safe.copy()

        lyap_info_storage.append(info)

        if use_lyap:
            total_checked += 1
            checked_in_block += 1
            if info.get("correction_mode") == "optimized_correction" or info.get("constrained_mpc_applied", False):
                total_filtered += 1
                filtered_in_block += 1
            if str(info.get("correction_mode", "")).startswith("fallback_mpc") or info.get("constrained_mpc_failed_applied_candidate", False):
                total_fallback_mpc += 1
                fallback_in_block += 1

        u_safe_dev_store[k, :] = u_dev_safe

        a_used = inv_map_from_bounds(u_dev_safe, u_min, u_max).astype(np.float32)

        u_scaled_applied[k, :] = u_dev_safe + ss_scaled_u
        u_plant = reverse_min_max(u_scaled_applied[k, :], data_min[:n_u], data_max[:n_u])

        delta_u = u_scaled_applied[k, :] - u_prev_scaled

        _set_system_input_phys(system, steady_states, u_plant)
        system.step()

        if mode == "disturb":
            system.hA = ha[k]
            system.Qs = qs[k]
            system.Qi = qi[k]

        _u_phys_next, y_phys_next = _system_io_phys(system, steady_states)
        y_system[k + 1, :] = y_phys_next

        y_next_dev = apply_min_max(y_phys_next, data_min[n_u:], data_max[n_u:]) - ss_scaled_y
        e_next = y_next_dev - y_sp_kp1
        e_store[k + 1, :] = e_next

        innov = y_prev_dev - y_hat_k
        xhat_aug_store[:, k + 1] = (
            (MPC_obj.A @ xhat_aug_store[:, k])
            + (MPC_obj.B @ u_dev_safe)
            + (L @ innov)
        )

        delta_y = y_next_dev - y_sp_k
        y_sp_phys = reverse_min_max(y_sp_k + ss_scaled_y, data_min[n_u:], data_max[n_u:])
        r = reward_fn(delta_y, delta_u, y_sp_phys)
        rewards[k] = float(r)

        next_u_dev = u_scaled_applied[k, :] - ss_scaled_u
        # Keep the TD3 transition tied to the setpoint active when the
        # action was chosen and rewarded. Using y_sp_kp1 here would mix
        # two different tasks at a setpoint-change boundary.
        next_state = apply_rl_scaled(min_max_dict, xhat_aug_store[:, k + 1], y_sp_k, next_u_dev)

        done = 0.0
        if not test:
            agent.push(rl_state, a_used, float(r), next_state, float(done))
            if k >= WARM_START:
                _ = agent.train_step(actor_update=True)

        if k in sub_changes:
            start = max(0, k - time_in_sub_episodes + 1)
            avg_rewards.append(float(np.mean(rewards[start:k + 1])))
            print("Sub_Episode:", sub_changes[k], "| avg. reward:", avg_rewards[-1])

            block_ratio = filtered_in_block / checked_in_block if checked_in_block > 0 else 0.0
            total_ratio = total_filtered / total_checked if total_checked > 0 else 0.0
            fallback_ratio = fallback_in_block / checked_in_block if checked_in_block > 0 else 0.0
            if projection_backend == "first_step_contraction_mpc":
                print(
                    "Constrained MPC applied in block:",
                    filtered_in_block, "/", checked_in_block,
                    "(ratio:", block_ratio, ")",
                    "| constrained MPC failed, candidate applied:",
                    fallback_in_block, "/", checked_in_block,
                    "(ratio:", fallback_ratio, ")",
                    "| total constrained replacements:",
                    total_filtered, "/", total_checked,
                    "(ratio:", total_ratio, ")",
                )
            else:
                print(
                    "Lyap corrected in block:",
                    filtered_in_block, "/", checked_in_block,
                    "(ratio:", block_ratio, ")",
                    "| fallback MPC in block:",
                    fallback_in_block, "/", checked_in_block,
                    "(ratio:", fallback_ratio, ")",
                    "| total corrected:",
                    total_filtered, "/", total_checked,
                    "(ratio:", total_ratio, ")",
                )

            last = lyap_info_storage[-1]
            last_target = last.get("target_info", {})
            last_selector = {} if last_target is None else last_target.get("selector_debug", {})
            print(
                "Last Lyap mode:", last.get("correction_mode"),
                "| verified:", last.get("verified"),
                "| V_next:", last.get("V_next_cand"),
                "| V_bound:", last.get("V_bound"),
                "| constrained_status:", last.get("solver_status"),
                "| constrained_applied:", last.get("constrained_mpc_applied"),
                "| target_stage:", last_target.get("solve_stage") if last_target else None,
                "| target_slack_inf:", last_target.get("target_slack_inf") if last_target else None,
                "| selector_status:", last_selector.get("status"),
            )

            filtered_in_block = 0
            checked_in_block = 0
            fallback_in_block = 0

    u_applied_phys = reverse_min_max(u_scaled_applied, data_min[:n_u], data_max[:n_u])

    return (
        y_system,
        u_applied_phys,
        avg_rewards,
        rewards,
        xhat_aug_store,
        nFE,
        time_in_sub_episodes,
        y_sp,
        yhat,
        e_store,
        qi,
        qs,
        ha,
        lyap_info_storage,
        u_safe_dev_store,
    )
