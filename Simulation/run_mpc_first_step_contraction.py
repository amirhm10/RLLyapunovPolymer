import numpy as np

from Lyapunov.lyapunov_core import design_lyapunov_filter_ingredients, evaluate_candidate_action
from Lyapunov.target_selector import build_target_selector_config, prepare_filter_target
from Lyapunov.upstream_controllers import (
    build_repeated_input_bounds,
    default_mpc_initial_guess,
    solve_offset_free_mpc_candidate,
    solve_offset_free_mpc_candidate_with_first_step_contraction,
)
from Simulation.run_mpc_lyapunov import (
    _as_selector_config_dict,
    _coerce_supplied_lyapunov_matrix,
    _normalize_tracking_target_policy,
    _reset_system_on_entry,
    _resolve_effective_target,
    _select_mpc_tracking_target,
    _selector_decomposition,
    _system_io_phys,
    _set_system_input_phys,
    _target_selector_overrides,
)
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.scaling_helpers import apply_min_max, reverse_min_max


def _normalize_mpc_setup(MPC_obj, u_min, u_max, IC_opt, bnds, cons):
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


def _selector_config_overrides(target_selector_config, u_nom_tgt, Qs_tgt_diag, Ru_tgt_diag, w_x_tgt, Qdx_tgt_diag, Rmove_diag, target_solver_pref):
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


def run_mpc_first_step_contraction(
    system,
    MPC_obj,
    y_sp_scenario,
    n_tests,
    set_points_len,
    steady_states,
    IC_opt,
    bnds,
    cons,
    warm_start,
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
    mode="disturb",
    P_lyap=None,
    rho_lyap=0.99,
    lyap_eps=1e-9,
    lyap_tol=1e-10,
    Qy_track_diag=None,
    Rmove_diag=None,
    Qs_tgt_diag=None,
    Ru_tgt_diag=None,
    u_nom_tgt=None,
    w_x_tgt=1e-6,
    lambda_u_ric=1.0,
    pd_eps_ric=0.0,
    fallback_policy="offset_free_mpc",
    mpc_target_policy="raw_setpoint",
    tracking_target_policy=None,
    selector_mode=None,
    target_selector_config=None,
    selector_H=None,
    target_backup_policy="last_valid",
    selector_warm_start=True,
    Qdx_tgt_diag=None,
    reuse_mpc_solution_as_ic=False,
    reset_system_on_entry=True,
    first_step_contraction_on=True,
    target_solver_pref=None,
):
    if reset_system_on_entry:
        _reset_system_on_entry(system)
    if str(fallback_policy) != "offset_free_mpc":
        raise ValueError("fallback_policy must be 'offset_free_mpc' for the first-step-contraction MPC path.")

    (
        y_sp,
        nFE,
        sub_changes,
        time_in_sub_episodes,
        _test_train_dict,
        _warm_start_idx,
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
    tracking_target_policy = _normalize_tracking_target_policy(
        mpc_target_policy=mpc_target_policy,
        tracking_target_policy=tracking_target_policy,
    )

    ss_scaled_u = apply_min_max(steady_states["ss_inputs"], data_min[:n_u], data_max[:n_u])
    ss_scaled_y = apply_min_max(steady_states["y_ss"], data_min[n_u:], data_max[n_u:])

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

    selector_cfg = build_target_selector_config(
        selector_mode=selector_mode,
        user_overrides=_selector_config_overrides(
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

    u_min = np.array([float(lo) for (lo, _hi) in bnds[:n_u]], dtype=float)
    u_max = np.array([float(hi) for (_lo, hi) in bnds[:n_u]], dtype=float)

    mpc_ic, mpc_bnds, mpc_cons = _normalize_mpc_setup(
        MPC_obj=MPC_obj,
        u_min=u_min,
        u_max=u_max,
        IC_opt=IC_opt,
        bnds=bnds,
        cons=cons,
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

    prev_target_info = None

    for k in range(nFE):
        u_prev_phys, y_prev_phys = _system_io_phys(system, steady_states)

        u_prev_scaled = apply_min_max(u_prev_phys, data_min[:n_u], data_max[:n_u])
        u_prev_dev = u_prev_scaled - ss_scaled_u

        y_prev_dev = apply_min_max(y_prev_phys, data_min[n_u:], data_max[n_u:]) - ss_scaled_y
        y_hat_k = MPC_obj.C @ xhat_aug_store[:, k]
        yhat[:, k] = y_hat_k

        y_sp_k = np.asarray(y_sp[k, :], float).reshape(-1)
        y_sp_kp1 = y_sp_k.copy() if (k + 1) >= y_sp.shape[0] else np.asarray(y_sp[k + 1, :], float).reshape(-1)
        setpoint_changed = True if k == 0 else not np.array_equal(y_sp_k, np.asarray(y_sp[k - 1, :], float).reshape(-1))

        e_k = y_prev_dev - y_sp_k
        e_store[k, :] = e_k

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
            selector_mode=selector_mode,
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
        cx_s, cd_d_s = _selector_decomposition(MPC_obj.C, n_x, effective_target_info, xhat_aug_store[n_x:, k])

        constrained_info = None
        if first_step_contraction_on and effective_target_info is not None and effective_target_info.get("success", False):
            u_mpc_cand, constrained_info = solve_offset_free_mpc_candidate_with_first_step_contraction(
                MPC_obj=MPC_obj,
                y_sp=mpc_tracking_target,
                u_prev_dev=u_prev_dev,
                x0_model=xhat_aug_store[:, k],
                x_s=np.asarray(effective_target_info["x_s"], float).reshape(-1),
                P_x=lyap_model["P_x"],
                rho_lyap=rho_lyap,
                eps_lyap=lyap_eps,
                lyap_tol=lyap_tol,
                IC_opt=mpc_ic,
                bnds=mpc_bnds,
                cons=mpc_cons,
                return_debug=True,
            )
        else:
            u_mpc_cand, constrained_info = solve_offset_free_mpc_candidate(
                MPC_obj=MPC_obj,
                y_sp=mpc_tracking_target,
                u_prev_dev=u_prev_dev,
                x0_model=xhat_aug_store[:, k],
                IC_opt=mpc_ic,
                bnds=mpc_bnds,
                cons=mpc_cons,
                return_debug=True,
            )

        if reuse_mpc_solution_as_ic and constrained_info.get("IC_opt_next") is not None:
            mpc_ic = np.asarray(constrained_info["IC_opt_next"], float).reshape(-1).copy()

        u_cand = None if u_mpc_cand is None else np.clip(np.asarray(u_mpc_cand, float).reshape(-1), u_min, u_max)
        candidate_eval = evaluate_candidate_action(
            u_cand=u_cand if u_cand is not None else np.clip(u_prev_dev, u_min, u_max),
            xhat_aug=xhat_aug_store[:, k],
            target_info=effective_target_info,
            ingredients=lyap_model,
            rho=rho_lyap,
            eps_lyap=lyap_eps,
            u_min=u_min,
            u_max=u_max,
            u_prev=u_prev_dev,
            du_min=None,
            du_max=None,
            tol=lyap_tol,
        )

        target_warm = target_info.get("warm_start", {})
        selector_dbg = target_info.get("selector_debug", {})
        info = {
            "source": "mpc_first_step_contraction",
            "accepted": False,
            "accept_reason": None,
            "reject_reason": None,
            "candidate_bounds_ok": candidate_eval.get("candidate_bounds_ok"),
            "candidate_move_ok": candidate_eval.get("candidate_move_ok"),
            "candidate_lyap_ok": candidate_eval.get("candidate_lyap_ok"),
            "u_cand": None if u_cand is None else u_cand.copy(),
            "u_prev": u_prev_dev.copy(),
            "u_safe": None,
            "u_s": None if effective_target_info is None or effective_target_info.get("u_s") is None else np.asarray(effective_target_info["u_s"], float).reshape(-1).copy(),
            "x_s": None if effective_target_info is None or effective_target_info.get("x_s") is None else np.asarray(effective_target_info["x_s"], float).reshape(-1).copy(),
            "d_s": None if effective_target_info is None or effective_target_info.get("d_s") is None else np.asarray(effective_target_info["d_s"], float).reshape(-1).copy(),
            "y_s": None if effective_target_info is None or effective_target_info.get("y_s") is None else np.asarray(effective_target_info["y_s"], float).reshape(-1).copy(),
            "r_s": None if effective_target_info is None or effective_target_info.get("r_s") is None else np.asarray(effective_target_info["r_s"], float).reshape(-1).copy(),
            "e_x": None if candidate_eval.get("e_x") is None else np.asarray(candidate_eval["e_x"], float).reshape(-1).copy(),
            "V_k": constrained_info.get("V_k") if constrained_info.get("V_k") is not None else candidate_eval.get("V_k"),
            "V_next_first": constrained_info.get("V_next_first"),
            "V_next_cand": constrained_info.get("V_next_first") if constrained_info.get("V_next_first") is not None else candidate_eval.get("V_next_cand"),
            "V_bound": constrained_info.get("V_bound") if constrained_info.get("V_bound") is not None else candidate_eval.get("V_bound"),
            "contraction_margin": constrained_info.get("contraction_margin"),
            "first_step_contraction_satisfied": constrained_info.get("first_step_contraction_satisfied"),
            "contraction_constraint_violation": None if constrained_info.get("contraction_margin") is None else float(max(constrained_info["contraction_margin"], 0.0)),
            "rho": float(rho_lyap),
            "eps_lyap": float(lyap_eps),
            "solver_status": constrained_info.get("status"),
            "solver_name": constrained_info.get("solver_name"),
            "solver_residuals": {},
            "slack_v": 0.0,
            "slack_u": 0.0,
            "trust_region_violation": None,
            "correction_mode": None,
            "qcqp_attempted": False,
            "qcqp_solved": False,
            "qcqp_hard_accepted": False,
            "qcqp_status": "not_attempted",
            "verified": False,
            "target_success": bool(target_info.get("success", False)),
            "current_target_success": bool(target_info.get("success", False)),
            "current_target_stage": target_info.get("solve_stage"),
            "target_stage": target_info.get("solve_stage"),
            "target_source": "recomputed",
            "selector_mode": target_info.get("selector_mode"),
            "effective_target_success": bool(effective_target_info is not None and effective_target_info.get("success", False)),
            "effective_target_source": effective_target_source,
            "effective_target_stage": None if effective_target_info is None else effective_target_info.get("solve_stage"),
            "effective_target_reused": bool(effective_target_source == "last_valid_target"),
            "effective_selector_mode": None if effective_target_info is None else effective_target_info.get("selector_mode"),
            "d_s_minus_dhat_inf": None if effective_target_info is None else effective_target_info.get("d_s_minus_dhat_inf"),
            "d_s_frozen": None if effective_target_info is None else effective_target_info.get("d_s_frozen"),
            "d_s_optimized": None if effective_target_info is None else effective_target_info.get("d_s_optimized"),
            "backup_target_available": bool(prev_target_info is not None and prev_target_info.get("success", False)),
            "target_info": target_info,
            "effective_target_info": effective_target_info,
            "u_fallback_mpc": None,
            "fallback_mode": None,
            "fallback_verified": False,
            "fallback_solver_status": None,
            "fallback_solver_message": None,
            "fallback_objective_value": None,
            "fallback_bounds_ok": None,
            "fallback_move_ok": None,
            "fallback_lyap_ok": None,
            "fallback_ic_next": None,
            "fallback_upstream_info": None,
            "fallback_tracking_target_source": None,
            "fallback_target_mismatch_inf": None,
            "mpc_tracking_target": mpc_tracking_target.copy(),
            "mpc_tracking_target_source": mpc_tracking_target_source,
            "qcqp_tracking_target": mpc_tracking_target.copy(),
            "qcqp_tracking_target_source": mpc_tracking_target_source,
            "target_mismatch_inf": target_mismatch_inf,
            "selector_warm_start_enabled": bool(target_warm.get("enabled", selector_warm_start)),
            "selector_warm_start_available": bool(target_warm.get("available", False)),
            "selector_warm_start_used": bool(target_warm.get("used", False)),
            "selector_prev_input_term_active": bool(selector_dbg.get("prev_input_term_active", False)),
            "selector_prev_state_term_active": bool(selector_dbg.get("prev_state_term_active", False)),
            "selector_Qr_diag_used": selector_dbg.get("Qr_diag_used"),
            "selector_R_u_ref_diag_used": selector_dbg.get("R_u_ref_diag_used"),
            "selector_R_delta_u_sel_diag_used": selector_dbg.get("R_delta_u_sel_diag_used"),
            "selector_Q_delta_x_diag_used": selector_dbg.get("Q_delta_x_diag_used"),
            "selector_Q_x_ref_diag_used": selector_dbg.get("Q_x_ref_diag_used"),
            "selector_Qx_base_diag_used": selector_dbg.get("Qx_base_diag_used"),
            "selector_Rdu_diag_used": selector_dbg.get("Rdu_diag_used"),
            "selector_objective_terms": target_info.get("objective_terms"),
            "selector_objective_value": target_info.get("objective_value"),
            "cx_s": None if cx_s is None else cx_s.copy(),
            "cd_d_s": None if cd_d_s is None else cd_d_s.copy(),
            "upstream_candidate_info": {
                **constrained_info,
                "mpc_tracking_target": mpc_tracking_target.copy(),
                "mpc_tracking_target_source": mpc_tracking_target_source,
                "target_mismatch_inf": target_mismatch_inf,
                "first_step_contraction_on": bool(first_step_contraction_on),
            },
            "allow_trust_region_slack": False,
            "final_lyap_value": None,
            "final_lyap_margin": None,
            "final_lyap_ok": None,
            "final_lyap_bound": None,
            "final_y_next_pred": None,
            "final_lyap_target_source": effective_target_source,
            "setpoint_changed": bool(setpoint_changed),
            "first_step_contraction_on": bool(first_step_contraction_on),
        }

        constrained_success = bool(constrained_info.get("success", False))
        if not first_step_contraction_on:
            constrained_success = True
            info.update({
                "accepted": True,
                "accept_reason": "candidate_ok",
                "reject_reason": None if u_cand is not None else "solver_status",
                "u_safe": np.clip(u_prev_dev, u_min, u_max) if u_cand is None else u_cand.copy(),
                "correction_mode": "accepted_candidate",
                "verified": bool(candidate_eval.get("accepted", False)),
                "final_lyap_value": candidate_eval.get("V_next_cand"),
                "final_lyap_bound": candidate_eval.get("V_bound"),
                "final_lyap_margin": None if candidate_eval.get("V_next_cand") is None or candidate_eval.get("V_bound") is None else float(candidate_eval.get("V_bound")) - float(candidate_eval.get("V_next_cand")),
                "final_lyap_ok": candidate_eval.get("candidate_lyap_ok"),
                "final_y_next_pred": None if candidate_eval.get("y_next_pred") is None else np.asarray(candidate_eval["y_next_pred"], float).reshape(-1).copy(),
                "first_step_contraction_satisfied": constrained_info.get("first_step_contraction_satisfied", candidate_eval.get("candidate_lyap_ok")),
                "contraction_margin": constrained_info.get("contraction_margin", candidate_eval.get("lyap_margin")),
            })
        elif constrained_success and u_cand is not None:
            info.update({
                "accepted": True,
                "accept_reason": "candidate_ok",
                "reject_reason": None,
                "u_safe": u_cand.copy(),
                "correction_mode": "accepted_candidate",
                "verified": True,
                "final_lyap_value": constrained_info.get("V_next_first", candidate_eval.get("V_next_cand")),
                "final_lyap_bound": constrained_info.get("V_bound", candidate_eval.get("V_bound")),
                "final_lyap_margin": None if constrained_info.get("contraction_margin") is None else float(-constrained_info["contraction_margin"]),
                "final_lyap_ok": constrained_info.get("first_step_contraction_satisfied"),
                "final_y_next_pred": None if candidate_eval.get("y_next_pred") is None else np.asarray(candidate_eval["y_next_pred"], float).reshape(-1).copy(),
            })
        else:
            u_fallback, fallback_info = solve_offset_free_mpc_candidate(
                MPC_obj=MPC_obj,
                y_sp=mpc_tracking_target,
                u_prev_dev=u_prev_dev,
                x0_model=xhat_aug_store[:, k],
                IC_opt=mpc_ic,
                bnds=mpc_bnds,
                cons=mpc_cons,
                return_debug=True,
            )
            if reuse_mpc_solution_as_ic and fallback_info.get("IC_opt_next") is not None:
                mpc_ic = np.asarray(fallback_info["IC_opt_next"], float).reshape(-1).copy()
            u_fallback = np.clip(u_prev_dev, u_min, u_max) if u_fallback is None else np.clip(np.asarray(u_fallback, float).reshape(-1), u_min, u_max)
            fallback_eval = evaluate_candidate_action(
                u_cand=u_fallback,
                xhat_aug=xhat_aug_store[:, k],
                target_info=effective_target_info,
                ingredients=lyap_model,
                rho=rho_lyap,
                eps_lyap=lyap_eps,
                u_min=u_min,
                u_max=u_max,
                u_prev=u_prev_dev,
                du_min=None,
                du_max=None,
                tol=lyap_tol,
            )
            fallback_verified = bool(fallback_eval.get("accepted", False))
            info.update({
                "accepted": fallback_verified,
                "accept_reason": "fallback_mpc_verified" if fallback_verified else None,
                "reject_reason": None if fallback_verified else ("target_unavailable" if effective_target_info is None else "lyapunov"),
                "u_safe": u_fallback.copy(),
                "correction_mode": "fallback_mpc_verified" if fallback_verified else "fallback_mpc_unverified",
                "verified": fallback_verified,
                "u_fallback_mpc": u_fallback.copy(),
                "fallback_mode": fallback_policy,
                "fallback_verified": fallback_verified,
                "fallback_solver_status": fallback_info.get("status"),
                "fallback_solver_message": fallback_info.get("message"),
                "fallback_objective_value": fallback_info.get("objective_value"),
                "fallback_bounds_ok": fallback_eval.get("candidate_bounds_ok"),
                "fallback_move_ok": fallback_eval.get("candidate_move_ok"),
                "fallback_lyap_ok": fallback_eval.get("candidate_lyap_ok"),
                "fallback_ic_next": None if fallback_info.get("IC_opt_next") is None else np.asarray(fallback_info["IC_opt_next"], float).reshape(-1).copy(),
                "fallback_upstream_info": fallback_info,
                "fallback_tracking_target_source": mpc_tracking_target_source,
                "fallback_target_mismatch_inf": target_mismatch_inf,
                "final_lyap_value": fallback_eval.get("V_next_cand"),
                "final_lyap_bound": fallback_eval.get("V_bound"),
                "final_lyap_margin": None if fallback_eval.get("V_next_cand") is None or fallback_eval.get("V_bound") is None else float(fallback_eval.get("V_bound")) - float(fallback_eval.get("V_next_cand")),
                "final_lyap_ok": fallback_eval.get("candidate_lyap_ok"),
                "final_y_next_pred": None if fallback_eval.get("y_next_pred") is None else np.asarray(fallback_eval["y_next_pred"], float).reshape(-1).copy(),
            })

        u_dev_apply = np.asarray(info["u_safe"], float).reshape(-1)
        lyap_info_storage.append(info)

        u_safe_dev_store[k, :] = u_dev_apply
        u_scaled_applied[k, :] = u_dev_apply + ss_scaled_u
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
            + (MPC_obj.B @ u_dev_apply)
            + (L @ innov)
        )

        delta_y = y_next_dev - y_sp_k
        y_sp_phys = reverse_min_max(y_sp_k + ss_scaled_y, data_min[n_u:], data_max[n_u:])
        rewards[k] = float(reward_fn(delta_y, delta_u, y_sp_phys))

        if k in sub_changes:
            start = max(0, k - time_in_sub_episodes + 1)
            avg_rewards.append(float(np.mean(rewards[start:k + 1])))
            last = lyap_info_storage[-1]
            print("Sub_Episode:", sub_changes[k], "| avg. reward:", avg_rewards[-1])
            print(
                "Last mode:", last.get("correction_mode"),
                "| verified:", last.get("verified"),
                "| first-step ok:", last.get("first_step_contraction_satisfied"),
                "| contraction margin:", last.get("contraction_margin"),
                "| fallback status:", last.get("fallback_solver_status"),
                "| target stage:", last.get("target_stage"),
            )

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
