import numpy as np

from Lyapunov.lyapunov_core import design_lyapunov_filter_ingredients
from Lyapunov.safety_filter import apply_lyapunov_safety_filter
from Lyapunov.target_selector import prepare_filter_target_from_refined_selector
from Lyapunov.upstream_controllers import (
    build_repeated_input_bounds,
    default_mpc_initial_guess,
    solve_offset_free_mpc_candidate,
)
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.scaling_helpers import apply_min_max, reverse_min_max


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


def _select_mpc_tracking_target(y_sp_raw, target_info, policy="raw_setpoint"):
    y_sp_raw = np.asarray(y_sp_raw, float).reshape(-1)
    target_info = {} if target_info is None else dict(target_info)
    y_s = target_info.get("y_s")
    stage = target_info.get("solve_stage")

    if y_s is not None:
        y_s = np.asarray(y_s, float).reshape(-1)

    if policy == "raw_setpoint":
        return y_sp_raw.copy(), "raw_setpoint"
    if policy == "admissible_if_available":
        if y_s is not None and bool(target_info.get("success", False)):
            return y_s.copy(), "admissible_target"
        return y_sp_raw.copy(), "raw_setpoint"
    if policy == "admissible_on_fallback":
        if y_s is not None and bool(target_info.get("success", False)) and stage == "fallback":
            return y_s.copy(), "admissible_target_fallback"
        return y_sp_raw.copy(), "raw_setpoint"
    raise ValueError(
        "policy must be one of 'raw_setpoint', 'admissible_if_available', or 'admissible_on_fallback'."
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


def run_mpc_lyapunov(
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
    w_mpc=1.0,
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
    use_lyap=True,
    du_min=None,
    du_max=None,
    trust_region_delta=None,
    allow_lyap_slack=False,
    target_solver_pref=None,
    filter_solver_pref=None,
    fallback_policy="offset_free_mpc",
    mpc_target_policy="raw_setpoint",
    tracking_target_policy=None,
    target_backup_policy="last_valid",
    selector_warm_start=True,
    Qdx_tgt_diag=None,
    lyap_acceptance_mode="hard_only",
    allow_trust_region_slack=False,
    reuse_mpc_solution_as_ic=False,
    reset_system_on_entry=True,
):
    if reset_system_on_entry:
        _reset_system_on_entry(system)

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

    total_checked = 0
    total_filtered = 0
    total_fallback_mpc = 0
    checked_in_block = 0
    filtered_in_block = 0
    fallback_in_block = 0

    prev_target_info = None
    last_verified_safe_dev = None

    for k in range(nFE):
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

        target_info = prepare_filter_target_from_refined_selector(
            A_aug=MPC_obj.A,
            B_aug=MPC_obj.B,
            C_aug=MPC_obj.C,
            xhat_aug=xhat_aug_store[:, k],
            y_sp=y_sp_k,
            u_min=u_min,
            u_max=u_max,
            u_nom=u_nom_tgt,
            Ty_diag=Qs_tgt_diag,
            Ru_diag=Ru_tgt_diag,
            Qx_diag=None,
            w_x=w_x_tgt,
            prev_target=prev_target_info,
            x_s_prev=None,
            u_s_prev=None,
            Qdx_diag=Qdx_tgt_diag,
            Rdu_diag=Rmove_diag,
            solver_pref=target_solver_pref,
            warm_start=selector_warm_start,
            return_debug=False,
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

        u_mpc_cand, upstream_info = solve_offset_free_mpc_candidate(
            MPC_obj=MPC_obj,
            y_sp=mpc_tracking_target,
            u_prev_dev=u_prev_dev,
            x0_model=xhat_aug_store[:, k],
            IC_opt=mpc_ic,
            bnds=mpc_bnds,
            cons=mpc_cons,
            return_debug=True,
        )
        if reuse_mpc_solution_as_ic and upstream_info.get("IC_opt_next") is not None:
            mpc_ic = np.asarray(upstream_info["IC_opt_next"], float).reshape(-1).copy()

        if u_mpc_cand is None:
            u_mpc_cand = np.clip(u_prev_dev, u_min, u_max)

        if (k + 1) < y_sp.shape[0]:
            y_sp_kp1 = np.asarray(y_sp[k + 1, :], float).reshape(-1)
        else:
            y_sp_kp1 = y_sp_k.copy()

        if use_lyap:
            safe_filter_prev = last_verified_safe_dev if last_verified_safe_dev is not None else u_prev_dev
            u_dev_safe, info = apply_lyapunov_safety_filter(
                u_cand=u_mpc_cand,
                xhat_aug=xhat_aug_store[:, k],
                target_info=target_info,
                model_info=lyap_model,
                lyap_config={
                    "source": "mpc",
                    "rho": rho_lyap,
                    "eps_lyap": lyap_eps,
                    "tol": lyap_tol,
                    "selector_warm_start": bool(selector_warm_start),
                    "target_backup_policy": str(target_backup_policy),
                    "backup_target_info": prev_target_info,
                    "backup_target_source": "last_valid_target" if prev_target_info is not None else None,
                    "lyap_acceptance_mode": str(lyap_acceptance_mode),
                    "candidate_weight_diag": float(w_mpc) * np.ones(n_u, dtype=float),
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
                    "IC_opt": mpc_ic,
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
                mpc_ic = np.asarray(info["fallback_ic_next"], float).reshape(-1).copy()
            info["setpoint_changed"] = bool(setpoint_changed)
            info["target_source"] = "recomputed"
            info["target_stage"] = target_info.get("solve_stage")
            info["current_target_success"] = bool(target_info.get("success", False))
            info["current_target_stage"] = target_info.get("solve_stage")
            info["effective_target_success"] = bool(effective_target_info is not None and effective_target_info.get("success", False))
            info["effective_target_stage"] = None if effective_target_info is None else effective_target_info.get("solve_stage")
            info["effective_target_source"] = effective_target_source
            info["effective_target_reused"] = bool(effective_target_source == "last_valid_target")
            selector_warm = target_info.get("warm_start", {})
            selector_dbg = target_info.get("selector_debug", {})
            info["selector_warm_start_enabled"] = bool(selector_warm.get("enabled", selector_warm_start))
            info["selector_warm_start_available"] = bool(selector_warm.get("available", False))
            info["selector_warm_start_used"] = bool(selector_warm.get("used", False))
            info["selector_Qdx_diag_used"] = selector_dbg.get("Qdx_diag_used")
            info["selector_Rdu_diag_used"] = selector_dbg.get("Rdu_diag_used")
            upstream_info = dict(upstream_info)
            upstream_info["mpc_tracking_target"] = mpc_tracking_target.copy()
            upstream_info["mpc_tracking_target_source"] = mpc_tracking_target_source
            upstream_info["target_mismatch_inf"] = target_mismatch_inf
            info["upstream_candidate_info"] = upstream_info
            info["mpc_tracking_target"] = mpc_tracking_target.copy()
            info["mpc_tracking_target_source"] = mpc_tracking_target_source
            info["target_mismatch_inf"] = target_mismatch_inf
            info["qcqp_tracking_target"] = mpc_tracking_target.copy()
            info["qcqp_tracking_target_source"] = mpc_tracking_target_source
            if info.get("verified", False):
                last_verified_safe_dev = u_dev_safe.copy()
        else:
            u_dev_safe = np.clip(u_mpc_cand, u_min, u_max)
            info = {
                "source": "mpc",
                "accepted": True,
                "accept_reason": "bypass",
                "reject_reason": None,
                "candidate_bounds_ok": True,
                "candidate_move_ok": True,
                "candidate_lyap_ok": None,
                "u_cand": u_mpc_cand.copy(),
                "u_safe": u_dev_safe.copy(),
                "u_prev": u_prev_dev.copy(),
                "u_s": None if effective_target_info is None else effective_target_info["u_s"].copy(),
                "x_s": None if effective_target_info is None else effective_target_info["x_s"].copy(),
                "d_s": None if effective_target_info is None else effective_target_info["d_s"].copy(),
                "y_s": None if effective_target_info is None else effective_target_info["y_s"].copy(),
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
                "target_info": target_info,
                "setpoint_changed": bool(setpoint_changed),
                "target_source": "recomputed",
                "target_stage": target_info.get("solve_stage"),
                "selector_warm_start_enabled": bool(selector_warm_start),
                "selector_warm_start_available": bool(target_info.get("warm_start", {}).get("available", False)),
                "selector_warm_start_used": bool(target_info.get("warm_start", {}).get("used", False)),
                "selector_Qdx_diag_used": target_info.get("selector_debug", {}).get("Qdx_diag_used"),
                "selector_Rdu_diag_used": target_info.get("selector_debug", {}).get("Rdu_diag_used"),
                "fallback_mode": None,
                "fallback_verified": False,
                "fallback_solver_status": None,
                "fallback_objective_value": None,
                "fallback_bounds_ok": None,
                "fallback_lyap_ok": None,
                "upstream_candidate_info": {
                    **upstream_info,
                    "mpc_tracking_target": mpc_tracking_target.copy(),
                    "mpc_tracking_target_source": mpc_tracking_target_source,
                    "target_mismatch_inf": target_mismatch_inf,
                },
                "mpc_tracking_target": mpc_tracking_target.copy(),
                "mpc_tracking_target_source": mpc_tracking_target_source,
                "target_mismatch_inf": target_mismatch_inf,
                "qcqp_tracking_target": mpc_tracking_target.copy(),
                "qcqp_tracking_target_source": mpc_tracking_target_source,
            }
            last_verified_safe_dev = u_dev_safe.copy()

        lyap_info_storage.append(info)

        total_checked += 1
        checked_in_block += 1
        if info.get("correction_mode") == "optimized_correction":
            total_filtered += 1
            filtered_in_block += 1
        if str(info.get("correction_mode", "")).startswith("fallback_mpc"):
            total_fallback_mpc += 1
            fallback_in_block += 1

        u_safe_dev_store[k, :] = u_dev_safe
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
        rewards[k] = float(reward_fn(delta_y, delta_u, y_sp_phys))

        if k in sub_changes:
            start = max(0, k - time_in_sub_episodes + 1)
            avg_rewards.append(float(np.mean(rewards[start:k + 1])))
            print("Sub_Episode:", sub_changes[k], "| avg. reward:", avg_rewards[-1])

            block_ratio = filtered_in_block / checked_in_block if checked_in_block > 0 else 0.0
            fallback_ratio = fallback_in_block / checked_in_block if checked_in_block > 0 else 0.0
            total_ratio = total_filtered / total_checked if total_checked > 0 else 0.0
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
