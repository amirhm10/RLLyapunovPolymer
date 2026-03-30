import numpy as np

from Lyapunov.lyapunov_core import compute_terminal_alpha_input_only
from Lyapunov.target_selector import compute_ss_target_refined_rawlings
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.lyapunov_utils import DEFAULT_CVXPY_SOLVERS, get_y_sp_step, shift_input_guess
from utils.scaling_helpers import apply_min_max, reverse_min_max


def run_standard_tracking_lyapunov_mpc_rawlings_target(
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
    Qs_tgt_diag=None,
    Ru_tgt_diag=None,
    u_nom_tgt=None,
    w_x_tgt=1e-6,
    mode="disturb",
    use_external_target_for_tracking=True,
    disturbance_after_step=True,
    skip_terminal_if_alpha_small=True,
    alpha_terminal_min=1e-8,
    use_target_on_solver_fail=False,
    Ty_tgt_diag=None,
    Qx_tgt_diag=None,
    Qdx_tgt_diag=None,
    Rdu_tgt_diag=None,
    u_tight_tgt=None,
    y_tight_tgt=None,
):
    if Ty_tgt_diag is None and Qs_tgt_diag is not None:
        Ty_tgt_diag = Qs_tgt_diag

    y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, _, _, qi, qs, ha = generate_setpoints_training_rl_gradually(
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

    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    u_dev_min = np.array([bnds[j][0] for j in range(n_inputs)], dtype=float)
    u_dev_max = np.array([bnds[j][1] for j in range(n_inputs)], dtype=float)

    y_mpc = np.zeros((nFE + 1, n_outputs), dtype=float)
    y_mpc[0, :] = system.current_output
    u_mpc = np.zeros((nFE, n_inputs), dtype=float)
    yhat = np.zeros((n_outputs, nFE), dtype=float)
    xhatdhat = np.zeros((n_aug, nFE + 1), dtype=float)

    rewards = np.zeros(nFE, dtype=float)
    avg_rewards = []
    delta_y_storage = []
    delta_u_storage = []
    lmpc_info_storage = []
    target_info_storage = []

    IC_opt = np.asarray(IC_opt, float).copy()
    x_s_prev = None
    u_s_prev = None

    for step_idx in range(nFE):
        x0_aug = xhatdhat[:, step_idx].copy()

        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        u_prev_dev = scaled_current_input - ss_scaled_inputs

        y_sp_k = get_y_sp_step(y_sp, step_idx, n_outputs)
        y_prev_scaled = apply_min_max(y_mpc[step_idx, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        yhat_now = np.asarray(LMPC_obj.C @ x0_aug, float).reshape(-1)
        innovation = y_prev_scaled - yhat_now

        x_s, u_s, d_s, dbg_tgt = compute_ss_target_refined_rawlings(
            A_aug=LMPC_obj.A,
            B_aug=LMPC_obj.B,
            C_aug=LMPC_obj.C,
            xhat_aug=x0_aug,
            y_sp=y_sp_k,
            u_min=u_dev_min,
            u_max=u_dev_max,
            u_applied_k=u_prev_dev,
            u_nom=u_nom_tgt,
            Ty_diag=Ty_tgt_diag,
            Ru_diag=Ru_tgt_diag,
            Qx_diag=Qx_tgt_diag,
            w_x=w_x_tgt,
            x_s_prev=x_s_prev,
            u_s_prev=u_s_prev,
            Qdx_diag=Qdx_tgt_diag,
            Rdu_diag=Rdu_tgt_diag,
            y_min=None,
            y_max=None,
            u_tight=u_tight_tgt,
            y_tight=y_tight_tgt,
            soft_output_bounds=True,
            solver_pref=DEFAULT_CVXPY_SOLVERS,
            return_debug=True,
        )

        dbg_tgt = {} if dbg_tgt is None else dict(dbg_tgt)
        dbg_tgt.update({
            "step": int(step_idx),
            "y_sp": y_sp_k.copy(),
            "x0_aug": x0_aug.copy(),
            "yhat_now": yhat_now.copy(),
            "innovation": innovation.copy(),
        })
        target_info_storage.append(dbg_tgt)

        step_info = {
            "step": int(step_idx),
            "success": False,
            "method": None,
            "status": None,
            "message": None,
            "fun": None,
            "solver_nit": None,
            "tracking_solver": None,
            "tracking_error": None,
            "x0_aug": x0_aug.copy(),
            "x_s": None if x_s is None else np.asarray(x_s, float).copy(),
            "u_s": None if u_s is None else np.asarray(u_s, float).copy(),
            "d_s": None if d_s is None else np.asarray(d_s, float).copy(),
            "x_s_aug": None if (x_s is None or d_s is None) else np.concatenate([x_s, d_s]).copy(),
            "y_sp": y_sp_k.copy(),
            "y_target": dbg_tgt.get("y_s"),
            "u_prev_dev": u_prev_dev.copy(),
            "yhat_now": yhat_now.copy(),
            "innovation": innovation.copy(),
            "target_success": bool(dbg_tgt.get("success", False)),
            "target_solver": dbg_tgt.get("solver"),
            "target_status": dbg_tgt.get("status"),
            "target_objective": dbg_tgt.get("objective_value"),
            "target_slack_inf": dbg_tgt.get("target_slack_inf"),
            "alpha_terminal_raw": None,
            "alpha_terminal": None,
            "alpha_terminal_used": None,
            "terminal_constraint_skipped": None,
            "terminal_value": None,
            "terminal_margin": None,
            "terminal_set_violated": None,
            "u_apply": None,
            "x_pred_path": None,
            "y_pred_path": None,
            "e_x_path": None,
            "e_y_path": None,
            "u_seq_opt": None,
            "du_seq_opt": None,
            "u_err_path": None,
            "y_meas_for_observer": y_prev_scaled.copy(),
            "y_current_scaled": None,
            "xhat_next_openloop": None,
            "observer_correction": None,
            "xhat_next": None,
            "reward": None,
            "delta_y": None,
            "delta_u": None,
        }

        if bool(dbg_tgt.get("success", False)) and x_s is not None and u_s is not None:
            x_s_aug = np.concatenate([x_s, d_s])
            if use_external_target_for_tracking:
                y_target = np.asarray(LMPC_obj.C @ x_s_aug, float).reshape(-1)
            else:
                y_target = y_sp_k.copy()

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

            x_s_prev = x_s.copy()
            u_s_prev = u_s.copy()

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
                )
            finally:
                LMPC_obj.terminal_set_on = terminal_set_on_prev

            step_info.update({
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
            })

            if getattr(sol, "success", False):
                u_dev_apply = np.asarray(sol.x[:n_inputs], float).reshape(-1)
                u_dev_apply = np.clip(u_dev_apply, u_dev_min, u_dev_max)
                IC_opt = shift_input_guess(sol.x, n_inputs, LMPC_obj.NC)

                report = LMPC_obj.standard_tracking_report(
                    x_opt=sol.x,
                    x0_aug=x0_aug,
                    x_s=x_s,
                    u_s=u_s,
                    y_target=y_target,
                    u_prev_dev=u_prev_dev,
                    alpha_terminal=alpha_for_solver,
                )
                step_info.update({
                    "success": True,
                    "method": "standard_tracking_lyapunov_mpc_rawlings_target",
                    "u_apply": u_dev_apply.copy(),
                    **report,
                })
                step_info["alpha_terminal_raw"] = float(alpha_terminal_raw)
                step_info["alpha_terminal"] = float(alpha_terminal)
                step_info["alpha_terminal_used"] = None if alpha_for_solver is None else float(alpha_for_solver)
                step_info["terminal_constraint_skipped"] = terminal_constraint_skipped
            else:
                if use_target_on_solver_fail:
                    u_dev_apply = np.clip(u_s, u_dev_min, u_dev_max)
                    fail_method = "solver_fail_use_us"
                else:
                    u_dev_apply = np.clip(u_prev_dev, u_dev_min, u_dev_max)
                    fail_method = "solver_fail_hold_prev"
                IC_opt = np.tile(u_dev_apply, LMPC_obj.NC)
                step_info.update({
                    "success": False,
                    "method": fail_method,
                    "u_apply": u_dev_apply.copy(),
                })
        else:
            u_dev_apply = np.clip(u_prev_dev, u_dev_min, u_dev_max)
            IC_opt = np.tile(u_dev_apply, LMPC_obj.NC)
            step_info.update({
                "success": False,
                "method": "target_fail_hold_prev",
                "message": "Refined target solve failed",
                "u_apply": u_dev_apply.copy(),
            })

        u_mpc[step_idx, :] = u_dev_apply + ss_scaled_inputs
        u_plant = reverse_min_max(u_mpc[step_idx, :], data_min[:n_inputs], data_max[:n_inputs])
        delta_u = u_mpc[step_idx, :] - scaled_current_input

        if mode == "disturb" and not disturbance_after_step:
            system.hA = ha[step_idx]
            system.Qs = qs[step_idx]
            system.Qi = qi[step_idx]

        system.current_input = u_plant
        system.step()

        if mode == "disturb" and disturbance_after_step:
            system.hA = ha[step_idx]
            system.Qs = qs[step_idx]
            system.Qi = qi[step_idx]

        y_mpc[step_idx + 1, :] = system.current_output

        y_current_scaled = apply_min_max(y_mpc[step_idx + 1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        delta_y = y_current_scaled - y_sp_k

        yhat[:, step_idx] = yhat_now
        xhat_next_openloop = LMPC_obj.A @ x0_aug + LMPC_obj.B @ u_dev_apply
        observer_correction = L @ innovation
        xhatdhat[:, step_idx + 1] = xhat_next_openloop + observer_correction

        y_sp_phys = reverse_min_max(y_sp_k + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])
        reward = reward_fn(delta_y, delta_u, y_sp_phys)
        rewards[step_idx] = reward

        delta_y_storage.append(delta_y)
        delta_u_storage.append(delta_u)

        step_info.update({
            "y_current_scaled": y_current_scaled.copy(),
            "xhat_next_openloop": xhat_next_openloop.copy(),
            "observer_correction": observer_correction.copy(),
            "xhat_next": xhatdhat[:, step_idx + 1].copy(),
            "reward": float(reward),
            "delta_y": np.asarray(delta_y, float).copy(),
            "delta_u": np.asarray(delta_u, float).copy(),
        })
        lmpc_info_storage.append(step_info)

        if step_idx in sub_episodes_changes_dict:
            avg_rewards.append(np.mean(rewards[step_idx - time_in_sub_episodes + 1:step_idx + 1]))
            last = lmpc_info_storage[-1]
            print(
                "Sub_Episode:", sub_episodes_changes_dict[step_idx],
                "| avg. reward:", avg_rewards[-1],
                "| method:", last.get("method"),
                "| success:", last.get("success"),
                "| alpha_terminal:", last.get("alpha_terminal"),
                "| terminal_margin:", last.get("terminal_margin"),
                "| target_slack_inf:", last.get("target_slack_inf"),
                "| nit:", last.get("solver_nit"),
            )

    u_mpc = reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs])

    return (
        y_mpc,
        u_mpc,
        avg_rewards,
        rewards,
        xhatdhat,
        nFE,
        time_in_sub_episodes,
        y_sp,
        yhat,
        delta_y_storage,
        delta_u_storage,
        lmpc_info_storage,
        target_info_storage,
    )
