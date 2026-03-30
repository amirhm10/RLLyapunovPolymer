import numpy as np
from utils.scaling_helpers import apply_min_max, reverse_min_max
from utils.helpers import generate_setpoints_training_rl_gradually
import scipy.optimize as spo


def run_mpc(system, MPC_obj, y_sp_scenario, n_tests, set_points_len,
            steady_states, IC_opt, bnds, cons, warm_start,
            L, data_min, data_max,
            test_cycle, reward_fn,
            nominal_qi, nominal_qs, nominal_ha,
            qi_change, qs_change, ha_change,
            mode="disturb"):
    # --- setpoints generation ---
    y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, test_train_dict, warm_start, qi, qs, ha = \
        generate_setpoints_training_rl_gradually(
            y_sp_scenario, n_tests, set_points_len, warm_start, test_cycle,
            nominal_qi, nominal_qs, nominal_ha,
            qi_change, qs_change, ha_change
        )

    # inputs and outputs of the system dimensions
    n_inputs = MPC_obj.B.shape[1]
    n_outputs = MPC_obj.C.shape[0]
    n_states = MPC_obj.A.shape[0]

    # Scaled steady states inputs and outputs
    ss_scaled_inputs = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])

    y_mpc = np.zeros((nFE + 1, n_outputs))
    y_mpc[0, :] = system.current_output
    u_mpc = np.zeros((nFE, n_inputs))
    yhat = np.zeros((n_outputs, nFE))
    xhatdhat = np.zeros((n_states, nFE + 1))
    rewards = np.zeros(nFE)
    rewards_mpc = np.zeros(nFE)
    avg_rewards = []
    avg_rewards_mpc = []

    # Recording
    delta_y_storage, delta_u_storage = [], []

    for i in range(nFE):
        # Current scaled input & deviation
        scaled_current_input = apply_min_max(system.current_input, data_min[:n_inputs], data_max[:n_inputs])
        scaled_current_input_dev = scaled_current_input - ss_scaled_inputs

        # Solving MPC optimization problem
        sol = spo.minimize(
            lambda x: MPC_obj.mpc_opt_fun(x, y_sp[i, :], scaled_current_input_dev,
                                          xhatdhat[:, i]), IC_opt, bounds=bnds, constraints=cons)

        # take the first control action (this is in scaled deviation form)
        u_mpc[i, :] = sol.x[:MPC_obj.B.shape[1]] + ss_scaled_inputs

        # u (reverse scaling of the mpc)
        u_plant = reverse_min_max(u_mpc[i, :], data_min[:n_inputs], data_max[:n_inputs])

        # delta u cost variables
        delta_u = u_mpc[i, :] - scaled_current_input

        # Apply to plant and step
        system.current_input = u_plant
        system.step()

        # Disturbance
        if mode == "disturb":
            # disturbances
            system.hA = ha[i]
            system.Qs = qs[i]
            system.Qi = qi[i]

        # Record the system output
        y_mpc[i+1, :] = system.current_output

        # ----- Observer & model roll -----
        y_current_scaled = apply_min_max(y_mpc[i+1, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled
        y_prev_scaled = apply_min_max(y_mpc[i, :], data_min[n_inputs:], data_max[n_inputs:]) - y_ss_scaled

        # Calculate Delta y in deviation form
        delta_y = y_current_scaled - y_sp[i, :]

        # Calculate the next state in deviation form
        yhat[:, i] = np.dot(MPC_obj.C, xhatdhat[:, i])
        xhatdhat[:, i+1] = (np.dot(MPC_obj.A, xhatdhat[:, i]) + np.dot(MPC_obj.B, (u_mpc[i, :] - ss_scaled_inputs))
                            + np.dot(L, (y_prev_scaled - yhat[:, i])).T)

        # y_sp in physical band
        y_sp_phys = reverse_min_max(y_sp[i, :] + y_ss_scaled, data_min[n_inputs:], data_max[n_inputs:])

        # Reward Calculation
        reward = reward_fn(delta_y, delta_u, y_sp_phys)

        # Recording
        rewards[i] = reward
        delta_y_storage.append(delta_y)
        delta_u_storage.append(delta_u)

        # Calculate average reward and printing
        if i in sub_episodes_changes_dict.keys():
            # Averaging the rewards from the last setpoint change till curtrent
            avg_rewards.append(np.mean(rewards[i - time_in_sub_episodes + 1: i]))

            # printing
            print('Sub_Episode : ', sub_episodes_changes_dict[i], ' | avg. reward :', avg_rewards[-1])

    u_mpc = reverse_min_max(u_mpc, data_min[:n_inputs], data_max[:n_inputs])

    return y_mpc, u_mpc, avg_rewards, rewards, xhatdhat, nFE, time_in_sub_episodes, y_sp, yhat, delta_y_storage, delta_u_storage