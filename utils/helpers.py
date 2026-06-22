import numpy as np


def apply_rl_scaled(min_max_dict, x_d_states, y_sp, u):
    """
    This function will apply RL scaling for the neural networks
    :param min_max_dict:
    :param state:
    :return: rl scaled of the state
    """

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]

    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]

    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    x_d_states_scaled = 2 * ((x_d_states - x_min) / (x_max - x_min)) - 1

    y_sp_scaled = 2 * ((y_sp - y_sp_min) / (y_sp_max - y_sp_min)) - 1

    u_scaled = 2 * ((u - u_min) / (u_max - u_min)) - 1

    states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))

    return states


def generate_setpoints_training_rl_gradually(y_sp_scenario, n_tests, set_points_len, warm_start, test_cycle,
                                             nominal_qi, nominal_qs, nominal_ha,
                                             qi_change, qs_change, ha_change,
                                             *,
                                             force_final_test=True,
                                             setpoint_profile=None,
                                             disturbance_profile=None):
    y_sp_scenario = np.asarray(y_sp_scenario, dtype=float)
    if y_sp_scenario.ndim != 2:
        raise ValueError("y_sp_scenario must be a 2-D array")

    time_in_sub_episodes = set_points_len * len(y_sp_scenario)

    if setpoint_profile is None:
        # For each scenario, create a block of size (set_points_len, n_outputs)
        blocks = [np.full((set_points_len, y_sp_scenario.shape[1]), scenario)
                  for scenario in y_sp_scenario]

        # Concatenate the blocks to form one cycle
        cycle = np.concatenate(blocks, axis=0)
        # Repeat the cycle 'repetitions' times
        y_sp = np.concatenate([cycle] * n_tests, axis=0)
    else:
        y_sp = np.asarray(setpoint_profile, dtype=float)
        if y_sp.ndim != 2:
            raise ValueError("setpoint_profile must be a 2-D array")
        if y_sp.shape[1] != y_sp_scenario.shape[1]:
            raise ValueError(
                "setpoint_profile output dimension must match y_sp_scenario; "
                f"got {y_sp.shape[1]} and {y_sp_scenario.shape[1]}"
            )
        expected_len = int(n_tests) * int(time_in_sub_episodes)
        if len(y_sp) != expected_len:
            raise ValueError(
                f"setpoint_profile length must equal n_tests * time_in_sub_episodes={expected_len}; "
                f"got {len(y_sp)}"
            )
        if not np.all(np.isfinite(y_sp)):
            raise ValueError("setpoint_profile contains non-finite values")

    # Test/train scenario. Repeat and slice so short patterns such as [False]
    # or [True] can define any number of episodes without changing callers.
    if len(test_cycle) == 0:
        raise ValueError("test_cycle must contain at least one boolean value")
    repetitions = int(np.ceil(n_tests / len(test_cycle)))
    test_cycle = list(test_cycle) * repetitions
    test_cycle = test_cycle[:n_tests]
    if force_final_test:
        test_cycle[-1] = True

    nFE = int(y_sp.shape[0])
    idxs_setpoints = np.arange(time_in_sub_episodes - 1, nFE, time_in_sub_episodes)
    idxs_tests = np.arange(0, nFE, time_in_sub_episodes)
    sub_episodes_changes = np.arange(1, len(idxs_setpoints) + 1)
    sub_episodes_changes_dict = {}
    test_train_dict = {}
    for i in range(len(idxs_setpoints)):
        sub_episodes_changes_dict[idxs_setpoints[i]] = sub_episodes_changes[i]
    for i in range(len(idxs_tests)):
        test_train_dict[idxs_tests[i]] = test_cycle[i]
    warm_start = list(test_train_dict.keys())[warm_start]

    if disturbance_profile is None:
        qi = np.linspace(nominal_qi, nominal_qi * qi_change, nFE)
        qs = np.linspace(nominal_qs, nominal_qs * qs_change, nFE)
        ha = np.linspace(nominal_ha, nominal_ha * ha_change, int(nFE / 2))
        ha = np.hstack((ha, np.tile(nominal_ha * ha_change, int(nFE/ 2))))
        if len(ha) < nFE:
            ha = np.hstack((ha, np.tile(nominal_ha * ha_change, nFE - len(ha))))
    else:
        missing = [name for name in ("qi", "qs", "ha") if name not in disturbance_profile]
        if missing:
            raise ValueError(f"disturbance_profile is missing required keys: {missing}")
        qi = np.asarray(disturbance_profile["qi"], dtype=float).reshape(-1)
        qs = np.asarray(disturbance_profile["qs"], dtype=float).reshape(-1)
        ha = np.asarray(disturbance_profile["ha"], dtype=float).reshape(-1)
        for name, values in (("qi", qi), ("qs", qs), ("ha", ha)):
            if len(values) != nFE:
                raise ValueError(
                    f"disturbance_profile['{name}'] length must equal nFE={nFE}; got {len(values)}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"disturbance_profile['{name}'] contains non-finite values")

    return y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, test_train_dict, warm_start, qi, qs, ha
