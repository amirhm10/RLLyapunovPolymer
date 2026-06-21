import os
import pickle

import scipy.optimize as spo
import numpy as np
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset

from Simulation.mpc import augment_state_space, augment_state_space_rawlings
from utils.path_helpers import resolve_repo_path
from utils.polymer_td3_defaults import DEFAULT_TD3_SETPOINT_SCALER_Y_PHYS, default_td3_state_bounds
from utils.scaling_helpers import apply_min_max, apply_min_max_pm1


def _resolve_system_dict_path(data_dir="Data", system_dict_path=None):
    if system_dict_path is not None:
        path = os.fspath(resolve_repo_path(system_dict_path))
        if os.path.exists(path):
            return path
        alt_path = path + ".pickle"
        if os.path.exists(alt_path):
            return alt_path
        raise FileNotFoundError(
            f"Could not find system_dict at '{path}' or '{alt_path}'."
        )

    full_data_dir = os.fspath(resolve_repo_path(data_dir, create=True))

    system_dict_path = os.path.join(full_data_dir, "system_dict")
    if os.path.exists(system_dict_path):
        return system_dict_path

    alt_path = system_dict_path + ".pickle"
    if os.path.exists(alt_path):
        return alt_path

    raise FileNotFoundError(
        f"Could not find system_dict at '{system_dict_path}' or '{alt_path}'."
    )


def _rawlings_disturbance_matrices(A, B, C, augmentation_mode):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)

    n = A.shape[0]
    p = C.shape[0]
    m = B.shape[1]
    mode = "output_disturbance" if augmentation_mode is None else str(augmentation_mode)

    if mode == "output_disturbance":
        return np.zeros((n, p), dtype=float), np.eye(p, dtype=float)

    if m != p:
        raise ValueError(
            f"Augmentation mode '{mode}' requires n_inputs == n_outputs so B can be used as Bd. "
            f"Got n_inputs={m}, n_outputs={p}."
        )

    if mode == "state_disturbance_via_B":
        return B.copy(), np.zeros((p, p), dtype=float)
    if mode == "mixed_B_I":
        return B.copy(), np.eye(p, dtype=float)
    if mode == "mixed_0.1B_I":
        return 0.1 * B.copy(), np.eye(p, dtype=float)
    if mode == "mixed_0.5B_I":
        return 0.5 * B.copy(), np.eye(p, dtype=float)

    raise ValueError(
        "augmentation_mode must be one of "
        "'output_disturbance', 'state_disturbance_via_B', 'mixed_B_I', "
        "'mixed_0.1B_I', or 'mixed_0.5B_I'."
    )


def load_and_prepare_system_data(
    steady_states,
    setpoint_y,
    u_min,
    u_max,
    data_dir="Data",
    n_inputs=2,
    system_dict_path=None,
    augmentation_style="legacy",
    augmentation_mode=None,
    Bd=None,
    Cd=None,
    state_min=None,
    state_max=None,
    setpoint_range_y=None,
):
    """
    Loads system matrices, scaling factors, and min-max state info from files,
    augments the state space, and applies min-max scaling to the steady states
    and setpoint. Returns a dictionary with the processed data.
    """
    full_data_dir = os.fspath(resolve_repo_path(data_dir, create=True))

    system_dict_path = _resolve_system_dict_path(data_dir=data_dir, system_dict_path=system_dict_path)

    with open(system_dict_path, "rb") as file:
        system_dict = pickle.load(file)

    A = system_dict["A"]
    B = system_dict["B"]
    C = system_dict["C"]

    augmentation_style = str(augmentation_style).lower()
    if augmentation_style == "legacy":
        A_aug, B_aug, C_aug = augment_state_space(A, B, C)
        Bd_used = np.zeros((A.shape[0], C.shape[0]), dtype=float)
        Cd_used = np.eye(C.shape[0], dtype=float)
    elif augmentation_style == "rawlings":
        if Bd is None and Cd is None:
            Bd_used, Cd_used = _rawlings_disturbance_matrices(A, B, C, augmentation_mode)
        else:
            if Bd is None or Cd is None:
                raise ValueError("Provide both Bd and Cd, or neither, for Rawlings augmentation.")
            Bd_used = np.asarray(Bd, float)
            Cd_used = np.asarray(Cd, float)
        A_aug, B_aug, C_aug, Bd_used, Cd_used = augment_state_space_rawlings(
            A, B, C, Bd=Bd_used, Cd=Cd_used
        )
    else:
        raise ValueError("augmentation_style must be 'legacy' or 'rawlings'.")

    scaling_factor_path = os.path.join(full_data_dir, "scaling_factor.pickle")
    with open(scaling_factor_path, "rb") as file:
        scaling_factor = pickle.load(file)
    data_min = scaling_factor["min"]
    data_max = scaling_factor["max"]

    min_max_states_path = os.path.join(full_data_dir, "min_max_states.pickle")
    with open(min_max_states_path, "rb") as file:
        min_max_states = pickle.load(file)
    if state_min is None and state_max is None:
        state_min_used, state_max_used = default_td3_state_bounds(A_aug.shape[0])
        state_bounds_source = "default_polymer_td3"
    elif state_min is None or state_max is None:
        raise ValueError("Provide both state_min and state_max, or neither.")
    else:
        state_min_used = np.asarray(state_min, dtype=float).reshape(-1)
        state_max_used = np.asarray(state_max, dtype=float).reshape(-1)
        state_bounds_source = "caller_override"

    if state_min_used.shape != (A_aug.shape[0],) or state_max_used.shape != (A_aug.shape[0],):
        raise ValueError(
            "State scaling bounds must match the augmented state dimension "
            f"{A_aug.shape[0]}; got {state_min_used.shape} and {state_max_used.shape}."
        )
    if np.any(state_max_used <= state_min_used):
        raise ValueError("Each state_max entry must be greater than state_min.")

    setpoint_y = np.atleast_2d(np.asarray(setpoint_y, dtype=float))
    if setpoint_range_y is None:
        setpoint_range_y_used = DEFAULT_TD3_SETPOINT_SCALER_Y_PHYS.copy()
        setpoint_bounds_source = "default_polymer_td3_scaler"
    else:
        setpoint_range_y_used = np.atleast_2d(np.asarray(setpoint_range_y, dtype=float))
        setpoint_bounds_source = "caller_override"

    y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    if setpoint_range_y_used.shape[1] != y_ss_scaled.size:
        raise ValueError(
            "Setpoint scaling envelope must match the number of outputs "
            f"{y_ss_scaled.size}; got shape {setpoint_range_y_used.shape}."
        )
    y_sp_scaled = apply_min_max(setpoint_y, data_min[n_inputs:], data_max[n_inputs:])
    y_sp_scaled_deviation = y_sp_scaled - y_ss_scaled
    y_sp_range_scaled = apply_min_max(setpoint_range_y_used, data_min[n_inputs:], data_max[n_inputs:])
    y_sp_range_scaled_deviation = y_sp_range_scaled - y_ss_scaled
    y_sp_bounds_min = np.min(y_sp_range_scaled_deviation, axis=0)
    y_sp_bounds_max = np.max(y_sp_range_scaled_deviation, axis=0)

    u_ss_scaled = apply_min_max(steady_states["ss_inputs"], data_min[:n_inputs], data_max[:n_inputs])
    b_min = apply_min_max(u_min, data_min[:n_inputs], data_max[:n_inputs]) - u_ss_scaled
    b_max = apply_min_max(u_max, data_min[:n_inputs], data_max[:n_inputs]) - u_ss_scaled

    min_max_dict = {
        "x_max": state_max_used,
        "x_min": state_min_used,
        "y_sp_min": y_sp_bounds_min,
        "y_sp_max": y_sp_bounds_max,
        "u_max": b_max,
        "u_min": b_min,
    }

    return {
        "A": A,
        "B": B,
        "C": C,
        "A_aug": A_aug,
        "B_aug": B_aug,
        "C_aug": C_aug,
        "Bd_used": Bd_used,
        "Cd_used": Cd_used,
        "system_dict_path": system_dict_path,
        "augmentation_style": augmentation_style,
        "augmentation_mode": augmentation_mode,
        "data_min": data_min,
        "data_max": data_max,
        "min_max_states": min_max_states,
        "state_bounds_source": state_bounds_source,
        "state_min_used": state_min_used,
        "state_max_used": state_max_used,
        "setpoint_bounds_source": setpoint_bounds_source,
        "setpoint_range_y_used": setpoint_range_y_used,
        "y_sp_range_scaled_deviation": y_sp_range_scaled_deviation,
        "y_ss_scaled": y_ss_scaled,
        "y_sp_scaled": y_sp_scaled,
        "y_sp_scaled_deviation": y_sp_scaled_deviation,
        "u_ss_scaled": u_ss_scaled,
        "b_min": b_min,
        "b_max": b_max,
        "min_max_dict": min_max_dict,
    }


def print_accuracy(agent, n_samples=1000, device="cpu"):
    s = torch.from_numpy(agent.buffer.states[:n_samples]).to(device)
    a = agent.buffer.actions[:n_samples]
    p_a = agent.actor(s).detach().cpu().numpy()

    entire_accuracy = r2_score(a, p_a)
    accuracy_input1 = r2_score(a[:, 0], p_a[:, 0])
    accuracy_input2 = r2_score(a[:, 1], p_a[:, 1])

    print(f"Agent r2 score for the predicted inputs compare to MPC inputs: {entire_accuracy:6f}")
    print(f"Agent r2 score for the predicted input 1 compare to MPC input 1: {accuracy_input1:6f}")
    print(f"Agent r2 score for the predicted input 2 compare to MPC input 2: {accuracy_input2:6f}")


def optimize_sample(i, MPC_obj, y_sp, u, x0_model, IC_opt, bnds, cons):
    sol = spo.minimize(
        lambda x: MPC_obj.mpc_opt_fun(x, y_sp, u, x0_model),
        IC_opt,
        bounds=bnds,
        constraints=cons,
    )
    return sol.x[:MPC_obj.B.shape[1]]


def filling_the_buffer(
    min_max_dict,
    A, B, C,
    MPC_obj,
    mpc_pretrain_samples_numbers,
    Q_penalty, R_penalty,
    agent,
    IC_opt, bnds, cons,
    chunk_size=10000,
):
    """
    Fill the replay buffer in batches.
    """

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]
    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]
    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    num_full_chunks = mpc_pretrain_samples_numbers // chunk_size
    remaining_samples = mpc_pretrain_samples_numbers % chunk_size

    chunk_sizes = [chunk_size] * num_full_chunks
    if remaining_samples > 0:
        chunk_sizes.append(remaining_samples)

    total_done = 0
    for chunk_idx, curr_chunk_size in enumerate(chunk_sizes):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunk_sizes)} (size={curr_chunk_size})")

        x_d_states = np.random.uniform(low=x_min, high=x_max, size=(curr_chunk_size, A.shape[0]))
        x_d_states_scaled = apply_min_max_pm1(x_d_states, x_min, x_max)

        y_sp = np.random.uniform(low=y_sp_min, high=y_sp_max, size=(curr_chunk_size, C.shape[0]))
        y_sp_scaled = apply_min_max_pm1(y_sp, y_sp_min, y_sp_max)

        u = np.random.uniform(low=u_min, high=u_max, size=(curr_chunk_size, B.shape[1]))
        u_scaled = apply_min_max_pm1(u, u_min, u_max)

        results = Parallel(n_jobs=-1, backend="loky", max_nbytes=None)(
            delayed(optimize_sample)(
                i + total_done,
                MPC_obj,
                y_sp[i, :],
                u[i, :],
                x_d_states[i, :],
                IC_opt,
                bnds,
                cons,
            )
            for i in range(curr_chunk_size)
        )
        u_mpc = np.array(results)

        next_x_d_states = np.dot(A, x_d_states.T) + np.dot(B, u_mpc.T)
        y_pred = np.dot(C, next_x_d_states)

        next_x_d_states_scaled = apply_min_max_pm1(next_x_d_states.T, x_min, x_max)
        u_mpc_scaled = apply_min_max_pm1(u_mpc, u_min, u_max)

        rewards = np.zeros(curr_chunk_size)
        for k in range(curr_chunk_size):
            dy = y_pred[:, k] - y_sp[k, :]
            du = u[k, :] - u_mpc[k, :]
            rewards[k] = -1.0 * (dy.T @ Q_penalty @ dy + du.T @ R_penalty @ du)

        actions = u_mpc_scaled.copy()
        states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))
        next_states = np.hstack((next_x_d_states_scaled, y_sp_scaled, u_mpc_scaled))

        agent.buffer.pretrain_add(states, actions, rewards, next_states)
        total_done += curr_chunk_size

    print("Replay buffer has been filled with generated samples.")


def add_steady_state_samples(
    min_max_dict,
    A, B, C,
    MPC_obj,
    steady_state_samples_numbers,
    Q_penalty, R_penalty,
    agent,
    IC_opt, bnds, cons,
    chunk_size=10000,
):
    """
    Add near steady-state samples (y_sp approx 0 and u approx 0 in deviation form).
    """

    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]
    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]
    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    mu = 0.0
    sigma = (x_max - x_min) / 1e5

    num_full_chunks = steady_state_samples_numbers // chunk_size
    remaining_samples = steady_state_samples_numbers % chunk_size

    chunk_sizes = [chunk_size] * num_full_chunks
    if remaining_samples > 0:
        chunk_sizes.append(remaining_samples)

    total_done = 0
    for chunk_idx, curr_chunk_size in enumerate(chunk_sizes):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunk_sizes)} (size={curr_chunk_size})")

        x_d_states = np.random.normal(mu, sigma, size=(curr_chunk_size, A.shape[0]))
        x_d_states_scaled = apply_min_max_pm1(x_d_states, x_min, x_max)

        y_sp = np.zeros((curr_chunk_size, C.shape[0]))
        y_sp_scaled = apply_min_max_pm1(y_sp, y_sp_min, y_sp_max)

        u = np.random.uniform(low=0.0, high=1e-8, size=(curr_chunk_size, B.shape[1]))
        u_scaled = apply_min_max_pm1(u, u_min, u_max)

        results = Parallel(n_jobs=-1, backend="loky", max_nbytes=None)(
            delayed(optimize_sample)(
                i + total_done,
                MPC_obj,
                y_sp[i, :],
                u[i, :],
                x_d_states[i, :],
                IC_opt,
                bnds,
                cons,
            )
            for i in range(curr_chunk_size)
        )
        u_mpc = np.array(results)

        next_x_d_states = np.dot(A, x_d_states.T) + np.dot(B, u_mpc.T)
        y_pred = np.dot(C, next_x_d_states)

        next_x_d_states_scaled = apply_min_max_pm1(next_x_d_states.T, x_min, x_max)
        u_mpc_scaled = apply_min_max_pm1(u_mpc, u_min, u_max)

        rewards = np.zeros(curr_chunk_size)
        for k in range(curr_chunk_size):
            dy = y_pred[:, k] - y_sp[k, :]
            du = u[k, :] - u_mpc[k, :]
            rewards[k] = -1.0 * (dy.T @ Q_penalty @ dy + du.T @ R_penalty @ du)

        actions = u_mpc_scaled.copy()
        states = np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))
        next_states = np.hstack((next_x_d_states_scaled, y_sp_scaled, u_mpc_scaled))

        agent.buffer.pretrain_add(states, actions, rewards, next_states)
        total_done += curr_chunk_size

    print("Replay buffer has been filled up with the steady_state values.")


class ReplayDataset(Dataset):
    def __init__(self, s, a, r, ns, d=None):
        self.s = np.ascontiguousarray(np.asarray(s, dtype=np.float32))
        self.a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
        self.r = np.ascontiguousarray(np.asarray(r, dtype=np.float32).reshape(-1))
        self.ns = np.ascontiguousarray(np.asarray(ns, dtype=np.float32))
        self.d = None if d is None else np.ascontiguousarray(
            np.asarray(d, dtype=np.float32).reshape(-1)
        )

        n_samples = int(self.s.shape[0])
        if (
            self.a.shape[0] != n_samples
            or self.r.shape[0] != n_samples
            or self.ns.shape[0] != n_samples
        ):
            raise ValueError(
                "ReplayDataset arrays must have matching first dimensions: "
                f"s={self.s.shape}, a={self.a.shape}, r={self.r.shape}, ns={self.ns.shape}."
            )
        if self.d is not None and self.d.shape[0] != n_samples:
            raise ValueError(
                "ReplayDataset done array must match the state count: "
                f"s={self.s.shape}, d={self.d.shape}."
            )

    def __len__(self):
        return int(self.s.shape[0])

    def __getitem__(self, i):
        if self.d is None:
            return self.s[i], self.a[i], self.r[i], self.ns[i]
        return self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i]
