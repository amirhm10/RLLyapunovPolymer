# Auto-generated safe-MPC filter research bundle
# This file is intended for sharing with ChatGPT or other research assistants.
# It is a concatenation of the current repository source files that define the MPC-upstream safety-filter path.

"""
Bundled files:
- Simulation/system_functions.py
- Simulation/mpc.py
- Simulation/run_mpc_lyapunov.py
- Lyapunov/target_selector.py
- Lyapunov/lyapunov_core.py
- Lyapunov/safety_filter.py
- Lyapunov/upstream_controllers.py
- Lyapunov/safety_debug.py
- utils/lyapunov_utils.py
- utils/scaling_helpers.py
- utils/helpers.py
- TD3Agent/reward_functions.py
- LyapunovSafetyFilterMPC.ipynb (run_config excerpt)
"""

# ==================== BEGIN FILE: Simulation/system_functions.py ====================
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


class PolymerCSTR:
    def __init__(self, params, design_params, ss_inputs, delta_t, deviation_form=False):
        self.params = params
        self.ss_inputs = ss_inputs

        # Parameter Design of the reactor
        self.CIf, self.CMf, self.Qi, self.Qs, self.Tf, self.Tcf, self.V, self.Vc = design_params
        self.Ad, self.Ed, self.Ap, self.Ep, self.At, self.Et, self.fi, self.m_delta_H_r, self.hA, self.rhocp, self.rhoccpc, self.Mm = params
        self.delta_t = delta_t
        self.steady_trajectory = self.ss_params()
        self.viscosity_ss = 0.0012 * (self.params[-1] *
                                      self.steady_trajectory[-1] / self.steady_trajectory[-2]) ** 0.71
        self.y_ss = np.array([self.viscosity_ss, self.steady_trajectory[2]])

        self.deviation_form = deviation_form

        if deviation_form:
            # Since we are in deviation form
            self.current_state = np.zeros(len(self.steady_trajectory))
            self.current_input = np.zeros(len(self.ss_inputs))
            self.current_viscosity = 0.0
        else:
            # out of deviation form
            self.current_state = self.steady_trajectory
            self.current_input = self.ss_inputs
            self.current_viscosity = self.viscosity_ss
        self.current_output = np.array([self.current_viscosity, self.current_state[2]])

    def odes_deviation(self, t, x, u):
        """
        dCI/dt = (Qi*CIf - Qt*CI)/V - k*dCI
        dCM/dt = (Qm*CMf - Qt*CM)/V - kp*CM*CP
        dT/dt  = Qt*(Tf-T)/V - (Delta_H/(rhocp))*kp*CM*CP - hA/(rhocp*V)*(T - Tc)
        dTc/dt = Qc*(Tcf-Tc)/Vc + hA/(rhoccpc*Vc)*(T-Tc)
        dD0/dt = 0.5*kt*CP**2 - Qt*D0/V
        dD1/dt = Mm*kp*CM*CP - Qt*D1/V
        dD2/dt = 5*Mm*kp*CM*CP + (3*Mm*kp**2)/kt*CM**2 - Qt*D2/V
        CP = sqrt(2*fi*kd*CI/kt)
        Qt = Qi + Qs + Qm
        """
        # Instead of modifying in place, create new arrays for the absolute state and input:
        x_abs = x + self.steady_trajectory
        u_abs = u + self.ss_inputs

        return self.odes(t, x_abs, u_abs)

    def odes(self, t, x, u):
        # current inputs and previous states
        Qc, Qm = u
        CI, CM, T, Tc, D0, D1, D2 = x

        # Calculating dependent parameters in the odes
        kd = self.Ad * np.exp(-self.Ed / T)
        kp = self.Ap * np.exp(-self.Ep / T)
        kt = self.At * np.exp(-self.Et / T)
        CP = (2 * self.fi * kd * CI / kt) ** 0.5
        Qt = self.Qi + self.Qs + Qm

        # System of ODEs
        dCIdt = (self.Qi * self.CIf - Qt * CI) / self.V - kd * CI
        dCMdt = (Qm * self.CMf - Qt * CM) / self.V - kp * CM * CP
        dTdt = Qt * (self.Tf - T) / self.V - (self.m_delta_H_r / self.rhocp) * kp * CM * CP - self.hA / (self.rhocp * self.V) * (T - Tc)
        # In the paper itself for the dTcdt first term was Qt, but instead it should be Qc
        dTcdt = Qc * (self.Tcf - Tc) / self.Vc + self.hA / (self.rhoccpc * self.Vc) * (T - Tc)
        dD0dt = 0.5 * kt * CP ** 2 - Qt * D0 / self.V
        dD1dt = self.Mm * kp * CM * CP - Qt * D1 / self.V
        dD2dt = 5 * self.Mm * kp * CM * CP + (3 * self.Mm * kp ** 2) / kt * CM ** 2 - Qt * D2 / self.V
        return dCIdt, dCMdt, dTdt, dTcdt, dD0dt, dD1dt, dD2dt

    def ss_params(self):
        x_0 = np.array([6e-2, 3.3, 320, 310, 0, 0, 1])

        x_ss = fsolve(lambda x: self.odes(0.0, x, self.ss_inputs), x_0)

        return x_ss

    def step(self):
        if self.deviation_form:
            sol = solve_ivp(self.odes_deviation, [0, self.delta_t], self.current_state, args=(self.current_input,))

            self.current_state = sol.y[:, -1]

            viscosity = (0.0012 * (self.params[-1] *
                                   (self.current_state[-1] + self.steady_trajectory[-1]) / (self.current_state[-2]
                                                                                            + self.steady_trajectory[
                                                                                                -2]))
                         ** 0.71)

            self.current_viscosity = viscosity - self.viscosity_ss
        else:
            sol = solve_ivp(self.odes, [0, self.delta_t], self.current_state, args=(self.current_input,))

            self.current_state = sol.y[:, -1]

            viscosity = (0.0012 * (self.params[-1] *
                                   (self.current_state[-1]) / (self.current_state[-2])) ** 0.71)

            self.current_viscosity = viscosity

        self.current_output = np.array([self.current_viscosity, self.current_state[2]])
# ==================== END FILE: Simulation/system_functions.py ======================

# ==================== BEGIN FILE: Simulation/mpc.py ====================
import numpy as np
import control
from scipy import signal


class MpcSolver(object):
    def __init__(self, A, B, C, Q_out, R_in, NP, NC, D=None):
        self.A = np.asarray(A, float)
        self.B = np.asarray(B, float)
        self.C = np.asarray(C, float)
        self.D = None if D is None else np.asarray(D, float)

        self.NP = int(NP)
        self.NC = int(NC)

        self.Q_out = np.asarray(Q_out, float).reshape(-1)
        self.R_in = np.asarray(R_in, float).reshape(-1)

    def mpc_opt_fun(self, x, y_sp, u_prev_dev, x0_model):
        n_inputs = self.B.shape[1]
        n_outputs = self.C.shape[0]

        U = np.asarray(x[:n_inputs * self.NC], float).reshape(self.NC, n_inputs)

        y_sp = np.asarray(y_sp, float).reshape(n_outputs, )
        u_prev_dev = np.asarray(u_prev_dev, float).reshape(n_inputs, )
        x0_model = np.asarray(x0_model, float)

        x_pred = np.zeros((self.A.shape[0], self.NP + 1), dtype=float)
        x_pred[:, 0] = x0_model

        for j in range(self.NP):
            idx = j if j < self.NC else self.NC - 1
            x_pred[:, j + 1] = self.A @ x_pred[:, j] + self.B @ U[idx, :]

        y_pred = self.C @ x_pred  # (n_outputs, NP+1)
        y_dev = y_pred[:, 1:] - y_sp[:, None]  # (n_outputs, NP)

        U_prev = np.vstack([u_prev_dev.reshape(1, -1), U[:-1, :]])
        du = U - U_prev  # (NC, n_inputs)

        obj = 0.0
        for i in range(n_outputs):
            obj += float(self.Q_out[i]) * float(np.sum(y_dev[i, :] ** 2))
        for j in range(n_inputs):
            obj += float(self.R_in[j]) * float(np.sum(du[:, j] ** 2))

        return float(obj)


def augment_state_space(A, B, C):
    """
    Augments a state-space model for offset-free MPC

    Parameters
    ----------
    A : np.ndarray
        The state matrix of size (n_states, n_states).
    B : np.ndarray
        The input matrix of size (n_states, n_inputs).
    C : np.ndarray
        The output matrix of size (n_outputs, n_states).

    Returns
    -------
    A_aug : np.ndarray
        The augmented state matrix of size ((n_states+n_outputs), (n_states+n_outputs)).
    B_aug : np.ndarray
        The augmented input matrix of size ((n_states+n_outputs), n_inputs).
    C_aug : np.ndarray
        The augmented output matrix of size (n_outputs, (n_states+n_outputs)).
    """
    n_states = A.shape[0]
    n_outputs = C.shape[0]

    # Construct integrator part for offset-free formulation
    # Bd: zeros for the integrator dynamics (n_states x n_outputs)
    Bd = np.zeros((n_states, n_outputs))
    # Augment A: Top block is [A, Bd], bottom block is [zeros, I]
    zeros_A = np.zeros((n_outputs, n_states))
    ident_A = np.eye(n_outputs)
    A_aug = np.vstack((np.hstack((A, Bd)),
                       np.hstack((zeros_A, ident_A))))

    # Augment B: Append zeros for the integrator states
    zeros_B = np.zeros((n_outputs, B.shape[1]))
    B_aug = np.vstack((B, zeros_B))

    # Augment C: Append identity so that the integrator states appear in the output
    Cd = np.eye(n_outputs)
    C_aug = np.hstack((C, Cd))

    return A_aug, B_aug, C_aug


def compute_observer_gain(A, C, desired_poles):
    """
    Compute an observer gain L for the given MPC system using the desired poles.
    Also performs an observability check.

    Parameters:
    -----------
    A, C : np.ndarray
        System Matrices
    desired_poles : np.ndarray
        A vector of desired observer poles.

    Returns:
    --------
    L : np.ndarray
        The observer gain matrix.
    """
    # Compute the observer gain using pole placement
    obs_gain_calc = signal.place_poles(A.T, C.T, desired_poles, method='KNV0')
    L = np.squeeze(obs_gain_calc.gain_matrix).T

    # Check observability
    observability_matrix = control.obsv(A, C)
    rank = np.linalg.matrix_rank(observability_matrix)
    if rank == A.shape[0]:
        print("The system is observable.")
    else:
        print("The system is not observable.")
    return L


# ==================== END FILE: Simulation/mpc.py ======================

# ==================== BEGIN FILE: Simulation/run_mpc_lyapunov.py ====================
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
            Qdx_diag=None,
            Rdu_diag=Rmove_diag,
            solver_pref=target_solver_pref,
            return_debug=False,
        )
        if target_info.get("success", False):
            prev_target_info = target_info
        final_lyap_target_info = target_info if target_info.get("success", False) else prev_target_info
        final_lyap_target_source = "current_target" if target_info.get("success", False) else (
            "last_valid_target" if final_lyap_target_info is not None else None
        )

        mpc_tracking_target, mpc_tracking_target_source = _select_mpc_tracking_target(
            y_sp_raw=y_sp_k,
            target_info=target_info,
            policy=mpc_target_policy,
        )
        target_mismatch_inf = None
        if target_info.get("y_s") is not None:
            target_mismatch_inf = float(
                np.max(np.abs(np.asarray(target_info["y_s"], float).reshape(-1) - y_sp_k))
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
                    "allow_lyap_slack": bool(allow_lyap_slack),
                    "lyap_slack_weight": 1e6,
                    "solver_pref": filter_solver_pref,
                    "use_output_tracking_term": True,
                    "tracking_output_target": mpc_tracking_target.copy(),
                    "tracking_output_target_source": mpc_tracking_target_source,
                    "final_lyap_target_info": final_lyap_target_info,
                    "final_lyap_target_source": final_lyap_target_source,
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
                "u_s": None if not target_info.get("success", False) else target_info["u_s"].copy(),
                "x_s": None if not target_info.get("success", False) else target_info["x_s"].copy(),
                "d_s": None if not target_info.get("success", False) else target_info["d_s"].copy(),
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
                "verified": True,
                "target_success": bool(target_info.get("success", False)),
                "target_info": target_info,
                "setpoint_changed": bool(setpoint_changed),
                "target_source": "recomputed",
                "target_stage": target_info.get("solve_stage"),
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
# ==================== END FILE: Simulation/run_mpc_lyapunov.py ======================

# ==================== BEGIN FILE: Lyapunov/target_selector.py ====================
import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

from utils.lyapunov_utils import DEFAULT_CVXPY_SOLVERS, diag_psd_from_vector, vector_or_zeros


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
_DYN_TOL_BY_STATUS = {
    "optimal": 1e-6,
    "optimal_inaccurate": 1e-5,
}


def _solver_sequence(solver_pref):
    if solver_pref is None:
        return DEFAULT_CVXPY_SOLVERS
    if isinstance(solver_pref, str):
        return (solver_pref,)
    return tuple(solver_pref)


def _reset_variable_values(variables):
    for var in variables:
        if var is not None:
            var.value = None


def _solve_problem_with_preferences(problem, variables, solver_pref):
    last_status = None
    last_solver = None
    last_err = None

    for solver_name in _solver_sequence(solver_pref):
        try:
            _reset_variable_values(variables)
            problem.solve(solver=solver_name, warm_start=False, verbose=False)
            last_status = problem.status
            last_solver = solver_name

            if any(var is not None and var.value is None for var in variables):
                continue
            if problem.status in _OPTIMAL_STATUSES:
                return {
                    "accepted_by_status": True,
                    "status": problem.status,
                    "solver": solver_name,
                    "error": None,
                    "objective_value": float(problem.value) if problem.value is not None else None,
                }
        except Exception as exc:
            last_err = repr(exc)

    return {
        "accepted_by_status": False,
        "status": last_status,
        "solver": last_solver,
        "error": last_err,
        "objective_value": float(problem.value) if problem.value is not None else None,
    }


def _bound_violation_inf(value, lower=None, upper=None):
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(np.max(np.maximum(lower - value, 0.0))))
    if upper is not None:
        violation = max(violation, float(np.max(np.maximum(value - upper, 0.0))))
    return violation


def _output_slack_value(var, size):
    if var is None or var.value is None:
        return np.zeros(size, dtype=float)
    return np.asarray(var.value, float).reshape(-1)


def compute_ss_target_refined_rawlings(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    Ty_diag=None,
    Ru_diag=None,
    Qx_diag=None,
    w_x=1e-6,
    x_s_prev=None,
    u_s_prev=None,
    Qdx_diag=None,
    Rdu_diag=None,
    y_min=None,
    y_max=None,
    u_tight=None,
    y_tight=None,
    soft_output_bounds=True,
    Wy_low_diag=None,
    Wy_high_diag=None,
    solver_pref=DEFAULT_CVXPY_SOLVERS,
    return_debug=False,
    H=None,
):
    """
    Compute a steady-state offset-free target using a two-stage Rawlings-style selector.

    The expected convention is that all quantities are expressed in the same coordinates.
    In this codebase that should usually mean scaled deviation coordinates, so `u_nom`
    is typically the zero vector.

    Stage 1 solves an exact steady-state target problem with hard equality on the
    controlled outputs. Stage 2 is only used if Stage 1 is infeasible or numerically
    unacceptable; it computes the closest feasible equilibrium target. When
    `soft_output_bounds` is enabled, only output inequality bounds are softened. The
    exact target equality in Stage 1 remains hard.
    """
    if not HAS_CVXPY:
        raise ImportError("CVXPY is required for the refined target selector.")

    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)
    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)
    y_sp = np.asarray(y_sp, float).reshape(-1)
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)

    if A_aug.ndim != 2 or A_aug.shape[0] != A_aug.shape[1]:
        raise ValueError("A_aug must be square.")
    if B_aug.ndim != 2 or B_aug.shape[0] != A_aug.shape[0]:
        raise ValueError("B_aug has incompatible shape.")
    if C_aug.ndim != 2 or C_aug.shape[1] != A_aug.shape[0]:
        raise ValueError("C_aug has incompatible shape.")
    if xhat_aug.size != A_aug.shape[0]:
        raise ValueError("xhat_aug has incorrect size.")

    n_aug = A_aug.shape[0]
    n_y = C_aug.shape[0]
    n_x = n_aug - n_y
    n_u = B_aug.shape[1]

    if n_x <= 0:
        raise ValueError("Invalid augmented model: inferred physical state dimension must be positive.")
    if u_min.size != n_u or u_max.size != n_u:
        raise ValueError("u_min and u_max must both have size n_u.")

    if H is None:
        H_arr = None
        n_c = n_y
    else:
        H_arr = np.asarray(H, float)
        if H_arr.ndim != 2 or H_arr.shape[1] != n_y:
            raise ValueError("H must have shape (n_c, n_y).")
        n_c = H_arr.shape[0]

    if y_sp.size != n_c:
        raise ValueError(f"y_sp has incorrect size. Expected {n_c}, got {y_sp.size}.")

    A = A_aug[:n_x, :n_x]
    Bd = A_aug[:n_x, n_x:]
    B = B_aug[:n_x, :]
    C = C_aug[:, :n_x]
    Cd = C_aug[:, n_x:]
    d_hat = xhat_aug[n_x:].copy()

    if d_hat.size != n_y:
        raise ValueError(
            "This selector assumes the augmented state is ordered as [x; d] with len(d) == n_y."
        )

    if u_nom is None:
        u_nom = np.zeros(n_u, dtype=float)
    else:
        u_nom = np.asarray(u_nom, float).reshape(-1)
        if u_nom.size != n_u:
            raise ValueError("u_nom has incorrect size.")

    u_tight = np.maximum(vector_or_zeros(u_tight, n_u), 0.0)
    y_tight = np.maximum(vector_or_zeros(y_tight, n_y), 0.0)

    u_lo = u_min + u_tight
    u_hi = u_max - u_tight
    if np.any(u_lo > u_hi):
        raise ValueError("Input tightening is too large. Tightened bounds are infeasible.")

    use_y_lo = y_min is not None
    use_y_hi = y_max is not None
    if use_y_lo:
        y_min = np.asarray(y_min, float).reshape(-1)
        if y_min.size != n_y:
            raise ValueError("y_min has incorrect size.")
        y_lo = y_min + y_tight
    else:
        y_lo = None
    if use_y_hi:
        y_max = np.asarray(y_max, float).reshape(-1)
        if y_max.size != n_y:
            raise ValueError("y_max has incorrect size.")
        y_hi = y_max - y_tight
    else:
        y_hi = None
    if y_lo is not None and y_hi is not None and np.any(y_lo > y_hi):
        raise ValueError("Output tightening is too large. Tightened output bounds are infeasible.")

    Ty, Ty_used = diag_psd_from_vector(Ty_diag, n_c, eps=1e-12, default=1.0)
    Ru, Ru_used = diag_psd_from_vector(Ru_diag, n_u, eps=1e-12, default=1.0)

    if Qx_diag is None:
        Qx = float(w_x) * np.eye(n_x, dtype=float)
        Qx_used = np.full(n_x, float(w_x), dtype=float)
    else:
        Qx, Qx_used = diag_psd_from_vector(Qx_diag, n_x, eps=1e-12, default=max(float(w_x), 1e-12))

    use_x_prev = x_s_prev is not None and Qdx_diag is not None
    use_u_prev = u_s_prev is not None and Rdu_diag is not None

    if use_x_prev:
        x_s_prev = np.asarray(x_s_prev, float).reshape(-1)
        if x_s_prev.size != n_x:
            raise ValueError("x_s_prev has incorrect size.")
        Qdx, Qdx_used = diag_psd_from_vector(Qdx_diag, n_x, eps=1e-12, default=0.0)
    else:
        x_s_prev = None
        Qdx = None
        Qdx_used = None

    if use_u_prev:
        u_s_prev = np.asarray(u_s_prev, float).reshape(-1)
        if u_s_prev.size != n_u:
            raise ValueError("u_s_prev has incorrect size.")
        Rdu, Rdu_used = diag_psd_from_vector(Rdu_diag, n_u, eps=1e-12, default=0.0)
    else:
        u_s_prev = None
        Rdu = None
        Rdu_used = None

    I_minus_A = np.eye(n_x, dtype=float) - A

    def build_stage_problem(stage_name):
        x = cp.Variable(n_x)
        u = cp.Variable(n_u)
        y_expr = C @ x + Cd @ d_hat
        yc_expr = y_expr if H_arr is None else H_arr @ y_expr

        constraints = [
            I_minus_A @ x - B @ u - Bd @ d_hat == 0.0,
            u >= u_lo,
            u <= u_hi,
        ]

        objective = cp.quad_form(u - u_nom, Ru) + cp.quad_form(x, Qx)
        s_y_low = None
        s_y_high = None
        Wy_low_used = None
        Wy_high_used = None

        if stage_name == "exact":
            constraints.append(yc_expr == y_sp)
        else:
            objective += cp.quad_form(yc_expr - y_sp, Ty)
            if use_x_prev:
                objective += cp.quad_form(x - x_s_prev, Qdx)
            if use_u_prev:
                objective += cp.quad_form(u - u_s_prev, Rdu)

        if y_lo is not None:
            if soft_output_bounds:
                s_y_low = cp.Variable(n_y, nonneg=True)
                Wy_low, Wy_low_used = diag_psd_from_vector(Wy_low_diag, n_y, eps=1e-12, default=1e3)
                objective += cp.quad_form(s_y_low, Wy_low)
                constraints.append(y_expr + s_y_low >= y_lo)
            else:
                constraints.append(y_expr >= y_lo)

        if y_hi is not None:
            if soft_output_bounds:
                s_y_high = cp.Variable(n_y, nonneg=True)
                Wy_high, Wy_high_used = diag_psd_from_vector(Wy_high_diag, n_y, eps=1e-12, default=1e3)
                objective += cp.quad_form(s_y_high, Wy_high)
                constraints.append(y_expr - s_y_high <= y_hi)
            else:
                constraints.append(y_expr <= y_hi)

        return {
            "stage_name": stage_name,
            "problem": cp.Problem(cp.Minimize(objective), constraints),
            "x": x,
            "u": u,
            "y_expr": y_expr,
            "yc_expr": yc_expr,
            "s_y_low": s_y_low,
            "s_y_high": s_y_high,
            "Wy_low_diag_used": Wy_low_used,
            "Wy_high_diag_used": Wy_high_used,
        }

    def evaluate_stage(stage_bundle, solve_info):
        status = solve_info["status"]
        tol = _DYN_TOL_BY_STATUS.get(status)
        result = {
            "solve_info": solve_info,
            "accepted": False,
            "reject_reason": None,
            "x_s": None,
            "u_s": None,
            "y_s": None,
            "yc_s": None,
            "s_y_low": np.zeros(n_y, dtype=float),
            "s_y_high": np.zeros(n_y, dtype=float),
            "dyn_residual_inf": None,
            "target_eq_residual_inf": None,
            "bound_violation_inf": None,
        }

        if not solve_info["accepted_by_status"]:
            result["reject_reason"] = "solver_status"
            return result
        if tol is None:
            result["reject_reason"] = "unsupported_status"
            return result

        x_s = np.asarray(stage_bundle["x"].value, float).reshape(-1)
        u_s = np.asarray(stage_bundle["u"].value, float).reshape(-1)
        y_s = np.asarray(C @ x_s + Cd @ d_hat, float).reshape(-1)
        yc_s = np.asarray(y_s if H_arr is None else H_arr @ y_s, float).reshape(-1)

        dyn_residual = I_minus_A @ x_s - B @ u_s - Bd @ d_hat
        dyn_residual_inf = float(np.max(np.abs(dyn_residual)))
        target_eq_residual = yc_s - y_sp
        target_eq_residual_inf = float(np.max(np.abs(target_eq_residual)))

        bound_violation_inf = _bound_violation_inf(u_s, lower=u_lo, upper=u_hi)
        if not soft_output_bounds:
            bound_violation_inf = max(
                bound_violation_inf,
                _bound_violation_inf(y_s, lower=y_lo, upper=y_hi),
            )

        if dyn_residual_inf > tol:
            result["reject_reason"] = "dyn_residual"
        elif stage_bundle["stage_name"] == "exact" and target_eq_residual_inf > tol:
            result["reject_reason"] = "target_eq_residual"
        elif bound_violation_inf > tol:
            result["reject_reason"] = "bound_violation"
        else:
            result["accepted"] = True

        result.update({
            "x_s": x_s,
            "u_s": u_s,
            "y_s": y_s,
            "yc_s": yc_s,
            "s_y_low": _output_slack_value(stage_bundle["s_y_low"], n_y),
            "s_y_high": _output_slack_value(stage_bundle["s_y_high"], n_y),
            "dyn_residual_inf": dyn_residual_inf,
            "target_eq_residual_inf": target_eq_residual_inf,
            "bound_violation_inf": float(bound_violation_inf),
        })
        return result

    exact_stage = build_stage_problem("exact")
    exact_solve = _solve_problem_with_preferences(
        exact_stage["problem"],
        [exact_stage["x"], exact_stage["u"], exact_stage["s_y_low"], exact_stage["s_y_high"]],
        solver_pref,
    )
    exact_eval = evaluate_stage(exact_stage, exact_solve)

    fallback_stage = None
    fallback_solve = None
    fallback_eval = None
    final_stage = None
    final_eval = None
    final_stage_name = None

    if exact_eval["accepted"]:
        final_stage = exact_stage
        final_eval = exact_eval
        final_stage_name = "exact"
    else:
        fallback_stage = build_stage_problem("fallback")
        fallback_solve = _solve_problem_with_preferences(
            fallback_stage["problem"],
            [fallback_stage["x"], fallback_stage["u"], fallback_stage["s_y_low"], fallback_stage["s_y_high"]],
            solver_pref,
        )
        fallback_eval = evaluate_stage(fallback_stage, fallback_solve)
        if fallback_eval["accepted"]:
            final_stage = fallback_stage
            final_eval = fallback_eval
            final_stage_name = "fallback"

    primary_failure = fallback_eval if fallback_eval is not None else exact_eval
    primary_solve = primary_failure["solve_info"]

    dbg = {
        "success": bool(final_eval is not None),
        "status": primary_solve["status"],
        "solver": primary_solve["solver"],
        "error": primary_solve["error"],
        "solve_stage": final_stage_name,
        "assumed_augmented_state_order": "[x; d]",
        "assumed_disturbance_dimension_equals_ny": True,
        "Ty_diag_used": Ty_used,
        "Ru_diag_used": Ru_used,
        "Qx_diag_used": Qx_used,
        "Qdx_diag_used": Qdx_used,
        "Rdu_diag_used": Rdu_used,
        "u_tight": u_tight.copy(),
        "y_tight": y_tight.copy(),
        "soft_output_bounds": bool(soft_output_bounds),
        "stage1_status": exact_solve["status"],
        "stage1_solver": exact_solve["solver"],
        "stage1_error": exact_solve["error"],
        "stage1_objective_value": exact_solve["objective_value"],
        "stage1_reject_reason": exact_eval["reject_reason"],
        "stage2_status": None if fallback_solve is None else fallback_solve["status"],
        "stage2_solver": None if fallback_solve is None else fallback_solve["solver"],
        "stage2_error": None if fallback_solve is None else fallback_solve["error"],
        "stage2_objective_value": None if fallback_solve is None else fallback_solve["objective_value"],
        "stage2_reject_reason": None if fallback_eval is None else fallback_eval["reject_reason"],
    }

    if final_eval is None:
        dbg.update({
            "x_s": None,
            "u_s": None,
            "d_s": d_hat.copy(),
            "x_s_aug": None,
            "y_s": None,
            "objective_value": primary_solve["objective_value"],
            "objective": primary_solve["objective_value"],
            "target_error": None,
            "target_error_inf": None,
            "target_error_norm": None,
            "target_slack": None,
            "target_slack_inf": None,
            "target_slack_2": None,
            "target_eq_residual_inf": primary_failure["target_eq_residual_inf"],
            "dyn_residual_inf": primary_failure["dyn_residual_inf"],
            "bound_violation_inf": primary_failure["bound_violation_inf"],
            "target_move_u_inf": None,
            "target_move_x_inf": None,
            "margin_to_u_min": None,
            "margin_to_u_max": None,
            "tight_margin_to_u_min": None,
            "tight_margin_to_u_max": None,
            "s_y_low": None if primary_failure is None else primary_failure["s_y_low"].copy(),
            "s_y_high": None if primary_failure is None else primary_failure["s_y_high"].copy(),
            "Wy_low_diag_used": None if fallback_stage is None else fallback_stage["Wy_low_diag_used"],
            "Wy_high_diag_used": None if fallback_stage is None else fallback_stage["Wy_high_diag_used"],
            "slack_y_inf": None,
            "slack_inf": None,
        })
        if y_min is not None:
            dbg["margin_to_y_min"] = None
            dbg["tight_margin_to_y_min"] = None
        if y_max is not None:
            dbg["margin_to_y_max"] = None
            dbg["tight_margin_to_y_max"] = None
        if return_debug:
            return None, None, d_hat.copy(), dbg
        return None, None, d_hat.copy()

    x_s = final_eval["x_s"]
    u_s = final_eval["u_s"]
    y_s = final_eval["y_s"]
    target_err = final_eval["yc_s"] - y_sp
    target_err_inf = float(np.max(np.abs(target_err)))
    target_err_norm = float(np.linalg.norm(target_err))
    target_slack_2 = float(np.linalg.norm(target_err))
    x_s_aug = np.concatenate([x_s, d_hat])

    final_solve = final_eval["solve_info"]
    final_stage_bundle = exact_stage if final_stage_name == "exact" else fallback_stage

    dbg.update({
        "success": True,
        "status": final_solve["status"],
        "solver": final_solve["solver"],
        "error": final_solve["error"],
        "x_s": x_s.copy(),
        "u_s": u_s.copy(),
        "d_s": d_hat.copy(),
        "x_s_aug": x_s_aug.copy(),
        "y_s": y_s.copy(),
        "objective_value": final_solve["objective_value"],
        "objective": final_solve["objective_value"],
        "target_error": target_err.copy(),
        "target_error_inf": target_err_inf,
        "target_error_norm": target_err_norm,
        "target_slack": target_err.copy(),
        "target_slack_inf": target_err_inf,
        "target_slack_2": target_slack_2,
        "target_eq_residual_inf": final_eval["target_eq_residual_inf"],
        "dyn_residual_inf": final_eval["dyn_residual_inf"],
        "bound_violation_inf": final_eval["bound_violation_inf"],
        "target_move_x_inf": None if x_s_prev is None else float(np.max(np.abs(x_s - x_s_prev))),
        "target_move_u_inf": None if u_s_prev is None else float(np.max(np.abs(u_s - u_s_prev))),
        "margin_to_u_min": (u_s - u_min).copy(),
        "margin_to_u_max": (u_max - u_s).copy(),
        "tight_margin_to_u_min": (u_s - u_lo).copy(),
        "tight_margin_to_u_max": (u_hi - u_s).copy(),
        "s_y_low": final_eval["s_y_low"].copy(),
        "s_y_high": final_eval["s_y_high"].copy(),
        "Wy_low_diag_used": final_stage_bundle["Wy_low_diag_used"],
        "Wy_high_diag_used": final_stage_bundle["Wy_high_diag_used"],
        "slack_y_inf": target_err_inf,
        "slack_inf": target_err_inf,
    })

    if y_min is not None:
        dbg["margin_to_y_min"] = (y_s - y_min).copy()
        dbg["tight_margin_to_y_min"] = (y_s - y_lo).copy()
    if y_max is not None:
        dbg["margin_to_y_max"] = (y_max - y_s).copy()
        dbg["tight_margin_to_y_max"] = (y_hi - y_s).copy()

    if return_debug:
        return x_s, u_s, d_hat.copy(), dbg
    return x_s, u_s, d_hat.copy()


def prepare_filter_target_from_refined_selector(
    A_aug,
    B_aug,
    C_aug,
    xhat_aug,
    y_sp,
    u_min,
    u_max,
    u_nom=None,
    Ty_diag=None,
    Ru_diag=None,
    Qx_diag=None,
    w_x=1e-6,
    prev_target=None,
    x_s_prev=None,
    u_s_prev=None,
    Qdx_diag=None,
    Rdu_diag=None,
    y_min=None,
    y_max=None,
    u_tight=None,
    y_tight=None,
    soft_output_bounds=True,
    Wy_low_diag=None,
    Wy_high_diag=None,
    solver_pref=DEFAULT_CVXPY_SOLVERS,
    return_debug=False,
    H=None,
):
    if prev_target is not None:
        if x_s_prev is None and prev_target.get("x_s") is not None:
            x_s_prev = prev_target["x_s"]
        if u_s_prev is None and prev_target.get("u_s") is not None:
            u_s_prev = prev_target["u_s"]

    x_s, u_s, d_s, dbg = compute_ss_target_refined_rawlings(
        A_aug=A_aug,
        B_aug=B_aug,
        C_aug=C_aug,
        xhat_aug=xhat_aug,
        y_sp=y_sp,
        u_min=u_min,
        u_max=u_max,
        u_nom=u_nom,
        Ty_diag=Ty_diag,
        Ru_diag=Ru_diag,
        Qx_diag=Qx_diag,
        w_x=w_x,
        x_s_prev=x_s_prev,
        u_s_prev=u_s_prev,
        Qdx_diag=Qdx_diag,
        Rdu_diag=Rdu_diag,
        y_min=y_min,
        y_max=y_max,
        u_tight=u_tight,
        y_tight=y_tight,
        soft_output_bounds=soft_output_bounds,
        Wy_low_diag=Wy_low_diag,
        Wy_high_diag=Wy_high_diag,
        solver_pref=solver_pref,
        return_debug=True,
        H=H,
    )

    requested_y_sp = np.asarray(y_sp, float).reshape(-1)
    y_s_dbg = None if dbg.get("y_s") is None else np.asarray(dbg["y_s"], float).reshape(-1)
    if H is None or y_s_dbg is None:
        yc_s_dbg = None if y_s_dbg is None else y_s_dbg.copy()
    else:
        yc_s_dbg = np.asarray(H, float) @ y_s_dbg

    target_info = {
        "success": bool(dbg.get("success", False) and x_s is not None and u_s is not None),
        "x_s": None if x_s is None else np.asarray(x_s, float).reshape(-1),
        "u_s": None if u_s is None else np.asarray(u_s, float).reshape(-1),
        "d_s": None if d_s is None else np.asarray(d_s, float).reshape(-1),
        "x_s_aug": None if dbg.get("x_s_aug") is None else np.asarray(dbg["x_s_aug"], float).reshape(-1),
        "y_s": y_s_dbg,
        "yc_s": yc_s_dbg,
        "requested_y_sp": requested_y_sp.copy(),
        "solve_stage": dbg.get("solve_stage"),
        "target_error": None if dbg.get("target_error") is None else np.asarray(dbg["target_error"], float).reshape(-1),
        "target_error_inf": dbg.get("target_error_inf"),
        "target_error_norm": dbg.get("target_error_norm"),
        "target_slack_inf": dbg.get("target_slack_inf"),
        "dyn_residual_inf": dbg.get("dyn_residual_inf"),
        "bound_violation_inf": dbg.get("bound_violation_inf"),
        "warm_start": {
            "x_s_prev": None if x_s is None else np.asarray(x_s, float).reshape(-1).copy(),
            "u_s_prev": None if u_s is None else np.asarray(u_s, float).reshape(-1).copy(),
        },
        "selector_debug": dbg,
    }

    if return_debug:
        return target_info, dbg
    return target_info
# ==================== END FILE: Lyapunov/target_selector.py ======================

# ==================== BEGIN FILE: Lyapunov/lyapunov_core.py ====================
from types import SimpleNamespace

import numpy as np
from scipy.linalg import solve_discrete_are

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

from utils.lyapunov_utils import (
    compute_du_sequence,
    diag_psd_from_vector,
    reshape_u_sequence,
    safety_filter_solver_sequence,
    tracking_solver_sequence,
    vector_or_zeros,
)


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
_TRACKING_TOL_BY_STATUS = {
    "optimal": 1e-7,
    "optimal_inaccurate": 1e-5,
}


def design_standard_tracking_terminal_ingredients(
    A_aug,
    B_aug,
    C_aug,
    Qy_diag,
    Su_diag=None,
    u_min=None,
    u_max=None,
    lambda_u=1.0,
    qx_eps=1e-10,
    eps_r=1e-9,
    return_debug=False,
):
    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)
    Qy_diag = np.asarray(Qy_diag, float).reshape(-1)

    n_aug = A_aug.shape[0]
    n_u = B_aug.shape[1]
    n_y = C_aug.shape[0]
    n_x = n_aug - n_y

    A = A_aug[:n_x, :n_x]
    B = B_aug[:n_x, :]
    C = C_aug[:, :n_x]

    Qy = np.diag(np.maximum(Qy_diag, 0.0))
    Qx = C.T @ Qy @ C + float(qx_eps) * np.eye(n_x)

    if Su_diag is None:
        if u_min is None or u_max is None:
            su = lambda_u * np.ones(n_u, dtype=float)
        else:
            u_min = np.asarray(u_min, float).reshape(-1)
            u_max = np.asarray(u_max, float).reshape(-1)
            span = np.maximum(0.5 * (u_max - u_min), eps_r)
            su = lambda_u / np.maximum(span ** 2, eps_r)
    else:
        su = np.maximum(np.asarray(Su_diag, float).reshape(-1), eps_r)

    Su = np.diag(su)
    P_x = solve_discrete_are(A, B, Qx, Su)
    K_x = -np.linalg.solve(Su + B.T @ P_x @ B, B.T @ P_x @ A)

    if return_debug:
        dbg = {
            "A_phys": A.copy(),
            "B_phys": B.copy(),
            "C_phys": C.copy(),
            "Qx": Qx.copy(),
            "Su": Su.copy(),
            "K_x": K_x.copy(),
            "eig_cl": np.linalg.eigvals(A + B @ K_x),
        }
        return P_x, K_x, dbg
    return P_x, K_x


def compute_terminal_alpha_input_only(P_x, K_x, u_s, u_min, u_max, alpha_scale=1.0):
    P_x = np.asarray(P_x, float)
    K_x = np.asarray(K_x, float)
    u_s = np.asarray(u_s, float).reshape(-1)
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)

    P_inv = np.linalg.inv(P_x)

    alphas = []
    for row_idx in range(K_x.shape[0]):
        k_row = K_x[row_idx:row_idx + 1, :]
        gamma_sq = float((k_row @ P_inv @ k_row.T).item())
        gamma_sq = max(gamma_sq, 0.0)
        gamma = float(np.sqrt(gamma_sq))

        headroom = float(min(u_max[row_idx] - u_s[row_idx], u_s[row_idx] - u_min[row_idx]))
        if headroom < 0.0:
            return 0.0

        if gamma <= 1e-14:
            alphas.append(np.inf)
        else:
            alphas.append((headroom / gamma) ** 2)

    alpha = float(min(alphas)) if alphas else 0.0
    return max(float(alpha_scale) * alpha, 0.0)


def _bound_violation_inf(value, lower=None, upper=None):
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(np.max(np.maximum(lower - value, 0.0))))
    if upper is not None:
        violation = max(violation, float(np.max(np.maximum(value - upper, 0.0))))
    return violation


def _bounds_to_horizon_matrices(bnds, n_u, horizon_control):
    if bnds is None:
        return None, None

    if len(bnds) == n_u:
        bnds = list(bnds) * int(horizon_control)
    elif len(bnds) != n_u * horizon_control:
        raise ValueError(
            f"bnds must have length {n_u} or {n_u * horizon_control}, got {len(bnds)}."
        )

    lower = np.full((horizon_control, n_u), -np.inf, dtype=float)
    upper = np.full((horizon_control, n_u), np.inf, dtype=float)

    for idx, bound in enumerate(bnds):
        row = idx // n_u
        col = idx % n_u
        lo, hi = bound
        if lo is not None:
            lower[row, col] = float(lo)
        if hi is not None:
            upper[row, col] = float(hi)

    return lower, upper


def _extract_num_iters(problem):
    solver_stats = getattr(problem, "solver_stats", None)
    if solver_stats is None:
        return None
    return getattr(solver_stats, "num_iters", None)


def split_augmented_model(A_aug, B_aug, C_aug):
    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    C_aug = np.asarray(C_aug, float)

    n_aug = A_aug.shape[0]
    if A_aug.shape != (n_aug, n_aug):
        raise ValueError("A_aug must be square.")
    if B_aug.ndim != 2 or B_aug.shape[0] != n_aug:
        raise ValueError("B_aug row dimension must match A_aug.")
    if C_aug.ndim != 2 or C_aug.shape[1] != n_aug:
        raise ValueError("C_aug column dimension must match A_aug.")

    n_y = C_aug.shape[0]
    n_x = n_aug - n_y
    if n_x <= 0:
        raise ValueError("Invalid augmented model: inferred physical state dimension must be positive.")

    return {
        "A_phys": A_aug[:n_x, :n_x].copy(),
        "B_phys": B_aug[:n_x, :].copy(),
        "Bd_phys": A_aug[:n_x, n_x:].copy(),
        "C_phys": C_aug[:, :n_x].copy(),
        "Cd_phys": C_aug[:, n_x:].copy(),
        "n_x": int(n_x),
        "n_y": int(n_y),
        "n_u": int(B_aug.shape[1]),
        "n_aug": int(n_aug),
    }


def factor_psd_left(P, neg_tol=1e-10):
    P = np.asarray(P, float)
    P = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P)
    min_eig = float(np.min(eigvals))
    if min_eig < -float(neg_tol):
        raise ValueError(f"P is not positive semidefinite. Min eigenvalue = {min_eig:.3e}.")

    eigvals = np.where(eigvals > 0.0, eigvals, 0.0)
    return np.diag(np.sqrt(eigvals)) @ eigvecs.T


def design_lyapunov_filter_ingredients(
    A_aug,
    B_aug,
    C_aug,
    Qy_diag,
    Ru_diag=None,
    u_min=None,
    u_max=None,
    u_nom=None,
    lambda_u=1.0,
    qx_eps=1e-10,
    eps_r=1e-9,
    return_debug=False,
):
    model = split_augmented_model(A_aug, B_aug, C_aug)
    A = model["A_phys"]
    B = model["B_phys"]
    C = model["C_phys"]
    n_u = model["n_u"]
    n_y = model["n_y"]

    Qy_diag = np.asarray(Qy_diag, float).reshape(-1)
    if Qy_diag.size != n_y:
        raise ValueError(f"Qy_diag size mismatch. Expected {n_y}, got {Qy_diag.size}.")

    if u_nom is None:
        u_nom = np.zeros(n_u, dtype=float)
    else:
        u_nom = np.asarray(u_nom, float).reshape(-1)
        if u_nom.size != n_u:
            raise ValueError(f"u_nom size mismatch. Expected {n_u}, got {u_nom.size}.")

    Qy = np.diag(np.maximum(Qy_diag, 0.0))
    Qx = C.T @ Qy @ C + float(qx_eps) * np.eye(model["n_x"], dtype=float)
    Qx = 0.5 * (Qx + Qx.T)

    if Ru_diag is None:
        if u_min is None or u_max is None:
            ru_diag = float(lambda_u) * np.ones(n_u, dtype=float)
        else:
            u_min = np.asarray(u_min, float).reshape(-1)
            u_max = np.asarray(u_max, float).reshape(-1)
            if u_min.size != n_u or u_max.size != n_u:
                raise ValueError("u_min/u_max size mismatch.")
            span = np.maximum(np.minimum(u_nom - u_min, u_max - u_nom), eps_r)
            ru_diag = float(lambda_u) / np.maximum(span ** 2, eps_r)
    else:
        ru_diag = np.asarray(Ru_diag, float).reshape(-1)
        if ru_diag.size != n_u:
            raise ValueError(f"Ru_diag size mismatch. Expected {n_u}, got {ru_diag.size}.")
        ru_diag = np.maximum(ru_diag, eps_r)

    R_lyap = np.diag(np.maximum(ru_diag, eps_r))
    P_x = solve_discrete_are(A, B, Qx, R_lyap)
    P_x = 0.5 * (P_x + P_x.T)
    K_x = -np.linalg.solve(R_lyap + B.T @ P_x @ B, B.T @ P_x @ A)
    S_x = factor_psd_left(P_x)

    ingredients = {
        **model,
        "Qy_diag": Qy_diag.copy(),
        "Qx": Qx.copy(),
        "R_lyap_diag": ru_diag.copy(),
        "R_lyap": R_lyap.copy(),
        "u_nom": u_nom.copy(),
        "P_x": P_x.copy(),
        "K_x": K_x.copy(),
        "S_x": S_x.copy(),
        "eig_cl": np.linalg.eigvals(A + B @ K_x),
    }

    if return_debug:
        dbg = {
            "A_phys": A.copy(),
            "B_phys": B.copy(),
            "C_phys": C.copy(),
            "Qx": Qx.copy(),
            "R_lyap": R_lyap.copy(),
            "P_x": P_x.copy(),
            "K_x": K_x.copy(),
            "eig_cl": ingredients["eig_cl"].copy(),
        }
        return ingredients, dbg
    return ingredients


def physical_error_from_target(xhat_aug, target_info):
    xhat_aug = np.asarray(xhat_aug, float).reshape(-1)
    if target_info is None or not target_info.get("success", False):
        raise ValueError("target_info must contain a successful target package.")

    x_s = np.asarray(target_info["x_s"], float).reshape(-1)
    n_x = x_s.size
    if xhat_aug.size < n_x:
        raise ValueError("xhat_aug is shorter than the physical target state.")
    return xhat_aug[:n_x] - x_s


def predict_next_physical_error(ingredients, e_x, u, u_s):
    A = np.asarray(ingredients["A_phys"], float)
    B = np.asarray(ingredients["B_phys"], float)
    e_x = np.asarray(e_x, float).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    u_s = np.asarray(u_s, float).reshape(-1)
    return A @ e_x + B @ (u - u_s)


def lyapunov_value(e_x, P_x):
    e_x = np.asarray(e_x, float).reshape(-1, 1)
    P_x = np.asarray(P_x, float)
    return float((e_x.T @ P_x @ e_x).item())


def lyapunov_bound(V_k, rho, eps_lyap):
    rho = float(rho)
    eps_lyap = float(eps_lyap)
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must satisfy 0 < rho < 1.")
    if eps_lyap < 0.0:
        raise ValueError("eps_lyap must be nonnegative.")
    return rho * float(V_k) + eps_lyap


def _bound_ok(value, lower=None, upper=None, tol=1e-9):
    violation = _bound_violation_inf(value, lower=lower, upper=upper)
    return bool(violation <= float(tol)), float(violation)


def evaluate_candidate_action(
    u_cand,
    xhat_aug,
    target_info,
    ingredients,
    rho=0.99,
    eps_lyap=1e-9,
    u_min=None,
    u_max=None,
    u_prev=None,
    du_min=None,
    du_max=None,
    tol=1e-9,
):
    if target_info is None or not target_info.get("success", False):
        return {
            "accepted": False,
            "accept_reason": None,
            "reject_reason": "target_unavailable",
            "candidate_bounds_ok": False,
            "candidate_move_ok": False,
            "candidate_lyap_ok": False,
        }

    u_cand = np.asarray(u_cand, float).reshape(-1)
    u_s = np.asarray(target_info["u_s"], float).reshape(-1)
    d_s = np.asarray(target_info["d_s"], float).reshape(-1)
    x_s = np.asarray(target_info["x_s"], float).reshape(-1)
    e_x = physical_error_from_target(xhat_aug, target_info)
    e_u = u_cand - u_s

    e_x_next = predict_next_physical_error(ingredients, e_x=e_x, u=u_cand, u_s=u_s)
    V_k = lyapunov_value(e_x, ingredients["P_x"])
    V_next_cand = lyapunov_value(e_x_next, ingredients["P_x"])
    V_bound = lyapunov_bound(V_k, rho=rho, eps_lyap=eps_lyap)
    lyap_margin = float(V_next_cand - V_bound)
    candidate_lyap_ok = bool(lyap_margin <= float(tol))

    bounds_ok, bounds_violation = _bound_ok(u_cand, lower=u_min, upper=u_max, tol=tol)

    if u_prev is None or (du_min is None and du_max is None):
        candidate_move_ok = True
        move_violation = 0.0
    else:
        delta_u = u_cand - np.asarray(u_prev, float).reshape(-1)
        candidate_move_ok, move_violation = _bound_ok(delta_u, lower=du_min, upper=du_max, tol=tol)

    C = np.asarray(ingredients["C_phys"], float)
    Cd = np.asarray(ingredients["Cd_phys"], float)
    x_next_pred = x_s + e_x_next
    y_next_pred = C @ x_next_pred + Cd @ d_s
    y_s = np.asarray(target_info["y_s"], float).reshape(-1)

    accepted = bool(bounds_ok and candidate_move_ok and candidate_lyap_ok)
    accept_reason = "candidate_ok" if accepted else None
    if accepted:
        reject_reason = None
    elif not bounds_ok:
        reject_reason = "input_bounds"
    elif not candidate_move_ok:
        reject_reason = "move_bounds"
    else:
        reject_reason = "lyapunov"

    return {
        "accepted": accepted,
        "accept_reason": accept_reason,
        "reject_reason": reject_reason,
        "candidate_bounds_ok": bool(bounds_ok),
        "candidate_move_ok": bool(candidate_move_ok),
        "candidate_lyap_ok": bool(candidate_lyap_ok),
        "u_cand": u_cand.copy(),
        "u_s": u_s.copy(),
        "x_s": x_s.copy(),
        "d_s": d_s.copy(),
        "e_x": e_x.copy(),
        "e_u": e_u.copy(),
        "e_x_next_pred": e_x_next.copy(),
        "y_next_pred": y_next_pred.copy(),
        "y_s": y_s.copy(),
        "V_k": float(V_k),
        "V_next_cand": float(V_next_cand),
        "V_bound": float(V_bound),
        "lyap_margin": float(lyap_margin),
        "candidate_bounds_violation": float(bounds_violation),
        "candidate_move_violation": float(move_violation),
        "rho": float(rho),
        "eps_lyap": float(eps_lyap),
    }


class StandardTrackingLyapunovMpcRawlingsTargetSolver:
    def __init__(
        self,
        A_aug,
        B_aug,
        C_aug,
        Qy_diag,
        Su_diag,
        NP,
        NC,
        P_x,
        K_x,
        Rdu_diag=None,
        terminal_set_on=True,
        terminal_alpha_scale=1.0,
        terminal_cost_scale=1.0,
        D=None,
        solver_pref_qp=None,
        solver_pref_conic=None,
    ):
        self.A = np.asarray(A_aug, float)
        self.B = np.asarray(B_aug, float)
        self.C = np.asarray(C_aug, float)
        self.D = None if D is None else np.asarray(D, float)

        self.Qy = np.asarray(Qy_diag, float).reshape(-1)
        self.Su = np.asarray(Su_diag, float).reshape(-1)
        self.Rdu = None if Rdu_diag is None else np.asarray(Rdu_diag, float).reshape(-1)

        self.NP = int(NP)
        self.NC = int(NC)

        self.P_x = np.asarray(P_x, float)
        self.K_x = np.asarray(K_x, float)
        self.terminal_set_on = bool(terminal_set_on)
        self.terminal_alpha_scale = float(terminal_alpha_scale)
        self.terminal_cost_scale = float(terminal_cost_scale)

        self.n_aug = self.A.shape[0]
        self.n_u = self.B.shape[1]
        self.n_y = self.C.shape[0]
        self.n_x = self.n_aug - self.n_y

        self.A_phys = self.A[:self.n_x, :self.n_x]
        self.B_phys = self.B[:self.n_x, :]
        self.C_phys = self.C[:, :self.n_x]

        self.Qy_mat = np.diag(np.maximum(self.Qy, 0.0))
        self.Su_mat = np.diag(np.maximum(self.Su, 0.0))
        self.Rdu_mat = None if self.Rdu is None else np.diag(np.maximum(self.Rdu, 0.0))

        self.solver_pref_qp = solver_pref_qp
        self.solver_pref_conic = solver_pref_conic

        if self.P_x.shape != (self.n_x, self.n_x):
            raise ValueError("P_x shape mismatch.")
        if self.K_x.shape != (self.n_u, self.n_x):
            raise ValueError("K_x shape mismatch.")
        if self.Qy.size != self.n_y:
            raise ValueError("Qy_diag size mismatch.")
        if self.Su.size != self.n_u:
            raise ValueError("Su_diag size mismatch.")
        if self.Rdu is not None and self.Rdu.size != self.n_u:
            raise ValueError("Rdu_diag size mismatch.")

    def _predict_from_sequence(self, u_sequence, x0_aug):
        x0_aug = np.asarray(x0_aug, float).reshape(-1)
        x_pred = np.zeros((self.n_aug, self.NP + 1), dtype=float)
        x_pred[:, 0] = x0_aug
        for step_idx in range(self.NP):
            ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
            x_pred[:, step_idx + 1] = self.A @ x_pred[:, step_idx] + self.B @ u_sequence[ctrl_idx, :]

        y_pred = self.C @ x_pred
        if self.D is not None:
            for step_idx in range(self.NP):
                ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
                y_pred[:, step_idx + 1] = y_pred[:, step_idx + 1] + self.D @ u_sequence[ctrl_idx, :]
        return x_pred, y_pred

    def _tracking_cost(self, u_sequence, y_target, u_prev_dev, x0_aug, x_s, u_s):
        y_target = np.asarray(y_target, float).reshape(self.n_y)
        u_prev_dev = np.asarray(u_prev_dev, float).reshape(self.n_u)
        x_s = np.asarray(x_s, float).reshape(self.n_x)
        u_s = np.asarray(u_s, float).reshape(self.n_u)

        x_pred, y_pred = self._predict_from_sequence(u_sequence, x0_aug)
        y_err = y_pred[:, 1:] - y_target[:, None]
        du = compute_du_sequence(u_sequence, u_prev_dev)
        u_err = u_sequence - u_s.reshape(1, -1)

        obj = 0.0
        for output_idx in range(self.n_y):
            obj += float(self.Qy[output_idx]) * float(np.sum(y_err[output_idx, :] ** 2))
        for input_idx in range(self.n_u):
            obj += float(self.Su[input_idx]) * float(np.sum(u_err[:, input_idx] ** 2))
        if self.Rdu is not None:
            for input_idx in range(self.n_u):
                obj += float(self.Rdu[input_idx]) * float(np.sum(du[:, input_idx] ** 2))

        e_terminal = x_pred[:self.n_x, -1] - x_s
        obj += self.terminal_cost_scale * float(e_terminal.T @ self.P_x @ e_terminal)
        return float(obj)

    def _terminal_value_from_sequence(self, u_sequence, x0_aug, x_s):
        x_pred, _ = self._predict_from_sequence(u_sequence, x0_aug)
        e_terminal = x_pred[:self.n_x, -1] - np.asarray(x_s, float).reshape(self.n_x)
        return float(e_terminal.T @ self.P_x @ e_terminal)

    def _terminal_value(self, x_opt, x0_aug, x_s):
        u_sequence = reshape_u_sequence(x_opt, self.n_u, self.NC)
        return self._terminal_value_from_sequence(u_sequence, x0_aug=x0_aug, x_s=x_s)

    def _evaluate_tracking_solution(
        self,
        u_sequence,
        x_pred,
        x0_aug,
        x_s,
        lower,
        upper,
        alpha_terminal,
        terminal_constraint_active,
        status,
    ):
        tol = _TRACKING_TOL_BY_STATUS.get(status)
        result = {
            "accepted": False,
            "reject_reason": None,
            "dyn_residual_inf": None,
            "initial_residual_inf": None,
            "bound_violation_inf": None,
            "terminal_value": None,
            "terminal_constraint_violation": None,
        }

        if tol is None:
            result["reject_reason"] = "solver_status"
            return result

        x0_aug = np.asarray(x0_aug, float).reshape(-1)
        x_s = np.asarray(x_s, float).reshape(-1)
        u_sequence = np.asarray(u_sequence, float)
        x_pred = np.asarray(x_pred, float)

        initial_residual_inf = float(np.max(np.abs(x_pred[:, 0] - x0_aug)))
        dyn_residual_inf = initial_residual_inf
        for step_idx in range(self.NP):
            ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
            residual = x_pred[:, step_idx + 1] - (self.A @ x_pred[:, step_idx] + self.B @ u_sequence[ctrl_idx, :])
            dyn_residual_inf = max(dyn_residual_inf, float(np.max(np.abs(residual))))

        bound_violation_inf = _bound_violation_inf(u_sequence, lower=lower, upper=upper)
        terminal_value = self._terminal_value_from_sequence(u_sequence, x0_aug=x0_aug, x_s=x_s)
        terminal_constraint_violation = 0.0
        if terminal_constraint_active:
            terminal_constraint_violation = max(float(terminal_value) - float(alpha_terminal), 0.0)

        if dyn_residual_inf > tol:
            result["reject_reason"] = "dyn_residual"
        elif bound_violation_inf > tol:
            result["reject_reason"] = "bound_violation"
        elif terminal_constraint_active and terminal_constraint_violation > tol:
            result["reject_reason"] = "terminal_constraint"
        else:
            result["accepted"] = True

        result.update({
            "dyn_residual_inf": dyn_residual_inf,
            "initial_residual_inf": initial_residual_inf,
            "bound_violation_inf": float(bound_violation_inf),
            "terminal_value": float(terminal_value),
            "terminal_constraint_violation": float(terminal_constraint_violation),
        })
        return result

    def standard_tracking_report(self, x_opt, x0_aug, x_s, u_s, y_target, u_prev_dev, alpha_terminal):
        u_sequence = reshape_u_sequence(x_opt, self.n_u, self.NC)
        du = compute_du_sequence(u_sequence, u_prev_dev)
        x_pred, y_pred = self._predict_from_sequence(u_sequence, x0_aug)

        x_s = np.asarray(x_s, float).reshape(self.n_x)
        u_s = np.asarray(u_s, float).reshape(self.n_u)
        y_target = np.asarray(y_target, float).reshape(self.n_y)

        e_x_path = x_pred[:self.n_x, :] - x_s[:, None]
        e_y_path = y_pred - y_target[:, None]
        u_err_path = u_sequence - u_s.reshape(1, -1)
        terminal_value = float(e_x_path[:, -1].T @ self.P_x @ e_x_path[:, -1])

        if alpha_terminal is None:
            terminal_margin = None
            terminal_set_violated = False
            alpha_terminal_used = None
        else:
            alpha_terminal_used = float(alpha_terminal)
            terminal_margin = float(alpha_terminal_used - terminal_value)
            terminal_set_violated = bool(terminal_value > alpha_terminal_used + 1e-8)

        return {
            "x_pred_path": x_pred.copy(),
            "y_pred_path": y_pred.copy(),
            "e_x_path": e_x_path.copy(),
            "e_y_path": e_y_path.copy(),
            "u_seq_opt": u_sequence.copy(),
            "du_seq_opt": du.copy(),
            "u_err_path": u_err_path.copy(),
            "terminal_value": terminal_value,
            "alpha_terminal_used": alpha_terminal_used,
            "terminal_margin": terminal_margin,
            "terminal_set_violated": terminal_set_violated,
        }

    def solve_tracking_mpc_step(
        self,
        IC_opt,
        bnds,
        y_target,
        u_prev_dev,
        x0_aug,
        x_s,
        u_s,
        alpha_terminal,
        options=None,
    ):
        if not HAS_CVXPY:
            raise ImportError("CVXPY is required for the standard Lyapunov tracking MPC solver.")

        options = {} if options is None else dict(options)
        solver_pref_override = options.pop("solver_pref", None)
        warm_start = bool(options.pop("warm_start", True))
        verbose = bool(options.pop("verbose", False))
        solve_kwargs = dict(options.pop("solve_kwargs", {}))

        x0_aug = np.asarray(x0_aug, float).reshape(-1)
        x_s = np.asarray(x_s, float).reshape(self.n_x)
        u_s = np.asarray(u_s, float).reshape(self.n_u)
        y_target = np.asarray(y_target, float).reshape(self.n_y)
        u_prev_dev = np.asarray(u_prev_dev, float).reshape(self.n_u)
        lower, upper = _bounds_to_horizon_matrices(bnds, self.n_u, self.NC)

        active_terminal_constraint = (
            self.terminal_set_on
            and alpha_terminal is not None
            and np.isfinite(float(alpha_terminal))
        )

        u_var = cp.Variable((self.NC, self.n_u))
        x_var = cp.Variable((self.n_aug, self.NP + 1))

        constraints = [x_var[:, 0] == x0_aug]
        if lower is not None:
            lower_rows, lower_cols = np.where(np.isfinite(lower))
            for row_idx, col_idx in zip(lower_rows, lower_cols):
                constraints.append(u_var[row_idx, col_idx] >= float(lower[row_idx, col_idx]))
        if upper is not None:
            upper_rows, upper_cols = np.where(np.isfinite(upper))
            for row_idx, col_idx in zip(upper_rows, upper_cols):
                constraints.append(u_var[row_idx, col_idx] <= float(upper[row_idx, col_idx]))

        objective = 0.0
        for step_idx in range(self.NP):
            ctrl_idx = step_idx if step_idx < self.NC else self.NC - 1
            constraints.append(
                x_var[:, step_idx + 1] == self.A @ x_var[:, step_idx] + self.B @ u_var[ctrl_idx, :]
            )

            y_expr = self.C @ x_var[:, step_idx + 1]
            if self.D is not None:
                y_expr = y_expr + self.D @ u_var[ctrl_idx, :]
            objective += cp.quad_form(y_expr - y_target, self.Qy_mat)

        for ctrl_idx in range(self.NC):
            objective += cp.quad_form(u_var[ctrl_idx, :] - u_s, self.Su_mat)

        if self.Rdu_mat is not None:
            objective += cp.quad_form(u_var[0, :] - u_prev_dev, self.Rdu_mat)
            for ctrl_idx in range(1, self.NC):
                objective += cp.quad_form(u_var[ctrl_idx, :] - u_var[ctrl_idx - 1, :], self.Rdu_mat)

        terminal_error = x_var[:self.n_x, self.NP] - x_s
        terminal_value_expr = cp.quad_form(terminal_error, self.P_x)
        objective += self.terminal_cost_scale * terminal_value_expr
        if active_terminal_constraint:
            constraints.append(terminal_value_expr <= float(alpha_terminal))

        problem = cp.Problem(cp.Minimize(objective), constraints)

        ic_flat = np.asarray(IC_opt, float).reshape(-1)
        if ic_flat.size == self.n_u * self.NC:
            try:
                u_guess = reshape_u_sequence(ic_flat, self.n_u, self.NC)
                u_var.value = u_guess
                x_guess, _ = self._predict_from_sequence(u_guess, x0_aug)
                x_var.value = x_guess
            except Exception:
                pass

        if solver_pref_override is None:
            solver_pref = self.solver_pref_conic if active_terminal_constraint else self.solver_pref_qp
        else:
            solver_pref = solver_pref_override
        solver_sequence = tracking_solver_sequence(
            active_terminal_constraint,
            solver_pref=solver_pref,
        )

        last_status = None
        last_solver = None
        last_error = None
        last_objective = None
        last_nit = None
        last_eval = None

        for solver_name in solver_sequence:
            try:
                problem.solve(
                    solver=solver_name,
                    warm_start=warm_start,
                    verbose=verbose,
                    **solve_kwargs,
                )
                last_status = problem.status
                last_solver = solver_name
                last_nit = _extract_num_iters(problem)
                if problem.value is not None:
                    last_objective = float(problem.value)

                if u_var.value is None or x_var.value is None:
                    continue

                u_value = np.asarray(u_var.value, float)
                x_value = np.asarray(x_var.value, float)
                last_eval = self._evaluate_tracking_solution(
                    u_sequence=u_value,
                    x_pred=x_value,
                    x0_aug=x0_aug,
                    x_s=x_s,
                    lower=lower,
                    upper=upper,
                    alpha_terminal=alpha_terminal,
                    terminal_constraint_active=active_terminal_constraint,
                    status=problem.status,
                )
                if problem.status in _OPTIMAL_STATUSES and last_eval["accepted"]:
                    return SimpleNamespace(
                        success=True,
                        x=u_value.reshape(-1),
                        status=problem.status,
                        message="optimal",
                        fun=last_objective,
                        nit=last_nit,
                        solver=solver_name,
                        error=None,
                        objective_value=last_objective,
                        dyn_residual_inf=last_eval["dyn_residual_inf"],
                        bound_violation_inf=last_eval["bound_violation_inf"],
                        terminal_value=last_eval["terminal_value"],
                        terminal_constraint_violation=last_eval["terminal_constraint_violation"],
                    )
            except Exception as exc:
                last_error = repr(exc)

        reject_reason = None if last_eval is None else last_eval.get("reject_reason")
        if reject_reason is None and last_error is not None:
            reject_reason = "solver_error"
        if reject_reason is None:
            reject_reason = "solver_status"

        return SimpleNamespace(
            success=False,
            x=None,
            status=last_status,
            message=reject_reason,
            fun=last_objective,
            nit=last_nit,
            solver=last_solver,
            error=last_error,
            objective_value=last_objective,
            dyn_residual_inf=None if last_eval is None else last_eval["dyn_residual_inf"],
            bound_violation_inf=None if last_eval is None else last_eval["bound_violation_inf"],
            terminal_value=None if last_eval is None else last_eval["terminal_value"],
            terminal_constraint_violation=None
            if last_eval is None
            else last_eval["terminal_constraint_violation"],
        )
# ==================== END FILE: Lyapunov/lyapunov_core.py ======================

# ==================== BEGIN FILE: Lyapunov/safety_filter.py ====================
import numpy as np

try:
    import cvxpy as cp

    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

from Lyapunov.lyapunov_core import evaluate_candidate_action, lyapunov_bound
from Lyapunov.upstream_controllers import solve_offset_free_mpc_candidate
from utils.lyapunov_utils import diag_psd_from_vector, safety_filter_solver_sequence


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


def _as_1d(name, value, expected_size=None):
    arr = np.asarray(value, float).reshape(-1)
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(f"{name} has size {arr.size}, expected {expected_size}.")
    return arr


def _weight_diag(diag_vals, size, default):
    if diag_vals is not None:
        diag_vals = np.asarray(diag_vals, float).reshape(-1)
        if diag_vals.size == 1:
            diag_vals = np.full(size, float(diag_vals.item()), dtype=float)
    mat, diag_used = diag_psd_from_vector(diag_vals, size, eps=1e-12, default=default)
    return mat, diag_used


def _maybe_vector(values, size):
    if values is None:
        return None
    values = np.asarray(values, float).reshape(-1)
    if values.size == 1:
        return np.full(size, float(values.item()), dtype=float)
    return _as_1d("vector", values, expected_size=size)


def _qcqp_output_target(target_info, lyap_config, n_y):
    y_target = lyap_config.get("tracking_output_target")
    if y_target is not None:
        return _as_1d("tracking_output_target", y_target, expected_size=n_y)
    if target_info is not None and target_info.get("y_s") is not None:
        return np.asarray(target_info["y_s"], float).reshape(-1)
    return None


def _postcheck_action(u_try, xhat_aug, target_info, model_info, lyap_config, bounds_info, u_prev):
    tol = float(lyap_config.get("tol", 1e-9))
    return evaluate_candidate_action(
        u_cand=u_try,
        xhat_aug=xhat_aug,
        target_info=target_info,
        ingredients=model_info,
        rho=float(lyap_config.get("rho", 0.99)),
        eps_lyap=float(lyap_config.get("eps_lyap", 1e-9)),
        u_min=bounds_info.get("u_min"),
        u_max=bounds_info.get("u_max"),
        u_prev=u_prev,
        du_min=bounds_info.get("du_min"),
        du_max=bounds_info.get("du_max"),
        tol=tol,
    )


def _final_lyap_target_info(target_info, lyap_config):
    if target_info is not None and target_info.get("success", False):
        return target_info, "current_target"

    diagnostic_target = lyap_config.get("final_lyap_target_info")
    if isinstance(diagnostic_target, dict) and diagnostic_target.get("success", False):
        return diagnostic_target, str(lyap_config.get("final_lyap_target_source", "last_valid_target"))

    return None, None


def _attach_final_lyap(
    base_debug,
    u_try,
    xhat_aug,
    target_info,
    model_info,
    lyap_config,
    bounds_info,
    u_prev,
):
    base_debug.update({
        "final_lyap_value": None,
        "final_lyap_margin": None,
        "final_lyap_ok": None,
        "final_lyap_bound": None,
        "final_y_next_pred": None,
        "final_lyap_target_source": None,
    })

    if u_try is None:
        return base_debug

    lyap_target, lyap_target_source = _final_lyap_target_info(target_info, lyap_config)
    if lyap_target is None:
        return base_debug

    post = _postcheck_action(
        u_try=u_try,
        xhat_aug=xhat_aug,
        target_info=lyap_target,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )

    V_next = post.get("V_next_cand")
    V_bound = post.get("V_bound")
    base_debug.update({
        "final_lyap_value": V_next,
        "final_lyap_bound": V_bound,
        "final_lyap_margin": None if V_next is None or V_bound is None else float(V_bound) - float(V_next),
        "final_lyap_ok": post.get("candidate_lyap_ok"),
        "final_y_next_pred": None if post.get("y_next_pred") is None else np.asarray(post["y_next_pred"], float).reshape(-1).copy(),
        "final_lyap_target_source": lyap_target_source,
    })
    return base_debug


def _fallback_candidates(target_info, bounds_info, u_prev):
    u_min = bounds_info.get("u_min")
    u_max = bounds_info.get("u_max")
    out = []

    fallback_safe_input = bounds_info.get("fallback_safe_input")
    if fallback_safe_input is not None:
        u_prev_safe = np.asarray(fallback_safe_input, float).reshape(-1)
        if u_min is not None and u_max is not None:
            u_prev_safe = np.clip(u_prev_safe, u_min, u_max)
        out.append(("fallback_previous_secondary", u_prev_safe))
    elif u_prev is not None:
        u_prev = np.asarray(u_prev, float).reshape(-1)
        if u_min is not None and u_max is not None:
            u_prev = np.clip(u_prev, u_min, u_max)
        out.append(("fallback_previous_secondary", u_prev))

    if target_info is not None and target_info.get("u_s") is not None:
        u_s = np.asarray(target_info["u_s"], float).reshape(-1)
        if u_min is not None and u_max is not None:
            u_s = np.clip(u_s, u_min, u_max)
        out.append(("fallback_steady_secondary", u_s))

    return out


def _attempt_secondary_fallbacks(base_debug, xhat_aug, target_info, model_info, lyap_config, bounds_info, u_prev):
    for mode, fallback in _fallback_candidates(target_info, bounds_info, u_prev):
        if target_info is None or not target_info.get("success", False):
            base_debug.update({
                "u_safe": fallback.copy(),
                "correction_mode": mode,
                "fallback_mode": mode,
                "fallback_verified": False,
                "verified": False,
            })
            _attach_final_lyap(
                base_debug=base_debug,
                u_try=fallback,
                xhat_aug=xhat_aug,
                target_info=target_info,
                model_info=model_info,
                lyap_config=lyap_config,
                bounds_info=bounds_info,
                u_prev=u_prev,
            )
            return fallback.copy(), base_debug

        post = _postcheck_action(
            u_try=fallback,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        if post.get("accepted", False):
            base_debug.update({
                "accepted": True,
                "accept_reason": mode,
                "u_safe": fallback.copy(),
                "correction_mode": mode,
                "fallback_mode": mode,
                "fallback_verified": True,
                "fallback_bounds_ok": bool(post.get("candidate_bounds_ok", False)),
                "fallback_move_ok": bool(post.get("candidate_move_ok", False)),
                "fallback_lyap_ok": bool(post.get("candidate_lyap_ok", False)),
                "verified": True,
            })
            _attach_final_lyap(
                base_debug=base_debug,
                u_try=fallback,
                xhat_aug=xhat_aug,
                target_info=target_info,
                model_info=model_info,
                lyap_config=lyap_config,
                bounds_info=bounds_info,
                u_prev=u_prev,
            )
            return fallback.copy(), base_debug

    return None, base_debug


def _attempt_mpc_fallback(
    base_debug,
    xhat_aug,
    target_info,
    model_info,
    lyap_config,
    bounds_info,
    u_prev,
    fallback_config,
):
    if fallback_config is None:
        return None, base_debug

    fallback_config = dict(fallback_config)
    mode = fallback_config.get("mode", "offset_free_mpc")
    if mode != "offset_free_mpc":
        return None, base_debug

    y_sp = fallback_config.get("y_sp")
    MPC_obj = fallback_config.get("MPC_obj")
    if y_sp is None or MPC_obj is None:
        return None, base_debug

    u_mpc, mpc_info = solve_offset_free_mpc_candidate(
        MPC_obj=MPC_obj,
        y_sp=y_sp,
        u_prev_dev=u_prev if u_prev is not None else fallback_config.get("u_prev_dev", np.zeros(model_info["n_u"], dtype=float)),
        x0_model=fallback_config.get("x0_model", xhat_aug),
        IC_opt=fallback_config.get("IC_opt"),
        bnds=fallback_config.get("bnds"),
        cons=fallback_config.get("cons"),
        return_debug=True,
    )

    base_debug.update({
        "u_fallback_mpc": None if u_mpc is None else np.asarray(u_mpc, float).reshape(-1).copy(),
        "fallback_mode": "offset_free_mpc",
        "fallback_solver_status": mpc_info.get("status"),
        "fallback_solver_message": mpc_info.get("message"),
        "fallback_objective_value": mpc_info.get("objective_value"),
        "fallback_ic_next": None if mpc_info.get("IC_opt_next") is None else np.asarray(mpc_info["IC_opt_next"], float).reshape(-1).copy(),
        "fallback_upstream_info": mpc_info,
        "fallback_tracking_target_source": fallback_config.get("tracking_target_source"),
        "fallback_target_mismatch_inf": fallback_config.get("target_mismatch_inf"),
    })

    if u_mpc is None:
        return None, base_debug

    if target_info is None or not target_info.get("success", False):
        if bool(fallback_config.get("allow_unverified", True)):
            u_mpc = np.asarray(u_mpc, float).reshape(-1)
            if bounds_info.get("u_min") is not None and bounds_info.get("u_max") is not None:
                u_mpc = np.clip(u_mpc, bounds_info["u_min"], bounds_info["u_max"])
            base_debug.update({
                "accepted": False,
                "accept_reason": None,
                "reject_reason": "target_unavailable",
                "u_safe": u_mpc.copy(),
                "correction_mode": "fallback_mpc_unverified",
                "fallback_verified": False,
                "verified": False,
            })
            _attach_final_lyap(
                base_debug=base_debug,
                u_try=u_mpc,
                xhat_aug=xhat_aug,
                target_info=target_info,
                model_info=model_info,
                lyap_config=lyap_config,
                bounds_info=bounds_info,
                u_prev=u_prev,
            )
            return u_mpc.copy(), base_debug
        return None, base_debug

    post = _postcheck_action(
        u_try=u_mpc,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )
    base_debug.update({
        "fallback_bounds_ok": bool(post.get("candidate_bounds_ok", False)),
        "fallback_move_ok": bool(post.get("candidate_move_ok", False)),
        "fallback_lyap_ok": bool(post.get("candidate_lyap_ok", False)),
    })

    if post.get("accepted", False):
        base_debug.update({
            "accepted": True,
            "accept_reason": "fallback_mpc_verified",
            "u_safe": np.asarray(u_mpc, float).reshape(-1).copy(),
            "correction_mode": "fallback_mpc_verified",
            "fallback_verified": True,
            "verified": True,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=u_mpc,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return np.asarray(u_mpc, float).reshape(-1).copy(), base_debug

    if bool(fallback_config.get("allow_unverified", True)):
        u_mpc = np.asarray(u_mpc, float).reshape(-1)
        if bounds_info.get("u_min") is not None and bounds_info.get("u_max") is not None:
            u_mpc = np.clip(u_mpc, bounds_info["u_min"], bounds_info["u_max"])
        base_debug.update({
            "accepted": False,
            "accept_reason": None,
            "u_safe": u_mpc.copy(),
            "correction_mode": "fallback_mpc_unverified",
            "fallback_verified": False,
            "verified": False,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=u_mpc,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return u_mpc.copy(), base_debug

    return None, base_debug


def apply_lyapunov_safety_filter(
    u_cand,
    xhat_aug,
    target_info,
    model_info,
    lyap_config,
    u_prev=None,
    bounds_info=None,
    fallback_config=None,
    return_debug=False,
):
    if bounds_info is None:
        bounds_info = {}
    else:
        bounds_info = dict(bounds_info)
    lyap_config = {} if lyap_config is None else dict(lyap_config)

    n_u = int(model_info["n_u"])
    n_y = int(model_info["n_y"])

    u_cand = _as_1d("u_cand", u_cand, expected_size=n_u)
    xhat_aug = _as_1d("xhat_aug", xhat_aug)
    u_prev = None if u_prev is None else _as_1d("u_prev", u_prev, expected_size=n_u)

    bounds_info["u_min"] = _maybe_vector(bounds_info.get("u_min"), n_u)
    bounds_info["u_max"] = _maybe_vector(bounds_info.get("u_max"), n_u)
    bounds_info["du_min"] = _maybe_vector(bounds_info.get("du_min"), n_u)
    bounds_info["du_max"] = _maybe_vector(bounds_info.get("du_max"), n_u)

    candidate_eval = _postcheck_action(
        u_try=u_cand,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )

    base_debug = {
        "source": str(lyap_config.get("source", "unknown")),
        "accepted": bool(candidate_eval.get("accepted", False)),
        "accept_reason": candidate_eval.get("accept_reason"),
        "reject_reason": candidate_eval.get("reject_reason"),
        "candidate_bounds_ok": candidate_eval.get("candidate_bounds_ok"),
        "candidate_move_ok": candidate_eval.get("candidate_move_ok"),
        "candidate_lyap_ok": candidate_eval.get("candidate_lyap_ok"),
        "u_cand": u_cand.copy(),
        "u_prev": None if u_prev is None else u_prev.copy(),
        "u_safe": None,
        "u_s": None if target_info is None or target_info.get("u_s") is None else np.asarray(target_info["u_s"], float).reshape(-1).copy(),
        "x_s": None if target_info is None or target_info.get("x_s") is None else np.asarray(target_info["x_s"], float).reshape(-1).copy(),
        "d_s": None if target_info is None or target_info.get("d_s") is None else np.asarray(target_info["d_s"], float).reshape(-1).copy(),
        "e_x": None if candidate_eval.get("e_x") is None else np.asarray(candidate_eval["e_x"], float).reshape(-1).copy(),
        "V_k": candidate_eval.get("V_k"),
        "V_next_cand": candidate_eval.get("V_next_cand"),
        "V_bound": candidate_eval.get("V_bound"),
        "rho": float(lyap_config.get("rho", 0.99)),
        "eps_lyap": float(lyap_config.get("eps_lyap", 1e-9)),
        "solver_status": None,
        "solver_name": None,
        "solver_residuals": {},
        "trust_region_violation": None,
        "slack_v": 0.0,
        "slack_u": 0.0,
        "correction_mode": None,
        "verified": False,
        "target_success": bool(target_info is not None and target_info.get("success", False)),
        "target_info": target_info,
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
        "qcqp_tracking_target": None,
        "qcqp_tracking_target_source": lyap_config.get("tracking_output_target_source"),
        "final_lyap_value": None,
        "final_lyap_margin": None,
        "final_lyap_ok": None,
        "final_lyap_bound": None,
        "final_y_next_pred": None,
        "final_lyap_target_source": None,
    }

    if candidate_eval.get("accepted", False):
        base_debug.update({
            "u_safe": u_cand.copy(),
            "correction_mode": "accepted_candidate",
            "verified": True,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=u_cand,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return u_cand.copy(), base_debug

    if target_info is None or not target_info.get("success", False):
        u_safe, base_debug = _attempt_mpc_fallback(
            base_debug=base_debug,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
            fallback_config=fallback_config,
        )
        if u_safe is not None:
            return u_safe, base_debug

        u_safe, base_debug = _attempt_secondary_fallbacks(
            base_debug=base_debug,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        if u_safe is not None:
            return u_safe, base_debug

        base_debug.update({
            "u_safe": u_cand.copy(),
            "correction_mode": "target_unavailable_unverified",
            "verified": False,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=u_cand,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return u_cand.copy(), base_debug

    if not HAS_CVXPY:
        u_safe, base_debug = _attempt_mpc_fallback(
            base_debug=base_debug,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
            fallback_config=fallback_config,
        )
        if u_safe is not None:
            return u_safe, base_debug

        u_safe, base_debug = _attempt_secondary_fallbacks(
            base_debug=base_debug,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        if u_safe is not None:
            return u_safe, base_debug

        fallback = np.asarray(target_info["u_s"], float).reshape(-1).copy()
        if bounds_info["u_min"] is not None and bounds_info["u_max"] is not None:
            fallback = np.clip(fallback, bounds_info["u_min"], bounds_info["u_max"])
        base_debug.update({
            "u_safe": fallback.copy(),
            "correction_mode": "no_cvxpy_unverified",
            "solver_status": "no_cvxpy",
            "verified": False,
        })
        _attach_final_lyap(
            base_debug=base_debug,
            u_try=fallback,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        return fallback.copy(), base_debug

    rho = float(lyap_config.get("rho", 0.99))
    eps_lyap = float(lyap_config.get("eps_lyap", 1e-9))

    W_cand, _ = _weight_diag(lyap_config.get("candidate_weight_diag"), n_u, default=1.0)
    W_move, _ = _weight_diag(lyap_config.get("move_weight_diag"), n_u, default=1.0)
    W_steady, _ = _weight_diag(lyap_config.get("steady_weight_diag"), n_u, default=1.0)
    W_output, _ = _weight_diag(lyap_config.get("output_weight_diag"), n_y, default=1.0)

    use_output_tracking = bool(lyap_config.get("use_output_tracking_term", True))
    allow_lyap_slack = bool(lyap_config.get("allow_lyap_slack", False))
    lyap_slack_weight = float(lyap_config.get("lyap_slack_weight", 1e6))
    trust_region_delta = lyap_config.get("trust_region_delta")
    trust_region_weight = float(lyap_config.get("trust_region_weight", 1e4))

    e_x = np.asarray(candidate_eval["e_x"], float).reshape(-1)
    u_s = np.asarray(target_info["u_s"], float).reshape(-1)
    d_s = np.asarray(target_info["d_s"], float).reshape(-1)
    y_s = np.asarray(target_info["y_s"], float).reshape(-1)
    y_track = _qcqp_output_target(target_info, lyap_config, n_y=n_y)
    A = np.asarray(model_info["A_phys"], float)
    B = np.asarray(model_info["B_phys"], float)
    C = np.asarray(model_info["C_phys"], float)
    Cd = np.asarray(model_info["Cd_phys"], float)
    P_x = np.asarray(model_info["P_x"], float)
    V_k = float(candidate_eval["V_k"])
    V_bound = float(lyapunov_bound(V_k, rho=rho, eps_lyap=eps_lyap))
    base_debug["qcqp_tracking_target"] = None if y_track is None else y_track.copy()
    if base_debug["qcqp_tracking_target_source"] is None:
        base_debug["qcqp_tracking_target_source"] = "selector_target"

    u_var = cp.Variable(n_u)
    slack_v = cp.Variable(nonneg=True) if allow_lyap_slack else None
    slack_u = cp.Variable(nonneg=True) if trust_region_delta is not None else None

    e_next_expr = A @ e_x + B @ (u_var - u_s)
    y_next_expr = C @ (np.asarray(target_info["x_s"], float).reshape(-1) + e_next_expr) + Cd @ d_s
    V_next_expr = cp.quad_form(e_next_expr, P_x)

    objective = cp.quad_form(u_var - u_cand, W_cand)
    if u_prev is not None:
        objective += cp.quad_form(u_var - u_prev, W_move)
    objective += cp.quad_form(u_var - u_s, W_steady)
    if use_output_tracking and y_track is not None:
        objective += cp.quad_form(y_next_expr - y_track, W_output)
    if slack_v is not None:
        objective += lyap_slack_weight * cp.square(slack_v)
    if slack_u is not None:
        objective += trust_region_weight * cp.square(slack_u)

    constraints = []
    if bounds_info["u_min"] is not None:
        constraints.append(u_var >= bounds_info["u_min"])
    if bounds_info["u_max"] is not None:
        constraints.append(u_var <= bounds_info["u_max"])
    if u_prev is not None and bounds_info["du_min"] is not None:
        constraints.append(u_var - u_prev >= bounds_info["du_min"])
    if u_prev is not None and bounds_info["du_max"] is not None:
        constraints.append(u_var - u_prev <= bounds_info["du_max"])

    lyap_rhs = V_bound if slack_v is None else V_bound + slack_v
    constraints.append(V_next_expr <= lyap_rhs)

    trust_region_violation = None
    if trust_region_delta is not None:
        trust_region_delta = np.maximum(_maybe_vector(trust_region_delta, n_u), 0.0)
        trust_rhs = trust_region_delta if slack_u is None else trust_region_delta + slack_u
        constraints.append(u_var - u_cand <= trust_rhs)
        constraints.append(u_cand - u_var <= trust_rhs)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    solver_sequence = safety_filter_solver_sequence(
        quadratic_constraint_active=True,
        solver_pref=lyap_config.get("solver_pref"),
    )

    best_action = None
    best_debug = None
    last_error = None

    for solver_name in solver_sequence:
        try:
            u_var.value = None
            if slack_v is not None:
                slack_v.value = None
            if slack_u is not None:
                slack_u.value = None
            problem.solve(solver=solver_name, warm_start=True, verbose=False)
        except Exception as exc:
            last_error = repr(exc)
            continue

        if u_var.value is None or problem.status not in _OPTIMAL_STATUSES:
            continue

        u_try = np.asarray(u_var.value, float).reshape(-1)
        post = _postcheck_action(
            u_try=u_try,
            xhat_aug=xhat_aug,
            target_info=target_info,
            model_info=model_info,
            lyap_config=lyap_config,
            bounds_info=bounds_info,
            u_prev=u_prev,
        )
        trust_region_violation = 0.0
        if trust_region_delta is not None:
            trust_region_violation = float(
                np.max(np.maximum(np.abs(u_try - u_cand) - trust_region_delta, 0.0))
            )

        trial_debug = {
            "solver_status": problem.status,
            "solver_name": solver_name,
            "solver_residuals": {
                "lyap_margin_post": float(post.get("lyap_margin", np.nan)),
                "input_bounds_violation": float(post.get("candidate_bounds_violation", np.nan)),
                "move_bounds_violation": float(post.get("candidate_move_violation", np.nan)),
            },
            "trust_region_violation": trust_region_violation,
            "slack_v": 0.0 if slack_v is None or slack_v.value is None else float(np.asarray(slack_v.value).item()),
            "slack_u": 0.0 if slack_u is None or slack_u.value is None else float(np.asarray(slack_u.value).item()),
            "objective_value": None if problem.value is None else float(problem.value),
            "V_next_post": post.get("V_next_cand"),
            "y_next_post": None if post.get("y_next_pred") is None else np.asarray(post["y_next_pred"], float).reshape(-1).copy(),
        }

        best_action = u_try
        best_debug = trial_debug
        if post.get("accepted", False):
            base_debug.update({
                "accepted": True,
                "accept_reason": "optimized_correction",
                "reject_reason": candidate_eval.get("reject_reason"),
                "u_safe": u_try.copy(),
                "correction_mode": "optimized_correction",
                "solver_status": trial_debug["solver_status"],
                "solver_name": trial_debug["solver_name"],
                "solver_residuals": trial_debug["solver_residuals"],
                "trust_region_violation": trial_debug["trust_region_violation"],
                "slack_v": trial_debug["slack_v"],
                "slack_u": trial_debug["slack_u"],
                "verified": True,
            })
            _attach_final_lyap(
                base_debug=base_debug,
                u_try=u_try,
                xhat_aug=xhat_aug,
                target_info=target_info,
                model_info=model_info,
                lyap_config=lyap_config,
                bounds_info=bounds_info,
                u_prev=u_prev,
            )
            return u_try.copy(), base_debug

    if best_debug is not None:
        base_debug.update({
            "solver_status": best_debug["solver_status"],
            "solver_name": best_debug["solver_name"],
            "solver_residuals": best_debug["solver_residuals"],
            "trust_region_violation": best_debug["trust_region_violation"],
            "slack_v": best_debug["slack_v"],
            "slack_u": best_debug["slack_u"],
        })

    u_safe, base_debug = _attempt_mpc_fallback(
        base_debug=base_debug,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
        fallback_config=fallback_config,
    )
    if u_safe is not None:
        return u_safe, base_debug

    u_safe, base_debug = _attempt_secondary_fallbacks(
        base_debug=base_debug,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )
    if u_safe is not None:
        return u_safe, base_debug

    if best_action is None:
        best_action = np.asarray(target_info["u_s"], float).reshape(-1).copy()
    if bounds_info["u_min"] is not None and bounds_info["u_max"] is not None:
        best_action = np.clip(best_action, bounds_info["u_min"], bounds_info["u_max"])

    base_debug.update({
        "accepted": False,
        "accept_reason": None,
        "reject_reason": candidate_eval.get("reject_reason"),
        "u_safe": best_action.copy(),
        "correction_mode": "unverified_fallback",
        "solver_residuals": {
            **base_debug["solver_residuals"],
            "solver_error": last_error,
        },
        "verified": False,
    })
    _attach_final_lyap(
        base_debug=base_debug,
        u_try=best_action,
        xhat_aug=xhat_aug,
        target_info=target_info,
        model_info=model_info,
        lyap_config=lyap_config,
        bounds_info=bounds_info,
        u_prev=u_prev,
    )
    return best_action.copy(), base_debug
# ==================== END FILE: Lyapunov/safety_filter.py ======================

# ==================== BEGIN FILE: Lyapunov/upstream_controllers.py ====================
from types import SimpleNamespace

import numpy as np
import scipy.optimize as spo


def build_repeated_input_bounds(u_min, u_max, horizon_control):
    u_min = np.asarray(u_min, float).reshape(-1)
    u_max = np.asarray(u_max, float).reshape(-1)
    if u_min.size != u_max.size:
        raise ValueError("u_min and u_max must have the same size.")
    if np.any(u_min > u_max):
        raise ValueError("u_min must be <= u_max elementwise.")

    bounds = []
    for _ in range(int(horizon_control)):
        for idx in range(u_min.size):
            bounds.append((float(u_min[idx]), float(u_max[idx])))
    return tuple(bounds)


def default_mpc_initial_guess(n_inputs, horizon_control, fill_value=0.0):
    return np.full(int(n_inputs) * int(horizon_control), float(fill_value), dtype=float)


def solve_offset_free_mpc_candidate(
    MPC_obj,
    y_sp,
    u_prev_dev,
    x0_model,
    IC_opt=None,
    bnds=None,
    cons=None,
    return_debug=False,
):
    n_u = int(MPC_obj.B.shape[1])
    horizon_control = int(getattr(MPC_obj, "NC", 1))

    y_sp = np.asarray(y_sp, float).reshape(-1)
    u_prev_dev = np.asarray(u_prev_dev, float).reshape(-1)
    x0_model = np.asarray(x0_model, float).reshape(-1)

    if IC_opt is None:
        ic_used = default_mpc_initial_guess(n_u, horizon_control)
    else:
        ic_used = np.asarray(IC_opt, float).reshape(-1)

    if ic_used.size != n_u * horizon_control:
        raise ValueError(
            f"IC_opt has size {ic_used.size}, expected {n_u * horizon_control}."
        )

    constraints = () if cons is None else tuple(cons)

    sol = spo.minimize(
        lambda x: MPC_obj.mpc_opt_fun(x, y_sp, u_prev_dev, x0_model),
        ic_used,
        bounds=bnds,
        constraints=constraints,
    )

    x_opt = None if getattr(sol, "x", None) is None else np.asarray(sol.x, float).reshape(-1)
    if x_opt is None or x_opt.size < n_u:
        u_candidate = None
        ic_next = ic_used.copy()
    else:
        u_candidate = x_opt[:n_u].copy()
        ic_next = x_opt.copy()

    info = {
        "success": bool(getattr(sol, "success", False)),
        "status": getattr(sol, "status", None),
        "message": None if getattr(sol, "message", None) is None else str(sol.message),
        "objective_value": None if getattr(sol, "fun", None) is None else float(sol.fun),
        "nit": getattr(sol, "nit", None),
        "candidate_available": bool(u_candidate is not None),
        "u_candidate": None if u_candidate is None else u_candidate.copy(),
        "x_opt": None if x_opt is None else x_opt.copy(),
        "IC_opt_used": ic_used.copy(),
        "IC_opt_next": ic_next.copy(),
        "bounds_used": bnds,
        "num_constraints": len(constraints),
    }

    if return_debug:
        return u_candidate, info
    return u_candidate
# ==================== END FILE: Lyapunov/upstream_controllers.py ======================

# ==================== BEGIN FILE: Lyapunov/safety_debug.py ====================
import csv
import json
import os
import pickle
from datetime import datetime

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from utils.scaling_helpers import apply_min_max, reverse_min_max


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _array_or_none(info, key):
    value = info.get(key)
    if value is None:
        return None
    return np.asarray(value, float).reshape(-1)


def _stack_vectors(lyap_info_storage, key, width, fill_value=np.nan):
    out = np.full((len(lyap_info_storage), width), float(fill_value), dtype=float)
    for idx, info in enumerate(lyap_info_storage):
        arr = _array_or_none(info, key)
        if arr is None:
            continue
        use = min(width, arr.size)
        out[idx, :use] = arr[:use]
    return out


def _selector_debug(info):
    target_info = info.get("target_info", {})
    if not isinstance(target_info, dict):
        return {}
    selector = target_info.get("selector_debug", {})
    return selector if isinstance(selector, dict) else {}


def _target_info(info):
    target_info = info.get("target_info", {})
    return target_info if isinstance(target_info, dict) else {}


def make_safety_filter_step_records(lyap_info_storage):
    records = []
    for step_idx, info in enumerate(lyap_info_storage):
        target_info = _target_info(info)
        selector = _selector_debug(info)
        upstream_info = info.get("upstream_candidate_info", {})
        target_success = bool(info.get("target_success", False))
        row = {
            "step": int(step_idx),
            "step_idx": int(step_idx),
            "source": info.get("source"),
            "accepted": bool(info.get("accepted", False)),
            "verified": bool(info.get("verified", False)),
            "accept_reason": info.get("accept_reason"),
            "reject_reason": info.get("reject_reason"),
            "correction_mode": info.get("correction_mode"),
            "projection_active": bool(str(info.get("correction_mode", "")) == "optimized_correction"),
            "candidate_bounds_ok": info.get("candidate_bounds_ok"),
            "candidate_move_ok": info.get("candidate_move_ok"),
            "candidate_lyap_ok": info.get("candidate_lyap_ok"),
            "fallback_mode": info.get("fallback_mode"),
            "fallback_verified": info.get("fallback_verified"),
            "fallback_solver_status": info.get("fallback_solver_status"),
            "fallback_objective_value": info.get("fallback_objective_value"),
            "fallback_bounds_ok": info.get("fallback_bounds_ok"),
            "fallback_move_ok": info.get("fallback_move_ok"),
            "fallback_lyap_ok": info.get("fallback_lyap_ok"),
            "fallback_tracking_target_source": info.get("fallback_tracking_target_source"),
            "fallback_target_mismatch_inf": info.get("fallback_target_mismatch_inf"),
            "solver_status": info.get("solver_status"),
            "solver_name": info.get("solver_name"),
            "slack_v": info.get("slack_v"),
            "slack_u": info.get("slack_u"),
            "trust_region_violation": info.get("trust_region_violation"),
            "V_k": info.get("V_k"),
            "V_next_cand": info.get("V_next_cand"),
            "V_bound": info.get("V_bound"),
            "final_lyap_value": info.get("final_lyap_value"),
            "final_lyap_bound": info.get("final_lyap_bound"),
            "final_lyap_margin": info.get("final_lyap_margin"),
            "final_lyap_ok": info.get("final_lyap_ok"),
            "final_lyap_target_source": info.get("final_lyap_target_source"),
            "rho": info.get("rho"),
            "eps_lyap": info.get("eps_lyap"),
            "target_success": target_success,
            "target_failure": (not target_success),
            "target_stage": info.get("target_stage"),
            "target_source": info.get("target_source"),
            "mpc_tracking_target_source": info.get("mpc_tracking_target_source"),
            "qcqp_tracking_target_source": info.get("qcqp_tracking_target_source"),
            "target_mismatch_inf": info.get("target_mismatch_inf"),
            "target_error_inf": target_info.get("target_error_inf"),
            "target_slack_inf": target_info.get("target_slack_inf"),
            "selector_status": selector.get("status"),
            "selector_solver": selector.get("solver"),
            "selector_stage": target_info.get("solve_stage"),
            "setpoint_changed": info.get("setpoint_changed"),
            "u_cand": json.dumps(_jsonable(info.get("u_cand"))),
            "u_safe": json.dumps(_jsonable(info.get("u_safe"))),
            "u_prev": json.dumps(_jsonable(info.get("u_prev"))),
            "u_s": json.dumps(_jsonable(info.get("u_s"))),
            "u_fallback_mpc": json.dumps(_jsonable(info.get("u_fallback_mpc"))),
            "mpc_tracking_target": json.dumps(_jsonable(info.get("mpc_tracking_target"))),
            "qcqp_tracking_target": json.dumps(_jsonable(info.get("qcqp_tracking_target"))),
            "x_s": json.dumps(_jsonable(info.get("x_s"))),
            "d_s": json.dumps(_jsonable(info.get("d_s"))),
            "solver_residuals": json.dumps(_jsonable(info.get("solver_residuals", {}))),
            "upstream_candidate_info": json.dumps(_jsonable(upstream_info)),
        }
        records.append(row)
    return records


def make_safety_filter_df(lyap_info_storage):
    records = make_safety_filter_step_records(lyap_info_storage)
    if not HAS_PANDAS:
        raise ImportError("pandas is required to build a DataFrame.")
    return pd.DataFrame(records)


def make_lyap_df(lyap_info_storage, slack_thr=1e-9, du_thr=1e-10):
    df = make_safety_filter_df(lyap_info_storage)
    if "slack_v" in df.columns:
        df["slack_v_active"] = df["slack_v"].fillna(0.0).astype(float) > float(slack_thr)
    if "u_cand" in df.columns and "u_safe" in df.columns:
        df["du_filter_active"] = df.apply(
            lambda row: (
                np.max(
                    np.abs(
                        np.asarray(json.loads(row["u_safe"]), float)
                        - np.asarray(json.loads(row["u_cand"]), float)
                    )
                )
                > float(du_thr)
            ),
            axis=1,
        )
    return df


def summarize_safety_filter_bundle(bundle):
    lyap_info_storage = bundle["lyap_info_storage"]
    modes = [str(info.get("correction_mode", "none")) for info in lyap_info_storage]
    solver_statuses = [str(info.get("solver_status")) for info in lyap_info_storage if info.get("solver_status") is not None]
    fallback_statuses = [
        str(info.get("fallback_solver_status"))
        for info in lyap_info_storage
        if info.get("fallback_solver_status") is not None
    ]
    summary = {
        "source": bundle.get("source"),
        "n_steps": int(bundle.get("nFE", len(lyap_info_storage))),
        "n_verified": int(sum(bool(info.get("verified", False)) for info in lyap_info_storage)),
        "n_target_success": int(sum(bool(info.get("target_success", False)) for info in lyap_info_storage)),
        "n_accepted_candidate": int(sum(mode == "accepted_candidate" for mode in modes)),
        "n_optimized_correction": int(sum(mode == "optimized_correction" for mode in modes)),
        "n_fallback_mpc_verified": int(sum(mode == "fallback_mpc_verified" for mode in modes)),
        "n_fallback_mpc_unverified": int(sum(mode == "fallback_mpc_unverified" for mode in modes)),
        "n_secondary_fallbacks": int(sum(mode.endswith("_secondary") for mode in modes)),
        "n_target_failures": int(sum(info.get("reject_reason") == "target_unavailable" for info in lyap_info_storage)),
        "reward_mean": float(np.mean(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "reward_min": float(np.min(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "reward_max": float(np.max(bundle["rewards"])) if len(bundle["rewards"]) > 0 else None,
        "target_error_inf_max": float(np.nanmax(bundle["target_error_inf"])) if bundle["target_error_inf"].size > 0 else None,
        "target_slack_inf_max": float(np.nanmax(bundle["target_slack_inf"])) if bundle["target_slack_inf"].size > 0 else None,
        "lyapunov_margin_min": float(np.nanmin(bundle["lyapunov_margin"])) if bundle["lyapunov_margin"].size > 0 else None,
        "target_mismatch_inf_max": float(np.nanmax(bundle["target_mismatch_inf"])) if bundle["target_mismatch_inf"].size > 0 else None,
    }
    summary["mode_counts"] = {mode: int(modes.count(mode)) for mode in sorted(set(modes))}
    summary["solver_status_counts"] = {
        status: int(solver_statuses.count(status)) for status in sorted(set(solver_statuses))
    }
    summary["fallback_solver_status_counts"] = {
        status: int(fallback_statuses.count(status)) for status in sorted(set(fallback_statuses))
    }
    return summary


def build_safety_filter_run_bundle(
    source,
    results,
    steady_states=None,
    config=None,
    min_max_dict=None,
    data_min=None,
    data_max=None,
    extra=None,
):
    (
        y_system,
        u_applied_phys,
        avg_rewards,
        rewards,
        xhatdhat,
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
    ) = results

    y_system = np.asarray(y_system, float)
    u_applied_phys = np.asarray(u_applied_phys, float)
    rewards = np.asarray(rewards, float)
    xhatdhat = np.asarray(xhatdhat, float)
    y_sp = np.asarray(y_sp, float)
    yhat = np.asarray(yhat, float)
    e_store = np.asarray(e_store, float)
    qi = np.asarray(qi, float)
    qs = np.asarray(qs, float)
    ha = np.asarray(ha, float)
    u_safe_dev_store = np.asarray(u_safe_dev_store, float)

    n_u = u_safe_dev_store.shape[1]
    n_y = y_system.shape[1]

    bundle = {
        "source": str(source),
        "config": {} if config is None else config,
        "steady_states": steady_states,
        "min_max_dict": min_max_dict,
        "data_min": None if data_min is None else np.asarray(data_min, float),
        "data_max": None if data_max is None else np.asarray(data_max, float),
        "nFE": int(nFE),
        "time_in_sub_episodes": int(time_in_sub_episodes),
        "avg_rewards": list(avg_rewards),
        "rewards": rewards.copy(),
        "y_system": y_system.copy(),
        "u_applied_phys": u_applied_phys.copy(),
        "xhatdhat": xhatdhat.copy(),
        "y_sp": y_sp.copy(),
        "yhat": yhat.copy(),
        "e_store": e_store.copy(),
        "qi": qi.copy(),
        "qs": qs.copy(),
        "ha": ha.copy(),
        "lyap_info_storage": lyap_info_storage,
        "u_safe_dev_store": u_safe_dev_store.copy(),
        "u_cand_dev_store": _stack_vectors(lyap_info_storage, "u_cand", n_u),
        "u_prev_dev_store": _stack_vectors(lyap_info_storage, "u_prev", n_u),
        "u_target_dev_store": _stack_vectors(lyap_info_storage, "u_s", n_u),
        "u_fallback_mpc_dev_store": _stack_vectors(lyap_info_storage, "u_fallback_mpc", n_u),
        "x_target_store": _stack_vectors(lyap_info_storage, "x_s", xhatdhat.shape[0] - n_y),
        "V_k": np.array([info.get("V_k", np.nan) for info in lyap_info_storage], dtype=float),
        "V_next_cand": np.array([info.get("V_next_cand", np.nan) for info in lyap_info_storage], dtype=float),
        "V_bound": np.array([info.get("V_bound", np.nan) for info in lyap_info_storage], dtype=float),
        "final_lyap_value": np.array([info.get("final_lyap_value", np.nan) for info in lyap_info_storage], dtype=float),
        "final_lyap_bound": np.array([info.get("final_lyap_bound", np.nan) for info in lyap_info_storage], dtype=float),
        "final_lyap_margin": np.array([info.get("final_lyap_margin", np.nan) for info in lyap_info_storage], dtype=float),
        "final_lyap_target_source": [info.get("final_lyap_target_source") for info in lyap_info_storage],
        "lyapunov_margin": np.array(
            [
                np.nan
                if info.get("V_next_cand") is None or info.get("V_bound") is None
                else float(info.get("V_bound")) - float(info.get("V_next_cand"))
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "target_error_inf": np.array(
            [info.get("target_info", {}).get("target_error_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_slack_inf": np.array(
            [info.get("target_info", {}).get("target_slack_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_mismatch_inf": np.array(
            [info.get("target_mismatch_inf", np.nan) for info in lyap_info_storage],
            dtype=float,
        ),
        "target_success_flags": np.array(
            [1.0 if bool(info.get("target_success", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "target_failure_flags": np.array(
            [0.0 if bool(info.get("target_success", False)) else 1.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "target_stage_code": np.array(
            [
                2.0 if str(info.get("target_stage", "")) == "exact"
                else 1.0 if str(info.get("target_stage", "")) == "fallback"
                else 0.0
                for info in lyap_info_storage
            ],
            dtype=float,
        ),
        "verified_flags": np.array(
            [1.0 if bool(info.get("verified", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "accepted_flags": np.array(
            [1.0 if bool(info.get("accepted", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "fallback_verified_flags": np.array(
            [1.0 if bool(info.get("fallback_verified", False)) else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "projection_active_flags": np.array(
            [1.0 if str(info.get("correction_mode", "")) == "optimized_correction" else 0.0 for info in lyap_info_storage],
            dtype=float,
        ),
        "correction_modes": [str(info.get("correction_mode", "none")) for info in lyap_info_storage],
        "accept_reasons": [info.get("accept_reason") for info in lyap_info_storage],
        "reject_reasons": [info.get("reject_reason") for info in lyap_info_storage],
        "solver_statuses": [info.get("solver_status") for info in lyap_info_storage],
        "fallback_solver_statuses": [info.get("fallback_solver_status") for info in lyap_info_storage],
        "extra": {} if extra is None else extra,
    }
    bundle["summary"] = summarize_safety_filter_bundle(bundle)
    return bundle


def _write_csv(path, records):
    if not records:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([])
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _save_npz(path, bundle):
    np.savez_compressed(
        path,
        y_system=bundle["y_system"],
        u_applied_phys=bundle["u_applied_phys"],
        xhatdhat=bundle["xhatdhat"],
        y_sp=bundle["y_sp"],
        yhat=bundle["yhat"],
        e_store=bundle["e_store"],
        qi=bundle["qi"],
        qs=bundle["qs"],
        ha=bundle["ha"],
        u_safe_dev_store=bundle["u_safe_dev_store"],
        u_cand_dev_store=bundle["u_cand_dev_store"],
        u_prev_dev_store=bundle["u_prev_dev_store"],
        u_target_dev_store=bundle["u_target_dev_store"],
        u_fallback_mpc_dev_store=bundle["u_fallback_mpc_dev_store"],
        V_k=bundle["V_k"],
        V_next_cand=bundle["V_next_cand"],
        V_bound=bundle["V_bound"],
        final_lyap_value=bundle["final_lyap_value"],
        final_lyap_bound=bundle["final_lyap_bound"],
        final_lyap_margin=bundle["final_lyap_margin"],
        lyapunov_margin=bundle["lyapunov_margin"],
        target_error_inf=bundle["target_error_inf"],
        target_slack_inf=bundle["target_slack_inf"],
        target_mismatch_inf=bundle["target_mismatch_inf"],
        target_success_flags=bundle["target_success_flags"],
        target_failure_flags=bundle["target_failure_flags"],
        target_stage_code=bundle["target_stage_code"],
        verified_flags=bundle["verified_flags"],
        accepted_flags=bundle["accepted_flags"],
        fallback_verified_flags=bundle["fallback_verified_flags"],
        projection_active_flags=bundle["projection_active_flags"],
    )


def plot_safety_filter_bundle(bundle, output_dir):
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting debug artifacts.")

    os.makedirs(output_dir, exist_ok=True)

    y_system = bundle["y_system"]
    u_applied_phys = bundle["u_applied_phys"]
    y_sp = np.asarray(bundle["y_sp"], float)
    V_k = bundle["V_k"]
    V_next_cand = bundle["V_next_cand"]
    V_bound = bundle["V_bound"]
    final_lyap_value = bundle["final_lyap_value"]
    final_lyap_bound = bundle["final_lyap_bound"]
    final_lyap_margin = bundle["final_lyap_margin"]
    lyapunov_margin = bundle["lyapunov_margin"]
    target_error_inf = bundle["target_error_inf"]
    target_slack_inf = bundle["target_slack_inf"]
    target_mismatch_inf = bundle["target_mismatch_inf"]
    target_success_flags = np.asarray(bundle["target_success_flags"], float)
    target_failure_flags = np.asarray(bundle["target_failure_flags"], float)
    target_stage_code = np.asarray(bundle["target_stage_code"], float)
    u_cand_dev = bundle["u_cand_dev_store"]
    u_safe_dev = bundle["u_safe_dev_store"]
    rewards = bundle["rewards"]
    projection_active = np.asarray(bundle["projection_active_flags"], float)

    if bundle.get("steady_states") is not None and bundle.get("data_min") is not None and bundle.get("data_max") is not None:
        steady_states = bundle["steady_states"]
        data_min = bundle["data_min"]
        data_max = bundle["data_max"]
        y_ss_scaled = apply_min_max(steady_states["y_ss"], data_min[u_applied_phys.shape[1]:], data_max[u_applied_phys.shape[1]:])
        y_sp_plot = reverse_min_max(y_sp + y_ss_scaled, data_min[u_applied_phys.shape[1]:], data_max[u_applied_phys.shape[1]:])
    else:
        y_sp_plot = y_sp

    time_y = np.arange(y_system.shape[0])
    time_u = np.arange(u_applied_phys.shape[0])

    plt.figure(figsize=(10, 6))
    for idx in range(y_system.shape[1]):
        plt.subplot(y_system.shape[1], 1, idx + 1)
        plt.plot(time_y, y_system[:, idx], label="output", linewidth=2)
        plt.step(time_u, y_sp_plot[:, idx] if y_sp_plot.ndim == 2 else y_sp_plot, where="post", linestyle="--", label="setpoint", linewidth=2)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outputs_vs_setpoint.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(y_system.shape[1] + 1, 1, figsize=(10, 3.0 * (y_system.shape[1] + 1)), sharex=False)
    axes = np.atleast_1d(axes)
    for idx in range(y_system.shape[1]):
        ax = axes[idx]
        ax.plot(time_y, y_system[:, idx], label="output", linewidth=2)
        ax.step(
            time_u,
            y_sp_plot[:, idx] if y_sp_plot.ndim == 2 else y_sp_plot,
            where="post",
            linestyle="--",
            label="setpoint",
            linewidth=2,
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    ax = axes[-1]
    ax.step(time_u, projection_active, where="post", linewidth=2, label="projection_active")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outputs_vs_setpoint_projection.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    for idx in range(u_applied_phys.shape[1]):
        plt.step(time_u, u_applied_phys[:, idx], where="post", linewidth=2, label=f"u{idx}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "applied_inputs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    for idx in range(u_safe_dev.shape[1]):
        plt.plot(time_u, u_cand_dev[:, idx], linestyle="--", linewidth=1.5, label=f"cand_{idx}")
        plt.plot(time_u, u_safe_dev[:, idx], linewidth=2.0, label=f"safe_{idx}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "candidate_vs_safe_dev.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, final_lyap_value, linewidth=2, label="final_lyap_value")
    plt.plot(time_u, final_lyap_bound, linewidth=2, linestyle="--", label="final_lyap_bound")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lyapunov_values.png"), dpi=300, bbox_inches="tight")
    plt.close()

    last_len = int(bundle.get("time_in_sub_episodes", len(time_u)))
    if last_len <= 0:
        last_len = len(time_u)
    start_idx = max(0, len(time_u) - last_len)
    last_steps = time_u[start_idx:]
    V_final_last = final_lyap_value[start_idx:]
    delta_V_last = np.full_like(V_final_last, np.nan)
    if V_final_last.size >= 2:
        delta_V_last[1:] = V_final_last[1:] - V_final_last[:-1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(last_steps, V_final_last, linewidth=2, label="final_lyap_value")
    axes[0].plot(last_steps, final_lyap_bound[start_idx:], linewidth=2, linestyle="--", label="final_lyap_bound")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()
    axes[1].plot(last_steps, delta_V_last, linewidth=2, label="delta final_lyap_value")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lyapunov_last_episode.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, final_lyap_margin, linewidth=2, label="final_lyap_margin")
    plt.plot(time_u, lyapunov_margin, linewidth=1.5, linestyle=":", label="candidate_lyap_margin")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lyapunov_margin.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, target_error_inf, linewidth=2, label="target_error_inf")
    plt.plot(time_u, target_slack_inf, linewidth=2, linestyle="--", label="target_slack_inf")
    if np.any(np.isfinite(target_mismatch_inf)):
        plt.plot(time_u, target_mismatch_inf, linewidth=2, linestyle=":", label="target_mismatch_inf")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_selector_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].step(time_u, target_failure_flags, where="post", linewidth=2, label="target_selector_failed")
    axes[0].step(time_u, target_success_flags, where="post", linewidth=1.5, linestyle="--", label="target_selector_success")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_yticks([0.0, 1.0])
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()
    axes[1].step(time_u, target_stage_code, where="post", linewidth=2, label="target_stage_code")
    axes[1].set_ylim(-0.1, 2.1)
    axes[1].set_yticks([0.0, 1.0, 2.0])
    axes[1].set_yticklabels(["failed", "fallback", "exact"])
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_selector_status.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(10, 5))
    plt.plot(time_u, rewards, linewidth=2, label="reward")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_trace.png"), dpi=300, bbox_inches="tight")
    plt.close()

    mode_counts = bundle["summary"]["mode_counts"]
    if mode_counts:
        labels = list(mode_counts.keys())
        values = [mode_counts[k] for k in labels]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correction_modes.png"), dpi=300, bbox_inches="tight")
        plt.close()

    solver_counts = bundle["summary"].get("solver_status_counts", {})
    if solver_counts:
        labels = list(solver_counts.keys())
        values = [solver_counts[k] for k in labels]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "solver_status_counts.png"), dpi=300, bbox_inches="tight")
        plt.close()

    fallback_counts = bundle["summary"].get("fallback_solver_status_counts", {})
    if fallback_counts:
        labels = list(fallback_counts.keys())
        values = [fallback_counts[k] for k in labels]
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fallback_solver_status_counts.png"), dpi=300, bbox_inches="tight")
        plt.close()


def save_safety_filter_debug_artifacts(
    bundle,
    directory=None,
    prefix_name="safety_filter_debug",
    save_plots=True,
):
    if directory is None:
        directory = os.getcwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(bundle["summary"]), f, indent=2)

    summary_csv_records = [
        {"key": key, "value": json.dumps(_jsonable(value))}
        for key, value in bundle["summary"].items()
    ]
    _write_csv(os.path.join(out_dir, "summary.csv"), summary_csv_records)

    step_records = make_safety_filter_step_records(bundle["lyap_info_storage"])
    _write_csv(os.path.join(out_dir, "step_table.csv"), step_records)

    _save_npz(os.path.join(out_dir, "arrays.npz"), bundle)

    if HAS_PANDAS:
        df = pd.DataFrame(step_records)
        df.to_pickle(os.path.join(out_dir, "step_table.pkl"))

    if save_plots:
        plot_safety_filter_bundle(bundle, out_dir)

    return out_dir


def save_lyap_debug_artifacts(bundle, directory=None, prefix_name="safety_filter_debug", save_plots=True):
    return save_safety_filter_debug_artifacts(
        bundle=bundle,
        directory=directory,
        prefix_name=prefix_name,
        save_plots=save_plots,
    )


def load_safety_filter_debug_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_lyap_debug_artifacts(path):
    return load_safety_filter_debug_bundle(path)
# ==================== END FILE: Lyapunov/safety_debug.py ======================

# ==================== BEGIN FILE: utils/lyapunov_utils.py ====================
import numpy as np


DEFAULT_CVXPY_SOLVERS = ("OSQP", "CLARABEL", "SCS")
DEFAULT_TRACKING_QP_SOLVERS = ("CLARABEL", "SCS", "OSQP")
DEFAULT_TRACKING_CONIC_SOLVERS = ("CLARABEL", "SCS")
DEFAULT_SAFETY_FILTER_QP_SOLVERS = ("OSQP", "CLARABEL", "SCS")
DEFAULT_SAFETY_FILTER_QCQP_SOLVERS = ("CLARABEL", "SCS")


def diag_psd_from_vector(diag_vals, size, eps=1e-12, default=1.0):
    if diag_vals is None:
        vals = default * np.ones(size, dtype=float)
    else:
        vals = np.asarray(diag_vals, float).reshape(-1)
        if vals.size != size:
            raise ValueError(f"Diagonal size mismatch. Expected {size}, got {vals.size}.")
        vals = np.maximum(vals, eps)
    return np.diag(vals), vals


def vector_or_zeros(values, size):
    if values is None:
        return np.zeros(size, dtype=float)
    out = np.asarray(values, float).reshape(-1)
    if out.size != size:
        raise ValueError(f"Size mismatch. Expected {size}, got {out.size}.")
    return out


def tracking_solver_sequence(terminal_constraint_active, solver_pref=None):
    if solver_pref is None:
        seq = (
            DEFAULT_TRACKING_CONIC_SOLVERS
            if terminal_constraint_active
            else DEFAULT_TRACKING_QP_SOLVERS
        )
    elif isinstance(solver_pref, str):
        seq = (solver_pref,)
    else:
        seq = tuple(solver_pref)

    seq = tuple(str(name).upper() for name in seq if name is not None)
    if terminal_constraint_active:
        seq = tuple(name for name in seq if name != "OSQP")
        if not seq:
            seq = DEFAULT_TRACKING_CONIC_SOLVERS
    return seq


def safety_filter_solver_sequence(quadratic_constraint_active=True, solver_pref=None):
    if solver_pref is None:
        seq = (
            DEFAULT_SAFETY_FILTER_QCQP_SOLVERS
            if quadratic_constraint_active
            else DEFAULT_SAFETY_FILTER_QP_SOLVERS
        )
    elif isinstance(solver_pref, str):
        seq = (solver_pref,)
    else:
        seq = tuple(solver_pref)

    seq = tuple(str(name).upper() for name in seq if name is not None)
    if quadratic_constraint_active:
        seq = tuple(name for name in seq if name != "OSQP")
        if not seq:
            seq = DEFAULT_SAFETY_FILTER_QCQP_SOLVERS
    elif not seq:
        seq = DEFAULT_SAFETY_FILTER_QP_SOLVERS
    return seq


def get_y_sp_step(y_sp, step_idx, n_outputs):
    y_sp = np.asarray(y_sp, float)
    if y_sp.ndim == 1:
        if y_sp.size != n_outputs:
            raise ValueError(f"y_sp vector has size {y_sp.size}, expected {n_outputs}.")
        return y_sp.reshape(-1)

    if y_sp.ndim != 2:
        raise ValueError(f"y_sp must be 1D or 2D, got shape {y_sp.shape}.")

    if y_sp.shape[0] == n_outputs:
        return y_sp[:, step_idx].reshape(-1)

    if y_sp.shape[1] == n_outputs:
        return y_sp[step_idx, :].reshape(-1)

    raise ValueError(f"Cannot infer y_sp orientation from shape {y_sp.shape} with n_outputs = {n_outputs}.")


def reshape_u_sequence(x_opt, n_inputs, horizon_control):
    return np.asarray(x_opt[:n_inputs * horizon_control], float).reshape(horizon_control, n_inputs)


def compute_du_sequence(u_sequence, u_prev_dev):
    u_prev_dev = np.asarray(u_prev_dev, float).reshape(1, -1)
    u_prev_path = np.vstack([u_prev_dev, u_sequence[:-1, :]])
    return u_sequence - u_prev_path


def shift_input_guess(x_opt, n_inputs, horizon_control):
    u_sequence = reshape_u_sequence(x_opt, n_inputs, horizon_control)
    shifted = np.vstack([u_sequence[1:, :], u_sequence[-1:, :]])
    return shifted.reshape(-1)
# ==================== END FILE: utils/lyapunov_utils.py ======================

# ==================== BEGIN FILE: utils/scaling_helpers.py ====================
import numpy as np


def apply_min_max(data, min_val, max_val, eps=1e-12):
    """Min-max scale data to [0, 1].

    If max_val == min_val (or extremely close), scaling is undefined.
    In that case, this returns 0 for that element to avoid NaNs/Infs.
    """
    x = np.asarray(data, dtype=float)
    mn = np.asarray(min_val, dtype=float)
    mx = np.asarray(max_val, dtype=float)

    diff = mx - mn
    mask = np.abs(diff) < eps
    denom = np.where(mask, 1.0, diff)
    out = (x - mn) / denom
    return np.where(mask, 0.0, out)


def reverse_min_max(scaled_data, min_val, max_val):
    """Inverse of apply_min_max (maps [0, 1] back to the original scale)."""
    z = np.asarray(scaled_data, dtype=float)
    mn = np.asarray(min_val, dtype=float)
    mx = np.asarray(max_val, dtype=float)
    return z * (mx - mn) + mn

def apply_min_max_pm1(data, min_val, max_val, eps=1e-12):
    """Min-max scale to [-1, 1] using apply_min_max([0, 1]) then affine map."""
    return 2.0 * apply_min_max(data, min_val, max_val, eps=eps) - 1.0

def apply_rl_scaled(min_max_dict, x_d_states, y_sp, u):
    x_min, x_max = min_max_dict["x_min"], min_max_dict["x_max"]
    y_sp_min, y_sp_max = min_max_dict["y_sp_min"], min_max_dict["y_sp_max"]
    u_min, u_max = min_max_dict["u_min"], min_max_dict["u_max"]

    x_d_states_scaled = apply_min_max_pm1(x_d_states, x_min, x_max)
    y_sp_scaled = apply_min_max_pm1(y_sp, y_sp_min, y_sp_max)
    u_scaled = apply_min_max_pm1(u, u_min, u_max)

    return np.hstack((x_d_states_scaled, y_sp_scaled, u_scaled))
# ==================== END FILE: utils/scaling_helpers.py ======================

# ==================== BEGIN FILE: utils/helpers.py ====================
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
                                             qi_change, qs_change, ha_change):
    # For each scenario, create a block of size (set_points_len, n_outputs)
    blocks = [np.full((set_points_len, y_sp_scenario.shape[1]), scenario)
              for scenario in y_sp_scenario]

    # Concatenate the blocks to form one cycle
    cycle = np.concatenate(blocks, axis=0)
    # Repeat the cycle 'repetitions' times
    y_sp = np.concatenate([cycle] * n_tests, axis=0)

    # Test train scenario
    test_cycle = test_cycle * int(n_tests / len(test_cycle))
    # Try making everything trainable but te end cycle should be only for testing
    test_cycle[-1] = True

    time_in_sub_episodes = set_points_len * len(y_sp_scenario)

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

    qi = np.linspace(nominal_qi, nominal_qi * qi_change, nFE)
    qs = np.linspace(nominal_qs, nominal_qs * qs_change, nFE)
    ha = np.linspace(nominal_ha, nominal_ha * ha_change, int(nFE / 2))
    ha = np.hstack((ha, np.tile(nominal_ha * ha_change, int(nFE/ 2))))

    return y_sp, nFE, sub_episodes_changes_dict, time_in_sub_episodes, test_train_dict, warm_start, qi, qs, ha
# ==================== END FILE: utils/helpers.py ======================

# ==================== BEGIN FILE: TD3Agent/reward_functions.py ====================
import numpy as np


def make_reward_fn_relative_QR(
    data_min, data_max, n_inputs,
    k_rel, band_floor_phys,
    Q_diag, R_diag,
    tau_frac=0.7,
    gamma_out=0.5, gamma_in=0.5,
    beta=5.0, gate="geom", lam_in=1.0,
    bonus_kind="exp", bonus_k=12.0, bonus_p=0.6, bonus_c=20.0,
):
    """
    Reward with relative tracking bands.

    data_min, data_max : arrays for [u_min..., y_min...], [u_max..., y_max...]
    n_inputs           : number of inputs (so outputs start at index n_inputs)
    k_rel              : per-output relative tolerance factors (len = n_outputs)
    band_floor_phys    : per-output minimum band in physical units (len = n_outputs)
    Q_diag, R_diag     : quadratic weights (len = n_outputs, len = n_inputs)
    """

    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)

    dy = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)

    k_rel = np.asarray(k_rel, float)
    band_floor_phys = np.asarray(band_floor_phys, float)
    Q_diag = np.asarray(Q_diag, float)
    R_diag = np.asarray(R_diag, float)

    band_floor_scaled = band_floor_phys / np.maximum(dy, 1e-12)

    def _sigmoid(x):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _phi(z):
        z = np.clip(z, 0.0, 1.0)
        if bonus_kind == "linear":
            return 1.0 - z
        if bonus_kind == "quadratic":
            return (1.0 - z) ** 2
        if bonus_kind == "exp":
            return (np.exp(-bonus_k * z) - np.exp(-bonus_k)) / (1.0 - np.exp(-bonus_k))
        if bonus_kind == "power":
            return 1.0 - np.power(z, bonus_p)
        if bonus_kind == "log":
            return np.log1p(bonus_c * (1.0 - z)) / np.log1p(bonus_c)
        raise ValueError("unknown bonus_kind")

    def reward_fn(e_scaled, du_scaled, y_sp_phys=None):
        e_scaled = np.asarray(e_scaled, float)
        du_scaled = np.asarray(du_scaled, float)

        if y_sp_phys is None:
            band_scaled = band_floor_scaled
        else:
            y_sp_phys = np.asarray(y_sp_phys, float)
            band_phys = np.maximum(k_rel * np.abs(y_sp_phys), band_floor_phys)
            band_scaled = band_phys / np.maximum(dy, 1e-12)

        tau_scaled = tau_frac * band_scaled

        abs_e = np.abs(e_scaled)
        s_i = _sigmoid((band_scaled - abs_e) / np.maximum(tau_scaled, 1e-12))

        if gate == "prod":
            w_in = float(np.prod(s_i, dtype=np.float64))
        elif gate == "mean":
            w_in = float(np.mean(s_i))
        elif gate == "geom":
            w_in = float(np.prod(s_i, dtype=np.float64) ** (1.0 / max(1, len(s_i))))
        else:
            raise ValueError("gate must be 'prod'|'mean'|'geom'")

        err_quad = np.sum(Q_diag * (e_scaled ** 2))
        err_eff = (1.0 - w_in) * err_quad + w_in * (lam_in * err_quad)

        move = np.sum(R_diag * (du_scaled ** 2))

        slope_at_edge = 2.0 * Q_diag * band_scaled

        overflow = np.maximum(abs_e - band_scaled, 0.0)
        lin_out = (1.0 - w_in) * np.sum(gamma_out * slope_at_edge * overflow)

        inside_mag = np.minimum(abs_e, band_scaled)
        lin_in = w_in * np.sum(gamma_in * slope_at_edge * inside_mag)

        qb2 = Q_diag * (band_scaled ** 2)
        z = abs_e / np.maximum(band_scaled, 1e-12)
        bonus = w_in * beta * np.sum(qb2 * _phi(z))

        return float(-(err_eff + move + lin_out + lin_in) + bonus)

    params = dict(
        k_rel=k_rel,
        band_floor_phys=band_floor_phys,
        band_floor_scaled=band_floor_scaled,
        Q_diag=Q_diag,
        R_diag=R_diag,
        tau_frac=tau_frac,
        gamma_out=gamma_out,
        gamma_in=gamma_in,
        beta=beta,
        gate=gate,
        lam_in=lam_in,
        bonus_kind=bonus_kind,
        bonus_k=bonus_k,
        bonus_p=bonus_p,
        bonus_c=bonus_c,
    )
    return params, reward_fn


def make_reward_fn_mpc_quadratic(Q_diag, R_diag):
    """
    One-step MPC-style quadratic stage cost in scaled deviation coordinates:
      r = - (sum_i Q_i e_i^2 + sum_j R_j du_j^2)

    Matches the signature of the relative reward:
      reward_fn(e_scaled, du_scaled, y_sp_phys=None)
    """

    Q_diag = np.asarray(Q_diag, float)
    R_diag = np.asarray(R_diag, float)

    def reward_fn(e_scaled, du_scaled, y_sp_phys=None):
        e = np.asarray(e_scaled, float)
        du = np.asarray(du_scaled, float)
        err = np.sum(Q_diag * (e ** 2))
        move = np.sum(R_diag * (du ** 2))
        return float(-(err + move))

    params = dict(Q_diag=Q_diag, R_diag=R_diag)
    return params, reward_fn
# ==================== END FILE: TD3Agent/reward_functions.py ======================

# ==================== BEGIN NOTEBOOK EXCERPT: LyapunovSafetyFilterMPC.ipynb run_config ====================
# "run_config = {\n",
#     "    \"rho_lyap\": 0.98,\n",
#     "    \"lyap_eps\": 1e-9,\n",
#     "    \"lyap_tol\": 1e-10,\n",
#     "    \"w_mpc\": 1.0,\n",
#     "    \"w_track\": 1.0,\n",
#     "    \"w_move\": 0.2,\n",
#     "    \"w_ss\": 0.0,\n",
#     "    \"fallback_policy\": \"offset_free_mpc\",\n",
#     "    \"mpc_target_policy\": \"raw_setpoint\",\n",
#     "    \"reuse_mpc_solution_as_ic\": False,\n",
#     "    \"reset_system_on_entry\": True,\n",
#     "    \"allow_lyap_slack\": True,\n",
#     "    \"trust_region_delta\": 0.15,\n",
#     "}\n",
#     "\n",
#     "# Recreate the plant before each run so repeated executions match the baseline MPC notebook.\n",
#     "cstr = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t, deviation_form=False)\n",
#     "\n",
#     
# ==================== END NOTEBOOK EXCERPT: LyapunovSafetyFilterMPC.ipynb run_config ======================
