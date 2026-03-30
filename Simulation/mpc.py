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


def augment_state_space_rawlings(A, B, C, Bd=None, Cd=None):
    """
    Rawlings-style offset-free augmentation:

        x_{k+1} = A x_k + B u_k + Bd d_k
        d_{k+1} = d_k
        y_k = C x_k + Cd d_k

    If Bd and Cd are both omitted, this defaults to an output-disturbance
    model with disturbance dimension equal to the number of outputs.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    if A.shape != (n, n):
        raise ValueError("A must be square.")
    if B.shape[0] != n:
        raise ValueError("B row dimension must match A.")
    if C.shape[1] != n:
        raise ValueError("C column dimension must match A.")

    if Bd is None and Cd is None:
        nd = p
        Bd = np.zeros((n, nd), dtype=float)
        Cd = np.eye(p, dtype=float)
    elif Bd is None or Cd is None:
        raise ValueError("Provide both Bd and Cd, or neither.")
    else:
        Bd = np.asarray(Bd, float)
        Cd = np.asarray(Cd, float)
        nd = Bd.shape[1]

        if Bd.shape[0] != n:
            raise ValueError("Bd row dimension must match A.")
        if Cd.shape[0] != p:
            raise ValueError("Cd row dimension must match C.")
        if Cd.shape[1] != nd:
            raise ValueError("Cd column dimension must match Bd.")

    A_aug = np.block([
        [A, Bd],
        [np.zeros((nd, n), dtype=float), np.eye(nd, dtype=float)],
    ])
    B_aug = np.vstack([B, np.zeros((nd, m), dtype=float)])
    C_aug = np.hstack([C, Cd])

    return A_aug, B_aug, C_aug, Bd, Cd


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


