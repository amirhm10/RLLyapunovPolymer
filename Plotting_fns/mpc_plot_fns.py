import os
from contextlib import nullcontext
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

plt.style.use('default')

from utils.plot_style import PAPER_COLORS, paper_plot_context
from utils.scaling_helpers import apply_min_max, reverse_min_max


def _normalize_step_matrix(values, n_steps, width, name):
    values = np.asarray(values, float)
    if values.ndim == 1:
        if values.size != width:
            raise ValueError(f"{name} vector has size {values.size}, expected {width}.")
        return np.tile(values.reshape(1, -1), (n_steps, 1))
    if values.shape == (width, n_steps):
        return values.T.copy()
    if values.shape == (n_steps, width):
        return values.copy()
    raise ValueError(f"Cannot normalize {name} with shape {values.shape} into ({n_steps}, {width}).")


def _default_output_labels(n_outputs):
    if int(n_outputs) == 2:
        return [r"$\eta$ (L/g)", r"$T$ (K)"]
    return [f"y[{idx}]" for idx in range(int(n_outputs))]


def _default_input_labels(n_inputs):
    if int(n_inputs) == 2:
        return [r"$Q_c$ (L/h)", r"$Q_m$ (L/h)"]
    return [f"u[{idx}]" for idx in range(int(n_inputs))]


def plot_mpc_results_cstr(
    y_sp, steady_states, nFE, delta_t, time_in_sub_episodes,
    y_mpc, u_mpc, data_min, data_max,
    directory=None, prefix_name="mpc_result",
    y_target=None,
    y_tracking_target=None,
    u_target=None,
    u_bounds=None,
    timestamp_subdir=True,
    paper_style=True,
    output_labels=None,
    input_labels=None,
):
    """
    Paper-ready MPC figures for the polymer CSTR.

    The base interface remains compatible with the existing notebooks, but the
    helper now also supports:
    - steady target output overlays (`y_target`)
    - stage-tracking output overlays (`y_tracking_target`)
    - steady input target overlays (`u_target`)
    - physical input bounds (`u_bounds`)
    - saving directly into an existing directory without an extra timestamp

    Returns: out_dir (str)
    """

    if directory is None:
        directory = os.getcwd()

    if timestamp_subdir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(directory, prefix_name, timestamp)
    else:
        out_dir = os.path.join(directory, prefix_name) if prefix_name else directory
    os.makedirs(out_dir, exist_ok=True)

    def _savefig(name_base):
        plt.tight_layout()
        png = os.path.join(out_dir, f"{name_base}.png")
        plt.savefig(png, bbox_inches="tight", dpi=300)
        plt.close()

    y_mpc = np.asarray(y_mpc, float)
    u_mpc = np.asarray(u_mpc, float)
    n_outputs = int(y_mpc.shape[1])
    n_inputs = int(u_mpc.shape[1])

    y_sp_dev = _normalize_step_matrix(y_sp, int(nFE), n_outputs, "y_sp")
    y_target_dev = None if y_target is None else _normalize_step_matrix(y_target, int(nFE), n_outputs, "y_target")
    y_tracking_dev = None if y_tracking_target is None else _normalize_step_matrix(
        y_tracking_target,
        int(nFE),
        n_outputs,
        "y_tracking_target",
    )
    u_target_plot = None if u_target is None else _normalize_step_matrix(u_target, int(nFE), n_inputs, "u_target")

    if u_bounds is not None:
        if len(u_bounds) != 2:
            raise ValueError("u_bounds must be a (lower, upper) pair.")
        u_lower = np.asarray(u_bounds[0], float).reshape(-1)
        u_upper = np.asarray(u_bounds[1], float).reshape(-1)
        if u_lower.size != n_inputs or u_upper.size != n_inputs:
            raise ValueError("u_bounds must match the number of inputs.")
    else:
        u_lower = None
        u_upper = None

    y_ss = apply_min_max(steady_states["y_ss"], data_min[n_inputs:], data_max[n_inputs:])
    y_sp_plot = reverse_min_max(y_sp_dev + y_ss, data_min[n_inputs:], data_max[n_inputs:])
    y_target_plot = None if y_target_dev is None else reverse_min_max(
        y_target_dev + y_ss,
        data_min[n_inputs:],
        data_max[n_inputs:],
    )
    y_tracking_plot = None if y_tracking_dev is None else reverse_min_max(
        y_tracking_dev + y_ss,
        data_min[n_inputs:],
        data_max[n_inputs:],
    )

    output_labels = _default_output_labels(n_outputs) if output_labels is None else list(output_labels)
    input_labels = _default_input_labels(n_inputs) if input_labels is None else list(input_labels)

    C_MPC = PAPER_COLORS["output"]
    C_SP = PAPER_COLORS["setpoint"]
    C_TARGET = PAPER_COLORS["target"]
    C_TRACK = "tab:gray"
    input_colors = [
        PAPER_COLORS.get(f"input_{idx}", f"tab:{['green', 'orange', 'blue', 'red'][idx % 4]}")
        for idx in range(n_inputs)
    ]

    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)
    time_plot_hour = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)

    ctx = paper_plot_context() if paper_style else nullcontext()
    with ctx:
        mpl.rcParams.update({"legend.frameon": True})

        # -------- Plot 1 (full horizon): outputs --------
        fig, axes = plt.subplots(n_outputs, 1, figsize=(7.6, 2.6 * n_outputs), sharex=True)
        axes = np.atleast_1d(axes)
        for idx, ax in enumerate(axes):
            ax.plot(time_plot, y_mpc[:, idx], "-", lw=2.2, color=C_MPC, label="MPC", zorder=2)
            ax.step(
                time_plot[:-1],
                y_sp_plot[:, idx],
                where="post",
                linestyle="--",
                lw=2.2,
                color=C_SP,
                alpha=0.95,
                label="Setpoint",
                zorder=3,
            )
            if y_target_plot is not None:
                ax.step(
                    time_plot[:-1],
                    y_target_plot[:, idx],
                    where="post",
                    linestyle="-.",
                    lw=1.9,
                    color=C_TARGET,
                    alpha=0.95,
                    label="Steady target",
                    zorder=3,
                )
            if y_tracking_plot is not None:
                ax.step(
                    time_plot[:-1],
                    y_tracking_plot[:, idx],
                    where="post",
                    linestyle=":",
                    lw=1.8,
                    color=C_TRACK,
                    alpha=0.95,
                    label="Tracking target",
                    zorder=3,
                )
            ax.set_ylabel(output_labels[idx])
            ax.set_xlim(0, time_plot[-1])
            ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
            ax.tick_params(axis="x", pad=4)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, facecolor="white")
        axes[-1].set_xlabel("Time (h)")
        plt.gcf().subplots_adjust(right=0.80)
        _savefig("fig_mpc_outputs_full")

        # -------- Plot 1b (last window): outputs --------
        fig, axes = plt.subplots(n_outputs, 1, figsize=(7.6, 2.6 * n_outputs), sharex=True)
        axes = np.atleast_1d(axes)
        output_start = max(0, nFE - time_in_sub_episodes)
        for idx, ax in enumerate(axes):
            ax.plot(
                time_plot_hour,
                y_mpc[output_start:, idx],
                "-",
                lw=2.2,
                color=C_MPC,
                label="MPC",
                zorder=2,
            )
            ax.step(
                time_plot_hour[:-1],
                y_sp_plot[output_start:, idx],
                where="post",
                linestyle="--",
                lw=2.2,
                color=C_SP,
                alpha=0.95,
                label="Setpoint",
                zorder=3,
            )
            if y_target_plot is not None:
                ax.step(
                    time_plot_hour[:-1],
                    y_target_plot[output_start:, idx],
                    where="post",
                    linestyle="-.",
                    lw=1.9,
                    color=C_TARGET,
                    alpha=0.95,
                    label="Steady target",
                    zorder=3,
                )
            if y_tracking_plot is not None:
                ax.step(
                    time_plot_hour[:-1],
                    y_tracking_plot[output_start:, idx],
                    where="post",
                    linestyle=":",
                    lw=1.8,
                    color=C_TRACK,
                    alpha=0.95,
                    label="Tracking target",
                    zorder=3,
                )
            ax.set_ylabel(output_labels[idx])
            ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
        axes[-1].set_xlabel("Time (h)")
        plt.gcf().subplots_adjust(right=0.80)
        _savefig(f"fig_mpc_outputs_last{time_in_sub_episodes}")

        # -------- Plot 2 (full horizon): inputs --------
        fig, axes = plt.subplots(n_inputs, 1, figsize=(7.6, 2.6 * n_inputs), sharex=True)
        axes = np.atleast_1d(axes)
        for idx, ax in enumerate(axes):
            ax.step(
                time_plot[:-1],
                u_mpc[:, idx],
                where="post",
                lw=2.2,
                color=input_colors[idx],
                label="Applied input",
                zorder=2,
            )
            if u_target_plot is not None:
                ax.step(
                    time_plot[:-1],
                    u_target_plot[:, idx],
                    where="post",
                    lw=1.9,
                    linestyle="--",
                    color=C_TARGET,
                    label="Steady target",
                    zorder=3,
                )
            if u_lower is not None and u_upper is not None:
                ax.axhline(u_lower[idx], color="tab:red", linewidth=1.2, linestyle=":", label="Lower bound" if idx == 0 else None)
                ax.axhline(u_upper[idx], color="tab:brown", linewidth=1.2, linestyle=":", label="Upper bound" if idx == 0 else None)
            ax.set_ylabel(input_labels[idx])
            ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, facecolor="white")
        axes[-1].set_xlabel("Time (h)")
        plt.gcf().subplots_adjust(right=0.80)
        _savefig("fig_mpc_inputs_full")

        # -------- Plot 2b (last window): inputs --------
        fig, axes = plt.subplots(n_inputs, 1, figsize=(7.6, 2.6 * n_inputs), sharex=True)
        axes = np.atleast_1d(axes)
        input_start = max(0, nFE - time_in_sub_episodes)
        input_time = np.linspace(0, min(time_in_sub_episodes, nFE) * delta_t, nFE - input_start)
        for idx, ax in enumerate(axes):
            ax.step(
                input_time,
                u_mpc[input_start:, idx],
                where="post",
                lw=2.2,
                color=input_colors[idx],
                label="Applied input",
            )
            if u_target_plot is not None:
                ax.step(
                    input_time,
                    u_target_plot[input_start:, idx],
                    where="post",
                    lw=1.9,
                    linestyle="--",
                    color=C_TARGET,
                    label="Steady target",
                )
            if u_lower is not None and u_upper is not None:
                ax.axhline(u_lower[idx], color="tab:red", linewidth=1.2, linestyle=":", label="Lower bound" if idx == 0 else None)
                ax.axhline(u_upper[idx], color="tab:brown", linewidth=1.2, linestyle=":", label="Upper bound" if idx == 0 else None)
            ax.set_ylabel(input_labels[idx])
            ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
            ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
        axes[-1].set_xlabel("Time (h)")
        plt.gcf().subplots_adjust(right=0.80)
        _savefig(f"fig_mpc_inputs_last{time_in_sub_episodes}")

    return out_dir


def plot_mpc_rl_results_cstr(
    y_sp, steady_states, nFE, delta_t, time_in_sub_episodes,
    y_mpc, u_mpc, y_rl, u_rl, data_min, data_max,
    directory=None, prefix_name="mpc_result"
):
    """
    Paper-ready MPC figures for CSTR; saves to directory/prefix_name/<timestamp>/.
    Same style and layout as distillation-column plot_mpc_results.
    Returns: out_dir (str)
    """

    if directory is None:
        directory = os.getcwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    def _savefig(name_base):
        plt.tight_layout()
        png = os.path.join(out_dir, f"{name_base}.png")
        plt.savefig(png, bbox_inches="tight", dpi=300)
        plt.close()

    # --- scaling logic (unchanged) ---
    y_ss = apply_min_max(steady_states["y_ss"], data_min[2:], data_max[2:])
    y_sp = (y_sp + y_ss)
    y_sp = (reverse_min_max(y_sp, data_min[2:], data_max[2:])).T  # (n_out, nFE)

    # --- style (exactly like distillation) ---
    mpl.rcParams.update({
        "font.size": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "legend.frameon": True,
    })

    C_MPC = "tab:blue"
    C_SP = "tab:red"
    C_U1 = "tab:green"
    C_U2 = "tab:orange"
    C_RL = "tab:purple"
    C_RL_U1 = "tab:brown"

    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)
    time_plot_hour = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)

    # -------- Plot 1 (full horizon): outputs --------
    plt.figure(figsize=(7.6, 5.2))

    ax = plt.subplot(2, 1, 1)
    ax.plot(time_plot, y_mpc[:, 0], "-", lw=2.2, color=C_MPC, label="MPC", zorder=2)
    ax.plot(time_plot, y_rl[:, 0], "-", lw=2.2, color=C_RL, label="RL", zorder=2)
    ax.step(time_plot[:-1], y_sp[0, :], where="post", linestyle="--", lw=2.2, color=C_SP, label="Setpoint", zorder=3)
    ax.set_ylabel(r"$\eta$ (L/g)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, time_plot[-1])
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", pad=4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0., facecolor="white")

    ax = plt.subplot(2, 1, 2)
    ax.plot(time_plot, y_mpc[:, 1], "-", lw=2.2, color=C_MPC, label="MPC", zorder=2)
    ax.plot(time_plot, y_rl[:, 1], "-", lw=2.2, color=C_RL, label="RL", zorder=2)
    ax.step(time_plot[:-1], y_sp[1, :], where="post", linestyle="--", lw=2.2, color=C_SP, label="Setpoint", zorder=3)
    ax.set_ylabel(r"$T$ (K)")
    ax.set_xlabel("Time (h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, time_plot[-1])
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", pad=4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0., facecolor="white")

    plt.gcf().subplots_adjust(right=0.82)
    _savefig("fig_mpc_outputs_full")

    # -------- Plot 1b (last window): outputs --------
    plt.figure(figsize=(7.6, 5.2))

    ax = plt.subplot(2, 1, 1)
    ax.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 0], "-", lw=2.2,
            color=C_MPC, label=r"MPC", zorder=2)
    ax.plot(time_plot_hour, y_rl[nFE - time_in_sub_episodes:, 0], "-", lw=2.2,
            color=C_RL, label=r"RL", zorder=2)
    ax.step(time_plot_hour[:-1], y_sp[0, nFE - time_in_sub_episodes:], where="post",
            linestyle="--", lw=2.2, color=C_SP, alpha=0.95, label=r"Setpoint", zorder=3)
    ax.set_ylabel(r"$\eta$ (L/g)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    # ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0., facecolor="white")

    ax = plt.subplot(2, 1, 2)
    ax.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 1], "-", lw=2.2,
            color=C_MPC, label=r"MPC", zorder=2)
    ax.plot(time_plot_hour, y_rl[nFE - time_in_sub_episodes:, 1], "-", lw=2.2,
            color=C_RL, label=r"RL", zorder=2)
    ax.step(time_plot_hour[:-1], y_sp[1, nFE - time_in_sub_episodes:], where="post",
            linestyle="--", lw=2.2, color=C_SP, alpha=0.95, label=r"Setpoint", zorder=3)
    ax.set_ylabel(r"$T$ (K)")
    ax.set_xlabel("Time (h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    # ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0., facecolor="white")

    plt.gcf().subplots_adjust(right=0.82)
    _savefig(f"fig_mpc_outputs_last{time_in_sub_episodes}")

    # -------- Plot 2 (full horizon): inputs --------
    plt.figure(figsize=(7.6, 5.2))

    ax = plt.subplot(2, 1, 1)
    ax.step(time_plot[:-1], u_mpc[:, 0], where="post", lw=2.2, color=C_U1, label="MPC Qc", zorder=2)
    ax.step(time_plot[:-1], u_rl[:, 0], where="post", lw=2.2, linestyle="--", color=C_U1, label="RL Qc", zorder=3)
    ax.set_ylabel(r"$Q_c$ (L/h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    # ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0., facecolor="white")

    ax = plt.subplot(2, 1, 2)
    ax.step(time_plot[:-1], u_mpc[:, 1], where="post", lw=2.2, color=C_U1, label="MPC Qc", zorder=2)
    ax.step(time_plot[:-1], u_rl[:, 1], where="post", lw=2.2, linestyle="--", color=C_U1, label="RL Qc", zorder=3)
    ax.set_ylabel(r"$Q_m$ (L/h)")
    ax.set_xlabel("Time (h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    # ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0., facecolor="white")

    plt.gcf().subplots_adjust(right=0.82)
    _savefig("fig_mpc_inputs_full")

    return out_dir
