import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
plt.style.use('default')

from utils.scaling_helpers import apply_min_max, reverse_min_max


def plot_mpc_results_cstr(
    y_sp, steady_states, nFE, delta_t, time_in_sub_episodes,
    y_mpc, u_mpc, data_min, data_max,
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

    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)
    time_plot_hour = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)

    # -------- Plot 1 (full horizon): outputs --------
    plt.figure(figsize=(7.6, 5.2))

    ax = plt.subplot(2, 1, 1)
    ax.plot(time_plot, y_mpc[:, 0], "-", lw=2.2, color=C_MPC, label=r"MPC", zorder=2)
    ax.step(time_plot[:-1], y_sp[0, :], where="post", linestyle="--", lw=2.2,
            color=C_SP, alpha=0.95, label=r"Setpoint", zorder=3)
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
    ax.plot(time_plot, y_mpc[:, 1], "-", lw=2.2, color=C_MPC, label=r"MPC", zorder=2)
    ax.step(time_plot[:-1], y_sp[1, :], where="post", linestyle="--", lw=2.2,
            color=C_SP, alpha=0.95, label=r"Setpoint", zorder=3)
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
    ax.step(time_plot[:-1], u_mpc[:, 0], where="post", lw=2.2, color=C_U1, label=r"$Q_c$", zorder=2)
    ax.set_ylabel(r"$Q_c$ (L/h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    # ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0., facecolor="white")

    ax = plt.subplot(2, 1, 2)
    ax.step(time_plot[:-1], u_mpc[:, 1], where="post", lw=2.2, color=C_U2, label=r"$Q_m$", zorder=2)
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