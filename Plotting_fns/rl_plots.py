import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import matplotlib as mpl
import matplotlib.ticker as mtick

from utils.path_helpers import repo_root, resolve_repo_path
from utils.scaling_helpers import reverse_min_max, apply_min_max

def plot_rl_results_disturbance(
    y_sp, steady_states, nFE, delta_t, time_in_sub_episodes,
    y_mpc, u_mpc, avg_rewards, data_min, data_max, warm_start_plot,
    directory=None, prefix_name="agent_result",
    agent=None,
    delta_y_storage=None,
    rewards=None,
    dist=None,
    start_plot_idx=10
):
    """
    Distillation-style plotting (same colors/fonts/no legends).
    Saves all figures to directory/prefix_name/<timestamp>.
    Handles:
      dist=None
      dist=1D array
      dist=dict with keys {"qi","qs","ha"}
    """


    if directory is None:
        directory = os.fspath(repo_root())
    else:
        directory = os.fspath(resolve_repo_path(directory))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    def _savefig(name):
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name), bbox_inches="tight", dpi=300)
        plt.close()

    y_sp_original = np.array(y_sp, copy=True)

    actor_losses = getattr(agent, "actor_losses", None) if agent is not None else None
    critic_losses = getattr(agent, "critic_losses", None) if agent is not None else None
    dy_arr = np.array(delta_y_storage) if delta_y_storage is not None else None
    rewards_arr = np.array(rewards) if rewards is not None else None

    # Canceling the deviation form (same logic)
    y_ss = apply_min_max(steady_states["y_ss"], data_min[2:], data_max[2:])
    y_sp = (y_sp + y_ss)
    y_sp = (reverse_min_max(y_sp, data_min[2:], data_max[2:])).T  # (n_out, nFE)

    # Distillation-style rcParams (no bold globals; bold comes from \mathbf in labels)
    mpl.rcParams.update({
        "font.size": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "legend.frameon": True
    })

    # Colors exactly like distillation code
    C_QC = "tab:green"
    C_QM = "tab:orange"
    C_RW = "tab:purple"

    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)
    warm_start_plot = np.atleast_1d(warm_start_plot) * delta_t
    ws_end = float(warm_start_plot.max()) if warm_start_plot.size > 0 else 0.0

    time_plot_hour = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)

    # -------- Plot 1: outputs (full) --------
    plt.figure(figsize=(10, 8))

    ax = plt.subplot(2, 1, 1)
    ax.plot(time_plot[start_plot_idx:], y_mpc[start_plot_idx:, 0], "b-", lw=2, zorder=2)
    ax.step(time_plot[start_plot_idx:-1], y_sp[0, start_plot_idx:], "r--", lw=2, where="post", zorder=3)
    for t_ws in warm_start_plot:
        ax.axvline(float(t_ws), color="k", linestyle="--", linewidth=1.2, zorder=1)
    if ws_end > 0.0:
        ax.axvspan(0.0, ws_end, facecolor="0.9", alpha=0.6, zorder=0)
    ax.set_ylabel(r"$\mathbf{\eta}$ (L/g)", fontsize=18)
    ax.set_xlim(0, time_plot[-1])
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    ax.tick_params(axis="x", pad=4)

    ax = plt.subplot(2, 1, 2)
    ax.plot(time_plot[start_plot_idx:], y_mpc[start_plot_idx:, 1], "b-", lw=2, zorder=2)
    ax.step(time_plot[start_plot_idx:-1], y_sp[1, start_plot_idx:], "r--", lw=2, where="post", zorder=3)
    for t_ws in warm_start_plot:
        ax.axvline(float(t_ws), color="k", linestyle="--", linewidth=1.2, zorder=1)
    if ws_end > 0.0:
        ax.axvspan(0.0, ws_end, facecolor="0.9", alpha=0.6, zorder=0)
    ax.set_ylabel(r"$\mathbf{T}$ (K)", fontsize=18)
    ax.set_xlabel(r"$\mathbf{Time}$ (hour)", fontsize=18)
    ax.set_xlim(0, time_plot[-1])
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    ax.tick_params(axis="x", pad=4)

    plt.subplot(2, 1, 1)
    plt.tick_params(axis="both", labelsize=16)
    plt.subplot(2, 1, 2)
    plt.tick_params(axis="both", labelsize=16)

    plt.gcf().subplots_adjust(right=0.95, bottom=0.12)
    _savefig("fig_rl_outputs_full.png")

    # -------- last window --------
    plt.figure(figsize=(7.6, 5.2))

    ax = plt.subplot(2, 1, 1)
    ax.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 0], "-", lw=2.2, color="b", zorder=2)
    ax.step(time_plot_hour[:-1], y_sp[0, nFE - time_in_sub_episodes:], where="post",
            linestyle="--", lw=2.2, color="r", alpha=0.95, zorder=3)
    ax.set_ylabel(r"$\eta$ (L/g)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = plt.subplot(2, 1, 2)
    ax.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 1], "-", lw=2.2, color="b", zorder=2)
    ax.step(time_plot_hour[:-1], y_sp[1, nFE - time_in_sub_episodes:], where="post",
            linestyle="--", lw=2.2, color="r", alpha=0.95, zorder=3)
    ax.set_ylabel(r"$T$ (K)")
    ax.set_xlabel("Time (h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.gcf().subplots_adjust(right=0.95)
    _savefig(f"fig_rl_outputs_last{time_in_sub_episodes}.png")

    # -------- last 4x window --------
    W4 = 4 * time_in_sub_episodes
    time_plot_4w = np.linspace(0, W4 * delta_t, W4 + 1)

    plt.figure(figsize=(7.6, 5.2))

    ax = plt.subplot(2, 1, 1)
    ax.plot(time_plot_4w, y_mpc[nFE - W4:, 0], "-", lw=2.2, color="b", zorder=2)
    ax.step(time_plot_4w[:-1], y_sp[0, nFE - W4:], where="post",
            linestyle="--", lw=2.2, color="r", alpha=0.95, zorder=3)
    ax.set_ylabel(r"$\eta$ (L/g)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = plt.subplot(2, 1, 2)
    ax.plot(time_plot_4w, y_mpc[nFE - W4:, 1], "-", lw=2.2, color="b", zorder=2)
    ax.step(time_plot_4w[:-1], y_sp[1, nFE - W4:], where="post",
            linestyle="--", lw=2.2, color="r", alpha=0.95, zorder=3)
    ax.set_ylabel(r"$T$ (K)")
    ax.set_xlabel("Time (h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.gcf().subplots_adjust(right=0.95)
    _savefig(f"fig_rl_outputs_last{W4}.png")

    # -------- Plot 2: inputs --------
    plt.figure(figsize=(7.6, 5.2))

    ax = plt.subplot(2, 1, 1)
    ax.step(time_plot[:-1], u_mpc[:, 0], where="post", lw=2.2, color=C_QC, zorder=2)
    ax.set_ylabel(r"$Q_c$ (L/h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = plt.subplot(2, 1, 2)
    ax.step(time_plot[:-1], u_mpc[:, 1], where="post", lw=2.2, color=C_QM, zorder=2)
    ax.set_ylabel(r"$Q_m$ (L/h)")
    ax.set_xlabel("Time (h)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.gcf().subplots_adjust(right=0.95)
    _savefig("fig_rl_inputs_full.png")

    # -------- Plot 3: reward per episode --------
    plt.figure(figsize=(7.2, 4.2))
    xep = np.arange(1, len(avg_rewards) + 1)
    plt.plot(xep, avg_rewards, "o-", lw=2.2, color=C_RW, zorder=2)
    plt.ylabel("Avg. Reward")
    plt.xlabel("Episode #")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig("fig_rl_rewards.png")

    # -------- optional losses --------
    if actor_losses is not None and len(actor_losses) > 0:
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(actor_losses, lw=1.8, color="tab:blue")
        plt.ylabel("Actor Loss")
        plt.xlabel("Update Step")
        plt.grid(True, linestyle="--", alpha=0.35)
        _savefig("loss_actor.png")

    if critic_losses is not None and len(critic_losses) > 0:
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(critic_losses, lw=1.8, color="tab:orange")
        plt.ylabel("Critic Loss")
        plt.xlabel("Update Step")
        plt.grid(True, linestyle="--", alpha=0.35)
        _savefig("loss_critic.png")

    # -------- optional delta_y windows (no legend) --------
    if dy_arr is not None and dy_arr.ndim == 2 and dy_arr.shape[1] >= 2:
        n = dy_arr.shape[0]

        i0 = max(0, n - 300)
        w = dy_arr[i0:n]
        if len(w) > 0:
            plt.figure(figsize=(7.6, 4.2))
            plt.plot(w[:, 0], c="r")
            plt.plot(w[:, 1], c="b")
            plt.ylabel(r"$\Delta y$")
            plt.xlabel("Step")
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("delta_y_last300.png")

        j0 = max(0, n - 700)
        j1 = max(0, n - 400)
        w2 = dy_arr[j0:j1]
        if len(w2) > 0:
            plt.figure(figsize=(7.6, 4.2))
            plt.plot(w2[:, 0], c="r")
            plt.plot(w2[:, 1], c="b")
            plt.ylabel(r"$\Delta y$")
            plt.xlabel("Step")
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("delta_y_700_400.png")

    # -------- optional per-step rewards (no legend) --------
    if rewards_arr is not None and rewards_arr.ndim == 1 and rewards_arr.size > 0:
        n = rewards_arr.size

        j0 = max(0, n - 700)
        j1 = max(0, n - 400)
        w = rewards_arr[j0:j1]
        if w.size > 0:
            plt.figure(figsize=(7.6, 4.2))
            plt.scatter(range(w.size), w, s=10)
            plt.ylabel("Reward")
            plt.xlabel("Step")
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("rewards_700_400.png")

        i0 = max(0, n - 300)
        w2 = rewards_arr[i0:n]
        if w2.size > 0:
            plt.figure(figsize=(7.6, 4.2))
            plt.scatter(range(w2.size), w2, s=10)
            plt.ylabel("Reward")
            plt.xlabel("Step")
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("rewards_last300.png")

        plt.figure(figsize=(7.6, 4.2))
        plt.scatter(range(rewards_arr.size), rewards_arr, s=10)
        plt.ylabel("Reward")
        plt.xlabel("Step")
        plt.grid(True, linestyle="--", alpha=0.35)
        _savefig("rewards_all.png")

    # -------- disturbance (no legend) --------
    if dist is not None:
        if isinstance(dist, dict) and all(k in dist for k in ["qi", "qs", "ha"]):
            qi_arr = np.asarray(dist["qi"]).squeeze()
            qs_arr = np.asarray(dist["qs"]).squeeze()
            ha_arr = np.asarray(dist["ha"]).squeeze()
            n_al = min(nFE, qi_arr.shape[0], qs_arr.shape[0], ha_arr.shape[0])

            def _dist_fig(t, q1, q2, hA, suffix):
                plt.figure(figsize=(7.6, 6.2))

                ax = plt.subplot(3, 1, 1)
                ax.plot(t, q1, "-", lw=2, color="tab:blue")
                ax.set_ylabel(r"$Q_i$ (L/h)")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
                ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
                ax.grid(True, linestyle="--", alpha=0.35)

                ax = plt.subplot(3, 1, 2)
                ax.plot(t, q2, "-", lw=2, color="tab:orange")
                ax.set_ylabel(r"$Q_s$ (L/h)")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
                ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
                ax.grid(True, linestyle="--", alpha=0.35)

                ax = plt.subplot(3, 1, 3)
                ax.plot(t, hA, "-", lw=2, color="tab:green")
                ax.set_xlabel("Time (h)")
                ax.set_ylabel(r"$h_a$ (J/Kh)")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.xaxis.set_major_locator(mtick.MaxNLocator(6))
                ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
                ax.grid(True, linestyle="--", alpha=0.35)

                plt.gcf().subplots_adjust(right=0.95, hspace=0.25)
                _savefig(f"fig_disturbances_{suffix}.png")

            _dist_fig(time_plot[:n_al], qi_arr[:n_al], qs_arr[:n_al], ha_arr[:n_al], suffix="full")

            if time_in_sub_episodes > 0:
                W = min(time_in_sub_episodes, n_al)
                t_lastW = np.linspace(0, W * delta_t, W, endpoint=False)
                _dist_fig(
                    t_lastW,
                    qi_arr[n_al - W:n_al],
                    qs_arr[n_al - W:n_al],
                    ha_arr[n_al - W:n_al],
                    suffix=f"last{W}"
                )
        else:
            dist_arr = np.asarray(dist).squeeze()
            n_al = min(nFE, dist_arr.shape[0])
            plt.figure(figsize=(7.2, 4.2))
            plt.plot(time_plot[start_plot_idx:n_al], dist_arr[start_plot_idx:n_al], lw=1.8, color="tab:blue")
            plt.ylabel("Disturbance")
            plt.xlabel("Time (h)")
            plt.grid(True, linestyle="--", alpha=0.35)
            _savefig("disturbance.png")

    return out_dir
