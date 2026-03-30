import os
import pickle
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import matplotlib as mpl
import matplotlib.ticker as mtick

from utils.scaling_helpers import reverse_min_max, apply_min_max
from utils.plot_style import PAPER_COLORS, paper_plot_context

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
    Saves all figures + input_data.pkl to directory/prefix_name/<timestamp>.
    Handles:
      dist=None
      dist=1D array
      dist=dict with keys {"qi","qs","ha"}
    """


    if directory is None:
        directory = os.getcwd()

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

    input_data = {
        "y_sp": y_sp_original,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y_mpc": y_mpc,
        "u_mpc": u_mpc,
        "avg_rewards": avg_rewards,
        "data_min": data_min,
        "data_max": data_max,
        "warm_start_plot": warm_start_plot,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "delta_y_storage": dy_arr,
        "rewards": rewards_arr,
        "dist": dist,
        "start_plot_idx": start_plot_idx
    }
    with open(os.path.join(out_dir, "input_data.pkl"), "wb") as f:
        pickle.dump(input_data, f)

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


def save_rl_summary_plots_from_bundle(
    bundle,
    output_dir,
    paper_style=True,
    save_input_data=True,
):
    """
    Render the classic RL summary plots directly into an existing debug-export
    directory using data stored in a safety-debug bundle.
    """
    os.makedirs(output_dir, exist_ok=True)

    extra = bundle.get("extra", {}) or {}
    data_min = bundle.get("data_min")
    data_max = bundle.get("data_max")
    steady_states = bundle.get("steady_states")

    if steady_states is None or data_min is None or data_max is None:
        raise ValueError("RL summary plots require steady_states, data_min, and data_max in the bundle.")

    y_sp = np.asarray(bundle["y_sp"], float)
    y_system = np.asarray(bundle["y_system"], float)
    u_applied_phys = np.asarray(bundle["u_applied_phys"], float)
    rewards = np.asarray(bundle["rewards"], float)
    qi = np.asarray(bundle.get("qi", []), float)
    qs = np.asarray(bundle.get("qs", []), float)
    ha = np.asarray(bundle.get("ha", []), float)
    avg_rewards = list(bundle.get("avg_rewards", []))
    delta_y_storage = np.asarray(bundle.get("e_store", []), float)

    nFE = int(bundle["nFE"])
    delta_t = float(extra.get("delta_t", bundle.get("delta_t", 1.0) or 1.0))
    time_in_sub_episodes = int(bundle["time_in_sub_episodes"])
    start_plot_idx = int(extra.get("start_plot_idx", bundle.get("start_plot_idx", 10)))
    warm_start_plot = np.atleast_1d(extra.get("warm_start_plot", bundle.get("warm_start_plot", 0))).astype(float)
    actor_losses = extra.get("actor_losses", bundle.get("actor_losses"))
    critic_losses = extra.get("critic_losses", bundle.get("critic_losses"))

    if actor_losses is not None:
        actor_losses = np.asarray(actor_losses, float)
    if critic_losses is not None:
        critic_losses = np.asarray(critic_losses, float)

    if save_input_data:
        input_data = {
            "y_sp": y_sp.copy(),
            "steady_states": steady_states,
            "nFE": nFE,
            "delta_t": delta_t,
            "time_in_sub_episodes": time_in_sub_episodes,
            "y_mpc": y_system.copy(),
            "u_mpc": u_applied_phys.copy(),
            "avg_rewards": avg_rewards,
            "data_min": data_min,
            "data_max": data_max,
            "warm_start_plot": warm_start_plot.copy(),
            "actor_losses": None if actor_losses is None else actor_losses.copy(),
            "critic_losses": None if critic_losses is None else critic_losses.copy(),
            "delta_y_storage": delta_y_storage.copy(),
            "rewards": rewards.copy(),
            "dist": {"qi": qi.copy(), "qs": qs.copy(), "ha": ha.copy()},
            "start_plot_idx": start_plot_idx,
        }
        with open(os.path.join(output_dir, "input_data.pkl"), "wb") as f:
            pickle.dump(input_data, f)

    ctx = paper_plot_context() if paper_style else nullcontext()

    with ctx:
        n_u = u_applied_phys.shape[1]
        y_ss = apply_min_max(steady_states["y_ss"], data_min[n_u:], data_max[n_u:])
        y_sp_phys = reverse_min_max(y_sp + y_ss, data_min[n_u:], data_max[n_u:]).T

        time_plot = np.linspace(0, nFE * delta_t, nFE + 1)
        time_plot_hour = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)
        ws_time = warm_start_plot * delta_t
        ws_end = float(ws_time.max()) if ws_time.size > 0 else 0.0

        def _savefig(name):
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, name), bbox_inches="tight", dpi=300)
            plt.close()

        c_out = PAPER_COLORS["output"]
        c_sp = PAPER_COLORS["setpoint"]
        c_qc = PAPER_COLORS["input_0"]
        c_qm = PAPER_COLORS["input_1"]
        c_rw = PAPER_COLORS["reward"]

        fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=False)
        axes[0].plot(time_plot[start_plot_idx:], y_system[start_plot_idx:, 0], color=c_out)
        axes[0].step(time_plot[start_plot_idx:-1], y_sp_phys[0, start_plot_idx:], where="post", linestyle="--", color=c_sp)
        for t_ws in ws_time:
            axes[0].axvline(float(t_ws), color="k", linestyle="--", linewidth=1.0)
        if ws_end > 0.0:
            axes[0].axvspan(0.0, ws_end, facecolor="0.92", alpha=0.7, zorder=0)
        axes[0].set_ylabel(r"$\eta$ (L/g)")

        axes[1].plot(time_plot[start_plot_idx:], y_system[start_plot_idx:, 1], color=c_out)
        axes[1].step(time_plot[start_plot_idx:-1], y_sp_phys[1, start_plot_idx:], where="post", linestyle="--", color=c_sp)
        for t_ws in ws_time:
            axes[1].axvline(float(t_ws), color="k", linestyle="--", linewidth=1.0)
        if ws_end > 0.0:
            axes[1].axvspan(0.0, ws_end, facecolor="0.92", alpha=0.7, zorder=0)
        axes[1].set_ylabel(r"$T$ (K)")
        axes[1].set_xlabel("Time (h)")
        _savefig("fig_rl_outputs_full.png")

        fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=False)
        axes[0].plot(time_plot_hour, y_system[nFE - time_in_sub_episodes:, 0], color=c_out)
        axes[0].step(time_plot_hour[:-1], y_sp_phys[0, nFE - time_in_sub_episodes:], where="post", linestyle="--", color=c_sp)
        axes[0].set_ylabel(r"$\eta$ (L/g)")
        axes[1].plot(time_plot_hour, y_system[nFE - time_in_sub_episodes:, 1], color=c_out)
        axes[1].step(time_plot_hour[:-1], y_sp_phys[1, nFE - time_in_sub_episodes:], where="post", linestyle="--", color=c_sp)
        axes[1].set_ylabel(r"$T$ (K)")
        axes[1].set_xlabel("Time (h)")
        _savefig(f"fig_rl_outputs_last{time_in_sub_episodes}.png")

        w4 = min(4 * time_in_sub_episodes, nFE)
        if w4 > 0:
            time_plot_4w = np.linspace(0, w4 * delta_t, w4 + 1)
            fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=False)
            axes[0].plot(time_plot_4w, y_system[nFE - w4:, 0], color=c_out)
            axes[0].step(time_plot_4w[:-1], y_sp_phys[0, nFE - w4:], where="post", linestyle="--", color=c_sp)
            axes[0].set_ylabel(r"$\eta$ (L/g)")
            axes[1].plot(time_plot_4w, y_system[nFE - w4:, 1], color=c_out)
            axes[1].step(time_plot_4w[:-1], y_sp_phys[1, nFE - w4:], where="post", linestyle="--", color=c_sp)
            axes[1].set_ylabel(r"$T$ (K)")
            axes[1].set_xlabel("Time (h)")
            _savefig(f"fig_rl_outputs_last{w4}.png")

        fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)
        axes[0].step(time_plot[:-1], u_applied_phys[:, 0], where="post", color=c_qc)
        axes[0].set_ylabel(r"$Q_c$ (L/h)")
        axes[1].step(time_plot[:-1], u_applied_phys[:, 1], where="post", color=c_qm)
        axes[1].set_ylabel(r"$Q_m$ (L/h)")
        axes[1].set_xlabel("Time (h)")
        _savefig("fig_rl_inputs_full.png")

        plt.figure(figsize=(7.2, 4.2))
        xep = np.arange(1, len(avg_rewards) + 1)
        plt.plot(xep, avg_rewards, "o-", color=c_rw)
        plt.ylabel("Avg. Reward")
        plt.xlabel("Episode #")
        _savefig("fig_rl_rewards.png")

        if actor_losses is not None and actor_losses.size > 0:
            plt.figure(figsize=(7.2, 4.2))
            plt.plot(actor_losses, color="tab:blue", linewidth=1.8)
            plt.ylabel("Actor Loss")
            plt.xlabel("Update Step")
            _savefig("loss_actor.png")

        if critic_losses is not None and critic_losses.size > 0:
            plt.figure(figsize=(7.2, 4.2))
            plt.plot(critic_losses, color="tab:orange", linewidth=1.8)
            plt.ylabel("Critic Loss")
            plt.xlabel("Update Step")
            _savefig("loss_critic.png")

        if delta_y_storage.ndim == 2 and delta_y_storage.shape[1] >= 2:
            dy_arr = delta_y_storage[:-1, :] if delta_y_storage.shape[0] > nFE else delta_y_storage
            n = dy_arr.shape[0]
            i0 = max(0, n - 300)
            if i0 < n:
                plt.figure(figsize=(7.6, 4.2))
                plt.plot(dy_arr[i0:n, 0], color="tab:red")
                plt.plot(dy_arr[i0:n, 1], color="tab:blue")
                plt.ylabel(r"$\Delta y$")
                plt.xlabel("Step")
                _savefig("delta_y_last300.png")
            j0 = max(0, n - 700)
            j1 = max(0, n - 400)
            if j0 < j1:
                plt.figure(figsize=(7.6, 4.2))
                plt.plot(dy_arr[j0:j1, 0], color="tab:red")
                plt.plot(dy_arr[j0:j1, 1], color="tab:blue")
                plt.ylabel(r"$\Delta y$")
                plt.xlabel("Step")
                _savefig("delta_y_700_400.png")

        if rewards.size > 0:
            n = rewards.size
            j0 = max(0, n - 700)
            j1 = max(0, n - 400)
            if j0 < j1:
                plt.figure(figsize=(7.6, 4.2))
                plt.scatter(range(j1 - j0), rewards[j0:j1], s=10, color=c_rw)
                plt.ylabel("Reward")
                plt.xlabel("Step")
                _savefig("rewards_700_400.png")

            i0 = max(0, n - 300)
            plt.figure(figsize=(7.6, 4.2))
            plt.scatter(range(n - i0), rewards[i0:n], s=10, color=c_rw)
            plt.ylabel("Reward")
            plt.xlabel("Step")
            _savefig("rewards_last300.png")

            plt.figure(figsize=(7.6, 4.2))
            plt.scatter(range(n), rewards, s=10, color=c_rw)
            plt.ylabel("Reward")
            plt.xlabel("Step")
            _savefig("rewards_all.png")

        if qi.size and qs.size and ha.size:
            n_al = min(nFE, qi.shape[0], qs.shape[0], ha.shape[0])

            def _dist_fig(t, q1, q2, hA_vals, suffix):
                fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.2), sharex=False)
                axes[0].plot(t, q1, color=PAPER_COLORS["dist_qi"])
                axes[0].set_ylabel(r"$Q_i$ (L/h)")
                axes[1].plot(t, q2, color=PAPER_COLORS["dist_qs"])
                axes[1].set_ylabel(r"$Q_s$ (L/h)")
                axes[2].plot(t, hA_vals, color=PAPER_COLORS["dist_ha"])
                axes[2].set_ylabel(r"$h_a$ (J/Kh)")
                axes[2].set_xlabel("Time (h)")
                _savefig(f"fig_disturbances_{suffix}.png")

            _dist_fig(time_plot[:n_al], qi[:n_al], qs[:n_al], ha[:n_al], "full")

            if time_in_sub_episodes > 0:
                w = min(time_in_sub_episodes, n_al)
                t_last = np.linspace(0, w * delta_t, w, endpoint=False)
                _dist_fig(
                    t_last,
                    qi[n_al - w:n_al],
                    qs[n_al - w:n_al],
                    ha[n_al - w:n_al],
                    f"last{w}",
                )

    return output_dir
