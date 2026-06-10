from __future__ import annotations

import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from utils.direct_lyapunov_study import (
    DIRECT_DISTURBANCE_N_TESTS,
    DIRECT_DISTURBANCE_SETPOINT_LEN,
    DIRECT_TWO_SETPOINT_Y_PHYS,
    direct_disturbance_test_cycle,
)
from utils.helpers import generate_setpoints_training_rl_gradually
from utils.of_mpc_td3_workflow import (
    HA_CHANGE,
    NOMINAL_HA,
    NOMINAL_QI,
    NOMINAL_QS,
    QI_CHANGE,
    QS_CHANGE,
)
from utils.path_helpers import repo_path


def main() -> None:
    n_tests = DIRECT_DISTURBANCE_N_TESTS
    set_points_len = DIRECT_DISTURBANCE_SETPOINT_LEN
    y_sp_scenario = DIRECT_TWO_SETPOINT_Y_PHYS.copy()
    test_cycle = direct_disturbance_test_cycle(n_tests)

    _, n_fe, _, time_in_sub_episodes, _, _, qi, qs, ha = generate_setpoints_training_rl_gradually(
        y_sp_scenario,
        n_tests,
        set_points_len,
        0,
        test_cycle,
        NOMINAL_QI,
        NOMINAL_QS,
        NOMINAL_HA,
        QI_CHANGE,
        QS_CHANGE,
        HA_CHANGE,
        force_final_test=False,
        disturbance_profile=None,
    )

    episode = np.arange(n_fe, dtype=float) / float(max(time_in_sub_episodes, 1))
    qi = np.asarray(qi[:n_fe], dtype=float)
    qs = np.asarray(qs[:n_fe], dtype=float)
    ha = np.asarray(ha[:n_fe], dtype=float)

    out_dir = repo_path("report", "figures", "2026-06-10_online_disturbance_runner")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "disturbance_profile.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.5, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.25]},
    )

    profiles = [
        ("Qi", qi, NOMINAL_QI, "tab:blue"),
        ("Qs", qs, NOMINAL_QS, "tab:orange"),
        ("hA", ha, NOMINAL_HA, "tab:green"),
    ]
    for ax, (name, values, nominal, color) in zip(axes[:3], profiles):
        ax.plot(episode, values, color=color, linewidth=2.0)
        ax.axhline(nominal, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(name)
        ax.text(
            0.01,
            0.78,
            f"{values[0]:.3g} -> {values[-1]:.3g}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )

    axes[3].plot(episode, qi / NOMINAL_QI, label="Qi / nominal", linewidth=2.0)
    axes[3].plot(episode, qs / NOMINAL_QS, label="Qs / nominal", linewidth=2.0)
    axes[3].plot(episode, ha / NOMINAL_HA, label="hA / nominal", linewidth=2.0)
    axes[3].axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    axes[3].set_ylabel("normalized")
    axes[3].set_xlabel("episode index")
    axes[3].legend(loc="best", ncols=3)

    fig.suptitle("Disturbance Profile Used By Disturbance-Only Online Runners", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved={out_path}")
    print(f"n_steps={n_fe}, episode_len={time_in_sub_episodes}, n_episodes={n_tests}")
    print(
        "final_ratios="
        f"Qi:{qi[-1] / NOMINAL_QI:.4f}, "
        f"Qs:{qs[-1] / NOMINAL_QS:.4f}, "
        f"hA:{ha[-1] / NOMINAL_HA:.4f}"
    )


if __name__ == "__main__":
    main()
