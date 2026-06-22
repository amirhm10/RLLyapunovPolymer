from __future__ import annotations

from pprint import pprint

from utils.gart_defaults import GART_FINAL_LYAP_EPS, GART_FINAL_RHO_LYAP
from utils.direct_lyapunov_study import DIRECT_DISTURBANCE_N_TESTS
from utils.online_disturbance_runner import (
    default_noisy_teacher_critic_warmup_overrides,
    run_online_td3_disturbance_preset,
)

# Cold-start online TD3 with the GART-LMPC safety gate.
#
# This root runner is intentionally editable, following the style of
# GARTLyapunovMPC.py. Change these values here for day-to-day experiments.

EPISODES = DIRECT_DISTURBANCE_N_TESTS
SET_POINTS_LEN = 400
SEED = 123
SAVE_PLOTS = True

# Set to None for an automatic timestamp, or use a fixed string to rerun into a
# predictable folder under results/OnlineTD3_ColdStart_SafetyGate/.
TIMESTAMP = None

# GART fallback gate with the same standard TD3 observation used by the other
# active online runners. Unsafe actor proposals are rejected and replaced by the
# GART-LMPC fallback, without the Section 16 projection QCQP.
RL_OBSERVATION_MODE = "standard"
PROJECTION_BACKEND = "direct_accept_or_fallback"

RHO_LYAP = GART_FINAL_RHO_LYAP
LYAP_EPS = GART_FINAL_LYAP_EPS
LYAP_TOL = None

REWARD_FALLBACK_PENALTY_ENABLED = False
GAMMA_FALLBACK = 0.0
FALLBACK_EVENT_PENALTY = 0.0

TRAINING_PHASE_OVERRIDES = default_noisy_teacher_critic_warmup_overrides(
    teacher_source="gart_lmpc",
    pretrained=False,
)

def run_configured_study() -> dict:
    config = {
        "preset": "cold_start_safety_gate",
        "episodes": int(EPISODES),
        "set_points_len": int(SET_POINTS_LEN),
        "seed": int(SEED),
        "save_plots": bool(SAVE_PLOTS),
        "timestamp": TIMESTAMP,
        "rl_observation_mode": RL_OBSERVATION_MODE,
        "projection_backend": PROJECTION_BACKEND,
        "rho_lyap": float(RHO_LYAP),
        "lyap_eps": float(LYAP_EPS),
        "lyap_tol": LYAP_TOL,
        "reward_fallback_penalty_enabled": bool(REWARD_FALLBACK_PENALTY_ENABLED),
        "gamma_fallback": float(GAMMA_FALLBACK),
        "fallback_event_penalty": float(FALLBACK_EVENT_PENALTY),
        "training_phase_overrides": dict(TRAINING_PHASE_OVERRIDES),
    }
    print("Online TD3 cold-start safety-gate configuration:")
    pprint(config)
    return run_online_td3_disturbance_preset(
        "cold_start_safety_gate",
        episodes=EPISODES,
        set_points_len=SET_POINTS_LEN,
        seed=SEED,
        save_plots=SAVE_PLOTS,
        timestamp=TIMESTAMP,
        rl_observation_mode=RL_OBSERVATION_MODE,
        projection_backend=PROJECTION_BACKEND,
        reward_fallback_penalty_enabled=REWARD_FALLBACK_PENALTY_ENABLED,
        gamma_fallback=GAMMA_FALLBACK,
        fallback_event_penalty=FALLBACK_EVENT_PENALTY,
        rho_lyap=RHO_LYAP,
        lyap_eps=LYAP_EPS,
        lyap_tol=LYAP_TOL,
        training_phase_overrides=TRAINING_PHASE_OVERRIDES,
    )


if __name__ == "__main__":
    run_configured_study()
