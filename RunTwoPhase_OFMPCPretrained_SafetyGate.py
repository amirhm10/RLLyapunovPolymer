from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from pprint import pprint

from RunOnlineTD3TwoPhaseStudy import run_two_phase_study
import utils.online_disturbance_runner as online_runner


# OF-MPC-pretrained online TD3 with active GART-LMPC safety gate.
METHOD = "ofmpc_pretrained_safety_gate"

# Paired paper seeds. Use SEEDS = None with N_SEEDS/SEED_START for quick sequential tests.
PAPER_SEEDS = (42, 7, 19, 73, 101, 203, 307, 401, 557, 809)
SEEDS: tuple[int, ...] | None = PAPER_SEEDS
N_SEEDS = len(PAPER_SEEDS)
SEED_START = 0

PHASE1_EPISODES = 200
PHASE2_EPISODES = 50
SET_POINTS_LEN = 400

OUTPUT_ROOT = Path.home() / "Desktop" / "Lyapunov_polymer_results"
TIMESTAMP = None
SAVE_PLOTS = True
EXPORT_PROFILE = "compact"  # "compact" or "debug"

AGENT_PATH = Path("results") / "PretrainOFMPC" / "20260621_203346" / "of_mpc_pretrained_td3_20260622_030149.pkl"
RESET_PRETRAINED_CRITIC = True
RL_OBSERVATION_MODE = "standard"
PROJECTION_BACKEND = "direct_accept_or_fallback"
REWARD_FALLBACK_PENALTY_ENABLED = False
GAMMA_FALLBACK = 0.0
FALLBACK_EVENT_PENALTY = 0.0
RHO_LYAP = online_runner.GART_FINAL_RHO_LYAP
LYAP_EPS = online_runner.GART_FINAL_LYAP_EPS
LYAP_TOL = None

ACTOR_LAYER_SIZES = (256, 256, 256)
CRITIC_LAYER_SIZES = (256, 256, 256)
REPLAY_BUFFER_CAPACITY = 80000
BATCH_SIZE = 256
ACTOR_LR = 5e-5
CRITIC_LR = 5e-4

TARGET_POLICY_SMOOTHING_NOISE_STD = 0.02
TARGET_POLICY_NOISE_CLIP = 0.04
EXPLORATION_STD_START = 0.02
EXPLORATION_STD_END = 0.005
TEACHER_SOURCE = "gart_lmpc"
TEACHER_CRITIC_WARMUP_EPISODES = 10
BC_EXPLORATION_STD = 0.02
HANDOFF_EXPLORATION_STD_START = 0.02
HANDOFF_EXPLORATION_STD_END = 0.005
HANDOFF_EPISODES = 10

PHASE1_SETPOINTS_Y_PHYS = (
    (4.5, 324.0),
    (3.4, 321.0),
)
PHASE2_SETPOINTS_Y_PHYS = (
    (3.35, 323.5),
)

NOMINAL_QI = 108.0
NOMINAL_QS = 459.0
NOMINAL_HA = 1.05e6

PHASE1_QI_MULTIPLIER = 0.95
PHASE1_QS_MULTIPLIER = 1.05
PHASE1_HA_MULTIPLIER = 0.92

PHASE2_QI_MULTIPLIER = 1.05
PHASE2_QS_MULTIPLIER = 0.95
PHASE2_HA_MULTIPLIER = 0.88


def _seed_arg() -> str | None:
    if SEEDS is None:
        return None
    return ",".join(str(seed) for seed in SEEDS)


def _timestamp() -> str:
    if TIMESTAMP is not None:
        return str(TIMESTAMP)
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{METHOD}"


def _build_args() -> Namespace:
    return Namespace(
        n_seeds=int(N_SEEDS),
        seed_start=int(SEED_START),
        seeds=_seed_arg(),
        methods=METHOD,
        output_root=str(OUTPUT_ROOT),
        timestamp=_timestamp(),
        save_plots=bool(SAVE_PLOTS),
        export_profile=str(EXPORT_PROFILE),
        agent_path=None if AGENT_PATH is None else str(AGENT_PATH),
        reset_pretrained_critic=bool(RESET_PRETRAINED_CRITIC),
        rl_observation_mode=RL_OBSERVATION_MODE,
        projection_backend=PROJECTION_BACKEND,
        reward_fallback_penalty_enabled=bool(REWARD_FALLBACK_PENALTY_ENABLED),
        gamma_fallback=float(GAMMA_FALLBACK),
        fallback_event_penalty=float(FALLBACK_EVENT_PENALTY),
        rho_lyap=float(RHO_LYAP),
        lyap_eps=float(LYAP_EPS),
        lyap_tol=LYAP_TOL,
        training_phase_overrides=_training_phase_overrides(),
        phase1_episodes=int(PHASE1_EPISODES),
        phase2_episodes=int(PHASE2_EPISODES),
        set_points_len=int(SET_POINTS_LEN),
        phase1_setpoints_y_phys=PHASE1_SETPOINTS_Y_PHYS,
        phase2_setpoints_y_phys=PHASE2_SETPOINTS_Y_PHYS,
        nominal_qi=float(NOMINAL_QI),
        nominal_qs=float(NOMINAL_QS),
        nominal_ha=float(NOMINAL_HA),
        phase1_qi_multiplier=float(PHASE1_QI_MULTIPLIER),
        phase1_qs_multiplier=float(PHASE1_QS_MULTIPLIER),
        phase1_ha_multiplier=float(PHASE1_HA_MULTIPLIER),
        phase2_qi_multiplier=float(PHASE2_QI_MULTIPLIER),
        phase2_qs_multiplier=float(PHASE2_QS_MULTIPLIER),
        phase2_ha_multiplier=float(PHASE2_HA_MULTIPLIER),
    )


def _apply_td3_defaults() -> None:
    online_runner.DEFAULT_ACTOR_LAYER_SIZES = tuple(int(v) for v in ACTOR_LAYER_SIZES)
    online_runner.DEFAULT_CRITIC_LAYER_SIZES = tuple(int(v) for v in CRITIC_LAYER_SIZES)
    online_runner.BUFFER_CAPACITY = int(REPLAY_BUFFER_CAPACITY)
    online_runner.BATCH_SIZE = int(BATCH_SIZE)
    online_runner.ACTOR_LR = float(ACTOR_LR)
    online_runner.CRITIC_LR = float(CRITIC_LR)
    online_runner.PRETRAINED_SMOOTHING_STD = float(TARGET_POLICY_SMOOTHING_NOISE_STD)
    online_runner.PRETRAINED_NOISE_CLIP = float(TARGET_POLICY_NOISE_CLIP)
    online_runner.PRETRAINED_EXPLORATION_STD_START = float(EXPLORATION_STD_START)
    online_runner.GLOBAL_EXPLORATION_STD_END = float(EXPLORATION_STD_END)
    online_runner.PRETRAINED_BC_EXPLORATION_STD = float(BC_EXPLORATION_STD)
    online_runner.PRETRAINED_HANDOFF_EXPLORATION_STD_START = float(HANDOFF_EXPLORATION_STD_START)
    online_runner.PRETRAINED_HANDOFF_EXPLORATION_STD_END = float(HANDOFF_EXPLORATION_STD_END)
    online_runner.PRETRAINED_HANDOFF_EPISODES = int(HANDOFF_EPISODES)


def _training_phase_overrides() -> dict:
    overrides = online_runner.noisy_teacher_critic_warmup_overrides(
        teacher_source=TEACHER_SOURCE,
        pretrained=True,
        teacher_episodes=int(TEACHER_CRITIC_WARMUP_EPISODES),
        handoff_episodes=int(HANDOFF_EPISODES),
    )
    overrides.update(
        {
            "bc_exploration_std": float(BC_EXPLORATION_STD),
            "handoff_exploration_std_start": float(HANDOFF_EXPLORATION_STD_START),
            "handoff_exploration_std_end": float(HANDOFF_EXPLORATION_STD_END),
            "full_rl_exploration_std_start": float(EXPLORATION_STD_START),
            "full_rl_exploration_std_end": float(EXPLORATION_STD_END),
            "exploration_std_start": float(EXPLORATION_STD_START),
            "exploration_std_end": float(EXPLORATION_STD_END),
        }
    )
    return overrides


def run_configured_study() -> dict:
    _apply_td3_defaults()
    args = _build_args()
    print("Two-phase OF-MPC-pretrained safety-gate configuration:")
    pprint(vars(args))
    return run_two_phase_study(args)


if __name__ == "__main__":
    run_configured_study()
