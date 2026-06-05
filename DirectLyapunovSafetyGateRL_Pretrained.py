# Auto-converted from Jupyter notebook.
# Source notebook archived at: archive/DirectLyapunovSafetyGateRL_Pretrained.ipynb
# Notebook outputs are intentionally not included.

# %% [markdown]
# # Direct Lyapunov Safety-Gate RL Study (Pretrained)
#
# This notebook evaluates the exact direct Lyapunov target family as a binary RL safety gate: accept the TD3 action if it passes the direct Lyapunov check, otherwise execute the direct Lyapunov MPC fallback. Each method starts from the same pretrained checkpoint.

# %%
from utils.path_helpers import repo_path
import os
import time
from datetime import datetime
from pprint import pprint

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

import torch

# %%
from TD3Agent.agent import TD3Agent
from TD3Agent.reward_functions import make_reward_fn_relative_QR
from Simulation.mpc import MpcSolver, compute_observer_gain
from Simulation.run_rl_lyapunov import run_rl_train
from Simulation.system_functions import PolymerCSTR
from Lyapunov.direct_lyapunov_mpc import design_direct_lyapunov_mpc_solver
from Lyapunov.safety_debug import (
    build_safety_filter_run_bundle,
    make_safety_filter_comparison_record,
    save_safety_filter_comparison_artifacts,
    save_safety_filter_debug_artifacts,
)
from utils.direct_lyapunov_study import (
    DIRECT_DISTURBANCE_N_TESTS,
    DIRECT_DISTURBANCE_SEED,
    DIRECT_DISTURBANCE_SETPOINT_LEN,
    DIRECT_TWO_SETPOINT_Y_PHYS,
    direct_disturbance_test_cycle,
    governed_reference_case_spec,
)
from utils.scaling_helpers import apply_min_max
from utils.td3_helpers import load_and_prepare_system_data

predict_h = 9
cont_h = 3
rho_lyap = 0.98
lyap_eps = 1e-5
lyap_tol = 1e-10
slack_penalty = 1e6
plant_mode = "disturb"
disturbance_after_step = False

# Governed-reference target regularization weights for RL runs.
u_prev_penalty_weight = 0.0
xs_prev_penalty_weight = 0.0

Ad = 2.142e17
Ed = 14897
Ap = 3.816e10
Ep = 3557
At = 4.50e12
Et = 843
fi = 0.6
m_delta_H_r = -6.99e4
hA = 1.05e6
rhocp = 1506
rhoccpc = 4043
Mm = 104.14
system_params = np.array([Ad, Ed, Ap, Ep, At, Et, fi, m_delta_H_r, hA, rhocp, rhoccpc, Mm])

CIf = 0.5888
CMf = 8.6981
Qi = 108.0
Qs = 459.0
Tf = 330.0
Tcf = 295.0
V = 3000.0
Vc = 3312.4
system_design_params = np.array([CIf, CMf, Qi, Qs, Tf, Tcf, V, Vc])

Qm_ss = 378.0
Qc_ss = 471.6
system_steady_state_inputs = np.array([Qc_ss, Qm_ss])
delta_t = 0.5

steady_states = {"ss_inputs": system_steady_state_inputs.copy()}
cstr_ss = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t, deviation_form=False)
steady_states["y_ss"] = cstr_ss.y_ss.copy()

u_min = np.array([71.6, 78.0])
u_max = np.array([870.0, 670.0])
setpoint_y_phys = DIRECT_TWO_SETPOINT_Y_PHYS.copy()

n_tests = DIRECT_DISTURBANCE_N_TESTS
set_points_len = DIRECT_DISTURBANCE_SETPOINT_LEN
TEST_CYCLE = direct_disturbance_test_cycle(n_tests)
FORCE_FINAL_TEST = True
warm_start = 0
WARMUP_EPISODES = 0
BC_TEACHER_EPISODES = 20
time_in_sub_episodes = setpoint_y_phys.shape[0] * set_points_len
ACTOR_FREEZE = 0 * set_points_len
phase_plot_boundaries = np.array(
    [
        (WARMUP_EPISODES + BC_TEACHER_EPISODES) * time_in_sub_episodes,
    ],
    dtype=int,
)
training_phase_config = {
    "episode_unit": "cycle",
    "warmup_buffer_only_episodes": WARMUP_EPISODES,
    "behavior_clone_teacher_episodes": BC_TEACHER_EPISODES,
    "bc_actor_updates_per_step": 4,
    "bc_exploration_std": 0.02,
    "full_rl_exploration_std_start": 0.02,
    "full_rl_exploration_std_end": 0.01,
    "full_rl_exploration_decay_mode": "linear",
    "bc_teacher_policy": "direct_lyapunov_mpc",
    "warmup_behavior_source": "direct_lyapunov_mpc",
    "bc_behavior_source": "policy_with_lmpc_teacher_demo",
    "handoff_episodes": 5,
    "handoff_blend": "linear",
    "warmup_behavior_noise": "none",
    "bc_behavior_noise": "gaussian",
    "full_rl_behavior_noise": "gaussian",
}

study_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(os.fspath(repo_path()), "results"), exist_ok=True)

system_data = load_and_prepare_system_data(
    steady_states=steady_states,
    setpoint_y=setpoint_y_phys,
    u_min=u_min,
    u_max=u_max,
    system_dict_path=os.path.join("Data", "system_dict"),
    augmentation_style="rawlings",
    augmentation_mode="output_disturbance",
)

A_aug = system_data["A_aug"]
B_aug = system_data["B_aug"]
C_aug = system_data["C_aug"]
data_min = system_data["data_min"]
data_max = system_data["data_max"]
min_max_dict = system_data["min_max_dict"]

inputs_number = int(B_aug.shape[1])
y_sp_scenario = apply_min_max(setpoint_y_phys, data_min[inputs_number:], data_max[inputs_number:]) - apply_min_max(
    steady_states["y_ss"],
    data_min[inputs_number:],
    data_max[inputs_number:],
)

poles = np.array([
    0.44619852,
    0.33547649,
    0.36380595,
    0.70467118,
    0.3562966,
    0.42900673,
    0.4228262,
    0.96916776,
    0.91230187,
])
L = compute_observer_gain(A_aug, C_aug, poles)

set_points_number = int(C_aug.shape[0])
STATE_DIM = int(A_aug.shape[0]) + set_points_number + inputs_number
ACTION_DIM = int(B_aug.shape[1])
ACTOR_LAYER_SIZES = [512, 512, 512, 512, 512]
CRITIC_LAYER_SIZES = [512, 512, 512, 512, 512]
BUFFER_CAPACITY = 40000
ACTOR_LR = 5e-5
CRITIC_LR = 5e-4
SMOOTHING_STD = 0.01
NOISE_CLIP = 0.01
GAMMA = 0.99
TAU = 0.005
MAX_ACTION = 1
POLICY_DELAY = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
STD_START = 0.0
STD_END = 0.0
STD_DECAY_RATE = 0.99992
STD_DECAY_MODE = "exp"

Qy_diag = np.array([12.0, 6.0])
Su_diag = np.array([1.0, 1.0])
Rdu_diag = np.array([1.0, 1.0])
k_rel = np.array([0.0015, 0.00015])
band_floor_phys = np.array([0.003, 0.035])
gamma_fallback = 3.0
fallback_event_penalty = 10.0
reward_config, reward_fn = make_reward_fn_relative_QR(
    data_min=data_min,
    data_max=data_max,
    n_inputs=inputs_number,
    k_rel=k_rel,
    band_floor_phys=band_floor_phys,
    Q_diag=Qy_diag,
    R_diag=Rdu_diag,
    tau_frac=0.5,
    gamma_out=1.0,
    gamma_in=3.0,
    beta=1.0,
    gate="geom",
    lam_in=3.0,
    bonus_kind="quadratic",
    gamma_fallback=gamma_fallback,
    fallback_event_penalty=fallback_event_penalty,
    R_fallback_diag=Rdu_diag,
    maintenance_band_scale=0.5,
    maintenance_move_weight=0.0,
    jitter_weight=0.0,
    dwell_bonus=0.0,
)

u_ss = apply_min_max(steady_states["ss_inputs"], data_min[:inputs_number], data_max[:inputs_number])
b_min = apply_min_max(u_min, data_min[:inputs_number], data_max[:inputs_number])
b_max = apply_min_max(u_max, data_min[:inputs_number], data_max[:inputs_number])
b1 = (b_min[0] - u_ss[0], b_max[0] - u_ss[0])
b2 = (b_min[1] - u_ss[1], b_max[1] - u_ss[1])
bnds = (b1, b2) * cont_h
IC_opt_template = np.zeros(inputs_number * cont_h)

u_min_scaled = apply_min_max(u_min, data_min[:inputs_number], data_max[:inputs_number])
u_max_scaled = apply_min_max(u_max, data_min[:inputs_number], data_max[:inputs_number])
u_dev_min = u_min_scaled - u_ss
u_dev_max = u_max_scaled - u_ss

LMPC_obj = design_direct_lyapunov_mpc_solver(
    A_aug=A_aug,
    B_aug=B_aug,
    C_aug=C_aug,
    Qy_diag=Qy_diag,
    NP=predict_h,
    NC=cont_h,
    Su_diag=Su_diag,
    u_min=u_dev_min,
    u_max=u_dev_max,
    Rdu_diag=Rdu_diag,
    terminal_set_on=True,
    terminal_alpha_scale=1.0,
)
MPC_obj = LMPC_obj  # Backward-compatible alias for direct safety-gate cases.
MPC_obj_offset_free = MpcSolver(
    A_aug,
    B_aug,
    C_aug,
    Q_out=Qy_diag,
    R_in=Rdu_diag,
    NP=predict_h,
    NC=cont_h,
)

nominal_qs = 459.0
nominal_qi = 108.0
nominal_hA = 1.05e6
qi_change = 0.95
qs_change = 1.05
ha_change = 0.92

case_specs = [
    governed_reference_case_spec(
        Qy_diag,
        case_name="rl_gate_governed_reference",
        controller_mode="direct_safety_gate",
        label="Pretrained RL with governed-reference safety gate",
        u_ref_weight=u_prev_penalty_weight,
        x_ref_weight=xs_prev_penalty_weight,
    ),
]


def make_td3_agent():
    return TD3Agent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        actor_hidden=ACTOR_LAYER_SIZES,
        critic_hidden=CRITIC_LAYER_SIZES,
        gamma=GAMMA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        batch_size=BATCH_SIZE,
        policy_delay=POLICY_DELAY,
        target_policy_smoothing_noise_std=SMOOTHING_STD,
        noise_clip=NOISE_CLIP,
        max_action=MAX_ACTION,
        tau=TAU,
        std_start=STD_START,
        std_end=STD_END,
        std_decay_rate=STD_DECAY_RATE,
        std_decay_mode=STD_DECAY_MODE,
        buffer_size=BUFFER_CAPACITY,
        device=DEVICE,
        actor_freeze=ACTOR_FREEZE,
    )

# %%
SAVE_TRAINED_AGENT = True

study_name = "Pretrain"
study_root = os.path.join(os.fspath(repo_path()), "results", study_name, study_timestamp)
os.makedirs(study_root, exist_ok=True)

agent_path = os.path.join(os.fspath(repo_path()), "Data", "agent_2507171027.pkl")
if not os.path.exists(agent_path):
    raise FileNotFoundError(f"TD3 checkpoint not found: {agent_path}")


def run_case(case_spec):
    case_name = case_spec["case_name"]
    case_target_config = dict(case_spec.get("target_config", {}))
    controller_mode = case_spec.get("controller_mode", "direct_safety_gate")
    is_mpc_only = controller_mode == "mpc_only"

    case_projection_backend = "mpc_only" if is_mpc_only else "direct_accept_or_fallback"
    case_use_lyap = not is_mpc_only
    case_mpc_obj = MPC_obj_offset_free if is_mpc_only else LMPC_obj
    case_training_phase_config = dict(training_phase_config)
    if is_mpc_only:
        case_training_phase_config.update(
            {
                "bc_teacher_policy": "offset_free_mpc",
                "bc_behavior_source": "offset_free_mpc",
                "warmup_behavior_source": "offset_free_mpc",
            }
        )

    case_agent = make_td3_agent()
    case_agent.load(agent_path)
    cstr_case = PolymerCSTR(
        system_params,
        system_design_params,
        system_steady_state_inputs,
        delta_t,
        deviation_form=False,
    )

    case_config = {
        "study_name": study_name,
        "case_name": case_name,
        "controller_mode": controller_mode,
        "projection_backend": case_projection_backend,
        "target_mode": case_spec["target_mode"],
        "target_config": case_target_config,
        "u_prev_penalty_weight": u_prev_penalty_weight,
        "xs_prev_penalty_weight": xs_prev_penalty_weight,
        "rho_lyap": rho_lyap,
        "lyap_eps": lyap_eps,
        "gamma_fallback": gamma_fallback,
        "fallback_event_penalty": fallback_event_penalty,
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "force_final_test": FORCE_FINAL_TEST,
        "disturbance_after_step": disturbance_after_step,
        "training_phase_config": dict(case_training_phase_config),
        "initial_agent_path": agent_path,
    }

    case_timer_start = time.perf_counter()
    results_case = run_rl_train(
        system=cstr_case,
        y_sp_scenario=y_sp_scenario,
        n_tests=n_tests,
        set_points_len=set_points_len,
        steady_states=steady_states,
        min_max_dict=min_max_dict,
        agent=case_agent,
        MPC_obj=case_mpc_obj,
        L=L,
        data_min=data_min,
        data_max=data_max,
        warm_start=warm_start,
        test_cycle=TEST_CYCLE,
        nominal_qi=nominal_qi,
        nominal_qs=nominal_qs,
        nominal_ha=nominal_hA,
        qi_change=qi_change,
        qs_change=qs_change,
        ha_change=ha_change,
        reward_fn=reward_fn,
        mode=plant_mode,
        rho_lyap=rho_lyap,
        lyap_eps=lyap_eps,
        lyap_tol=lyap_tol,
        seed=DIRECT_DISTURBANCE_SEED,
        use_lyap=case_use_lyap,
        IC_opt=IC_opt_template.copy(),
        bnds=bnds,
        cons=(),
        reuse_mpc_solution_as_ic=False,
        reset_system_on_entry=True,
        projection_backend=case_projection_backend,
        first_step_contraction_on=True,
        direct_target_mode=case_spec["target_mode"],
        direct_target_config=case_target_config,
        direct_tracking_use_target_output=False,
        diagnostic_lmpc_obj=LMPC_obj,
        disturbance_after_step=disturbance_after_step,
        training_phase_config=case_training_phase_config,
        force_final_test=FORCE_FINAL_TEST,
    )
    case_wall_clock_seconds = float(time.perf_counter() - case_timer_start)
    case_steps = int(results_case[5])
    case_episode_len = int(results_case[6])
    case_episodes = int(np.ceil(case_steps / float(case_episode_len))) if case_episode_len > 0 else 0
    timing_metadata = {
        "wall_clock_seconds": case_wall_clock_seconds,
        "wall_clock_seconds_per_episode": (
            None if case_episodes <= 0 else case_wall_clock_seconds / float(case_episodes)
        ),
        "wall_clock_seconds_per_step": (
            None if case_steps <= 0 else case_wall_clock_seconds / float(case_steps)
        ),
        "wall_clock_steps_per_second": (
            None if case_wall_clock_seconds <= 0.0 else case_steps / case_wall_clock_seconds
        ),
        "wall_clock_n_steps": case_steps,
        "wall_clock_n_episodes": case_episodes,
    }
    case_config.update(timing_metadata)

    bundle_case = build_safety_filter_run_bundle(
        source=case_name,
        results=results_case,
        steady_states=steady_states,
        config=case_config,
        min_max_dict=min_max_dict,
        data_min=data_min,
        data_max=data_max,
        extra={
            "delta_t": delta_t,
            "phase_plot_boundaries": phase_plot_boundaries,
            "start_plot_idx": 10,
            "agent_path": agent_path,
            "reward_config": reward_config,
            "actor_losses": case_agent.actor_losses,
            "critic_losses": case_agent.critic_losses,
            "timing": timing_metadata,
        },
    )
    debug_dir_case = save_safety_filter_debug_artifacts(
        bundle=bundle_case,
        directory=study_root,
        prefix_name=case_name,
        save_plots=True,
    )
    record_case = make_safety_filter_comparison_record(case_name, bundle_case, debug_dir_case)
    if SAVE_TRAINED_AGENT:
        trained_agent_path = case_agent.save(debug_dir_case, prefix="trained_agent", include_optim=False)
        record_case["trained_agent_path"] = trained_agent_path
        bundle_case["extra"]["trained_agent_path"] = trained_agent_path
        bundle_case["config"]["trained_agent_path"] = trained_agent_path
    record_case.update(timing_metadata)
    print(f"Completed {case_name}")
    pprint(record_case)
    return bundle_case, debug_dir_case, record_case


bundles_by_case = {}
debug_dirs_by_case = {}
comparison_records = []
for case_spec in case_specs:
    bundle_case, debug_dir_case, record_case = run_case(case_spec)
    case_name = case_spec["case_name"]
    bundles_by_case[case_name] = bundle_case
    debug_dirs_by_case[case_name] = debug_dir_case
    comparison_records.append(record_case)

comparison_artifacts = save_safety_filter_comparison_artifacts(
    comparison_records,
    bundles_by_case,
    study_root,
    save_plots=True,
)
print("Saved pretrained RL direct-gate study:")
pprint(comparison_artifacts)
comparison_df = pd.DataFrame(comparison_records) if pd is not None else comparison_records
comparison_df

# %%
