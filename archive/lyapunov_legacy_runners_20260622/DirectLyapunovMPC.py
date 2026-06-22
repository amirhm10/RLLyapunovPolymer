# Auto-converted from Jupyter notebook.
# Source notebook archived at: archive/DirectLyapunovMPC_FourMethodDisturbance.ipynb
# Notebook outputs are intentionally not included.

# %% [markdown]
# # Direct Lyapunov MPC Four-Method Disturbance Study
#
# This notebook runs the direct output-disturbance Lyapunov MPC across the four bounded target variants and saves case-level bundles plus a cross-method comparison export.

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

# %%
from TD3Agent.reward_functions import make_reward_fn_relative_QR
from Simulation.mpc import MpcSolver, compute_observer_gain
from Simulation.system_functions import PolymerCSTR
from Lyapunov.direct_lyapunov_mpc import (
    build_direct_lyapunov_run_bundle,
    design_direct_lyapunov_mpc_solver,
    make_direct_lyapunov_comparison_record,
    run_direct_output_disturbance_lyapunov_mpc,
    run_offset_free_mpc_with_direct_diagnostics,
    save_direct_lyapunov_comparison_artifacts,
    save_direct_lyapunov_debug_artifacts,
)
from utils.direct_lyapunov_study import (
    # DIRECT_DISTURBANCE_N_TESTS,
    # DIRECT_DISTURBANCE_SETPOINT_LEN,
    DIRECT_TWO_SETPOINT_Y_PHYS,
    direct_disturbance_test_cycle,
    governed_reference_case_spec,
)
from utils.scaling_helpers import apply_min_max
from utils.td3_helpers import load_and_prepare_system_data

# Direct Lyapunov four-method disturbance study configuration
predict_h = 9
cont_h = 3
rho_lyap = 0.99
lyap_eps = 5e-3
slack_penalty = 1e6
use_target_on_solver_fail = False
plant_mode = "disturb"
disturbance_after_step = False
use_target_output_for_tracking = False
save_case_bundles = True
save_case_plots = True



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

n_episodes = 300
n_tests = n_episodes
# set_points_len = DIRECT_DISTURBANCE_SETPOINT_LEN
set_points_len = 400
TEST_CYCLE = direct_disturbance_test_cycle(n_tests)
warm_start = 0

study_name = "directLyap"
study_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
study_root = os.path.join(os.fspath(repo_path()), "results", study_name, study_timestamp)
os.makedirs(study_root, exist_ok=True)

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

u_ss_scaled = apply_min_max(steady_states["ss_inputs"], data_min[:inputs_number], data_max[:inputs_number])
u_min_scaled = apply_min_max(u_min, data_min[:inputs_number], data_max[:inputs_number])
u_max_scaled = apply_min_max(u_max, data_min[:inputs_number], data_max[:inputs_number])

u_dev_min = u_min_scaled - u_ss_scaled
u_dev_max = u_max_scaled - u_ss_scaled
bnds = tuple((float(lo), float(hi)) for lo, hi in zip(u_dev_min, u_dev_max)) * cont_h
IC_opt_template = np.zeros(inputs_number * cont_h)

Qy_diag = np.array([5.0, 1.0])
Su_diag = np.array([1.0, 1.0])
Rdu_diag = np.array([1.0, 1.0])

k_rel = np.array([0.003, 0.0003])
band_floor_phys = np.array([0.006, 0.07])
reward_config, reward_fn = make_reward_fn_relative_QR(
    data_min=data_min,
    data_max=data_max,
    n_inputs=inputs_number,
    k_rel=k_rel,
    band_floor_phys=band_floor_phys,
    Q_diag=Qy_diag,
    R_diag=Rdu_diag,
    tau_frac=0.7,
    gamma_out=0.5,
    gamma_in=0.5,
    beta=7.0,
    gate="geom",
    lam_in=1.0,
    bonus_kind="exp",
    bonus_k=12.0,
    bonus_p=0.6,
    bonus_c=20.0,
)

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

# Governed-reference is the active target selector default.
# The MPC stage objective still tracks the raw setpoint, not y_s.
u_prev_penalty_weight = 0.0
xs_prev_penalty_weight = 0.0
governed_reference_target_config = governed_reference_case_spec(
    Qy_diag,
    case_name="lyap_governed_reference",
    label="Governed-reference LyapMPC",
    u_ref_weight=u_prev_penalty_weight,
    x_ref_weight=xs_prev_penalty_weight,
)["target_config"]

active_config = {
    "predict_h": predict_h,
    "cont_h": cont_h,
    "rho_lyap": rho_lyap,
    "lyap_eps": lyap_eps,
    "slack_penalty": slack_penalty,
    "n_tests": n_tests,
    "set_points_len": set_points_len,
    "plant_mode": plant_mode,
    "disturbance_after_step": disturbance_after_step,
    "use_target_output_for_tracking": use_target_output_for_tracking,
    "study_name": study_name,
    "u_prev_penalty_weight": u_prev_penalty_weight,
    "xs_prev_penalty_weight": xs_prev_penalty_weight,
    "setpoint_y_phys": setpoint_y_phys.tolist(),
    "governed_reference_target_config": dict(governed_reference_target_config),
}
case_specs = [
    governed_reference_case_spec(
        Qy_diag,
        case_name="lyap_governed_reference",
        label="Governed-reference LyapMPC",
        u_ref_weight=u_prev_penalty_weight,
        x_ref_weight=xs_prev_penalty_weight,
    ),
    governed_reference_case_spec(
        Qy_diag,
        case_name="mpc_only",
        controller_mode="mpc_only",
        lyapunov_mode="diagnostic_only",
        label="Offset-free MPC with governed-reference diagnostics",
        u_ref_weight=u_prev_penalty_weight,
        x_ref_weight=xs_prev_penalty_weight,
    ),
]

print("Effective no-RL direct case specs. Governed-reference target selector is the default.")
for spec in case_specs:
    print(
        f"  {spec['case_name']}: controller={spec.get('controller_mode', 'direct_lyapunov_mpc')}, "
        f"target_mode={spec['target_mode']}, target_config={spec.get('target_config', {})}"
    )

# %%
def run_case(case_spec):
    case_name = case_spec["case_name"]
    case_target_mode = case_spec["target_mode"]
    case_lyapunov_mode = case_spec.get("lyapunov_mode", "diagnostic_only")
    case_target_config = dict(case_spec.get("target_config", {}))
    controller_mode = case_spec.get("controller_mode", "direct_lyapunov_mpc")
    is_mpc_only = controller_mode == "mpc_only"
    case_config = {
        **active_config,
        "case_name": case_name,
        "controller_mode": controller_mode,
        "target_mode": case_target_mode,
        "lyapunov_mode": case_lyapunov_mode,
        "target_config": case_target_config,
    }
    print(
        f"Running {case_name}: controller_mode={controller_mode}, "
        f"target_mode={case_target_mode}, lyapunov_mode={case_lyapunov_mode}, "
        f"target_config={case_target_config}"
    )

    cstr_case = PolymerCSTR(
        system_params,
        system_design_params,
        system_steady_state_inputs,
        delta_t,
        deviation_form=False,
    )
    case_timer_start = time.perf_counter()
    if is_mpc_only:
        results_case = run_offset_free_mpc_with_direct_diagnostics(
            system=cstr_case,
            MPC_obj=MPC_obj_offset_free,
            diagnostic_LMPC_obj=LMPC_obj,
            y_sp_scenario=y_sp_scenario,
            n_tests=n_tests,
            set_points_len=set_points_len,
            steady_states=steady_states,
            IC_opt=IC_opt_template.copy(),
            bnds=bnds,
            L=L,
            data_min=data_min,
            data_max=data_max,
            test_cycle=TEST_CYCLE,
            reward_fn=reward_fn,
            nominal_qi=nominal_qi,
            nominal_qs=nominal_qs,
            nominal_ha=nominal_hA,
            qi_change=qi_change,
            qs_change=qs_change,
            ha_change=ha_change,
            target_mode=case_target_mode,
            target_config=case_target_config,
            target_H=None,
            mode=plant_mode,
            disturbance_after_step=disturbance_after_step,
            use_target_output_for_tracking=use_target_output_for_tracking,
            rho_lyap=rho_lyap,
            lyap_eps=lyap_eps,
            first_step_contraction_on=True,
            reset_system_on_entry=True,
            solver_options={"warm_start": True},
        )
    else:
        results_case = run_direct_output_disturbance_lyapunov_mpc(
            system=cstr_case,
            LMPC_obj=LMPC_obj,
            y_sp_scenario=y_sp_scenario,
            n_tests=n_tests,
            set_points_len=set_points_len,
            steady_states=steady_states,
            IC_opt=IC_opt_template.copy(),
            bnds=bnds,
            L=L,
            data_min=data_min,
            data_max=data_max,
            test_cycle=TEST_CYCLE,
            reward_fn=reward_fn,
            nominal_qi=nominal_qi,
            nominal_qs=nominal_qs,
            nominal_ha=nominal_hA,
            qi_change=qi_change,
            qs_change=qs_change,
            ha_change=ha_change,
            target_mode=case_target_mode,
            lyapunov_mode=case_lyapunov_mode,
            target_config=case_target_config,
            target_H=None,
            mode=plant_mode,
            disturbance_after_step=disturbance_after_step,
            use_target_output_for_tracking=use_target_output_for_tracking,
            skip_terminal_if_alpha_small=True,
            alpha_terminal_min=1e-8,
            use_target_on_solver_fail=use_target_on_solver_fail,
            rho_lyap=rho_lyap,
            lyap_eps=lyap_eps,
            slack_penalty=slack_penalty,
            first_step_contraction_on=True,
            reset_system_on_entry=True,
            solver_options={"warm_start": True},
        )
    case_wall_clock_seconds = float(time.perf_counter() - case_timer_start)
    case_steps = int(results_case["nFE"])
    case_episode_len = int(results_case["time_in_sub_episodes"])
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
    bundle_case = build_direct_lyapunov_run_bundle(
        source=case_name,
        results=results_case,
        steady_states=steady_states,
        config=case_config,
        data_min=data_min,
        data_max=data_max,
        extra={"reward_config": reward_config, "min_max_dict": min_max_dict, "timing": timing_metadata},
    )

    debug_dir_case = save_direct_lyapunov_debug_artifacts(
        bundle_case,
        directory=study_root,
        prefix_name=case_name,
        save_plots=save_case_plots,
    )
    record_case = make_direct_lyapunov_comparison_record(case_name, bundle_case, debug_dir_case)
    record_case.update(timing_metadata)
    print(f"Completed {case_name}")
    pprint(record_case)
    return results_case, bundle_case, debug_dir_case, record_case

# %%
results_by_case = {}
bundles_by_case = {}
debug_dirs_by_case = {}
comparison_records = []

for case_spec in case_specs:
    results_case, bundle_case, debug_dir_case, record_case = run_case(case_spec)
    case_name = case_spec["case_name"]
    results_by_case[case_name] = results_case
    bundles_by_case[case_name] = bundle_case
    debug_dirs_by_case[case_name] = debug_dir_case
    comparison_records.append(record_case)

comparison_artifacts = save_direct_lyapunov_comparison_artifacts(
    comparison_records,
    bundles_by_case,
    study_root,
    save_plots=True,
)

print("Saved direct four-method study:")
pprint(comparison_artifacts)
comparison_df = pd.DataFrame(comparison_records) if pd is not None else comparison_records
comparison_df

