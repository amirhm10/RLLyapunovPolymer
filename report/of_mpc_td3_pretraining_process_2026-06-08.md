# OF-MPC TD3 Pretraining Process

Date: 2026-06-08

## Objective

This report documents the repo-native offset-free MPC pretraining workflow for the TD3 agent in `Lyapunov_polymer`. The workflow now follows the `Polymer_example` organization:

- reusable setup, controller, training, artifact, and comparison helpers live in `utils/of_mpc_td3_workflow.py`
- `PretrainTD3OffsetFreeMPC.py` is only the pretraining and saving runner
- `ComparePretrainedTD3OffsetFreeMPC.py` is only the saved-agent versus OF-MPC comparison runner

This is a process report. It does not claim new closed-loop performance unless a generated checkpoint is later evaluated and reported.

## Coordinates

The workflow uses scaled-deviation coordinates.

- Manipulated inputs are $u = [Q_c, Q_m]^\top$.
- Outputs are $y = [\eta, T]^\top$.
- The steady-state input anchor is $u_{ss} = [471.6, 378.0]^\top$.
- The output steady state is computed from `PolymerCSTR`.
- Inputs and outputs are min-max scaled, and the controller operates on deviations from the steady-state anchors.

The TD3 state is

$$
s_k =
\begin{bmatrix}
\tilde{x}_{d,k} \\
\tilde{y}_{sp,k} \\
\tilde{u}_{k-1}
\end{bmatrix},
$$

where $\tilde{x}_{d,k}$ is the augmented state mapped to $[-1, 1]$, $\tilde{y}_{sp,k}$ is the setpoint deviation mapped to $[-1, 1]$, and $\tilde{u}_{k-1}$ is the previous input deviation mapped to $[-1, 1]$.

The dimensions are not hard-coded. They are computed from the augmented matrices:

```python
inputs_number = int(B_aug.shape[1])
set_points_number = int(C_aug.shape[0])
state_dim = int(A_aug.shape[0]) + set_points_number + inputs_number
action_dim = inputs_number
```

For the current polymer model this gives `state_dim = 13` and `action_dim = 2`, but those values are consequences of the matrices, not separate workflow constants.

The TD3 normalization ranges are centralized in `utils/polymer_td3_defaults.py` and are used by the pretraining runner, comparison runner, and active Lyapunov pretrained workflow through `utils.td3_helpers.load_and_prepare_system_data(...)`. The default augmented-state envelope is:

```python
x_max = [256.79686253, 256.01560603, 48.99447186, 144.79949103,
         2.82199733, 3.14014989, 2.78866348, 3.71691422, 6.2029936]
x_min = [-272.28060121, -1112.33972595, -76.63993491, -608.60327886,
         -3.94399122, -3.93115257, -2.9532091, -4.06547624, -28.25906582]
```

The default TD3 setpoint-scaling envelope is the broad Polymer-example pretraining envelope used by the saved `Data/agent_2507171027.pkl` checkpoint:

```python
[[2.8, 320.0],
 [5.0, 326.0]]
```

The direct two-setpoint polymer schedule remains the default rollout and comparison scenario:

```python
[[4.5, 324.0],
 [3.4, 321.0]]
```

This separation is important: the TD3 state feature $\tilde{y}_{sp,k}$ is scaled with the broad pretraining envelope, while the actual commanded setpoint sequence can be the direct Lyapunov comparison schedule.

## Offset-Free Expert

The pretraining expert is offset-free MPC, not Direct Lyapunov MPC. The augmentation is the Rawlings output-disturbance model:

$$
\begin{aligned}
x_{k+1} &= A x_k + B u_k + B_d d_k, \\
d_{k+1} &= d_k, \\
y_k &= C x_k + C_d d_k .
\end{aligned}
$$

The selected mode is output disturbance:

$$
B_d = 0, \qquad C_d = I.
$$

The OF-MPC expert uses:

- prediction horizon `NP = 9`
- control horizon `NC = 3`
- output weights $Q = \operatorname{diag}(5, 1)$
- input-move weights $R = \operatorname{diag}(1, 1)$
- input bounds $[71.6, 78.0] \le u \le [870.0, 670.0]$
- TD3 setpoint-scaling envelope `[[2.8, 320.0], [5.0, 326.0]]`

At each sampled state, the expert solves

$$
\begin{aligned}
\min_{U_0,\ldots,U_{N_C-1}} \quad
&\sum_{i=1}^{N_P}
\left(y_i - y_{sp}\right)^\top Q \left(y_i - y_{sp}\right)
+ \sum_{j=0}^{N_C-1}
\Delta u_j^\top R \Delta u_j \\
\text{s.t.} \quad
&x_{i+1}=A_{aug}x_i+B_{aug}u_i, \\
&y_i=C_{aug}x_i, \\
&u_{min}^{dev} \le u_j \le u_{max}^{dev}.
\end{aligned}
$$

The first optimized move $u_0^\star$ is stored as the TD3 action label.

## Replay Buffer and TD3 Pretraining

The runner reuses the existing low-level helpers:

- `utils.td3_helpers.filling_the_buffer(...)`
- `utils.td3_helpers.add_steady_state_samples(...)`
- `TD3Agent.pretrain_from_buffer(...)`

For sampled expert states:

$$
x_d \sim \mathcal{U}(x_{min}, x_{max}), \qquad
y_{sp} \sim \mathcal{U}(y_{sp,min}, y_{sp,max}), \qquad
u_{prev} \sim \mathcal{U}(u_{min}^{dev}, u_{max}^{dev}).
$$

The stored transition uses

$$
x_{d,k+1}=A_{aug}x_{d,k}+B_{aug}u_0^\star,
$$

and the stored reward is

$$
r_k =
-\left[
\left(y_{k+1}-y_{sp}\right)^\top Q \left(y_{k+1}-y_{sp}\right)
+
\left(u_{prev}-u_0^\star\right)^\top R \left(u_{prev}-u_0^\star\right)
\right].
$$

The TD3 pretraining stages are:

1. Behavioral cloning of the OF-MPC first move.
2. Critic TD warm-up under the cloned actor.

For clean OF-MPC-versus-LMPC expert-label comparisons, the LMPC pretraining helper has been aligned to the OF-MPC TD3 optimizer and TD-target defaults:

- `gamma = 0.995`
- `actor_lr = 1e-4`
- `critic_lr = 1e-4`
- `policy_delay = 4`
- `target_policy_smoothing_noise_std = 0.2`
- `noise_clip = 0.5`

The `policy_delay` setting is not active inside the offline `pretrain_from_buffer(...)` actor-BC and frozen-actor critic warm-up loops, because that routine does not run delayed actor TD3 updates. It is still saved as agent metadata and becomes relevant if the same constructed checkpoint is later used for online TD3 updates.

The current OF-MPC and LMPC pretraining runners use actor and critic hidden layers `[256, 256, 256]` by default. The layer sizes remain CLI-overridable for loading or generating checkpoints with older architectures.

## Default Workload

The pretraining runner has one production default configuration, matching the original `Polymer_example` workload sizes. These run-tunable defaults live in `PretrainTD3OffsetFreeMPC.py`, not in the reusable helper module, so they can be changed directly for a new run:

- `mpc_samples = 4_900_000`
- `steady_samples = 100_000`
- `chunk_size = 100_000`
- `actor_epochs = 1000`
- `critic_epochs = 500`
- `pretrain_batch_size = 8192`
- actor hidden layers `[256, 256, 256]`
- critic hidden layers `[256, 256, 256]`

There is no `smoke` preset and no `legacy-full` preset. Smoke runs are explicit overrides only:

```powershell
python PretrainTD3OffsetFreeMPC.py --mpc-samples 32 --steady-samples 8 --chunk-size 16 --actor-epochs 1 --critic-epochs 1
```

The runner also exposes CLI overrides for the pretraining batch size and architecture:

```powershell
python PretrainTD3OffsetFreeMPC.py --pretrain-batch-size 4096 --actor-layers 512,512,512 --critic-layers 512,512,512
```

The full default run is:

```powershell
python PretrainTD3OffsetFreeMPC.py
```

Artifacts are written under:

```text
results/PretrainOFMPC/<timestamp>/
```

Expected files include the checkpoint, `config.json`, `summary.json`, `loss_arrays.json`, `loss_arrays.csv`, `loss_summary.json`, and `pretraining_history.json`.

## Saved-Agent Comparison Runner

`ComparePretrainedTD3OffsetFreeMPC.py` mirrors `PretrainAgentPerformance.ipynb` from `Polymer_example`.

It loads a saved TD3 checkpoint, runs it with `Simulation.run_rl.run_rl_pre_trained(...)`, and compares it against OF-MPC for nominal and disturbance modes. The comparison scenario is:

```python
[[4.5, 324.0],
 [3.4, 321.0]]
```

Default comparison settings are:

- `n_tests = 2`
- `set_points_len = 400`
- modes: `nominal` and `disturb`

If an OF-MPC baseline pickle is missing, the runner generates it with `Simulation.mpc_run.run_mpc(...)` and caches it under:

```text
results/PretrainOFMPCComparison/baselines/
```

Comparison artifacts are written under:

```text
results/PretrainOFMPCComparison/<timestamp>/
```

Useful commands:

```powershell
python ComparePretrainedTD3OffsetFreeMPC.py
python ComparePretrainedTD3OffsetFreeMPC.py --agent-path results/PretrainOFMPC/<timestamp>/of_mpc_pretrained_td3_<timestamp>.pkl
python ComparePretrainedTD3OffsetFreeMPC.py --agent-path results/PretrainOFMPC/<timestamp>/of_mpc_pretrained_td3_<timestamp>.pkl --set-points-len 10 --modes nominal
python ComparePretrainedTD3OffsetFreeMPC.py --agent-path results/PretrainOFMPC/<timestamp>/of_mpc_pretrained_td3_<timestamp>.pkl --actor-layers 512,512,512 --critic-layers 512,512,512
```

## Current Result Check

The latest substantive pretraining bundle reviewed here is:

```text
results/PretrainOFMPC/20260609_002245/
```

Configuration:

- checkpoint: `of_mpc_pretrained_td3_20260609_134202.pkl`
- replay samples: `2,000,000` OF-MPC samples plus `100,000` near-steady samples
- architecture: actor and critic hidden layers `[256, 256, 256]`
- pretraining: `1000` actor behavioral-cloning epochs and `500` critic TD warm-up epochs
- scaler: broad Polymer TD3 setpoint scaler `[[2.8, 320.0], [5.0, 326.0]]`
- comparison rollout: direct two-setpoint scenario `[[4.5, 324.0], [3.4, 321.0]]`

The matching comparison bundle is:

```text
results/PretrainOFMPCComparison/20260609_155747/
```

Nominal comparison:

- TD3 mean RMSE: `0.3562`
- OF-MPC mean RMSE: `0.3554`
- TD3 minus OF-MPC mean RMSE: `0.0008`
- TD3 mean absolute input move: `0.6535`
- OF-MPC mean absolute input move: `0.6461`

Disturbance comparison:

- TD3 mean RMSE: `0.3594`
- OF-MPC mean RMSE: `0.3569`
- TD3 minus OF-MPC mean RMSE: `0.0025`
- TD3 mean absolute input move: `0.6679`
- OF-MPC mean absolute input move: `0.6776`

The learned policy is therefore very close to the OF-MPC expert on this two-setpoint comparison. Relative to the earlier smaller checkpoint `results/PretrainOFMPC/20260608_171525/`, which had mean RMSE around `2.19` nominal and `1.95` disturbed under the corrected scaler, this is a major improvement.

The TD3 and OF-MPC trajectories are also close in physical units. In the latest comparison, the mean absolute TD3-versus-OF-MPC input differences were approximately `[5.71, 0.99]` L/h in nominal mode and `[5.51, 1.36]` L/h in disturbance mode. Mean absolute output differences were approximately `[0.0098, 0.0343]` for nominal mode and `[0.0082, 0.0277]` for disturbance mode.

## Critic-Loss Audit

The current result bundle does not contain a usable critic-loss curve. The file `loss_arrays.json` exists, but its lists are empty:

```json
{
  "actor_bc_losses": [],
  "actor_losses": [],
  "critic_losses": []
}
```

This is a logging issue in the completed run, not direct evidence that the critic failed. The pretraining loop printed epoch losses during training, but the epoch-averaged actor and critic losses were not saved into the result bundle in an analyzable form.

The pretraining code has now been hardened for future OF-MPC and LMPC runs. The TD3 pretraining routine returns an explicit history, and the workflow writes:

- `loss_arrays.json`: backward-compatible raw actor and critic loss lists
- `loss_arrays.csv`: one row per epoch index for quick plotting
- `loss_summary.json`: counts and first, last, min, max, and mean loss statistics
- `pretraining_history.json`: loss, learning-rate, and sample-count history from the pretraining call

The workflow also validates loss logging before saving the checkpoint. If actor or critic epochs are requested and the corresponding loss history is empty or shorter than expected, the run now raises an error instead of silently producing `{}`-style empty loss artifacts.

Future valid runs should contain:

- one actor behavioral-cloning loss per actor epoch
- one critic TD loss per critic epoch

For the completed `20260609_002245` checkpoint, critic health can only be assessed indirectly. A proxy TD diagnostic on the saved comparison rollout gave:

- nominal mean absolute TD residual: about `570` for Q1 and `663` for Q2
- disturbance mean absolute TD residual: about `660` for Q1 and `601` for Q2
- mean Q-values: about `-1.22e5`

The Q-value scale is large because the critic was trained on broad synthetic replay samples where the stored reward mean was about `-604`; with `gamma = 0.995`, a continuing-value scale near `-604 / (1 - 0.995) \approx -1.21e5` is expected. The rollout TD residual is therefore roughly at the sub-percent scale relative to the Q magnitude. This proxy does not replace a real training loss curve, but it does not indicate an obvious critic divergence.

## Limitations

- The expert labels are OF-MPC labels, not LMPC labels.
- The replay state distribution is synthetic, so it may not match closed-loop visitation under the plant.
- The stored critic reward is the OF-MPC pretraining penalty, not the safety-gate RL reward used in Direct Lyapunov studies.
- The comparison runner is an OF-MPC versus saved-TD3 comparison. It does not include the Direct Lyapunov safety gate.

## Future LMPC Conversion

The future LMPC migration should keep this runner/helper split and replace only the expert-label source. The TD3 state representation, artifact format, and comparison runner can remain stable while the label generator changes from OF-MPC to Direct Lyapunov MPC.
