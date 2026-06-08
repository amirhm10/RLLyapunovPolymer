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
- pretraining setpoint envelope `[[2.8, 320.0], [5.0, 326.0]]`

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

For broad samples:

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

The actor and critic hidden layers are `[512, 512, 512, 512, 512]` for compatibility with the current `Lyapunov_polymer` checkpoints and runners.

## Default Workload

The pretraining runner has one production default configuration, matching the original `Polymer_example` workload sizes. These run-tunable defaults live in `PretrainTD3OffsetFreeMPC.py`, not in the reusable helper module, so they can be changed directly for a new run:

- `mpc_samples = 4_900_000`
- `steady_samples = 100_000`
- `chunk_size = 100_000`
- `actor_epochs = 1000`
- `critic_epochs = 500`
- `pretrain_batch_size = 8192`
- actor hidden layers `[512, 512, 512, 512, 512]`
- critic hidden layers `[512, 512, 512, 512, 512]`

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

Expected files include the checkpoint, `config.json`, `summary.json`, and loss arrays when available.

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

## Limitations

- The expert labels are OF-MPC labels, not LMPC labels.
- The replay state distribution is synthetic and broad, so it may not match closed-loop visitation under the plant.
- The stored critic reward is the OF-MPC pretraining penalty, not the safety-gate RL reward used in Direct Lyapunov studies.
- The comparison runner is an OF-MPC versus saved-TD3 comparison. It does not include the Direct Lyapunov safety gate.

## Future LMPC Conversion

The future LMPC migration should keep this runner/helper split and replace only the expert-label source. The TD3 state representation, artifact format, and comparison runner can remain stable while the label generator changes from OF-MPC to Direct Lyapunov MPC.
