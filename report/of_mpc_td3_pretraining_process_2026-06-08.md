# OF-MPC TD3 Pretraining Process

Date: 2026-06-08

## Objective

This report documents the migrated offset-free MPC pretraining workflow for the TD3 agent in `Lyapunov_polymer`. The migration adds a repo-native script that generates expert demonstrations with the existing offset-free MPC formulation, fills the TD3 replay buffer with synthetic samples, and saves a compatible TD3 checkpoint under `results/PretrainOFMPC/`.

This is a process report, not a new performance report. It does not claim that a new checkpoint improves the 300-episode Direct Lyapunov safety-gate study unless that checkpoint is later trained and evaluated in closed loop.

## Coordinates

The pretraining workflow uses the same scaled-deviation coordinate convention as the active RL and MPC code.

- Physical manipulated inputs are $u = [Q_c, Q_m]^\top$.
- Physical outputs are $y = [\eta, T]^\top$.
- The steady-state input anchor is $u_{ss} = [471.6, 378.0]^\top$.
- The output steady state is computed by `PolymerCSTR` from the same plant parameters used by the current pretrained runner.
- Inputs are min-max scaled and then represented as deviation from $u_{ss}$.
- The TD3 state is the concatenation

$$
s_k =
\begin{bmatrix}
\tilde{x}_{d,k} \\
\tilde{y}_{sp,k} \\
\tilde{u}_{k-1}
\end{bmatrix},
$$

where $\tilde{x}_{d,k}$ is the augmented model state mapped to $[-1, 1]$, $\tilde{y}_{sp,k}$ is the scaled setpoint deviation mapped to $[-1, 1]$, and $\tilde{u}_{k-1}$ is the previous input deviation mapped to $[-1, 1]$.

For the current Rawlings output-disturbance augmentation, this gives

- `STATE_DIM = 13`
- `ACTION_DIM = 2`
- actor hidden layers `[512, 512, 512, 512, 512]`
- critic hidden layers `[512, 512, 512, 512, 512]`

These dimensions match the active `DirectLyapunovSafetyGateRL_Pretrained.py` architecture.

## Offset-Free Augmentation

The migrated workflow uses the Rawlings-style output-disturbance model already available in `utils.td3_helpers.load_and_prepare_system_data(...)`:

$$
\begin{aligned}
x_{k+1} &= A x_k + B u_k + B_d d_k, \\
d_{k+1} &= d_k, \\
y_k &= C x_k + C_d d_k .
\end{aligned}
$$

For the migrated OF-MPC pretraining script, the selected mode is `augmentation_style="rawlings"` and `augmentation_mode="output_disturbance"`, so

$$
B_d = 0, \qquad C_d = I.
$$

This keeps the pretraining model consistent with the disturbance coordinates used by the active Direct Lyapunov workflow.

## Expert OF-MPC Problem

The expert controller is the legacy offset-free MPC, not the Direct Lyapunov MPC. For each synthetic sample, the script solves a finite-horizon tracking problem with

- prediction horizon `NP = 9`
- control horizon `NC = 3`
- output weights $Q = \operatorname{diag}(5, 1)$
- move weights $R = \operatorname{diag}(1, 1)$
- physical input bounds $[71.6, 78.0] \le u \le [870.0, 670.0]$
- broad pretraining setpoint envelope

$$
y_{sp}^{phys} \in
\left\{
[2.8, 320.0]^\top,\,
[5.0, 326.0]^\top
\right\}.
$$

In scaled-deviation coordinates, the first input move is selected by

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

For $i \ge N_C$, the final optimized move is held constant across the remaining prediction horizon. The expert label stored in the replay buffer is the first optimized input move $u_0^\star$.

## Replay Buffer Construction

The workflow reuses the existing helper functions:

- `utils.td3_helpers.filling_the_buffer(...)`
- `utils.td3_helpers.add_steady_state_samples(...)`

For the broad MPC samples, the helper samples

$$
x_d \sim \mathcal{U}(x_{min}, x_{max}), \qquad
y_{sp} \sim \mathcal{U}(y_{sp,min}, y_{sp,max}), \qquad
u_{prev} \sim \mathcal{U}(u_{min}^{dev}, u_{max}^{dev}).
$$

After the expert MPC solve, it stores

$$
a_k = \tilde{u}_0^\star,
$$

and propagates the model one step:

$$
x_{d,k+1}=A_{aug}x_{d,k}+B_{aug}u_0^\star.
$$

The stored reward is the negative tracking and input-move penalty:

$$
r_k =
-\left[
\left(y_{k+1}-y_{sp}\right)^\top Q \left(y_{k+1}-y_{sp}\right)
+
\left(u_{prev}-u_0^\star\right)^\top R \left(u_{prev}-u_0^\star\right)
\right].
$$

The near-steady samples use a narrow Gaussian around zero augmented-state deviation, zero setpoint deviation, and a nearly zero previous input deviation. These samples bias the actor toward the steady-state action near the governed target.

## TD3 Pretraining Stages

The migrated script calls `TD3Agent.pretrain_from_buffer(...)` with the generated replay data.

Stage 1 is behavioral cloning:

$$
\min_{\theta_\pi}
\frac{1}{N}
\sum_{k=1}^{N}
\left\|
\pi_{\theta_\pi}(s_k)-a_k^{MPC}
\right\|_2^2 .
$$

Stage 2 freezes the actor and trains the critic with TD targets under the cloned policy:

$$
y_k =
r_k + \gamma(1-d_k)
Q_{\bar{\theta}_Q}
\left(
s_{k+1},
\pi_{\bar{\theta}_\pi}(s_{k+1})+\epsilon
\right),
$$

with TD3 target-policy smoothing enabled for the critic warm-up. The saved checkpoint contains actor and critic state dictionaries in the current `TD3Agent.save(...)` format.

## Script Interface

New entrypoint:

```powershell
python PretrainTD3OffsetFreeMPC.py
```

The default preset is `smoke`, intended for fast validation rather than a useful production checkpoint.

Useful commands:

```powershell
python PretrainTD3OffsetFreeMPC.py --preset legacy-full
python PretrainTD3OffsetFreeMPC.py --mpc-samples 100000 --steady-samples 5000 --actor-epochs 50 --critic-epochs 20
python PretrainTD3OffsetFreeMPC.py --mpc-samples 32 --steady-samples 8 --chunk-size 16 --actor-epochs 1 --critic-epochs 1
```

The script writes artifacts under

```text
results/PretrainOFMPC/<timestamp>/
```

Expected files:

- `of_mpc_pretrained_td3_<timestamp>.pkl`
- `config.json`
- `summary.json`
- `loss_arrays.json`
- `loss_arrays.csv`, only when the TD3 agent exposes nonempty loss arrays

## Selecting a New Checkpoint

The pretrained Direct Lyapunov safety-gate runner keeps its existing default checkpoint:

```text
Data/agent_2507171027.pkl
```

To evaluate a migrated pretraining checkpoint, set:

```powershell
$env:PRETRAINED_TD3_AGENT_PATH='results/PretrainOFMPC/<timestamp>/of_mpc_pretrained_td3_<timestamp>.pkl'
python DirectLyapunovSafetyGateRL_Pretrained.py
```

Relative paths are resolved from the repository root. Absolute paths are also accepted.

## Limitations

- The expert labels are still OF-MPC labels. They are not Direct Lyapunov MPC labels.
- The synthetic state distribution is broad and model-based, so it can include states that are not equally likely under closed-loop plant operation.
- The reward stored for critic warm-up is the OF-MPC pretraining penalty, not the safety-gate RL training reward used in the 300-episode study.
- The smoke preset only validates plumbing and checkpoint compatibility. It is too small to produce a meaningful actor.
- `TD3Agent.pretrain_from_buffer(...)` currently prints epoch losses, but may not populate persistent loss arrays for this path. The script saves loss files when arrays are available.

## Future LMPC Conversion Path

The next migration should replace only the expert-label generator, not the TD3 checkpoint format or runner interface.

1. Keep the replay state and action representation fixed.
2. Replace `MpcSolver.mpc_opt_fun(...)` labels with Direct Lyapunov MPC first-step labels.
3. Store diagnostics for target feasibility, Lyapunov contraction residual, fallback feasibility, and slack use.
4. Compare OF-MPC-pretrained and LMPC-pretrained agents with the same safety-gate runner and the same disturbance test cycle.
5. Report both training reward and `reward_no_penalty`, because the safety-gate reward includes fallback/event penalties.

