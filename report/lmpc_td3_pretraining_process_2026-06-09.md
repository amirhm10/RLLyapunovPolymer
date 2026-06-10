# Direct LMPC TD3 Pretraining Process

Date: 2026-06-09

## Objective

This report documents the new repo-native TD3 pretraining workflow that uses the active Direct Output-Disturbance Lyapunov MPC as the expert. It sits beside the OF-MPC pretraining workflow:

- `utils/lmpc_td3_workflow.py` contains reusable setup, Direct LMPC construction, label generation, TD3 training orchestration, checkpoint architecture inference, and comparison helpers.
- `PretrainTD3LyapunovMPC.py` is the pretraining and checkpoint-saving runner.
- `ComparePretrainedTD3LyapunovMPC.py` is the saved-agent comparison runner against Direct LMPC and OF-MPC.

This is primarily a method and validation report. The only new numerical results reported here are tiny smoke validations, not production training performance.

## Coordinates and Scaling

The workflow keeps the corrected Polymer TD3 scaling convention already used by the OF-MPC migration.

The TD3 state is

$$
s_k =
\begin{bmatrix}
\tilde{x}_{d,k} \\
\tilde{y}_{sp,k} \\
\tilde{u}_{k-1}
\end{bmatrix},
$$

where each block is mapped to $[-1,1]$ before it is passed to the actor and critic.

The dimensions are computed from the augmented matrices:

```python
inputs_number = int(B_aug.shape[1])
set_points_number = int(C_aug.shape[0])
state_dim = int(A_aug.shape[0]) + set_points_number + inputs_number
action_dim = inputs_number
```

For the current polymer model, this gives `state_dim = 13` and `action_dim = 2`.

The TD3 setpoint scaler is the broad Polymer pretraining envelope:

```python
[[2.8, 320.0],
 [5.0, 326.0]]
```

The rollout and comparison setpoints remain the direct two-setpoint schedule:

```python
[[4.5, 324.0],
 [3.4, 321.0]]
```

The augmented-state bounds are the restored Polymer TD3 bounds in `utils/polymer_td3_defaults.py`. The same `load_and_prepare_system_data(...)` path is used by OF-MPC pretraining, LMPC pretraining, comparison runners, and active Lyapunov RL runners.

## Direct LMPC Expert

The expert controller is built with `design_direct_lyapunov_mpc_solver(...)` using the Rawlings output-disturbance augmentation:

$$
\begin{aligned}
x_{k+1} &= A x_k + B u_k + B_d d_k, \\
d_{k+1} &= d_k, \\
y_k &= C x_k + C_d d_k .
\end{aligned}
$$

The selected disturbance model is output disturbance:

$$
B_d = 0, \qquad C_d = I.
$$

The Direct LMPC expert uses:

- `NP = 9`
- `NC = 3`
- MPC/LMPC objective weights `Qy_mpc_diag = [5, 1]`
- `Su_diag = [1, 1]`
- `Rdu_diag = [1, 1]`
- `terminal_set_on = True`
- `terminal_alpha_scale = 1.0`
- target mode `governed_reference`
- hard Lyapunov tracking mode
- first-step contraction enabled
- `rho_lyap = 0.99`
- `lyap_eps = 5e-3`
- `use_target_output_for_tracking = False`
- `use_target_on_solver_fail = False`

For each sampled replay candidate, the workflow first solves the governed-reference target problem. Conceptually, it finds an admissible governed command $r_k$ and a steady target $(x_s,u_s,y_s)$ compatible with the augmented output-disturbance model and input limits.

The tracking MPC then solves a constrained finite-horizon problem of the form

$$
\begin{aligned}
\min_{\{u_i\}} \quad
& \sum_{i=0}^{N_P-1}
\left(C_{aug}x_{i+1} - y_{sp}\right)^\top Q_y
\left(C_{aug}x_{i+1} - y_{sp}\right)
+ \sum_{j=0}^{N_C-1}
\Delta u_j^\top R_{\Delta u}\Delta u_j \\
\text{s.t.} \quad
& x_{i+1}=A_{aug}x_i+B_{aug}u_i, \\
& u_{min}^{dev} \le u_j \le u_{max}^{dev}, \\
& (x_{N_P}-x_s)^\top P_x(x_{N_P}-x_s) \le \alpha, \\
& V(x_1-x_s) \le \rho V(x_0-x_s)+\epsilon .
\end{aligned}
$$

Only successful hard-LMPC solves are accepted as TD3 labels. Failed target solves, failed tracking solves, and hold-previous fallbacks are skipped and counted in diagnostics.

## Replay-Buffer Construction

Broad candidates are sampled from the same corrected TD3 ranges used by OF-MPC pretraining:

$$
x_d \sim \mathcal{U}(x_{min},x_{max}), \quad
y_{sp} \sim \mathcal{U}(y_{sp,min},y_{sp,max}), \quad
u_{prev} \sim \mathcal{U}(u_{min}^{dev},u_{max}^{dev}).
$$

Near-steady candidates use small Gaussian augmented-state perturbations, zero setpoint deviation, and near-zero previous-input deviations.

For an accepted LMPC label $u_0^\star$, the transition is stored as:

$$
\begin{aligned}
s_k &= \operatorname{scale}_{[-1,1]}(x_{d,k}, y_{sp,k}, u_{k-1}), \\
a_k &= \operatorname{scale}_{[-1,1]}(u_0^\star), \\
x_{d,k+1} &= A_{aug}x_{d,k}+B_{aug}u_0^\star, \\
s_{k+1} &= \operatorname{scale}_{[-1,1]}(x_{d,k+1}, y_{sp,k}, u_0^\star).
\end{aligned}
$$

The offline replay reward is not the online shaped RL reward. It is the same one-step quadratic stage cost used by the MPC objective weights:

$$
r_k =
-\left[
\left(y_{k+1}-y_{sp,k}\right)^\top Q_{MPC}
\left(y_{k+1}-y_{sp,k}\right)
+
\left(u_0^\star-u_{prev,k}\right)^\top R_{MPC}
\left(u_0^\star-u_{prev,k}\right)
\right],
$$

with $Q_{MPC}=\operatorname{diag}(5,1)$ and $R_{MPC}=\operatorname{diag}(1,1)$. The online safety-gate RL reward remains a separate shaped reward family and does not set the LMPC or OF-MPC objective weights.

## TD3 Pretraining

The LMPC pretraining runner now keeps the editable production defaults at the top of `PretrainTD3LyapunovMPC.py`, matching the current OF-MPC pretraining runner style:

- accepted broad LMPC labels: `2,000,000`
- accepted near-steady labels: `100,000`
- candidate chunk size: `100,000`
- parallel label workers: `-1` (`joblib` uses all available cores)
- parallel backend: `loky`
- candidate rows per parallel task and replay flush threshold: `8192`
- max attempt multiplier: `5`
- actor behavioral-cloning epochs: `1000`
- critic TD warm-up epochs: `500`
- pretraining batch size: `8192`
- actor hidden layers: `[256, 256, 256]`
- critic hidden layers: `[256, 256, 256]`

For clean OF-MPC-versus-LMPC scale-up comparisons, the LMPC TD3 constructor now matches the OF-MPC TD3 constructor:

- `gamma = 0.995`
- `actor_lr = 1e-4`
- `critic_lr = 1e-4`
- `policy_delay = 4`
- `target_policy_smoothing_noise_std = 0.2`
- `noise_clip = 0.5`

The `policy_delay` value is not active during offline actor behavioral cloning or frozen-actor critic warm-up, but it remains part of the constructed/saved TD3 agent and is relevant when the checkpoint is used for later online TD3 updates.

The runner exposes CLI overrides for all workload sizes, label-parallelism settings, architecture values, seed, device, and output root.

LMPC label generation is parallelized at the candidate-batch level. The parent process samples candidate states and owns the replay buffer. Each `loky` worker builds its own Direct LMPC solver from serializable matrices, labels its assigned candidates, and returns plain NumPy transitions plus diagnostics. This avoids passing a live CVXPY problem or mutating the replay buffer across processes.

During label generation, the workflow prints OF-MPC-style candidate chunk progress, for example `Processing broad LMPC candidate chunk 1/20`. The chunk count is based on the maximum allowed candidate attempts, so the run can finish earlier if enough accepted labels are collected.

The CVXPY warning `Solution may be inaccurate` is suppressed during LMPC pretraining label solves to keep long-run logs readable. Solver statuses, target stages, failure keys, and success rates are still saved in `label_diagnostics.json`, so the warning is not used as the diagnostic channel.

Because a parallel worker batch may finish after the requested accepted-label count is already reached, diagnostics distinguish kept replay labels from extra successful solves:

- `acceptance_rate`: accepted replay labels divided by attempted candidates
- `solve_success_rate`: successful LMPC solves divided by attempted candidates
- `discarded_successes`: successful solves not inserted because the requested count was already reached

Checkpoint metadata was also extended in `TD3Agent.save(...)`. New checkpoints store `state_dim`, `action_dim`, `actor_hidden`, and `critic_hidden` in the `hparams` block. The comparison runner can also infer architecture from older checkpoints by reading saved layer weights.

## Artifacts

Pretraining artifacts are written to:

```text
results/PretrainLMPC/<timestamp>/
```

Expected files:

- `lmpc_pretrained_td3_<timestamp>.pkl`
- `config.json`
- `summary.json`
- `label_diagnostics.json`
- `loss_arrays.json`
- `loss_arrays.csv`
- `loss_summary.json`
- `pretraining_history.json`

The loss writer is shared with the OF-MPC workflow. It validates that requested actor and critic epochs produced non-empty epoch histories before the checkpoint is saved, so a run with missing loss logs now fails clearly instead of leaving empty loss arrays for later analysis.

Comparison artifacts are written to:

```text
results/PretrainLMPCComparison/<timestamp>/
```

Baselines are cached under:

```text
results/PretrainLMPCComparison/baselines/
```

The comparison CSV/JSON include one record per controller:

- `td3`
- `direct_lmpc`
- `offset_free_mpc`

The Direct LMPC and OF-MPC diagnostic records include target-stage counts, solver success rates, contraction satisfaction rates, and diagnostic unsafe counts when those fields are available.

## Commands

Full default LMPC pretraining:

```powershell
python PretrainTD3LyapunovMPC.py
```

Small smoke run:

```powershell
python PretrainTD3LyapunovMPC.py --lmpc-samples 16 --steady-samples 4 --candidate-chunk-size 8 --actor-epochs 1 --critic-epochs 1
```

Larger run:

```powershell
python PretrainTD3LyapunovMPC.py --lmpc-samples 250000 --steady-samples 25000 --actor-epochs 800 --critic-epochs 300
```

Compare a saved LMPC-pretrained TD3 checkpoint:

```powershell
python ComparePretrainedTD3LyapunovMPC.py --agent-path results/PretrainLMPC/<timestamp>/<checkpoint>.pkl
```

Short nominal comparison:

```powershell
python ComparePretrainedTD3LyapunovMPC.py --agent-path results/PretrainLMPC/<timestamp>/<checkpoint>.pkl --set-points-len 10 --n-tests 1 --modes nominal
```

## Smoke Validation

The default `python` in this shell is Python 3.13 and does not have `cvxpy`, so runtime validation used the existing conda interpreter:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe
```

That environment had `cvxpy 1.8.2`.

Static validation passed for:

```powershell
python -m py_compile PretrainTD3LyapunovMPC.py ComparePretrainedTD3LyapunovMPC.py utils/lmpc_td3_workflow.py TD3Agent/agent.py
```

After aligning the Lyapunov contraction tolerance with `DirectLyapunovMPC.py`, smoke pretraining was run with 1 broad LMPC label, 1 near-steady label, and one actor/critic epoch. The generated bundle was:

```text
results/PretrainLMPCEpsSmoke/20260609_194353/
```

Observed corrected smoke diagnostics:

- `config.json` recorded `rho_lyap = 0.99` and `lyap_eps = 0.005`
- accepted labels: `2`
- attempted candidates: `2`
- total acceptance rate: `1.0`
- broad acceptance rate: `1.0`
- steady acceptance rate: `1.0`
- actor BC loss entries: `1`
- critic TD loss entries: `1`
- actor BC loss: `0.0620`
- critic TD loss: `215.22`

This confirms artifact writing and non-empty loss arrays. It is too small to say anything meaningful about final closed-loop performance.

A short nominal comparison was run for this smoke checkpoint:

```text
results/PretrainLMPCEpsComparisonSmoke/20260609_194410/
```

The metrics file contains TD3, Direct LMPC, and OF-MPC records. The Direct LMPC baseline had target success rate `1.0` and contraction satisfaction rate `1.0` on this short nominal run. The OF-MPC diagnostic baseline had target success rate `1.0`, contraction satisfaction rate `0.95`, and one diagnostic unsafe step. The baseline cache filenames now include the MPC objective-weight token and Lyapunov contraction token, for example `_q5_1_r1_1_rho0p99_eps0p005`, so older `[12,6]` or stricter-epsilon smoke baselines are not silently reused.

Failure handling was also tested by requesting 3 broad labels with only 3 candidate attempts. The run failed clearly after accepting 1 of 3 labels and wrote diagnostics:

```text
results/PretrainLMPC/20260609_163559/label_diagnostics.json
```

The existing OF-MPC workflow still compiled, and a tiny OF-MPC smoke pretraining run passed after allowing Windows joblib worker processes:

```text
results/PretrainOFMPCSmoke/20260609_191259/
```

Parallel LMPC label generation was also smoke-tested with `--label-n-jobs 2`, `--parallel-backend loky`, and `--worker-batch-size 2`:

```text
results/PretrainLMPCParallelSmoke/20260609_214340/
```

The run accepted 3 replay labels from 4 attempted candidates, recorded 4 successful LMPC solves, and discarded 1 extra successful solve after the requested count was reached. Loss logging remained valid with one actor BC loss and one critic TD loss.

## Limitations

- Broad random candidates can be rejected by hard LMPC. Acceptance rate must be monitored before committing to a large production run.
- Parallel LMPC workers rebuild their own CVXPY-backed solver per candidate batch. This is safer than sharing solver objects, but the batch size should be tuned so solver construction overhead does not dominate.
- Offline LMPC pretraining uses the one-step MPC quadratic reward. Online RL training uses the shaped relative-QR reward and should remain separate from the MPC/LMPC objective weights.
- The LMPC pretraining and comparison workflows now use the same proof-track contraction setting as `DirectLyapunovMPC.py`: `rho_lyap = 0.99`, `lyap_eps = 5e-3`.
- The smoke checkpoint is intentionally tiny and should not be interpreted as evidence of policy quality.

## Next Experiment

Run a calibration job before the first production LMPC pretraining run:

```powershell
python PretrainTD3LyapunovMPC.py --lmpc-samples 1000 --steady-samples 100 --candidate-chunk-size 64 --actor-epochs 2 --critic-epochs 2
```

Use `label_diagnostics.json` to estimate the broad-label acceptance rate. If the acceptance rate is too low, narrow the broad candidate state envelope for LMPC labels while keeping the TD3 scaler unchanged.
