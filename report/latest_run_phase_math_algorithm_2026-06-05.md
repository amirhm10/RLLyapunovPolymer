# Latest Run Phase, Mathematics, And Algorithm

Date: 2026-06-05

## Objective

This report summarizes the latest run state for the direct Lyapunov MPC and direct Lyapunov safety-gate RL workflow. The goal is to answer a practical question: which phase are we in right now?

The short answer is:

**We are in the saved-agent evaluation alignment phase.** The governed-reference method has already been promoted and tested in full 300-episode training runs. The next scientifically clean step is to evaluate the final saved cold-start and pretrained agents under a controlled saved-agent script. Before running that evaluation, the saved-agent script should be aligned with the governed-reference target mode and the intended plant mode.

## Files And Results Inspected

Source files:

- `DirectLyapunovMPC.py`
- `DirectLyapunovSafetyGateRL_ColdStart.py`
- `DirectLyapunovSafetyGateRL_Pretrained.py`
- `DirectLyapunovSavedAgentEvaluation.py`
- `Simulation/saved_agent_evaluation.py`
- `Simulation/run_rl_lyapunov.py`
- `Lyapunov/governed_reference_target.py`
- `TD3Agent/reward_functions.py`
- `utils/direct_lyapunov_study.py`

Reports and result summaries:

- `report/governed_reference_lyap_mpc_methodology_2026-05-22.md`
- `report/governed_reference_latest_three_run_analysis_2026-05-23.md`
- `results/directLyap/20260525_170004/lyap_governed_reference/summary.json`
- `results/ColdStart/20260525_170006/comparison_table.csv`
- `results/Pretrain/20260604_165431/comparison_table.csv`

## Current Phase

The current workflow has moved through these phases:

1. **Direct Lyapunov target feasibility phase**
   The hidden target mismatch problem was identified: the controller could satisfy Lyapunov contraction around $y_s$ while $y_s$ was not close enough to the raw requested setpoint $y_{sp}$.

2. **Governed-reference method phase**
   A two-stage governed reference layer was implemented:

   $$
   y_{sp,k}\rightarrow r_k\rightarrow (x_{s,k},u_{s,k},y_{s,k}).
   $$

3. **Full training-run phase**
   Full 300-episode direct, cold-start RL, and pretrained RL runs were completed with the governed-reference target mode active in the main runners.

4. **Current phase: saved-agent evaluation alignment**
   `DirectLyapunovSavedAgentEvaluation.py` is now set to:

   ```python
   plant_mode = "nominal"
   ```

   This is the latest run-setting modification. However, the reusable saved-agent helpers still hard-code:

   ```python
   direct_target_mode = "bounded"
   target_mode = "bounded"
   ```

   Therefore, the saved-agent evaluation is not yet aligned with the governed-reference training runs. If it is run now, it will evaluate the latest saved agents under a nominal plant but with the older bounded target selector.

## Latest Run Evidence

The latest direct Lyapunov run on disk is:

```text
results/directLyap/20260525_170004/lyap_governed_reference/
```

It used the governed-reference target, hard first-step Lyapunov contraction, raw-setpoint tracking, and disturbed plant mode.

| Metric | Value |
|---|---:|
| Steps | 240000 |
| Target success | 1.000 |
| Solver success | 1.000 |
| Hard contraction | 1.000 |
| Reward no penalty mean | -4.333 |
| Mean output reference error inf | 0.452 |
| Mean target mismatch inf | 0.557 |
| Governor active rate | 0.836 |
| Seconds per step | 0.0403 |

The latest cold-start safety-gate RL run on disk is:

```text
results/ColdStart/20260525_170006/
```

| Case | Mean RMSE | Reward no penalty | Fallback rate | Intervention rate |
|---|---:|---:|---:|---:|
| RL gate | 0.302 | -13.271 | 0.0093 | 0.0123 |
| `mpc_only` | 0.289 | -12.752 | 0.0000 | 0.0000 |

The latest pretrained safety-gate RL run on disk is:

```text
results/Pretrain/20260604_165431/
```

| Case | Mean RMSE | Reward no penalty | Fallback rate | Intervention rate |
|---|---:|---:|---:|---:|
| RL gate | 0.210 | -6.094 | 0.0187 | 0.0212 |
| `mpc_only` | 0.169 | -4.403 | 0.0000 | 0.0000 |

These latest runs show that the safety gate is reliable and the target selector solves consistently, but the RL policy is still not beating same-run `mpc_only` on raw RMSE in the latest exported evidence.

## Existing Figures Used

Direct governed-reference diagnostics:

![Direct governed-reference diagnostics](../results/directLyap/20260525_170004/lyap_governed_reference/plots/06_governed_reference_diagnostics.png)

Cold-start latest comparison:

![Cold-start output RMSE comparison](../results/ColdStart/20260525_170006/comparison_plots/comparison_output_rmse.png)

Pretrained latest comparison:

![Pretrained output RMSE comparison](../results/Pretrain/20260604_165431/comparison_plots/comparison_output_rmse.png)

No new figures were generated for this report. The interpretation above uses existing saved summary files and existing run plots.

## Mathematical Coordinates

The plant is the polymer CSTR with manipulated input:

$$
u =
\begin{bmatrix}
Q_c \\
Q_m
\end{bmatrix},
$$

and controlled output:

$$
y =
\begin{bmatrix}
\eta \\
T
\end{bmatrix}.
$$

The controller works in scaled deviation coordinates. For an input or output $z^{phys}$, the scaled-deviation variable is:

$$
z_k = S_z(z^{phys}_k)-S_z(z^{phys}_{ss}),
$$

where $S_z(\cdot)$ is the min-max scaling map and $z_{ss}^{phys}$ is the steady-state anchor.

The augmented offset-free model is:

$$
x^a_k=
\begin{bmatrix}
x_k \\
\hat d_k
\end{bmatrix},
\qquad
\hat y_k = Cx_k + C_d\hat d_k.
$$

The observer update used in the rollout has the form:

$$
\hat x^a_{k+1}
=
A_a\hat x^a_k + B_a u_k
+ L\left(y_k-C_a\hat x^a_k\right).
$$

## Governed-Reference Target Mathematics

The current active training runners use a governed-reference target selector. The first stage computes a governed command $r_k$ near the raw setpoint:

$$
\begin{aligned}
\min_{x_g,u_g}\quad
&\|Cx_g+C_d\hat d_k-y_{sp,k}\|_{W_r}^2
+\lambda_r\|Cx_g+C_d\hat d_k-r_{k-1}\|_2^2 \\
\text{s.t.}\quad
&x_g = Ax_g+B_d\hat d_k+Bu_g,\\
&u_{\min}^h\le u_g\le u_{\max}^h.
\end{aligned}
$$

The tightened input bounds are:

$$
u_{\min}^h = u_{\min}+\alpha_h(u_{\max}-u_{\min}),
\qquad
u_{\max}^h = u_{\max}-\alpha_h(u_{\max}-u_{\min}),
$$

with $\alpha_h=0.03$ in the active governed-reference default.

The second stage computes the steady Lyapunov target around the governed command:

$$
\begin{aligned}
\min_{x_s,u_s}\quad
&\|Cx_s+C_d\hat d_k-r_k\|_{Q_r}^2
+\lambda_u\|u_s-u_{k-1}\|_2^2
+\lambda_x\|x_s-x_{s,k-1}\|_2^2 \\
\text{s.t.}\quad
&x_s = Ax_s+B_d\hat d_k+Bu_s,\\
&u_{\min}^h\le u_s\le u_{\max}^h.
\end{aligned}
$$

The active governed-reference defaults in the main runners are:

```python
lambda_cmd_move = 1.0
u_ref_weight = 0.1
x_ref_weight = 0.1
input_headroom_frac = 0.03
one_step_probe = True
Qr_diag = Qy_diag
W_r_diag = Qy_diag
```

The target is a certificate center, not the reported process objective. The active training runners keep:

$$
y_{\mathrm{track},k}=y_{sp,k}.
$$

## Direct Lyapunov MPC Mathematics

For a selected target $(x_{s,k},u_{s,k},y_{s,k})$, the Lyapunov function is:

$$
V_k =
(\hat x_k-x_{s,k})^T P_x(\hat x_k-x_{s,k}).
$$

The hard first-step contraction test is:

$$
V_{k+1|k}\le \rho V_k+\epsilon.
$$

The direct Lyapunov MPC solves a finite-horizon tracking problem with the same raw process target used for reporting:

$$
\begin{aligned}
\min_{\mathbf u}\quad
&\sum_{i=1}^{N_p}
\|y_{k+i|k}-y_{sp,k}\|_{Q_y}^2
+\sum_{j=0}^{N_c-1}\|\Delta u_{k+j|k}\|_{R_{\Delta u}}^2 \\
\text{s.t.}\quad
&x^a_{k+i+1|k}=A_ax^a_{k+i|k}+B_au_{k+i|k},\\
&u_{\min}\le u_{k+j|k}\le u_{\max},\\
&V_{k+1|k}\le \rho V_k+\epsilon.
\end{aligned}
$$

For the latest main runners, the active Lyapunov parameters are:

```python
rho_lyap = 0.98
lyap_eps = 1e-9
```

The saved-agent evaluation script currently differs:

```python
rho_lyap = 0.99
lyap_eps = 1e-3
```

That is a material difference if the goal is an apples-to-apples final-agent evaluation.

## Safety-Gate RL Mathematics

The TD3 actor observes an RL state containing the augmented observer state, the active setpoint, and the previous input:

$$
s_k =
\begin{bmatrix}
\hat x^a_k \\
y_{sp,k} \\
u_{k-1}
\end{bmatrix}.
$$

The actor proposes a bounded action:

$$
a_k=\pi_\theta(s_k)\in[-1,1]^{n_u}.
$$

The action is mapped to the input-deviation box:

$$
u_k^{RL}
=
u_{\min}
+\frac{a_k+1}{2}\odot (u_{\max}-u_{\min}).
$$

The direct safety gate evaluates the candidate:

$$
u_k =
\begin{cases}
u_k^{RL}, & \text{if } u_k^{RL}\in U
\text{ and } V_{k+1|k}(u_k^{RL})\le \rho V_k+\epsilon,\\
u_k^{LMPC}, & \text{otherwise.}
\end{cases}
$$

Here $u_k^{LMPC}$ is the direct Lyapunov MPC fallback action solved around the same selected target.

The reward now stores two channels:

$$
r_k^{base}
=
-\ell_y(e_k)-\ell_{\Delta u}(\Delta u_k)+b(e_k),
$$

and

$$
r_k =
r_k^{base}
-\mathbf 1_{\mathrm{fallback}}
\left(
\gamma_f\|u_k^{RL}-u_k\|_{R_f}^2+c_f
\right).
$$

The logged `reward_no_penalty` equals $r_k^{base}$. This is the preferred channel for cross-method control-performance comparison because direct Lyapunov MPC does not train with a fallback event penalty.

## Current Algorithmic Flow

The active training flow is:

1. Scale the plant output, input, and setpoint into deviation coordinates.
2. Update the augmented observer state $\hat x^a_k$.
3. Solve the governed command $r_k$ from the raw setpoint $y_{sp,k}$.
4. Solve the steady target $(x_s,u_s,y_s)$ around $r_k$.
5. In direct MPC, solve the direct Lyapunov MPC and apply its first input.
6. In RL, let TD3 propose $u_k^{RL}$.
7. Check the RL proposal against input bounds and first-step Lyapunov contraction.
8. Apply the RL action if certified, otherwise apply direct Lyapunov MPC fallback.
9. Log both actual training reward and `reward_no_penalty`.
10. Save plots, arrays, summaries, trained agents, and comparison tables.

The intended saved-agent evaluation flow is:

1. Load the latest non-`mpc_only` cold-start and pretrained agents.
2. Freeze learning.
3. Run `cold_saved_rl`, `pretrained_saved_rl`, `mpc_only`, and `direct_lmpc`.
4. Compare raw setpoint RMSE, tail offset, actual fallback, would-be fallback, runtime, and `reward_no_penalty`.

The current saved-agent implementation needs one decision before that flow is run:

- either intentionally evaluate the saved agents under the older bounded target selector,
- or update `Simulation/saved_agent_evaluation.py` so the saved-agent helper uses the same governed-reference target mode as the training runners.

For the current research question, the second option is the fairer next step.

## Interpretation

The latest evidence supports four conclusions.

First, governed-reference direct Lyapunov MPC is numerically reliable. The latest direct run reports full target success, full solver success, full hard-contraction satisfaction, and zero Lyapunov slack.

Second, the latest RL safety-gate runs are also reliable from a safety-supervision perspective. Both cold-start and pretrained runs have 100% target success and no target failures. The safety gate intervenes on a small minority of steps.

Third, the latest same-run `mpc_only` baselines still beat the RL policies on raw output RMSE. This means the current evidence should not claim that RL outperforms MPC in tracking. The stronger claim is that RL can operate mostly inside the Lyapunov gate and is faster per step, while MPC remains the tracking reference.

Fourth, the current saved-agent evaluation script is not yet aligned with the latest governed-reference training setup. This is the main phase issue. Running it as-is would answer a different question: "How do the latest agents behave with a nominal plant and bounded target selector?"

## Bugs, Inconsistencies, And Risks

- `DirectLyapunovSavedAgentEvaluation.py` now sets `plant_mode = "nominal"`, so disturbance profiles are created but not applied to the plant unless the mode is switched back to `"disturb"`.
- `Simulation/saved_agent_evaluation.py` hard-codes `direct_target_mode="bounded"` for saved RL and `target_mode="bounded"` for `mpc_only` and `direct_lmpc`.
- Saved-agent evaluation uses `rho_lyap = 0.99` and `lyap_eps = 1e-3`, while the latest active training runners use `rho_lyap = 0.98` and `lyap_eps = 1e-9`.
- Saved-agent evaluation uses `Qy_diag = [8.0, 6.0]`, while the latest RL training runners use `Qy_diag = [12.0, 6.0]`.
- Direct Lyapunov `actual_intervention_rate = 1.0` is a controller-mode artifact, not an RL fallback rate. It should not be compared directly to RL intervention rates.
- The latest direct, cold-start, and pretrained runs do not all share the same timestamp. The latest pretrained run is from 2026-06-04, while the latest cold-start and direct runs are from 2026-05-25.

## Literature Connection

The current method remains in the model-based safe RL family: a learned TD3 actor proposes a continuous input, while an MPC/Lyapunov layer certifies or replaces the action. This is consistent with the local project narrative in `StatsControl2026/stats_control_2026_slides.tex` and `report/safe_rl_implementation_summary.md`.

No new external citation was added in this report. For a paper section, the citation pass should explicitly connect three ideas:

- reference governors for admissible command modification,
- stabilizing or Lyapunov MPC for contraction and recursive feasibility language,
- predictive safety filters or safe RL shields for certifying learned actions before plant actuation.

## Recommended Next Experiment

The next experiment should be a clean saved-agent evaluation in two stages.

Stage 1: nominal final-policy evaluation.

- File to edit: `Simulation/saved_agent_evaluation.py`
- Change: pass `target_mode="governed_reference"` and `direct_target_mode="governed_reference"` through the saved-agent helpers.
- Keep: `plant_mode = "nominal"` in `DirectLyapunovSavedAgentEvaluation.py`.
- Metric to watch: final-agent RMSE and `reward_no_penalty` against `mpc_only`.
- Failure mode to watch: RL looks good only because the target selector is not matching the training setup.
- Figure to generate: saved-agent output tracking and fallback/intervention counts.

Stage 2: disturbance final-policy evaluation.

- File to edit: `DirectLyapunovSavedAgentEvaluation.py`
- Change: set `plant_mode = "disturb"` after the nominal evaluation is complete.
- Metric to watch: disturbance RMSE, tail offset, fallback rate, and target mismatch.
- Failure mode to watch: nominal success that does not survive $Q_i$, $Q_s$, and $hA$ changes.
- Figure to generate: per-scenario tracking and fallback count plots for nominal, `qi_step`, `qs_step`, `ha_step`, and `all_step`.

## Remaining Uncertainty

The current run state is strong enough to say the method is stable and well-instrumented. It is not yet strong enough to say the RL agent is better than offset-free MPC.

The decisive missing result is a saved-agent evaluation that is:

- final-policy only,
- governed-reference aligned,
- explicit about nominal versus disturbed plant mode,
- compared against same-tuning `mpc_only`,
- reported with both `reward` and `reward_no_penalty`.

That is the phase we are currently in.
