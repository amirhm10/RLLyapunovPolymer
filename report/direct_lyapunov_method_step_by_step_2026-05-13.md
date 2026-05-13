# Direct Lyapunov Method Step-by-Step

Date: 2026-05-13

## Objective

This note reconstructs, mathematically and step by step, the direct Lyapunov controller family currently used in:

- [DirectLyapunovMPC_FourMethodDisturbance.ipynb](../DirectLyapunovMPC_FourMethodDisturbance.ipynb)
- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](../DirectLyapunovSafetyGateRL_ColdStart.ipynb)

The goal here is not result interpretation. The goal is to write down what the implemented algorithm actually is, in the same order as the code executes it, and in notation that we can later extend or modify.

## Files inspected

- [Simulation/run_rl_lyapunov.py](../Simulation/run_rl_lyapunov.py)
- [Lyapunov/direct_lyapunov_mpc.py](../Lyapunov/direct_lyapunov_mpc.py)
- [Lyapunov/lyapunov_core.py](../Lyapunov/lyapunov_core.py)
- [Lyapunov/frozen_output_disturbance_target.py](../Lyapunov/frozen_output_disturbance_target.py)
- [analysis/steady_state_debug_analysis.py](../analysis/steady_state_debug_analysis.py)
- [utils/helpers.py](../utils/helpers.py)
- [TD3Agent/reward_functions.py](../TD3Agent/reward_functions.py)
- [TD3Agent/agent.py](../TD3Agent/agent.py)

## 1. Scope of the algorithm family

The three notebooks share one common core:

1. An offset-free augmented model provides the observer state.
2. A direct output-disturbance target selector computes an admissible steady target.
3. A Lyapunov contraction test defines whether a candidate action is acceptable.
4. If RL is used, TD3 proposes the candidate action.
5. If the candidate fails, the controller falls back to direct Lyapunov MPC.

The difference between the three notebooks is only where the candidate action comes from:

- `DirectLyapunovMPC_FourMethodDisturbance.ipynb`:
  there is no RL proposal; the direct Lyapunov MPC action is applied every step.
- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`:
  the candidate comes from a TD3 actor initialized from MPC-style pretraining.
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`:
  the candidate comes from a TD3 actor that starts with teacher-driven warmup and then online RL.

## 2. Coordinate system and notation

Unless stated otherwise, the control code works in scaled deviation coordinates.

Let

- $x_k \in \mathbb{R}^{n_x}$ be the physical plant state in controller coordinates
- $d_k \in \mathbb{R}^{n_y}$ be the output-disturbance state
- $z_k = \begin{bmatrix} x_k \\ d_k \end{bmatrix}$ be the augmented state
- $\hat z_k = \begin{bmatrix} \hat x_k \\ \hat d_k \end{bmatrix}$ be the observer estimate
- $u_k \in \mathbb{R}^{n_u}$ be the input in scaled deviation coordinates
- $y_k \in \mathbb{R}^{n_y}$ be the output in scaled deviation coordinates
- $y_{\mathrm{sp},k} \in \mathbb{R}^{n_y}$ be the requested output setpoint
- $(x_{s,k}, u_{s,k}, d_{s,k}, y_{s,k})$ be the selected steady target at time $k$

For the Lyapunov layer we use only the physical-state error

$$
e_{x,k} = \hat x_k - x_{s,k}.
$$

## 3. Augmented plant and observer model

The direct target path used by the current notebooks assumes a frozen output-disturbance augmentation:

$$
x_{k+1} = A x_k + B u_k
$$

$$
d_{k+1} = d_k
$$

$$
y_k = C x_k + d_k.
$$

This structure is enforced in [Lyapunov/frozen_output_disturbance_target.py](../Lyapunov/frozen_output_disturbance_target.py). In particular, the target code assumes:

- no disturbance term in the physical-state dynamics
- zero lower-left augmentation block
- disturbance integrator dynamics $d_{k+1}=d_k$
- output disturbance enters as $C_d = I$

The observer update used in the rollout is

$$
\hat z_{k+1} = A_{\mathrm{aug}} \hat z_k + B_{\mathrm{aug}} u_k + L \bigl(y_k - C_{\mathrm{aug}} \hat z_k\bigr).
$$

In the implementation this is carried in `xhat_aug_store` and updated in [Simulation/run_rl_lyapunov.py](../Simulation/run_rl_lyapunov.py).

## 4. RL observation and action map

This step is used only in the two RL notebooks.

The RL observation is built by concatenating:

- the current augmented observer estimate $\hat z_k$
- the current raw setpoint $y_{\mathrm{sp},k}$
- the previous applied input $u_{k-1}$

after min-max scaling to $[-1,1]$.

So the implemented RL state is

$$
s_k = \mathcal{S}\!\left(\hat z_k,\; y_{\mathrm{sp},k},\; u_{k-1}\right),
$$

where $\mathcal{S}(\cdot)$ denotes the per-channel scaling in [utils/helpers.py](../utils/helpers.py).

The actor outputs

$$
a_k = \pi_\theta(s_k) \in [-1,1]^{n_u},
$$

and this is mapped back into the admissible deviation-input box:

$$
u_k^{\mathrm{RL}} = \mathcal{T}(a_k).
$$

For the current direct RL notebooks:

- `STATE_DIM = n_{\mathrm{aug}} + n_y + n_u`
- `ACTION_DIM = n_u`

so the actor interface is built around raw $y_{\mathrm{sp}}$, not around $y_s$.

## 5. Step A: steady target selection

At each control step, before the Lyapunov test is evaluated, the algorithm computes a local steady target.

### 5.1 Exact target equations

Because the disturbance is frozen in the output equation, the target equations are

$$
(I-A)x_s - B u_s = 0
$$

$$
C x_s = y_{\mathrm{sp},k} - \hat d_k
$$

$$
d_s = \hat d_k
$$

$$
y_s = C x_s + d_s.
$$

Equivalently, if $I-A$ is invertible, then

$$
x_s = (I-A)^{-1} B u_s
$$

and the steady-output matching equation becomes

$$
G u_s = y_{\mathrm{sp},k} - \hat d_k,
\qquad
G := C(I-A)^{-1}B.
$$

This is the exact admissible-target problem solved in [analysis/steady_state_debug_analysis.py](../analysis/steady_state_debug_analysis.py) and wrapped by [Lyapunov/frozen_output_disturbance_target.py](../Lyapunov/frozen_output_disturbance_target.py).

### 5.2 Bounded target problem

If the exact steady input violates bounds, the current notebooks use the bounded target mode. Then the target is computed from a bounded least-squares problem of the form

$$
\min_{x_s,u_s}
\left\|
\begin{bmatrix}
(I-A)x_s - B u_s \\
C x_s - (y_{\mathrm{sp},k} - \hat d_k)
\end{bmatrix}
\right\|_2^2
+
\|u_s - u_{\mathrm{ref}}\|_{W_u}^2
+
\|x_s - x_{\mathrm{ref}}\|_{W_x}^2
$$

subject to

$$
u_{\min} \le u_s \le u_{\max}.
$$

The optional regularization terms are exactly the mechanism behind the four direct methods:

- no regularization: `bounded_hard`
- input anchoring: `bounded_hard_u_prev_*`
- state anchoring: `bounded_hard_xs_prev_*`
- both: `bounded_hard_u_prev_*_xs_prev_*`

Here:

- $u_{\mathrm{ref}}$ is typically the previous applied input $u_{k-1}$
- $x_{\mathrm{ref}}$ is typically the previously successful target state
- $W_u$ and $W_x$ are the anchor weights

In reduced form, when $x_s = (I-A)^{-1}Bu_s$ is available, the bounded solve becomes

$$
\min_{u_s}
\|G u_s - (y_{\mathrm{sp},k} - \hat d_k)\|_2^2
+
\|u_s - u_{\mathrm{ref}}\|_{W_u}^2
+
\|(I-A)^{-1}Bu_s - x_{\mathrm{ref}}\|_{W_x}^2
$$

subject to

$$
u_{\min} \le u_s \le u_{\max}.
$$

The solver returns diagnostics such as:

- success flag
- whether the exact or bounded solution was used
- residual norms
- condition numbers
- active lower and upper bounds
- input-anchor and state-anchor penalties

## 6. Step B: Lyapunov ingredients

The direct gate and the direct MPC fallback both use a Lyapunov matrix for the physical-state error.

The code constructs

$$
Q_x = C^\top Q_y C + \varepsilon I
$$

and then solves the discrete-time Riccati equation to obtain

$$
P_x = \mathrm{DARE}(A,B,Q_x,R_u),
$$

with corresponding linear feedback

$$
K_x = -\left(R_u + B^\top P_x B\right)^{-1} B^\top P_x A.
$$

The Lyapunov value is

$$
V_k = V(e_{x,k}) = e_{x,k}^\top P_x e_{x,k}.
$$

This is implemented in [Lyapunov/lyapunov_core.py](../Lyapunov/lyapunov_core.py).

## 7. Step C: candidate-action Lyapunov gate

This step is used only in the RL notebooks.

Given the RL proposal $u_k^{\mathrm{RL}}$, the gate predicts the one-step physical-state error under the selected target:

$$
e_{x,k+1}^{\mathrm{cand}} = A e_{x,k} + B\bigl(u_k^{\mathrm{RL}} - u_{s,k}\bigr).
$$

Then it computes

$$
V_k = e_{x,k}^\top P_x e_{x,k}
$$

$$
V_{k+1}^{\mathrm{cand}} =
\left(e_{x,k+1}^{\mathrm{cand}}\right)^\top
P_x
e_{x,k+1}^{\mathrm{cand}}
$$

and the Lyapunov bound

$$
V_{\mathrm{bound},k} = \rho V_k + \varepsilon_{\mathrm{lyap}}.
$$

The candidate is accepted if and only if all three tests pass:

$$
u_{\min} \le u_k^{\mathrm{RL}} \le u_{\max}
$$

$$
\Delta u_{\min} \le u_k^{\mathrm{RL}} - u_{k-1} \le \Delta u_{\max}
$$

$$
V_{k+1}^{\mathrm{cand}} \le \rho V_k + \varepsilon_{\mathrm{lyap}}.
$$

In the current notebooks:

- $\rho = 0.99$
- $\varepsilon_{\mathrm{lyap}} = 10^{-9}$

and the gate returns explicit reject reasons:

- `input_bounds`
- `move_bounds`
- `lyapunov`
- `target_unavailable`

## 8. Step D: direct Lyapunov MPC fallback

If the RL candidate is rejected, the supervisor calls the direct Lyapunov tracking MPC solver.

The fallback solves a finite-horizon optimization over an input sequence $\{u_0,\dots,u_{N_C-1}\}$ and a predicted state trajectory $\{z_1,\dots,z_{N_P}\}$.

The implemented objective is of the form

$$
\min
\sum_{i=0}^{N_P-1}\|y_{k+i+1} - y_{\mathrm{target},k}\|_{Q_y}^2
+
\mathbf{1}_{S_u}\sum_{i=0}^{N_C-1}\|u_i-u_{s,k}\|_{S_u}^2
+
\|u_0-u_{k-1}\|_{R_{\Delta u}}^2
+
\sum_{i=1}^{N_C-1}\|u_i-u_{i-1}\|_{R_{\Delta u}}^2
+
\mathbf{1}_{P}\|x_{N_P}-x_{s,k}\|_{P_x}^2
$$

subject to

$$
z_{i+1} = A_{\mathrm{aug}} z_i + B_{\mathrm{aug}} u_{\min(i,N_C-1)}
$$

$$
u_{\min} \le u_i \le u_{\max}.
$$

The solver can also enforce:

1. A terminal set constraint

$$
(x_{N_P} - x_{s,k})^\top P_x (x_{N_P} - x_{s,k}) \le \alpha_k
$$

2. A first-step contraction constraint

$$
(x_1 - x_{s,k})^\top P_x (x_1 - x_{s,k}) \le \rho V_k + \varepsilon_{\mathrm{lyap}}.
$$

The most important target-definition choice is

$$
y_{\mathrm{target},k} =
\begin{cases}
y_{s,k}, & \text{if target-output tracking is enabled} \\
y_{\mathrm{sp},k}, & \text{if raw-setpoint tracking is enabled.}
\end{cases}
$$

For the current saved direct notebooks, the fallback is called with raw-setpoint tracking:

$$
y_{\mathrm{target},k} = y_{\mathrm{sp},k}.
$$

This is the key structural mismatch in the current method family:

- the Lyapunov certificate is centered on $(x_s,u_s)$
- the direct tracking objective still pushes on raw $y_{\mathrm{sp}}$

so safety and tracking are not perfectly aligned when $y_s \neq y_{\mathrm{sp}}$.

### Failure handling

With the current notebook settings:

- if target selection fails, the controller holds the previous input
- if fallback MPC fails, the controller also holds the previous input

because `use_target_on_solver_fail=False` in the direct path.

## 9. Step E: plant step, observer correction, reward, and replay

Once the safe input $u_k^{\mathrm{safe}}$ is chosen, the algorithm:

1. applies it to the plant
2. steps the plant
3. measures the next output
4. updates the observer
5. computes reward
6. stores the transition for RL, when RL is active

### 9.1 Applied input

The applied input is

$$
u_k^{\mathrm{safe}} =
\begin{cases}
u_k^{\mathrm{RL}}, & \text{if the candidate is accepted} \\
u_k^{\mathrm{MPC}}, & \text{if fallback succeeds} \\
u_{k-1}, & \text{if the fallback path fails.}
\end{cases}
$$

### 9.2 Reward

The RL notebooks use the relative-band reward built in [TD3Agent/reward_functions.py](../TD3Agent/reward_functions.py).

At a high level, with output error

$$
e_{y,k+1} = y_{k+1} - y_{\mathrm{sp},k}
$$

and move

$$
\Delta u_k = u_k^{\mathrm{safe}} - u_{k-1},
$$

the reward has the form

$$
r_k = -\text{tracking penalty}
       -\text{move penalty}
       -\text{outside-band penalty}
       -\text{inside-band linear penalty}
       +\text{inside-band bonus}.
$$

The exact formula depends on:

- relative output bands
- band floors in physical units
- geometric gating of inside-band behavior
- quadratic output and move weights

This reward is evaluated against raw $y_{\mathrm{sp},k}$, not against $y_{s,k}$.

### 9.3 Stored transition

The RL transition stored in the replay buffer is

$$
\bigl(s_k,\; a_k,\; r_k,\; s_{k+1}\bigr)
$$

with

$$
s_{k+1} = \mathcal{S}\!\left(\hat z_{k+1},\; y_{\mathrm{sp},k},\; u_k^{\mathrm{safe}}\right).
$$

The next state intentionally keeps the same active setpoint $y_{\mathrm{sp},k}$ at a setpoint boundary, so the action and reward remain tied to one task definition.

## 10. Training-phase differences between the RL notebooks

The direct safety-gate method is the same in both RL notebooks. What changes is how the actor is initialized and trained.

### 10.1 Pretrained RL notebook

- warmup behavior source: `policy`
- actor interface already fixed to the baseline MPC-pretrained state definition
- gate checks the candidate and falls back when needed

So this notebook is best interpreted as:

$$
\text{pretrained TD3 actor} \;+\; \text{direct Lyapunov gate} \;+\; \text{direct MPC fallback}.
$$

### 10.2 Cold-start RL notebook

- warmup behavior source: `direct_lyapunov_mpc`
- behavior-clone teacher episodes are used before full RL
- later full RL uses the same direct gate and fallback structure

So this notebook is best interpreted as:

$$
\text{teacher-driven direct MPC initialization} \;+\; \text{TD3 learning} \;+\; \text{direct Lyapunov gate}.
$$

This cold-start setup is the cleaner place to test observation changes later, because it is not tied to an MPC-pretrained actor interface.

## 11. Step-by-step algorithm summary

The common online algorithm can be written as:

```text
Given current observer estimate zhat_k, previous input u_{k-1}, and setpoint y_sp,k:

1. If RL is active, build s_k = scale(zhat_k, y_sp,k, u_{k-1}).

2. Solve the direct steady-target problem:
      (x_s,k, u_s,k, d_s,k, y_s,k) = TargetSelector(zhat_k, y_sp,k, u_{k-1}).

3. If RL is active, propose u_k^RL = T(pi_theta(s_k)).
   Otherwise skip directly to Step 5.

4. Evaluate the candidate:
      accept if
      - input bounds hold
      - move bounds hold
      - V_{k+1}^cand <= rho V_k + eps_lyap

5. If RL candidate is rejected, solve direct tracking MPC around the selected target.

6. Apply the final safe input to the plant.

7. Update the observer and compute reward.

8. If RL is active, store the transition and update TD3 according to the current training phase.
```

## 12. What the current mathematics already tells us

This reconstruction already exposes the main design tension in the present notebooks.

### 12.1 Safety is centered on the admissible target

The direct Lyapunov certificate is about contraction of

$$
e_{x,k} = \hat x_k - x_{s,k},
$$

not direct contraction of raw output error $y_k - y_{\mathrm{sp},k}$.

### 12.2 Reward and RL observation are centered on the raw setpoint

The RL actor observes raw $y_{\mathrm{sp},k}$ and the reward is also evaluated against raw $y_{\mathrm{sp},k}$.

### 12.3 The current fallback tracking objective is also centered on the raw setpoint

In the saved notebook settings,

$$
y_{\mathrm{target},k} = y_{\mathrm{sp},k},
$$

even though the admissible target package contains $y_{s,k}$.

### 12.4 Therefore one-step contraction does not imply exact raw-setpoint settling

If $y_{s,k} \neq y_{\mathrm{sp},k}$, the following can all be true at once:

- the Lyapunov check is satisfied
- the RL action is certified
- the fallback MPC is consistent with the certificate around $(x_s,u_s)$
- the closed loop still shows raw-setpoint offset or jitter

That is not a contradiction. It is a consequence of solving two different objectives:

- contraction around $(x_s,u_s)$
- tracking pressure toward raw $y_{\mathrm{sp}}$

## 13. Extension questions for the next report version

This note is the base method report. The next extension should probably answer these questions explicitly:

1. Should the fallback direct MPC track $y_s$ or $y_{\mathrm{sp}}$?
2. Should the RL observation stay at raw $y_{\mathrm{sp}}$, or should cold-start RL also receive admissible-target information?
3. Should persistent offset trigger fallback even when the one-step Lyapunov check passes?
4. Should gate interventions be reused as supervised teacher data so the actor becomes more Lyapunov-consistent online?
5. Should we save a compact per-step export of $y$, $y_{\mathrm{sp}}$, $y_s$, $u$, $u_s$, $V_k$, and $V_{k+1}$ for future proofs and figures?

Those are natural next steps because the mathematical structure is now explicit enough to modify one block at a time.
