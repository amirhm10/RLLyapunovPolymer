# Direct Lyapunov Method With And Without RL

Date: 2026-05-16

## Objective

This note explains the current direct Lyapunov method family in the repository at three levels:

1. direct Lyapunov MPC without RL
2. direct Lyapunov safety-gated RL with a pretrained TD3 initialization
3. direct Lyapunov safety-gated RL with a cold-start TD3 agent

It also explains how behavioral cloning works in two different places in this codebase:

- the offline MPC-to-TD3 pretraining notebook
- the online teacher-driven behavioral-cloning phase used inside the direct RL notebooks

The goal is to be mathematically explicit and implementation-faithful. This report follows the code paths that are active now, rather than describing an idealized variant.

## Primary files inspected

- [DirectLyapunovMPC_FourMethodDisturbance.ipynb](../DirectLyapunovMPC_FourMethodDisturbance.ipynb)
- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](../DirectLyapunovSafetyGateRL_ColdStart.ipynb)
- [pretraining_rl_controller.ipynb](../pretraining_rl_controller.ipynb)
- [Lyapunov/direct_lyapunov_mpc.py](../Lyapunov/direct_lyapunov_mpc.py)
- [Lyapunov/lyapunov_core.py](../Lyapunov/lyapunov_core.py)
- [Simulation/run_rl_lyapunov.py](../Simulation/run_rl_lyapunov.py)
- [TD3Agent/agent.py](../TD3Agent/agent.py)
- [TD3Agent/reward_functions.py](../TD3Agent/reward_functions.py)
- [utils/helpers.py](../utils/helpers.py)
- [utils/td3_helpers.py](../utils/td3_helpers.py)
- [utils/direct_lyapunov_study.py](../utils/direct_lyapunov_study.py)

## 1. What the three notebooks are actually doing

All three direct notebooks share the same backbone:

1. estimate an augmented offset-free state
2. compute an admissible steady target for the current setpoint
3. build a Lyapunov certificate on the physical-state error around that target
4. choose a control action
5. apply the action, step the plant, update the observer, and log reward

The only difference is where the candidate action comes from.

For [DirectLyapunovMPC_FourMethodDisturbance.ipynb](../DirectLyapunovMPC_FourMethodDisturbance.ipynb):

- there is no RL proposal
- the action comes directly from the direct Lyapunov MPC solve every step

For [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb):

- the candidate action comes from a TD3 actor
- the actor is initialized by loading `Data/agent_2507171027.pkl`
- the candidate is accepted only if it passes the direct Lyapunov gate
- otherwise the controller falls back to direct Lyapunov MPC

For [DirectLyapunovSafetyGateRL_ColdStart.ipynb](../DirectLyapunovSafetyGateRL_ColdStart.ipynb):

- the same gate and fallback logic are used
- the TD3 agent starts from fresh random weights
- a teacher-driven phase helps the actor before full RL takes over

## 2. Shared experiment structure

The current direct study helpers define a two-setpoint disturbance study with:

```text
y_{\mathrm{sp}}^{(1)} = \begin{bmatrix}4.5 \\ 324.0\end{bmatrix},
\qquad
y_{\mathrm{sp}}^{(2)} = \begin{bmatrix}3.4 \\ 321.0\end{bmatrix},
```

repeated over `n_tests = 200` cycles with `set_points_len = 800` time steps per setpoint block.

The active direct case generator in [utils/direct_lyapunov_study.py](../utils/direct_lyapunov_study.py) supports four target-selector variants:

- `bounded_hard`
- `bounded_hard_u_prev_*`
- `bounded_hard_xs_prev_*`
- `bounded_hard_u_prev_*_xs_prev_*`

The current direct notebook calls select only:

- `bounded_hard`
- the mixed anchor-and-smoothness case with `u_ref_weight = 0.25` and `x_ref_weight = 0.25`

All three notebooks also share the same main direct-MPC tuning:

- prediction horizon `N_P = 9`
- control horizon `N_C = 3`
- tracking weights `Q_y = diag(5, 1)`
- move weights `R_{\Delta u} = diag(1, 1)`
- Lyapunov contraction factor `rho = 0.99`
- Lyapunov tolerance term `eps_lyap = 1e-9`
- soft-contraction penalty parameter `1e6` even though the current notebooks run the hard-contraction path
- `first_step_contraction_on = True`
- `direct_tracking_use_target_output = False`
- `terminal_set_on = True` when the direct solver is constructed

That last point matters later: the direct method is centered on a first-step contraction idea, but the fallback solver is still built with the terminal-set machinery available.

## 3. Coordinates and notation

The implementation works mostly in scaled deviation coordinates.

Let:

- `x_k \in \mathbb{R}^{n_x}` be the physical plant state in controller coordinates
- `d_k \in \mathbb{R}^{n_y}` be the output-disturbance state
- `z_k = \begin{bmatrix}x_k \\ d_k\end{bmatrix}` be the augmented state
- `\hat z_k = \begin{bmatrix}\hat x_k \\ \hat d_k\end{bmatrix}` be the observer estimate
- `u_k \in \mathbb{R}^{n_u}` be the deviation input
- `y_k \in \mathbb{R}^{n_y}` be the deviation output
- `y_{\mathrm{sp},k} \in \mathbb{R}^{n_y}` be the requested output setpoint
- `(x_{s,k}, u_{s,k}, d_{s,k}, y_{s,k})` be the direct steady target selected at time step `k`

The Lyapunov layer uses the physical-state error only:

```text
e_{x,k} = \hat x_k - x_{s,k}.
```

The current RL interface uses:

- RL state dimension `n_aug + n_y + n_u`
- RL action dimension `n_u`

so the actor sees the augmented observer state, the raw requested setpoint, and the previous applied input.

## 4. Augmented model and observer

The direct notebooks call `load_and_prepare_system_data(...)` with:

- `augmentation_style = "rawlings"`
- `augmentation_mode = "output_disturbance"`

The effective linear augmented model used by the direct target and direct RL logic is:

```text
x_{k+1} = A x_k + B u_k
```

```text
d_{k+1} = d_k
```

```text
y_k = C x_k + d_k.
```

So the disturbance is frozen in the output equation and does not drive the physical-state dynamics.

The observer update used in the rollout is:

```text
\hat z_{k+1}
=
A_{\mathrm{aug}} \hat z_k
+
B_{\mathrm{aug}} u_k
+
L\left(y_k - C_{\mathrm{aug}} \hat z_k\right).
```

This is the common backbone for both the direct-only and RL-gated paths.

## 5. RL state and action map

The RL state is built by [utils/helpers.py](../utils/helpers.py) as:

```text
s_k = \mathcal{S}\!\left(\hat z_k,\; y_{\mathrm{sp},k},\; u_{k-1}\right),
```

where each channel is min-max mapped into `[-1,1]`.

More explicitly:

```text
s_k =
\begin{bmatrix}
2\frac{\hat z_k - z_{\min}}{z_{\max} - z_{\min}} - 1 \\
2\frac{y_{\mathrm{sp},k} - y_{\mathrm{sp},\min}}{y_{\mathrm{sp},\max} - y_{\mathrm{sp},\min}} - 1 \\
2\frac{u_{k-1} - u_{\min}}{u_{\max} - u_{\min}} - 1
\end{bmatrix}.
```

The actor outputs

```text
a_k = \pi_\theta(s_k) \in [-1,1]^{n_u}.
```

That action is mapped back into the admissible deviation-input box:

```text
u_k^{\mathrm{RL}} = \mathcal{T}(a_k),
```

where the code uses an affine map from `[-1,1]` to `[u_{\min}, u_{\max}]`.

## 6. Direct method without RL

This is the logic in [DirectLyapunovMPC_FourMethodDisturbance.ipynb](../DirectLyapunovMPC_FourMethodDisturbance.ipynb).

### 6.1 Step 1: estimate the current operating point

At each time step the controller forms:

- the augmented estimate `\hat z_k`
- the previous input `u_{k-1}`
- the requested setpoint `y_{\mathrm{sp},k}`

These are the inputs to the direct target selector.

### 6.2 Step 2: solve the direct steady target problem

The direct selector first tries to find an admissible steady target satisfying:

```text
(I-A)x_{s,k} - B u_{s,k} = 0
```

```text
C x_{s,k} = y_{\mathrm{sp},k} - \hat d_k
```

```text
d_{s,k} = \hat d_k
```

```text
y_{s,k} = C x_{s,k} + d_{s,k}.
```

If the exact target is not input-feasible, the selector solves a bounded least-squares problem:

```text
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
```

subject to:

```text
u_{\min} \le u_s \le u_{\max}.
```

In the current direct notebooks:

- `bounded_hard` means `W_u = 0` and `W_x = 0`
- the mixed case uses both an input anchor and a state anchor
- `u_ref` is the previous applied input
- `x_ref` is the previously successful target state

### 6.3 Step 3: construct the Lyapunov certificate

The direct method uses a physical-state Lyapunov function:

```text
V(e_x) = e_x^\top P_x e_x.
```

The matrix `P_x` comes from the discrete algebraic Riccati equation based on:

```text
Q_x = C^\top Q_y C + \varepsilon I,
```

with a corresponding local feedback gain

```text
K_x = -\left(R_u + B^\top P_x B\right)^{-1} B^\top P_x A.
```

The one-step contraction bound is:

```text
V_{k+1} \le \rho V_k + \varepsilon_{\mathrm{lyap}}.
```

The terminal admissible-level parameter is computed from the input headroom around `u_s`:

```text
\alpha
=
\min_i
\left(
\frac{\min(u_{\max,i} - u_{s,i},\; u_{s,i} - u_{\min,i})}{\gamma_i}
\right)^2,
```

where `\gamma_i^2 = k_i P_x^{-1} k_i^\top` for row `k_i` of `K_x`.

### 6.4 Step 4: solve the direct tracking MPC

After target selection, the no-RL notebook solves the direct tracking MPC:

```text
\min_{\{u_i\},\{z_i\}}
\sum_{i=0}^{N_P-1}
\|y_{k+i+1} - y_{\mathrm{target},k}\|_{Q_y}^2
+
\|u_0 - u_{k-1}\|_{R_{\Delta u}}^2
+
\sum_{i=1}^{N_C-1}\|u_i - u_{i-1}\|_{R_{\Delta u}}^2
```

subject to:

```text
z_{i+1} = A_{\mathrm{aug}} z_i + B_{\mathrm{aug}} u_{\min(i,N_C-1)}
```

```text
u_{\min} \le u_i \le u_{\max}
```

and, when active,

```text
(x_1 - x_{s,k})^\top P_x (x_1 - x_{s,k})
\le
\rho V_k + \varepsilon_{\mathrm{lyap}}
```

and sometimes also

```text
(x_{N_P} - x_{s,k})^\top P_x (x_{N_P} - x_{s,k}) \le \alpha_k.
```

An important implementation detail is:

```text
y_{\mathrm{target},k} =
\begin{cases}
y_{s,k}, & \text{if target-output tracking is enabled} \\
y_{\mathrm{sp},k}, & \text{otherwise.}
\end{cases}
```

The current direct notebooks set:

```text
y_{\mathrm{target},k} = y_{\mathrm{sp},k},
```

because `use_target_output_for_tracking = False`.

So the Lyapunov certificate is centered on `(x_s,u_s)`, while the tracking objective still pulls toward the raw requested setpoint.

### 6.5 Step 5: apply, update, and score

Once the direct MPC returns a first move `u_k`:

1. convert the scaled deviation move back to physical inputs
2. apply it to the plant
3. step the plant
4. update the observer
5. compute reward for logging

The direct-only notebook is not training an RL agent, but it still computes the same tracking-style reward so performance summaries remain comparable.

## 7. Direct RL method: common gate-and-fallback structure

Both RL notebooks use the same online direct safety architecture.

### 7.1 Step 1: build the RL state

The code forms:

```text
s_k = \mathcal{S}\!\left(\hat z_k,\; y_{\mathrm{sp},k},\; u_{k-1}\right).
```

### 7.2 Step 2: generate a candidate action

Depending on the training phase, the candidate comes from either:

- the TD3 actor
- or a teacher action computed by direct Lyapunov MPC

In either case the behavior is expressed first in actor coordinates `a_k \in [-1,1]^{n_u}` and then mapped to the physical controller coordinates:

```text
u_k^{\mathrm{cand}} = \mathcal{T}(a_k).
```

### 7.3 Step 3: recompute the direct steady target

Before the candidate is judged, the gate recomputes:

```text
(x_{s,k}, u_{s,k}, d_{s,k}, y_{s,k}).
```

So the candidate is never checked against an old nominal target. It is checked against the target selected from the current observer state and the current requested setpoint.

### 7.4 Step 4: evaluate the candidate analytically

The candidate check in [Lyapunov/lyapunov_core.py](../Lyapunov/lyapunov_core.py) predicts the next physical-state error:

```text
e_{x,k+1}^{\mathrm{cand}} = A e_{x,k} + B\left(u_k^{\mathrm{cand}} - u_{s,k}\right).
```

Then it computes:

```text
V_k = e_{x,k}^\top P_x e_{x,k}
```

```text
V_{k+1}^{\mathrm{cand}}
=
\left(e_{x,k+1}^{\mathrm{cand}}\right)^\top
P_x
e_{x,k+1}^{\mathrm{cand}}
```

```text
V_{\mathrm{bound},k} = \rho V_k + \varepsilon_{\mathrm{lyap}}.
```

The candidate is accepted if all active tests pass:

```text
u_{\min} \le u_k^{\mathrm{cand}} \le u_{\max}
```

```text
\Delta u_{\min} \le u_k^{\mathrm{cand}} - u_{k-1} \le \Delta u_{\max}
```

```text
V_{k+1}^{\mathrm{cand}} \le \rho V_k + \varepsilon_{\mathrm{lyap}}.
```

For the current direct RL notebooks, `du_min` and `du_max` are not passed, so the active checks reduce to:

```text
u_{\min} \le u_k^{\mathrm{cand}} \le u_{\max}
```

and

```text
V_{k+1}^{\mathrm{cand}} \le \rho V_k + \varepsilon_{\mathrm{lyap}}.
```

So the current reject reasons are effectively:

- `input_bounds`
- `lyapunov`
- `target_unavailable`

### 7.5 Step 5: either accept the candidate or fall back

If the candidate passes:

```text
u_k^{\mathrm{safe}} = u_k^{\mathrm{cand}}.
```

If the candidate fails, the supervisor calls the same direct tracking MPC described in Section 6 and uses its first move:

```text
u_k^{\mathrm{safe}} = u_k^{\mathrm{MPC}}.
```

If the direct target solve fails or the fallback solve fails, the current code holds the previous input:

```text
u_k^{\mathrm{safe}} = u_{k-1}.
```

### 7.6 Step 6: plant step, observer update, reward, and replay

After the safe action is chosen, the code:

1. applies the safe action to the plant
2. steps the plant
3. updates the observer
4. computes reward
5. stores the RL transition when the current cycle is a training cycle

The reward is computed against the raw requested setpoint:

```text
e_{y,k+1} = y_{k+1} - y_{\mathrm{sp},k}
```

```text
\Delta u_k = u_k^{\mathrm{safe}} - u_{k-1}.
```

The stored next RL state is:

```text
s_{k+1} = \mathcal{S}\!\left(\hat z_{k+1},\; y_{\mathrm{sp},k},\; u_k^{\mathrm{safe}}\right).
```

The code intentionally keeps the same active setpoint `y_{\mathrm{sp},k}` at the transition boundary so the action, reward, and next state all refer to the same task definition.

## 8. The exact reward used by the direct RL notebooks

The direct RL notebooks use `make_reward_fn_relative_QR(...)` with:

- `Q = diag(5, 1)`
- `R = diag(1, 1)`
- `k_rel = [0.003, 0.0003]`
- `band_floor_phys = [0.006, 0.07]`
- `tau_frac = 0.7`
- `gamma_out = 0.5`
- `gamma_in = 0.5`
- `beta = 7.0`
- `gate = "geom"`
- `lam_in = 1.0`
- `bonus_kind = "exp"`
- `bonus_k = 12.0`

For each output channel `i`, the physical tolerance band is:

```text
b_i^{\mathrm{phys}} =
\max\!\left(k_{\mathrm{rel},i}|y_{\mathrm{sp},i}^{\mathrm{phys}}|,\; b_{i,\mathrm{floor}}^{\mathrm{phys}}\right).
```

After scaling into controller coordinates:

```text
b_i = \frac{b_i^{\mathrm{phys}}}{\Delta y_i},
\qquad
\tau_i = 0.7\, b_i.
```

The smooth inside-band gate is:

```text
s_i = \sigma\!\left(\frac{b_i - |e_i|}{\tau_i}\right),
```

and for the current `geom` choice:

```text
w_{\mathrm{in}} =
\left(\prod_i s_i\right)^{1/n_y}.
```

The quadratic error and move penalties are:

```text
J_e = \sum_i Q_i e_i^2,
\qquad
J_u = \sum_j R_j (\Delta u_j)^2.
```

The code then adds linear penalties near and outside the band and an inside-band bonus. With

```text
z_i = \frac{|e_i|}{b_i},
```

the exponential bonus shape is:

```text
\phi(z_i) =
\frac{e^{-k z_i} - e^{-k}}{1 - e^{-k}},
\qquad k = 12.
```

The final reward is:

```text
r_k
=
-
\left(
J_{e,\mathrm{eff}}
+
J_u
+
J_{\mathrm{out}}
+
J_{\mathrm{in}}
\right)
+
J_{\mathrm{bonus}}.
```

So the direct RL agent is not optimizing the Lyapunov value directly. It is optimizing a shaped tracking-and-move reward while the gate enforces Lyapunov admissibility.

## 9. Pretrained direct RL notebook

The notebook [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb) loads a TD3 checkpoint:

```text
\theta \leftarrow \theta_{\mathrm{loaded}},
\qquad
\phi \leftarrow \phi_{\mathrm{loaded}},
```

by calling `case_agent.load(agent_path)`.

In the current `load(...)` implementation this restores:

- actor weights
- critic weights

and then rebuilds fresh AdamW optimizers. It does not restore optimizer state from the checkpoint.

So "pretrained" here means:

- network parameters start from a previously saved solution
- optimizer momentum history does not

The notebook still uses an online teacher phase:

- `WARMUP_EPISODES = 0`
- `BC_TEACHER_EPISODES = 20`
- `bc_actor_updates_per_step = 4`
- `bc_exploration_std = 0.005`

So the pretrained notebook is not "checkpoint then immediate pure TD3". It is:

```text
\text{loaded TD3 weights}
\;+\;
\text{20 teacher-driven BC cycles}
\;+\;
\text{full RL under the direct gate}.
```

## 10. Cold-start direct RL notebook

The notebook [DirectLyapunovSafetyGateRL_ColdStart.ipynb](../DirectLyapunovSafetyGateRL_ColdStart.ipynb) uses the same online gate, target selector, reward, and fallback solver, but the agent starts from freshly initialized weights.

Its current phase schedule is also:

- `WARMUP_EPISODES = 0`
- `BC_TEACHER_EPISODES = 20`
- `bc_actor_updates_per_step = 4`
- `bc_exploration_std = 0.005`

So the currently saved cold-start notebook begins with:

```text
\theta_0,\phi_0 \text{ random}
```

and then immediately enters the teacher-driven behavioral-cloning phase.

A subtle but important note:

- `warmup_behavior_source` differs between the pretrained and cold-start notebooks
- but `WARMUP_EPISODES = 0` in both

So that warmup-source difference does not currently affect execution.

## 11. Behavioral cloning: there are two different BC mechanisms here

This repository uses the phrase "behavioral cloning" in two distinct ways.

### 11.1 Offline BC in `pretraining_rl_controller.ipynb`

This notebook creates a large synthetic dataset using an MPC teacher.

The helper [utils/td3_helpers.py](../utils/td3_helpers.py) samples random tuples:

```text
(x_d,\; y_{\mathrm{sp}},\; u_{\mathrm{prev}})
```

and solves an MPC problem to get the teacher first move:

```text
u^{\mathrm{MPC}} = \pi_{\mathrm{MPC}}(x_d,\; y_{\mathrm{sp}},\; u_{\mathrm{prev}}).
```

The offline replay state is built as:

```text
s = \mathcal{S}(x_d,\; y_{\mathrm{sp}},\; u_{\mathrm{prev}})
```

and the label action is the scaled MPC move:

```text
a^{\star} = \mathcal{S}_u(u^{\mathrm{MPC}}).
```

The notebook fills about 4.9 million generic samples and 100,000 near-steady-state samples, then trains in two stages.

Stage 1: actor behavioral cloning

```text
\min_\theta
\mathbb{E}_{(s,a^\star)}
\left[
\|\pi_\theta(s) - a^\star\|_2^2
\right].
```

This is exactly what `TD3Agent.pretrain_from_buffer(...)` does in its actor stage.

Stage 2: critic TD fitting with the actor frozen

```text
y = r + \gamma Q_{\phi^-}(s', \pi_{\theta^-}(s'))
```

```text
\min_\phi
\mathbb{E}
\left[
\ell_{\mathrm{Huber}}(Q_{\phi,1}(s,a), y)
+
\ell_{\mathrm{Huber}}(Q_{\phi,2}(s,a), y)
\right].
```

So the offline pretraining notebook is best described as:

```text
\text{supervised imitation of MPC}
\;+\;
\text{offline critic fitting},
```

not as online RL on the plant.

### 11.2 Online BC inside the direct RL notebooks

The direct RL notebooks use a different behavioral-cloning mechanism during the first `BC_TEACHER_EPISODES` cycles.

During that phase, the code sets:

- `policy_phase = "behavior_clone_teacher"`
- `use_teacher_behavior = True`
- `run_critic_only_update = True`
- `run_actor_bc_update = True`
- `run_td3_full_update = False`

At each step:

1. solve the direct target problem
2. solve the direct Lyapunov tracking MPC
3. convert its first move to actor coordinates
4. optionally add Gaussian noise with standard deviation `0.005`
5. pass that action through the same direct safety gate and plant interface

The teacher action before noise is:

```text
a_k^{\mathrm{teacher}} = \mathcal{T}^{-1}(u_k^{\mathrm{direct}}).
```

If Gaussian teacher noise is active:

```text
\tilde a_k^{\mathrm{teacher}} = a_k^{\mathrm{teacher}} + \epsilon_k,
\qquad
\epsilon_k \sim \mathcal{N}(0,\sigma_{\mathrm{BC}}^2 I).
```

After mapping, clipping, and safety checking, the executed safe action is:

```text
a_k^{\mathrm{used}} = \mathcal{T}^{-1}(u_k^{\mathrm{safe}}).
```

This is the important implementation detail:

- the replay buffer stores the executed safe action
- the BC buffer also stores the executed safe action

not the raw teacher action before filtering.

So the online BC data pair is:

```text
(s_k,\; a_k^{\mathrm{used}}).
```

The BC buffer insertion is:

```text
\mathcal{D}_{\mathrm{BC}} \leftarrow \mathcal{D}_{\mathrm{BC}} \cup \{(s_k, a_k^{\mathrm{used}})\}.
```

The actor BC update is:

```text
\min_\theta
\mathbb{E}_{(s,a)\sim\mathcal{D}_{\mathrm{BC}}}
\left[
\|\pi_\theta(s) - a\|_2^2
\right].
```

The code performs this actor imitation step four times per plant step in the BC phase because:

- `bc_actor_updates_per_step = 4`

At the same time, the ordinary replay buffer receives:

```text
(s_k,\; a_k^{\mathrm{used}},\; r_k,\; s_{k+1},\; 0).
```

and the critic is updated with TD targets while the actor is updated only by BC, not by the TD3 policy-gradient step.

So the online BC phase is:

```text
\text{teacher-generated plant interaction}
\;+\;
\text{critic TD learning}
\;+\;
\text{actor MSE imitation of executed safe actions}.
```

That is different from the offline pretraining notebook even though both are called behavioral cloning.

## 12. What happens after the BC phase

Once the step index passes the BC block:

- `use_teacher_behavior = False`
- `run_critic_only_update = False`
- `run_actor_bc_update = False`
- `run_td3_full_update = True`

Now the actor generates the candidate directly, exploration switches to the full-RL schedule, and TD3 runs in the usual way:

```text
\nabla_\theta J
\approx
\nabla_\theta Q_{\phi,1}(s,\pi_\theta(s)).
```

But the direct gate remains active, so the plant still receives:

```text
u_k^{\mathrm{safe}}
=
\begin{cases}
u_k^{\mathrm{RL}}, & \text{if accepted} \\
u_k^{\mathrm{direct}}, & \text{if rejected and fallback succeeds} \\
u_{k-1}, & \text{if the fallback path fails.}
\end{cases}
```

## 13. Pretrained versus cold-start: the clean comparison

The current pretrained and cold-start direct RL notebooks differ mainly in initial network parameters.

Pretrained:

```text
(\theta_0,\phi_0) = (\theta_{\mathrm{checkpoint}}, \phi_{\mathrm{checkpoint}})
```

Cold-start:

```text
(\theta_0,\phi_0) = (\theta_{\mathrm{random}}, \phi_{\mathrm{random}})
```

After initialization, both notebooks currently run:

- zero warmup cycles
- twenty teacher-driven BC cycles
- then full RL under the same direct gate

So the real experimental question is not just:

```text
\text{pretrained vs cold-start}
```

but more precisely:

```text
\text{pretrained initialization plus online teacher BC}
\quad \text{vs} \quad
\text{random initialization plus online teacher BC}.
```

## 14. Important implementation caveats

### 14.1 The certificate target and the tracking target are not identical

The current direct notebooks certify contraction around:

```text
(x_s, u_s),
```

but the direct tracking objective and the reward both still reference:

```text
y_{\mathrm{sp}},
```

not necessarily:

```text
y_s.
```

So when the admissible target differs from the requested raw setpoint, the controller can be Lyapunov-consistent and still show raw-setpoint offset or jitter.

### 14.2 The current direct RL notebooks do not use hard move bounds in the gate

The generic gate supports `\Delta u` checks, but the current notebook calls do not activate them.

### 14.3 The direct fallback still has terminal-set machinery available

The current solver is created with `terminal_set_on = True`. The implementation may skip the terminal constraint online when the computed `\alpha` is too small, but the fallback is not literally "first-step contraction only" in every step.

### 14.4 The pretrained checkpoint origin is only partially verified

The pretrained direct RL notebook explicitly loads `Data/agent_2507171027.pkl`.

The repository also contains [pretraining_rl_controller.ipynb](../pretraining_rl_controller.ipynb), which documents how MPC-style TD3 checkpoints are generated. However, the saved filename shown in that notebook output is different from the filename loaded by the current direct RL notebook. So it is scientifically safer to say:

- the direct pretrained notebook uses a previously saved TD3 checkpoint
- the pretraining notebook shows the checkpoint-generation mechanism
- but this report does not claim that the notebook output displayed there is the exact same artifact as `agent_2507171027.pkl`

## 15. A compact algorithm summary

The direct-only notebook is:

```text
At each step:
1. estimate zhat_k
2. solve the direct steady target
3. solve the direct Lyapunov tracking MPC
4. apply the first move
5. update observer and logs
```

The direct RL notebooks are:

```text
At each step:
1. estimate zhat_k and build RL state s_k
2. generate a candidate action from either the teacher or the actor
3. recompute the direct steady target
4. check the candidate against bounds and first-step Lyapunov contraction
5. if accepted, apply it
6. if rejected, solve the direct fallback MPC and apply that result
7. update observer, reward, replay, and learning phase logic
```

The offline pretraining notebook is:

```text
1. sample synthetic states, setpoints, and previous inputs
2. label them with MPC first moves
3. train the actor by MSE imitation
4. freeze the actor and train the critic by TD targets
5. save the checkpoint
```

## 16. Final interpretation

The current direct method family is best understood as a layered architecture:

```text
\text{direct admissible target selection}
\rightarrow
\text{physical-state Lyapunov certificate}
\rightarrow
\text{tracking MPC fallback}
\rightarrow
\text{optional TD3 candidate policy}.
```

Without RL, the controller simply uses the direct target plus the direct Lyapunov MPC solve every step.

With RL, the TD3 policy is not trusted on its own. It is allowed to act only through a direct target-centered Lyapunov acceptance test, and it is corrected by direct MPC whenever that test fails.

Behavioral cloning then appears in two different roles:

- offline, to initialize a TD3 policy from MPC data
- online, to make the actor imitate safe teacher-driven actions during the early direct RL phase

That is the clearest way to read the current implementation mathematically and algorithmically.
