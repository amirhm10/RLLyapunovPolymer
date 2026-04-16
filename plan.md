# Codex implementation plan: clean rewrite to direct Lyapunov MPC with frozen output disturbance

## Goal

Replace the current notebook and code path built around `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb` with a cleaner method based on:

- a **frozen output-disturbance steady-state target calculator**
- a **direct Lyapunov MPC controller used at every step of the run**
- easy notebook-level switching between **unbounded** and **bounded** target modes
- strong diagnostics, plots, and export artifacts for later comparison and analysis
- a companion **LaTeX technical writeup** with full mathematics and step-by-step derivations

This rewrite should be treated as a new clean baseline, not as an incremental patch on top of the current transitional notebook logic.

---

## High-level design decisions

These choices are fixed for this rewrite.

1. **Output disturbance only**
   - The target calculator must use the output-disturbance interpretation only.
   - Do **not** include any `A_xd d_s` term in the steady-state dynamics.
   - The steady-state calculator must therefore use:
     - state equation: `(I - A) x_s - B u_s = 0`
     - output equation: `C x_s + d_s = y_sp`
     - frozen disturbance: `d_s = d_hat_k`

2. **Frozen disturbance target**
   - At each control step `k`, set `d_s = d_hat_k`.
   - The target calculator solves only for `x_s` and `u_s`.

3. **No x-reference or previous-target penalties for now**
   - Do **not** include terms like:
     - `||x_s - xhat_k||^2`
     - `||x_s - x_{s,prev}||^2`
     - `||u_s - u_{s,prev}||^2`
   - Keep the target calculator as clean and physics-based as possible.
   - If regularization becomes necessary later, it should be added only after the clean version is working and documented.

4. **Direct Lyapunov MPC for the full run**
   - Do **not** keep the current structure:
     - baseline MPC candidate
     - hard check
     - constrained replacement MPC
   - Instead, the controller used online at every step must be a **single direct Lyapunov MPC**.
   - The first-step Lyapunov contraction condition must be part of the MPC problem itself at every time step.

5. **Two target modes must remain visible and easy to switch in the notebook**
   - `unbounded`
   - `bounded`
   - The notebook should expose these clearly near the top in a short configuration block.

---

## Cleanup scope

### Remove these frozen-dhat first-step replacement artifacts

Delete or fully retire the notebook and code added specifically for the current transitional approach:

- `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`
- `Lyapunov/frozen_dhat_target.py`

Also remove the plumbing added only to support that notebook’s target generation modes in:

- `Simulation/run_mpc_first_step_contraction.py`

Specifically remove the frozen-target mode logic and the `target_generation_mode` / `frozen_target_config` path from this file if it is no longer used after the rewrite.

### Important cleanup rule

Do **not** delete older general Lyapunov or MPC code unless it becomes fully dead and is clearly tied only to the removed notebook.

Examples:

- Keep canonical standard Lyapunov code in `Lyapunov/` unless the new method intentionally supersedes a specific function.
- If `analysis/steady_state_debug_analysis.py` contains utilities still useful for the new method, refactor and keep the reusable parts rather than deleting the whole file.

### Required cleanup outcome

After cleanup, the repo should no longer contain a notebook or active code path that describes the method as:

- plain offset-free MPC candidate
- then check / replace with first-step contraction MPC
- with frozen-dhat target generation as a side path

The new code path should instead directly represent the new method below.

---

## New method to implement

# 1. Model and coordinates

Use the controller-side linear model in **scaled deviation coordinates** with **output disturbance only**:

$$
\begin{aligned}
x_{k+1} &= A x_k + B u_k, \\
y_k &= C x_k + d_k, \\
d_{k+1} &= d_k.
\end{aligned}
$$

Interpretation:

- `x_k` is the physical state used by the controller
- `u_k` is the input in controller coordinates
- `d_k` is an output disturbance / offset state only
- `y_k` is the controlled output
- `xhatdhat` or `xhat_aug` stores the observer estimate `[x_hat; d_hat]`

The steady-state target uses the observer disturbance estimate as a frozen constant:

$$
d_s = \hat d_k.
$$

Do **not** use any `A_xd d_s` term in this new method.

---

# 2. Steady-state target calculator

At each step `k`, given:

- current setpoint `y_{sp,k}`
- current disturbance estimate `\hat d_k`

solve for `x_s` and `u_s` only.

## 2.1 Unbounded target mode

Use the exact steady-state equations:

$$
\begin{aligned}
(I - A) x_s - B u_s &= 0, \\
C x_s &= y_{sp,k} - \hat d_k.
\end{aligned}
$$

Equivalent stacked system:

$$
\begin{bmatrix}
I - A & -B \\
C & 0
\end{bmatrix}
\begin{bmatrix}
x_s \\
u_s
\end{bmatrix}
=
\begin{bmatrix}
0 \\
y_{sp,k} - \hat d_k
\end{bmatrix}.
$$

Implementation guidance:

- If the stacked system is square and well-conditioned, solve directly.
- Otherwise allow a least-squares solve.
- Return diagnostics about rank, conditioning, residual norms, and whether the result is exact or least-squares.

## 2.2 Bounded target mode

Keep the same target equations, but enforce input bounds:

$$
u_{\min} \le u_s \le u_{\max}.
$$

Recommended formulation:

$$
\min_{x_s, u_s}
\left\|(I-A)x_s - Bu_s\right\|_{Q_{\mathrm{dyn}}}^2
+
\left\|C x_s - (y_{sp,k} - \hat d_k)\right\|_{Q_y}^2
$$

subject to

$$
u_{\min} \le u_s \le u_{\max}.
$$

Notes:

- Do **not** add `x_s` anchor terms.
- Do **not** add previous-target smoothing terms.
- Keep this version as clean as possible.
- If needed numerically, a very small regularization can be added later, but only if it is explicitly justified and documented.

## 2.3 Optional reduced form

If `I - A` is invertible, support a reduced representation for analysis and possibly for the bounded solve:

$$
x_s = (I-A)^{-1} B u_s,
$$

and therefore

$$
G u_s = y_{sp,k} - \hat d_k,
\qquad
G := C (I-A)^{-1} B.
$$

This reduced form can be used for diagnostics and possibly for the bounded input-only solve if it is numerically well-behaved.

---

# 3. Direct Lyapunov MPC controller

Implement a **single online controller** that uses the target `(x_s, u_s, d_s)` from the chosen target mode and solves a direct Lyapunov MPC at every step.

This controller should replace the current candidate-and-replacement architecture.

## 3.1 Prediction model used inside MPC

For the prediction horizon, freeze `d_s = \hat d_k` and use:

$$
\begin{aligned}
x_{i+1|k} &= A x_{i|k} + B u_{i|k}, \\
y_{i|k} &= C x_{i|k} + \hat d_k.
\end{aligned}
$$

Initial condition:

$$
x_{0|k} = \hat x_k.
$$

## 3.2 MPC objective

Use a practical tracking objective around the raw setpoint and optionally the steady target:

$$
J_k =
\sum_{i=0}^{N_p-1} \|y_{i|k} - y_{sp,k}\|_Q^2
+
\sum_{i=0}^{N_c-1} \|\Delta u_{i|k}\|_R^2
+
\|x_{N_p|k} - x_s\|_P^2.
$$

with

$$
\Delta u_{0|k} = u_{0|k} - u_{k-1},
\qquad
\Delta u_{i|k} = u_{i|k} - u_{i-1|k} \text{ for } i \ge 1.
$$

You may optionally include a small first-move anchor to `u_s` later, but do not add unnecessary terms in the first clean version unless needed.

## 3.3 Constraints

Use the standard input and optional move constraints:

$$
u_{\min} \le u_{i|k} \le u_{\max},
$$

and if move bounds are available,

$$
\Delta u_{\min} \le \Delta u_{i|k} \le \Delta u_{\max}.
$$

## 3.4 Lyapunov function and first-step contraction

Define the physical-state error relative to the target:

$$
e_{x,k} = \hat x_k - x_s.
$$

Use the Lyapunov function:

$$
V_k = e_{x,k}^\top P_x e_{x,k}.
$$

Enforce the hard first-step contraction inequality directly inside the MPC problem:

$$
(x_{1|k} - x_s)^\top P_x (x_{1|k} - x_s)
\le
\rho V_k + \varepsilon_{\mathrm{lyap}}.
$$

This must be part of the direct controller at every step.

## 3.5 Feasibility handling

Because the user wants Lyapunov MPC for the entire run, avoid returning to the old replacement structure.

Recommended approach:

- support a **hard** first-step contraction mode
- support an optional **soft** mode with slack

Soft version recommendation:

Introduce slack `\sigma_k \ge 0` and enforce

$$
(x_{1|k} - x_s)^\top P_x (x_{1|k} - x_s)
\le
\rho V_k + \varepsilon_{\mathrm{lyap}} + \sigma_k
$$

with a large penalty on `\sigma_k` in the objective.

The notebook should make this choice visible.

---

## Required software structure

### Create a new target calculator module

Suggested file:

- `Lyapunov/frozen_output_disturbance_target.py`

This module should contain clear public functions such as:

- `solve_target_unbounded_output_disturbance(...)`
- `solve_target_bounded_output_disturbance(...)`
- shared helper functions for conditioning, rank diagnostics, reduced form, and residual reporting

### Create a new direct Lyapunov MPC module

Suggested file:

- `Lyapunov/direct_lyapunov_mpc.py`

This module should contain:

- Lyapunov-ingredient helpers if needed
- direct controller solver used online at every step
- support for hard and optional soft contraction modes
- a rollout function for closed-loop simulations

### Notebook entrypoint

Create a clean new notebook dedicated to this method.

Suggested notebook name:

- `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`

The notebook should be short, organized, and focused on configuration, running experiments, and plotting.

---

## Notebook requirements

The notebook must expose all important modes clearly near the top so the user can change them easily.

### Required visible options

1. `target_mode`
   - `"unbounded"`
   - `"bounded"`

2. `lyapunov_mode`
   - `"hard"`
   - `"soft"`

3. horizon parameters
   - prediction horizon
   - control horizon

4. Lyapunov parameters
   - `rho_lyap`
   - `lyap_eps`
   - slack penalty if soft mode is used

5. disturbance and scenario settings
   - setpoint scenario
   - disturbance scenario
   - number of tests / steps

6. export / plotting toggles
   - save plots
   - save debug bundle
   - save CSV / pickle / summary tables if applicable

The notebook should also clearly print the active configuration before the run starts.

---

## Results and diagnostics to generate

The new method must produce proper outputs for later comparison and analysis.

### Save structured run bundles

Export a bundle that includes at least:

- plant outputs in physical units
- plant outputs in controller deviation coordinates
- applied inputs in physical units
- applied inputs in controller deviation coordinates
- observer state `xhatdhat`
- steady targets `x_s`, `u_s`, `d_s`, `y_s`
- Lyapunov quantities
  - `V_k`
  - first-step predicted Lyapunov value
  - Lyapunov bound
  - contraction margin
  - slack value if soft mode is active
- solver status / infeasibility status / residuals
- target-solver diagnostics
  - exact or bounded mode
  - rank and condition information
  - residual norms
  - whether exact solution was inside bounds

### Comparison-ready summary tables

Generate summary tables containing at least:

- average reward or cost per run / subepisode if still relevant
- average output tracking error
- maximum output tracking error
- average input movement
- maximum input movement
- percentage of time hard contraction is satisfied
- if soft mode is used:
  - average slack
  - maximum slack
  - fraction of steps with nonzero slack
- if bounded target mode is used:
  - average target residual
  - maximum target residual
  - fraction of steps where bounds are active

These summaries should make later comparison between:

- unbounded vs bounded target mode
- hard vs soft Lyapunov mode

straightforward.

---

## Plotting requirements

Create a proper plotting/export path similar in spirit to the existing debug tools.

### Required figures

1. **Outputs vs setpoints**
   - plant outputs
   - raw setpoint
   - steady target output `y_s`

2. **Inputs vs steady targets and bounds**
   - applied inputs
   - steady target inputs `u_s`
   - input bounds

3. **State-target mismatch**
   - `x_hat`
   - `x_s`
   - `x_hat - x_s`

4. **Disturbance / output-offset view**
   - `d_hat`
   - `d_s = d_hat`
   - `y_sp - d_hat`

5. **Lyapunov diagnostics**
   - `V_k`
   - first-step predicted Lyapunov value
   - Lyapunov bound
   - contraction margin
   - slack trajectory if soft mode is used

6. **Bounded-target diagnostics** (when `target_mode = "bounded"`)
   - target residual norms
   - active lower / upper bound indicators
   - exact vs bounded target comparison

7. **Tail-window / near-steady-state overview**
   - same spirit as previous debug plots, but adapted to the new direct Lyapunov MPC method

### Plotting style

- Keep plots paper-ready and readable.
- Use consistent labeling between notebook cells and saved files.
- Save all figures to a timestamped output directory.
- Keep filenames stable and descriptive.

---

## LaTeX technical writeup request

Create an extensive LaTeX file documenting the new method.

Suggested file:

- `report/direct_lyapunov_mpc_frozen_output_disturbance.tex`

This LaTeX document should be detailed and step-by-step. It must include:

1. **Motivation and scope**
   - why the previous notebook/code path is being retired
   - why a direct Lyapunov MPC is cleaner than candidate-then-replacement for this stage

2. **Model and coordinate conventions**
   - scaled deviation coordinates
   - output disturbance only
   - observer state split

3. **Steady-state target calculator**
   - derive the exact unbounded equations step by step
   - derive the stacked system
   - derive the reduced form `G u_s = y_sp - d_hat` when `I - A` is invertible
   - formulate the bounded problem carefully
   - explain what is optimized and what is frozen
   - explicitly state that no `x_s - x_ref` or previous-target penalties are used in this clean version

4. **Direct Lyapunov MPC formulation**
   - prediction model
   - objective function
   - constraints
   - Lyapunov function
   - hard first-step contraction condition
   - optional soft-contraction slack version

5. **Algorithm box / step-by-step procedure**
   - observer update
   - target solve
   - Lyapunov MPC solve
   - apply first move
   - logging/export

6. **Implementation mapping to repo files**
   - map equations to the actual Python modules and notebook entrypoint

7. **Discussion and interpretation**
   - meaning of bounded vs unbounded target mode
   - meaning of hard vs soft Lyapunov mode
   - what diagnostics should be used to judge performance and feasibility

The LaTeX should be mathematically explicit and suitable as an internal technical note or future manuscript appendix.

---

## Validation requirements

Use low-cost checks only. Do not run expensive training or long notebooks unless necessary.

At minimum:

- `python -m py_compile` on touched Python files
- import checks for the new modules
- small synthetic target-solver tests
- small synthetic direct Lyapunov MPC tests

If a change-report is appropriate, create one under `change-reports/` and keep it aligned with the code change.

---

## Suggested implementation sequence

1. Remove the current frozen-dhat first-step replacement notebook and its dedicated target module.
2. Remove frozen-target mode plumbing from `Simulation/run_mpc_first_step_contraction.py` if it is no longer needed.
3. Create the new target calculator module for output-disturbance-only targets.
4. Create the new direct Lyapunov MPC module.
5. Create the new notebook with visible mode switches for bounded/unbounded and hard/soft Lyapunov modes.
6. Build export bundles, summary tables, and plotting utilities for the new method.
7. Create the LaTeX technical note.
8. Run low-cost validation.
9. Create a descriptive commit and matching change report.

---

## Final reminder

This rewrite should intentionally move away from the current transitional logic and produce a cleaner method that is easy to explain mathematically:

- frozen output disturbance target
- two target modes: bounded and unbounded
- no `x_s` reference / previous-target penalties for now
- direct Lyapunov MPC at every step
- strong plots and exported diagnostics
- a full LaTeX technical note documenting the method
