# Codex Expansion Plan: Add Input Box-Constraint Analysis to the Existing Steady-State Debug Workflow

## Scope
This is an **expansion** of the existing steady-state debug analysis workflow. Do **not** remove or replace the previous implementation. Keep the current analysis pipeline intact and **add** a new analysis branch for bounded steady-state calculations.

The controller in the closed loop remains the existing **normal offset-free MPC**. This expansion is **only for analysis/debugging of steady-state target equations** at each time step. It must **not** feed back into MPC, Lyapunov logic, projection logic, or any safety-filter logic.

## Goal
At each simulation step `k`, we already have:
- the fixed linear model matrices `A, B, C`
- the current setpoint `y_sp[k]`
- the current disturbance estimate `dhat_k`

We want to analyze the steady-state equations under the **legacy augmentation assumption** with frozen disturbance target:

\[
d_s(k) = \hat d_k.
\]

The steady-state equations are:

\[
(I - A)x_s(k) - B u_s(k) = 0,
\]
\[
C x_s(k) = y_{sp}(k) - \hat d_k.
\]

Now expand the analysis to include a box constraint on the steady-state input target:

\[
u_{\min} \le u_s(k) \le u_{\max}.
\]

## Important Implementation Rule
Keep the **existing unbounded steady-state analysis** exactly as it is.

Add a second analysis layer that:
1. first checks the exact unbounded steady-state solution,
2. then checks whether that exact solution is inside the input box,
3. and if not, solves a **simple constrained fallback problem** for analysis.

Do not delete the old outputs, old plots, or old tables. Add new outputs alongside them.

## Mathematical Structure to Implement

### 1. Exact unbounded steady-state solve
At each time step define

\[
r_k := y_{sp}(k) - \hat d_k.
\]

Then the exact steady-state equations are

\[
(I - A)x_s(k) - B u_s(k) = 0,
\]
\[
C x_s(k) = r_k.
\]

Use the same non-optimization linear solve already discussed in the previous plan.

If `(I - A)` is invertible, also compute the reduced steady-state gain

\[
G = C (I - A)^{-1} B,
\]

so that the reduced equation becomes

\[
G u_s(k) = r_k.
\]

This is the **exact unbounded** steady-state target calculation.

### 2. Box check on the exact solution
After obtaining the exact unbounded solution `(x_s_exact, u_s_exact)`, check whether

\[
u_{\min} \le u_s^{exact}(k) \le u_{\max}.
\]

Classify each time step into one of the following categories:
- `exact_bounded`: exact steady-state solution exists and satisfies bounds
- `exact_unbounded`: exact steady-state solution exists but violates bounds
- `exact_unsolved`: exact steady-state system could not be solved reliably

### 3. Constrained fallback solve
If the exact solution either:
- violates the box constraint, or
- cannot be solved reliably,

then solve the following **simple constrained fallback problem**:

Reduced form preferred when available:

\[
\min_{u_s} \| G u_s - r_k \|_2^2
\]
subject to
\[
u_{\min} \le u_s \le u_{\max}.
\]

Then recover

\[
x_s(k) = (I - A)^{-1} B u_s(k)
\]

when `(I - A)` is invertible.

If reduced form is not available or not numerically reliable, solve the full form:

\[
\min_{x_s,u_s}
\left\|
\begin{bmatrix}
I - A & -B \\
C & 0
\end{bmatrix}
\begin{bmatrix}
x_s \\
u_s
\end{bmatrix}
-
\begin{bmatrix}
0 \\
r_k
\end{bmatrix}
\right\|_2^2
\]
subject to
\[
u_{\min} \le u_s \le u_{\max}.
\]

## Naming of the fallback problem
Use one of the following names consistently in code comments, docs, and report text:
- **box-constrained least-squares problem**
- **bounded least-squares problem**
- **constrained steady-state least-squares problem**

Avoid using only the generic phrase "constrained optimization problem" unless needed in prose. That phrase is correct but too broad. The most accurate name here is:

**box-constrained least-squares problem**

If you implement the full quadratic form explicitly, it is also correct to call it a:

**quadratic program (QP) with box constraints**

## What Codex should implement

### A. Keep all existing outputs
Do not remove any existing analysis outputs from the previous steady-state debug workflow.

### B. Add new per-step quantities
At each time step log at least the following:
- `r_k = y_sp[k] - dhat_k`
- `xs_exact`
- `us_exact`
- `exact_solve_success`
- `exact_within_bounds`
- `exact_bound_violation_inf`
- `exact_eq_residual_state_inf`
- `exact_eq_residual_output_inf`
- `xs_bounded`
- `us_bounded`
- `bounded_solve_success`
- `bounded_residual_norm`
- `bounded_active_lower_mask`
- `bounded_active_upper_mask`
- `solve_mode` with values such as:
  - `exact_bounded`
  - `exact_unbounded_fallback_bounded_ls`
  - `exact_unsolved_fallback_bounded_ls`
  - `failed`

### C. Add helper functions
Create new helper functions without breaking the old ones. Suggested structure:
- `solve_exact_steady_state_unbounded(...)`
- `check_box_bounds(u, u_min, u_max, tol=...)`
- `solve_bounded_steady_state_least_squares(...)`
- `run_parallel_steady_state_box_analysis(...)`

These can internally use reduced form or full form depending on numerical reliability.

### D. Numerical solver suggestions
For the bounded least-squares fallback, use a simple and reliable approach.

Preferred order:
1. `scipy.optimize.lsq_linear` for reduced form if possible, because it naturally handles bounds
2. `scipy.optimize.minimize` only if needed
3. a small QP formulation only if necessary

The implementation goal is clarity and debug value, not solver sophistication.

## Required Tables
Generate tables for one or more selected runs showing per-step or summary statistics.

### Table 1. Overall summary
Include:
- percentage of steps in each `solve_mode`
- percentage of exact solutions inside bounds
- average and max infinity-norm bound violation of the exact solution
- average and max residual norm of the bounded fallback
- average and max difference between `us_exact` and `us_bounded`
- average and max difference between `xs_exact` and `xs_bounded`

### Table 2. Per-input bound activity summary
For each input dimension:
- fraction of time lower bound active in bounded fallback
- fraction of time upper bound active in bounded fallback
- average exact violation below lower bound
- average exact violation above upper bound

### Table 3. Event-focused table
Around setpoint changes and any large `dhat` variations, show a compact time-window table with:
- time step index
- `y_sp`
- `dhat_k`
- `us_exact`
- `us_bounded`
- `solve_mode`
- exact residuals
- bounded residual norm

## Required Plots
Add plots specifically for the bounded analysis. Keep the previous plots too.

### Plot group 1. Inputs
For each input dimension:
- `us_exact` vs `us_bounded` vs `u_min` vs `u_max`
- optionally overlay the actual applied MPC input `u_applied`

### Plot group 2. Outputs
For each output dimension:
- exact target output vs bounded target output vs setpoint
- because in bounded fallback exact output matching may be lost, plot the mismatch clearly

### Plot group 3. Residuals
- state-equation residual norm for exact solve
- output-equation residual norm for exact solve
- bounded least-squares residual norm over time

### Plot group 4. Constraint activity
- binary time-series or heatmap showing when each input hits lower or upper bound in the bounded fallback

### Plot group 5. Comparison around events
Create zoomed plots around setpoint changes to compare:
- `us_exact`
- `us_bounded`
- `y_sp`
- bounded output error
- `dhat_k`

## Analysis interpretation Codex should include in comments and docs
The new branch is for answering the following questions:
1. How often does the exact steady-state target already satisfy the box constraints?
2. When it violates the bounds, how large are the violations?
3. How much does the bounded least-squares fallback distort the target relative to the exact unbounded solution?
4. Which inputs most often become active at the box limits?
5. During setpoint changes, does the bounded solve behave smoothly and consistently?

## Output files Codex should create
In the analysis output directory, create:
- a machine-readable results file such as `.pkl` or `.npz`
- a summary `.csv`
- figures for all required plot groups
- a compact markdown summary

## Additional documentation Codex must create

### 1. Parameter markdown file
Create a detailed markdown file documenting all parameters used in the bounded analysis.

Required filename suggestion:
- `steady_state_box_analysis_parameters.md`

It should document at least:
- tolerance for exact-solve residuals
- tolerance for bound checks
- solver choices and fallbacks
- whether reduced form or full form is used
- plotting configuration
- selected runs or episodes analyzed

### 2. LaTeX report in report directory
Create a detailed LaTeX file in the report directory explaining the mathematics and the exact step-by-step algorithm.

Required content:
- legacy augmentation model
- definition of `dhat_k` as frozen disturbance target for analysis
- exact unbounded steady-state equations
- stacked linear system form
- reduced form with `G = C(I-A)^{-1}B`
- input box constraints
- direct exact solve and bound-check logic
- bounded least-squares fallback derivation
- algorithmic flow at each time step
- list of diagnostic quantities, tables, and figures

Suggested filename:
- `report/steady_state_box_analysis.tex`

The LaTeX file should be mathematically explicit and written step by step. It should focus only on this steady-state analysis workflow and should not discuss Lyapunov or safety-filter logic.

## Final implementation instructions to Codex
1. Do not delete the previous steady-state debug analysis.
2. Add this bounded steady-state expansion as a parallel analysis branch.
3. Keep the MPC controller unchanged.
4. Do not feed the bounded steady-state calculation back into MPC.
5. Make the code easy to toggle on or off with a clear flag.
6. Prefer readable, modular functions over one large script.
7. Save enough diagnostics so the user can inspect exact vs bounded solutions at each time step.
