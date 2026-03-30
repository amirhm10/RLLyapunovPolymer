# Codex Implementation Plan: Replace All Prior Target-Selector Modes With a Single Refined Step A Selector

## Objective

Remove the previous target-selector mode system entirely and implement only one selector: a refined steady-state target selector based on the current Step A discussion.

This selector must preserve the useful part of the original method, namely good output-side target generation, while fixing the main structural failures observed in the original selector:

1. `x_s` collapses toward zero in scaled deviation coordinates.
2. `u_s` stays too close to nominal or nearly constant and does not reflect the actual operating region.
3. The returned target package is useful in output space (`r_s`, `y_s`) but poor as a center for projection / Lyapunov filtering.
4. Previous-target smoothing alone is not sufficient because it only preserves continuity and cannot move the selector away from an already bad center.

The new selector should therefore:

1. keep the same steady-state constraints and keep `d_s = d_hat_k` fixed for now,
2. keep the soft output-target structure,
3. anchor the steady-state input target to the currently applied input,
4. smooth the target over time,
5. weakly anchor the steady-state state target to the current observer state,
6. retain a full diagnostic payload,
7. replace all previous selector modes.

This plan is intentionally single-method only. Do not preserve the old mode-switching system.

---

## Required high-level outcome

Implement one and only one target-selector method in the codebase.

The old selector variants / mode switches must be removed or deprecated cleanly so that the code path is unambiguous.

The new selector must solve one single steady-state target optimization problem with:

- fixed disturbance target `d_s = d_hat_k`,
- soft output target term,
- input anchor to the actual applied input,
- weak state anchor to the current observer estimate,
- previous-target smoothing for both input and state,
- optional input and output bounds,
- diagnostics for feasibility, residuals, and objective-term breakdown.

---

## Mathematical formulation to implement

Work in the existing scaled deviation coordinates already used in the codebase.

At each time step `k`, with current estimated augmented state

- `xhat_k` = current estimated physical state in scaled deviation coordinates,
- `dhat_k` = current estimated disturbance state,
- `u_applied_k` = actual applied input at the current time step in scaled deviation coordinates,
- `x_s_prev`, `u_s_prev` = previous selector target if available,
- `y_sp` = desired output setpoint in scaled deviation coordinates,

solve the following steady-state target problem.

### Decision variables

- `x_s in R^{n_x}`
- `u_s in R^{n_u}`
- `r_s in R^{n_r}` if an output selection matrix `H` is used, otherwise `r_s = y_s`

### Fixed quantity

- `d_s = dhat_k`

### Steady-state equations

```math
x_s = A x_s + B u_s + B_d d_s
```

```math
y_s = C x_s + C_d d_s
```

```math
r_s = H y_s
```

If `H` is not used in the implementation path, use `r_s = y_s`.

### Optimization objective

Implement the selector as

```math
\begin{aligned}
\min_{x_s,u_s,r_s}\quad
& \|r_s - y_{sp}\|_{Q_r}^2
+ \|u_s - u_{applied,k}\|_{R_{u,ref}}^2 \\
& + \|u_s - u_s^{prev}\|_{R_{\Delta u}}^2
+ \|x_s - x_s^{prev}\|_{Q_{\Delta x}}^2 \\
& + \|x_s - \hat x_k\|_{Q_{x,ref}}^2
\end{aligned}
```

subject to

```math
x_s = A x_s + B u_s + B_d d_s
```

```math
r_s = H(C x_s + C_d d_s)
```

```math
u_{min} \le u_s \le u_{max}
```

and optional output bounds

```math
y_{min} \le C x_s + C_d d_s \le y_{max}
```

If no previous target is available, the corresponding smoothing terms must be disabled automatically for that solve.

If no `H` is provided, treat `r_s` as the full output target.

---

## Why this method is being implemented

Codex should preserve this rationale in comments and documentation.

### Failure of the original selector

The original selector effectively behaved like a soft output-target selector whose regularization was centered at or near:

- `x_s = 0`
- `u_s = u_nom`
- `d_s = dhat_k`

Empirically, that produced:

- useful `r_s` / `y_s` tracking,
- but `x_s` near zero,
- and `u_s` nearly constant or not representative of the actual operating region.

That means the original selector could be useful in output space while remaining poor as a state-space center for projection / Lyapunov correction.

### Why previous-target smoothing alone is not enough

The terms

```math
\|x_s - x_s^{prev}\|^2, \quad \|u_s - u_s^{prev}\|^2
```

provide only continuity. They do not provide direction.

If the previous target is already collapsed to a poor center, these terms keep the new solution near that same poor center.

### Why the new anchor terms are needed

The new selector adds:

- `u_s` anchor to `u_applied_k`, so the steady-state input target lives near the actual operating region,
- weak `x_s` anchor to `xhat_k`, so the state target can move after setpoint changes instead of remaining trapped near the old equilibrium.

The `xhat_k` anchor must remain weak because `xhat_k` is not itself a steady-state object and contains transient information.

---

## Design constraints

1. Keep `d_s = dhat_k` fixed for this implementation.
2. Do not implement disturbance optimization yet.
3. Do not preserve the old four-mode target-selector framework.
4. Do not introduce raw mode switches for target selector behavior.
5. Keep the solver path compatible with the existing scaled deviation coordinate system.
6. Reuse existing matrix shapes, scaling conventions, and diagnostics style wherever possible.
7. Maintain compatibility with the current Lyapunov / projection pipeline inputs.
8. Preserve previous-target persistence if it already exists in the code path.

---

## Parameters to expose

Implement these as user-adjustable parameters with defaults.

### Core target-tracking weight

- `Qr_tgt` or equivalent
- meaning: weight for `r_s - y_sp`
- default: reuse the current output-target weighting logic already used for the selector, e.g. the current `Qs_tgt_diag` / `Ty_diag` style.

### Input anchor weight

- `alpha_u_ref`
- meaning: multiplier for the applied-input anchor matrix
- default: `0.5`

Define

```math
R_{u,ref} = \alpha_{u,ref} R_{\Delta u}^{mpc}
```

where `R_delta_u_mpc` is the existing MPC move-penalty matrix structure if available.

### Input smoothing weight

- `alpha_du_sel`
- meaning: multiplier for previous-target input smoothing
- default: `0.5`

Define

```math
R_{\Delta u}^{sel} = \alpha_{du,sel} R_{\Delta u}^{mpc}
```

### State smoothing weight

- `alpha_dx_sel`
- meaning: multiplier for previous-target state smoothing
- default: `0.05`

### State anchor weight

- `alpha_x_ref`
- meaning: multiplier for weak anchoring of `x_s` to `xhat_k`
- default: `0.01`

### State weighting base choice

Expose an option

- `x_weight_base = "CtQC"` or `"identity"`

Default: `"CtQC"`

If `CtQC`, define

```math
Q_x^{base} = C^\top Q_r C
```

Then use

```math
Q_{\Delta x} = \alpha_{dx,sel} Q_x^{base}
```

```math
Q_{x,ref} = \alpha_{x,ref} Q_x^{base}
```

If `identity`, use

```math
Q_x^{base} = I
```

with correct dimension.

### Optional nominal input reference

Keep support for a nominal input vector if already used elsewhere, but do not use it as the primary anchor in this selector.

### Optional bounds

- `use_output_bounds_in_selector = True/False`
- default: `True` if the current code path already supports output bounds reliably, otherwise preserve current behavior.

### Solver options

Preserve current solver preference infrastructure.

### Diagnostics toggles

Keep a detailed debug dictionary by default.

---

## Default weight hierarchy

Codex must preserve this intended ordering in code comments and documentation.

The intended importance ordering is

```math
Q_r \gg R_{u,ref} \sim R_{\Delta u}^{sel} > Q_{\Delta x} > Q_{x,ref}
```

Interpretation:

1. output target quality remains the top objective,
2. input operating-region consistency and input smoothing come next,
3. state smoothing is weaker,
4. direct attraction to `xhat_k` is weakest.

---

## Exact implementation tasks

### Task 1. Remove the old multi-mode selector framework

- Delete or deprecate the previous target-selector modes.
- Ensure only one selector path remains active.
- Clean up mode-related configuration flags.
- Update any notebooks or config files that previously referenced multiple selector modes.

### Task 2. Implement the new selector function

Create or replace the selector function with a single clearly named function, for example:

- `compute_refined_step_a_target(...)`

or another consistent name.

This function should:

Inputs:
- `A, B, Bd, C, Cd`
- `dhat_k`
- `xhat_k`
- `u_applied_k`
- `y_sp`
- `u_min, u_max`
- optional `y_min, y_max`
- optional `H`
- optional `x_s_prev, u_s_prev`
- weighting parameters / matrices
- solver settings
- debug flag

Outputs:
- `success`
- `x_s`
- `u_s`
- `d_s = dhat_k`
- `y_s`
- `r_s`
- objective breakdown
- residuals
- bounds diagnostics
- solver metadata

### Task 3. Objective-term construction

Implement all five objective terms exactly as discussed.

Disable the two previous-target terms automatically when previous targets are unavailable.

Keep the `xhat_k` state anchor weak by default.

### Task 4. Weight-matrix builder

Create a helper that constructs:

- `R_u_ref`
- `R_delta_u_sel`
- `Q_delta_x`
- `Q_x_ref`

from exposed scalar multipliers and the chosen base matrices.

This helper should:
- accept user overrides,
- default to the values above,
- document how matrices are built.

### Task 5. Diagnostics

Return a rich debug dictionary including at least:

- solver status
- objective value
- each objective term value separately
- dynamic residual infinity norm
- output residual infinity norm
- input-bound violation infinity norm
- output-bound violation infinity norm if applicable
- whether previous-target terms were active
- actual matrices or scalar multipliers used
- `y_s - y_sp`
- `r_s - y_sp`
- `u_s - u_applied_k`
- `u_s - u_s_prev` when available
- `x_s - x_s_prev` when available
- `x_s - xhat_k`

### Task 6. Downstream integration

Replace all current selector calls with this new selector.

Update downstream code so the expected target package remains available:

- `x_s`
- `u_s`
- `d_s`
- `y_s`
- `r_s`
- source / success metadata

Do not change the downstream projection / Lyapunov code yet unless necessary for compatibility.

### Task 7. Logging and plotting support

Ensure the new selector outputs are stored so that the following plots can be produced easily:

- `y_s` vs `y_sp`
- `r_s` vs `y_sp`
- `xhat` vs `x_s`
- `dhat` vs `d_s`
- `u_applied` vs `u_s`
- objective-term trajectories over time
- selector residual trajectories

### Task 8. Backward compatibility / migration notes

If removing old selector modes breaks notebook code, update notebooks or config examples accordingly.

Do not leave stale options that appear user-selectable but no longer do anything.

---

## Initial parameter defaults to implement

Use these as the first defaults unless the existing code structure requires slight naming adjustments.

- `alpha_u_ref = 0.5`
- `alpha_du_sel = 0.5`
- `alpha_dx_sel = 0.05`
- `alpha_x_ref = 0.01`
- `x_weight_base = "CtQC"`
- `use_output_bounds_in_selector = True` if stable in the current code path

Matrix construction:

```math
R_{u,ref} = \alpha_{u,ref} R_{\Delta u}^{mpc}
```

```math
R_{\Delta u}^{sel} = \alpha_{du,sel} R_{\Delta u}^{mpc}
```

If `x_weight_base == "CtQC"`, define

```math
Q_x^{base} = C^\top Q_r C
```

else

```math
Q_x^{base} = I
```

Then

```math
Q_{\Delta x} = \alpha_{dx,sel} Q_x^{base}
```

```math
Q_{x,ref} = \alpha_{x,ref} Q_x^{base}
```

---

## Acceptance criteria

Codex should consider the implementation successful only if the following are satisfied.

1. The old selector modes are fully removed or clearly deprecated.
2. The new selector compiles and runs in the existing pipeline.
3. The selector objective contains all five intended terms.
4. The previous-target terms deactivate cleanly when no previous target exists.
5. The debug dictionary contains separate term values and residuals.
6. The code uses current scaled deviation coordinates consistently.
7. The resulting selector can produce and log `x_s`, `u_s`, `d_s`, `y_s`, and `r_s` for plotting.
8. The implementation includes concise but useful inline comments explaining the role of each term.

---

## Validation plan for Codex to implement or document

Codex should add or describe validation that checks:

1. At a fixed setpoint with stable operation, the selector returns smooth `x_s` and `u_s`.
2. After a setpoint change, `u_s` moves meaningfully instead of staying near nominal.
3. `x_s` does not collapse trivially to zero unless that is genuinely the equilibrium.
4. `r_s` remains near `y_sp` when feasible.
5. The objective-term breakdown reflects the intended tradeoff.
6. The selector remains numerically stable when previous-target information is absent at initialization.

---

## Deliverables Codex must produce in addition to code

### Deliverable A. Detailed parameter markdown file

Create a separate markdown file in the repository documentation or report area, for example:

- `reports/refined_step_a_selector_parameters.md`

This file must include:

1. all exposed selector parameters,
2. exact mathematical meaning of each parameter,
3. default values,
4. how the matrix weights are constructed,
5. recommended tuning order,
6. expected effect of increasing or decreasing each parameter,
7. notes on scaled deviation coordinates,
8. notes on when to reduce or increase the `xhat` anchor weight.

### Deliverable B. Detailed LaTeX report file

Create a detailed LaTeX report file in the report section, for example:

- `reports/refined_step_a_selector_method.tex`

This report must include:

1. a clear statement of the original selector failure mode,
2. the exact refined Step A optimization problem,
3. all variables and dimensions,
4. all assumptions,
5. the difference between tracking target quality and projection-center usefulness,
6. why previous-target smoothing alone is not enough,
7. why the input anchor to `u_applied_k` is introduced,
8. why the state anchor to `xhat_k` is weak rather than strong,
9. the role of each weight matrix,
10. expected behavior after a setpoint change,
11. interpretation of the selector in scaled deviation coordinates,
12. detailed discussion of implementation choices and diagnostics,
13. a discussion section explaining what this selector is trying to solve mathematically and what it is not yet trying to solve.

The LaTeX report should be extended and mathematically explicit, not just a short note.

---

## Reporting / code comments guidance

Codex should preserve the following interpretation in documentation and comments.

The refined Step A selector is trying to solve:

> Find a steady-state equilibrium target that remains close to the desired output target while also remaining close to the actual closed-loop operating region and evolving smoothly in time.

It is not yet trying to solve:

1. optimized disturbance targeting,
2. joint optimization over `d_s`,
3. full separation of tracking target and projection center,
4. explicit robust target selection under uncertainty.

Those are possible future extensions, but they are intentionally out of scope for this implementation.

---

## References to cite in the added documentation

Include citations to the offset-free MPC and target-calculation literature already discussed in this project, especially the standard steady-state target calculation perspective and the interpretation of augmented disturbance states.

At minimum, reference the key materials already identified in this project context, including the Rawlings / Mayne / Diehl MPC text and the previously discussed target-calculation and disturbance-model references used in prior notes.

Do not invent new claims. Keep the discussion grounded in the literature and in the observed behavior of the current codebase.

---

## Final instruction to Codex

Implement only the refined Step A selector described in this document.

Do not preserve the previous four-mode target-selector framework.

Update the codebase, configuration, and documentation so that the selector behavior is unified, parameterized, diagnosable, and mathematically documented.
