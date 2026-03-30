# Codex implementation plan: four target-selector modes for the safe MPC filter

## Goal

Refactor the target-selector portion of the safe MPC filter so that the codebase supports **four selectable target-selector modes** behind one common interface.

The four modes are:

1. `current_exact_fallback_frozen_d`
2. `free_disturbance_prior`
3. `compromised_reference`
4. `single_stage_robust_sstp`

The implementation must preserve the current behavior as one switchable mode, and allow the user to select any of the four modes through a configuration object or keyword arguments without editing the solver internals.

This plan is written for Codex. Follow it closely and implement the changes in a clean, testable, reversible way.

---

## High-level design requirement

Create a single user-facing selector API, for example:

```python
target_info = prepare_filter_target(
    selector_mode=selector_mode,
    A_aug=A_aug,
    B_aug=B_aug,
    C_aug=C_aug,
    xhat_aug=xhat_aug,
    y_sp=y_sp,
    u_min=u_min,
    u_max=u_max,
    config=target_selector_config,
    prev_target=prev_target_info,
    H=H,
    return_debug=False,
)
```

All four modes must return the same `target_info` schema so the rest of the code can consume them without mode-specific branching.

At minimum, `target_info` must include:

```python
{
    "success": bool,
    "selector_mode": str,
    "solve_stage": str or None,
    "x_s": np.ndarray or None,
    "u_s": np.ndarray or None,
    "d_s": np.ndarray or None,
    "x_s_aug": np.ndarray or None,
    "y_s": np.ndarray or None,
    "yc_s": np.ndarray or None,
    "r_s": np.ndarray or None,
    "requested_y_sp": np.ndarray,
    "target_error": np.ndarray or None,
    "target_error_inf": float or None,
    "target_error_norm": float or None,
    "dyn_residual_inf": float or None,
    "bound_violation_inf": float or None,
    "selector_debug": dict,
}
```

Notes:
- `r_s` means the admissible or compromised steady reference actually selected by the method.
- For methods that do not optimize a separate `r_s`, set `r_s = yc_s`.
- Keep backward-compatible aliases if needed, but make `r_s` part of the standard target info.

---

## Current method to preserve as baseline

### Mode 0: `current_exact_fallback_frozen_d`

This mode must preserve the current implementation as closely as possible.

### What this method solves

Given the current augmented estimate

```math
\hat z_k = [\hat x_k;\hat d_k],
```

freeze the pseudo steady disturbance at

```math
d_s = \hat d_k.
```

Then solve the two-stage target problem.

#### Stage 1: exact target

```math
\min_{x_s,u_s}
(u_s-u_{nom})^T R_u (u_s-u_{nom})
+ x_s^T Q_x x_s
+ \Phi_y^{soft}(y_s)
```

subject to

```math
(I-A)x_s - Bu_s - B_d \hat d_k = 0
```

```math
u_{lo} \le u_s \le u_{hi}
```

```math
y_{c,s} = y_{sp}
```

plus optional output bounds.

#### Stage 2: fallback target

```math
\min_{x_s,u_s}
(y_{c,s}-y_{sp})^T T_y (y_{c,s}-y_{sp})
+ (u_s-u_{nom})^T R_u (u_s-u_{nom})
+ x_s^T Q_x x_s
+ \Phi_\Delta(x_s,u_s)
+ \Phi_y^{soft}(y_s)
```

subject to

```math
(I-A)x_s - Bu_s - B_d \hat d_k = 0
```

```math
u_{lo} \le u_s \le u_{hi}
```

plus optional output bounds.

### What it is trying to solve

This method tries to find an exact offset-free equilibrium under the frozen estimated disturbance. If exact feasibility is not accepted, it then computes the closest admissible equilibrium under the same frozen disturbance.

### Why it is important

This is the current production baseline and must remain reproducible for comparison.

### Default parameters for this mode

Use these defaults unless the user overrides them:

```python
mode0_defaults = {
    "u_nom": None,              # interpreted as zero vector
    "Ty_diag": 1e8 * Q_out,
    "Ru_diag": np.ones(n_u),
    "Qx_diag": None,
    "w_x": 1e-6,
    "Qdx_diag": None,
    "Rdu_diag": Rmove_diag,
    "soft_output_bounds": True,
    "Wy_low_diag": 1e3 * np.ones(n_y),
    "Wy_high_diag": 1e3 * np.ones(n_y),
    "u_tight": np.zeros(n_u),
    "y_tight": np.zeros(n_y),
    "solver_pref": DEFAULT_CVXPY_SOLVERS,
    "accept_statuses": ("optimal", "optimal_inaccurate"),
    "tol_optimal": 1e-6,
    "tol_optimal_inaccurate": 1e-5,
}
```

Do not hard-code dimensions in defaults. Build them after `n_u`, `n_y`, and `Q_out` are known.

---

## Mode 1: `free_disturbance_prior`

### Core idea

Do not freeze the pseudo steady disturbance exactly at `d_hat`. Instead, treat the pseudo steady disturbance as a decision variable and penalize deviation from the observer estimate.

This mode exists because the closed-loop observer may use `dhat` as a bookkeeping state for steady bias, not as a physically trustworthy steady-state disturbance coordinate.

### What this method solves

Decision variables:

```math
x_s,\; u_s,\; d_s,\; r_s
```

with `r_s = yc_s` if no separate reference needs to be stored.

Solve one single target problem:

```math
\min_{x_s,u_s,d_s,r_s}
\frac{1}{2}\|r_s-y_{sp}\|_{Q_r}^2
+ \frac{1}{2}\|u_s-u_{nom}\|_{R_u}^2
+ \frac{1}{2}\|x_s-x_s^{prev}\|_{Q_{\Delta x}}^2
+ \frac{1}{2}\|u_s-u_s^{prev}\|_{R_{\Delta u}}^2
+ \frac{1}{2}\|d_s-\hat d_k\|_{Q_d}^2
+ \Phi_y^{soft}(y_s)
```

subject to

```math
(I-A)x_s - Bu_s - B_d d_s = 0
```

```math
y_s = Cx_s + C_d d_s
```

```math
r_s = H y_s
```

```math
u_{lo} \le u_s \le u_{hi}
```

plus optional output bounds on `y_s`.

Optional trust region:

```math
\|d_s-\hat d_k\|_\infty \le \Delta_d.
```

### What it is trying to solve

This method tries to compute the best admissible steady-state target while treating the observer disturbance estimate as a **prior** rather than an exact truth.

### Why this can solve the observed problem

In the current method, the target selector hard-freezes

```math
d_s = \hat d_k,
```

but your closed-loop analysis suggests that `dhat` may carry steady output bias while `xhat` returns near zero. In that situation, forcing exact steady-state equations around the frozen `dhat` can make the selector solve around an estimator artifact.

This method removes that rigidity. It still respects the disturbance estimate through `Q_d`, but does not assume the estimate is exact.

### What part of the current method conflicts with this idea

The frozen-disturbance assumption in the current method is the conflicting piece. The current selector treats `dhat` as a trustworthy steady-state parameter. Mode 1 instead treats it as a soft prior.

### Default parameters for this mode

```python
mode1_defaults = {
    "u_nom": None,
    "Qr_diag": 1e8 * Q_out,
    "Ru_diag": np.ones(n_u),
    "Qd_diag": 1e2 * np.ones(n_d),
    "Qdx_diag": 1e-3 * np.ones(n_x),
    "Rdu_diag": 1.0 * Rmove_diag,
    "soft_output_bounds": True,
    "Wy_low_diag": 1e3 * np.ones(n_y),
    "Wy_high_diag": 1e3 * np.ones(n_y),
    "u_tight": np.zeros(n_u),
    "y_tight": np.zeros(n_y),
    "delta_d_inf": None,        # optional trust region, disabled by default
    "solver_pref": DEFAULT_CVXPY_SOLVERS,
    "accept_statuses": ("optimal", "optimal_inaccurate"),
    "tol_optimal": 1e-6,
    "tol_optimal_inaccurate": 1e-5,
}
```

Notes on defaults:
- `Qd_diag` should be large enough to keep `d_s` near `d_hat`, but not so large that it reproduces the current rigid behavior.
- `Qdx_diag` is activated by default here because Mode 1 benefits from target continuity.
- The default `delta_d_inf` is `None`. Add it only if drift becomes too large.

---

## Mode 2: `compromised_reference`

### Core idea

Introduce an admissible steady controlled output `r_s` and treat that as the official steady reference when the raw setpoint cannot or should not be imposed exactly.

This is a target-consistency fix. It is especially important when the Lyapunov center and the tracked target drift apart.

### What this method solves

Decision variables:

```math
x_s,\; u_s,\; r_s
```

and optionally `d_s = \hat d_k` frozen, or `d_s` optimized only if the user turns that on. For the first implementation, keep this mode separate from Mode 1 and freeze `d_s = \hat d_k` by default.

Solve:

```math
\min_{x_s,u_s,r_s}
\frac{1}{2}\|r_s-y_{sp}\|_{Q_r}^2
+ \frac{1}{2}\|u_s-u_{nom}\|_{R_u}^2
+ \frac{1}{2}\|x_s-x_s^{prev}\|_{Q_{\Delta x}}^2
+ \frac{1}{2}\|u_s-u_s^{prev}\|_{R_{\Delta u}}^2
+ \Phi_y^{soft}(y_s)
```

subject to

```math
(I-A)x_s - Bu_s - B_d \hat d_k = 0
```

```math
y_s = Cx_s + C_d \hat d_k
```

```math
r_s = H y_s
```

```math
u_{lo} \le u_s \le u_{hi}
```

plus optional output bounds.

### What it is trying to solve

This method computes the admissible steady output that is closest to the requested setpoint, then uses that admissible output as the official target for all downstream layers.

### Why this can solve the observed problem

The current code can produce `y_s != y_sp` in fallback, while the upstream MPC and tracking objective may still target the raw setpoint depending on `mpc_target_policy`. That means the dynamic controller can be pushed toward one equilibrium while the Lyapunov certificate is centered around another.

Mode 2 removes that internal contradiction. The selected admissible output `r_s` becomes the target everywhere.

### What part of the current method conflicts with this idea

The current method permits target mismatch across layers:
- the target selector may return a fallback admissible output,
- but the upstream MPC can still use `raw_setpoint`,
- and the safety layer is centered on the selector equilibrium.

That inconsistency is the part that conflicts with Mode 2.

### Required downstream code change for Mode 2

When `selector_mode == "compromised_reference"`:
- make the upstream MPC candidate track `r_s`,
- make the safety-correction tracking term use `r_s`,
- keep the Lyapunov center at the same selected equilibrium.

Do not leave `mpc_target_policy` disconnected from the selector result in this mode.

### Default parameters for this mode

```python
mode2_defaults = {
    "u_nom": None,
    "Qr_diag": 1e8 * Q_out,
    "Ru_diag": np.ones(n_u),
    "Qdx_diag": 1e-3 * np.ones(n_x),
    "Rdu_diag": 1.0 * Rmove_diag,
    "soft_output_bounds": True,
    "Wy_low_diag": 1e3 * np.ones(n_y),
    "Wy_high_diag": 1e3 * np.ones(n_y),
    "u_tight": np.zeros(n_u),
    "y_tight": np.zeros(n_y),
    "freeze_d_at_estimate": True,
    "solver_pref": DEFAULT_CVXPY_SOLVERS,
    "accept_statuses": ("optimal", "optimal_inaccurate"),
    "tol_optimal": 1e-6,
    "tol_optimal_inaccurate": 1e-5,
}
```

---

## Mode 3: `single_stage_robust_sstp`

### Core idea

Replace the current exact-stage / fallback-stage split with one single steady-state target problem that always returns the best admissible target.

This is primarily an architectural and numerical simplification. It avoids binary switching between exact and fallback logic.

### What this method solves

For the first implementation, keep `d_s = \hat d_k` frozen by default, but structure the code so later you can enable free `d_s` if desired.

Decision variables:

```math
x_s,\; u_s,\; r_s,\; \epsilon_x,\; \epsilon_y
```

Solve:

```math
\min_{x_s,u_s,r_s,\epsilon_x,\epsilon_y}
\frac{1}{2}\|r_s-y_{sp}\|_{Q_r}^2
+ \frac{1}{2}\|u_s-u_{nom}\|_{R_u}^2
+ \frac{1}{2}\|x_s-x_s^{prev}\|_{Q_{\Delta x}}^2
+ \frac{1}{2}\|u_s-u_s^{prev}\|_{R_{\Delta u}}^2
+ \rho_x \|\epsilon_x\|_1
+ \rho_y \|\epsilon_y\|_1
+ \Phi_y^{soft}(y_s)
```

subject to

```math
(I-A)x_s - Bu_s - B_d \hat d_k = \epsilon_x
```

```math
y_s = Cx_s + C_d \hat d_k
```

```math
r_s = H y_s + \epsilon_y
```

```math
u_{lo} \le u_s \le u_{hi}
```

plus optional output bounds.

You can also implement a simpler version with `\epsilon_x = 0` and only soft reference mismatch. That is acceptable as a first code version, but the full structure above is preferred if it remains convex and numerically stable.

### What it is trying to solve

This method always computes one best admissible equilibrium target instead of trying exact feasibility first and then switching to a different problem.

### Why this can solve the observed problem

You observed that the current target selector is in fallback essentially all the time. That means the exact stage is not an active operating mode in practice. A single-stage target problem removes unnecessary mode switching and should behave more smoothly under estimator drift, mild model mismatch, and changing setpoints.

### What part of the current method conflicts with this idea

The current hard split between exact solve and fallback solve is the conflicting piece. That design assumes exact target feasibility is common or at least important enough to justify a separate mode. Your results suggest that assumption is not holding in the present closed loop.

### Default parameters for this mode

```python
mode3_defaults = {
    "u_nom": None,
    "Qr_diag": 1e8 * Q_out,
    "Ru_diag": np.ones(n_u),
    "Qdx_diag": 1e-3 * np.ones(n_x),
    "Rdu_diag": 1.0 * Rmove_diag,
    "rho_x": 1e5,
    "rho_y": 1e5,
    "soft_output_bounds": True,
    "Wy_low_diag": 1e3 * np.ones(n_y),
    "Wy_high_diag": 1e3 * np.ones(n_y),
    "u_tight": np.zeros(n_u),
    "y_tight": np.zeros(n_y),
    "freeze_d_at_estimate": True,
    "solver_pref": DEFAULT_CVXPY_SOLVERS,
    "accept_statuses": ("optimal", "optimal_inaccurate"),
    "tol_optimal": 1e-6,
    "tol_optimal_inaccurate": 1e-5,
}
```

Notes:
- Start with large `rho_x` and `rho_y` so the slacks are used only when necessary.
- If this creates numerical conditioning problems, scale them down gradually.

---

## Required software architecture

### 1. Add a selector-mode enum or string constants

Create one canonical mode list:

```python
TARGET_SELECTOR_MODES = (
    "current_exact_fallback_frozen_d",
    "free_disturbance_prior",
    "compromised_reference",
    "single_stage_robust_sstp",
)
```

Raise a clear error if the user provides any unsupported mode.

### 2. Add a configuration dataclass or dict-normalizer

Create a config object, for example:

```python
@dataclass
class TargetSelectorConfig:
    selector_mode: str = "current_exact_fallback_frozen_d"
    u_nom: Optional[np.ndarray] = None
    Ty_diag: Optional[np.ndarray] = None
    Qr_diag: Optional[np.ndarray] = None
    Ru_diag: Optional[np.ndarray] = None
    Qx_diag: Optional[np.ndarray] = None
    w_x: float = 1e-6
    Qdx_diag: Optional[np.ndarray] = None
    Rdu_diag: Optional[np.ndarray] = None
    Qd_diag: Optional[np.ndarray] = None
    delta_d_inf: Optional[float] = None
    rho_x: float = 1e5
    rho_y: float = 1e5
    soft_output_bounds: bool = True
    Wy_low_diag: Optional[np.ndarray] = None
    Wy_high_diag: Optional[np.ndarray] = None
    u_tight: Optional[np.ndarray] = None
    y_tight: Optional[np.ndarray] = None
    freeze_d_at_estimate: bool = True
    solver_pref: Optional[Sequence[str]] = None
    tol_optimal: float = 1e-6
    tol_optimal_inaccurate: float = 1e-5
```

Normalize dimensions after `n_x`, `n_u`, and `n_y` are known.

### 3. Shared helper functions

Refactor common code into shared helpers:
- matrix extraction from augmented model,
- bound tightening,
- weight construction,
- stage evaluation,
- output slack handling,
- debug dictionary assembly,
- standard residual and bound checks.

Do not duplicate these helpers four times.

### 4. Implement one function per mode

Implement:

```python
_compute_target_mode0_current_exact_fallback(...)
_compute_target_mode1_free_disturbance_prior(...)
_compute_target_mode2_compromised_reference(...)
_compute_target_mode3_single_stage_robust_sstp(...)
```

Then dispatch from one wrapper.

### 5. Standardize diagnostics

All modes must populate:
- selected mode,
- selected stage name,
- solver status,
- objective value,
- `dhat_used`,
- whether `d_s` was optimized or frozen,
- `target_error_inf`,
- `dyn_residual_inf`,
- `bound_violation_inf`,
- previous-target move sizes,
- and any slacks used.

---

## Required integration changes in the safe filter

### A. Upstream tracking target selection

Update the upstream target-selection logic so that it can use the selector result directly.

Recommended behavior:
- Mode 0: preserve current `mpc_target_policy` options.
- Mode 1: default to `admissible_if_available`.
- Mode 2: force use of `r_s`.
- Mode 3: default to `r_s`, with optional override if user explicitly wants raw setpoint.

### B. Safety-correction tracking term

Where the safety QCQP or tracking objective uses `y_sp`, make that target selectable:
- raw setpoint,
- `yc_s`,
- or `r_s`.

This must be consistent with the selector mode.

### C. Logging and plots

Store, at every time step:
- `selector_mode`,
- `r_s`,
- `d_s`,
- `y_s`,
- `solve_stage`,
- `d_s_minus_dhat_inf`,
- `target_error_inf`,
- and `mpc_tracking_target_source`.

These fields are needed to compare the four methods fairly.

---

## User-facing defaults and override behavior

### Default choice

Keep the default selector mode as the current method:

```python
selector_mode = "current_exact_fallback_frozen_d"
```

### User overrides

Allow every parameter to be overridden by the user. Never lock a parameter by mode. Instead:
- provide sensible defaults by mode,
- merge user-provided values on top,
- and record the final used values in debug output.

For example:

```python
cfg = build_target_selector_config(
    selector_mode="free_disturbance_prior",
    user_overrides={
        "Qd_diag": 5e1 * np.ones(n_d),
        "delta_d_inf": 0.02,
    },
)
```

---

## Validation and comparison tasks

Implement tests and comparisons for all four modes.

### Unit tests

At minimum:
1. shape checks for all returned arrays,
2. infeasible-tightening failure path,
3. `selector_mode` dispatch error path,
4. `r_s` existence and consistency,
5. frozen-`d` modes must satisfy `d_s == d_hat`,
6. free-`d` mode must satisfy finite `d_s` and log `d_s - d_hat`,
7. Mode 0 exact stage and fallback stage must both be reproducible on synthetic examples.

### Closed-loop comparison hooks

Add one script or notebook helper that runs the same scenario with all four modes and compares:
- target mismatch,
- Lyapunov acceptance rate,
- QCQP activation rate,
- fallback rate,
- total infeasibility rate,
- control effort,
- settled tracking error,
- and `d_s - d_hat`.

Do not change the rest of the controller between modes unless the mode definition explicitly requires a target-consistency change.

---

## References to include in code comments and report

Use these references in comments or the requested report where relevant:

1. Rawlings, Mayne, and Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd edition.
   Use for the classical steady-state target problem and offset-free target formulation.

2. Muske and Badgwell, “Disturbance modeling for offset-free linear model predictive control,” Journal of Process Control, 2002.
   Use for disturbance modeling, detectability, and steady-state target calculation under estimated disturbances.

3. Kuntz, *Towards a Turnkey Model Predictive Controller*, PhD thesis, UCSB, 2024.
   Use for the estimator / target-problem / regulator decomposition and offset-free target interpretation.

4. Yan et al., “A predictive control framework with guaranteed safety,” Automatica, 2023.
   Use for the pseudo steady disturbance as an optimization variable and compromised-reference logic.

5. Kassmann, Badgwell, and Hawkins, “Robust steady-state target calculation for model predictive control,” AIChE Journal, 2000.
   Use for the SSTP viewpoint and robust target-calculation framing.

---

## Required report generation task for Codex

After implementing the four modes, create a detailed LaTeX report in the repository that explains all four selector methods.

The report must include:

1. exact mathematics for each method,
2. all decision variables and constraints,
3. what each method is trying to solve,
4. how each method differs from the current baseline,
5. what problem each redesign is meant to fix,
6. default parameters for each method,
7. which downstream parts of the safe filter must be consistent with that method,
8. a comparison table of all four methods,
9. implementation notes mapping each method to the actual code functions,
10. a section on expected advantages, disadvantages, and likely failure modes.

The report should be self-contained and written in paper style, not just code documentation.

---

## Implementation order for Codex

Follow this order exactly:

1. add config object and selector-mode dispatch,
2. preserve Mode 0 exactly,
3. implement Mode 2 next because it is the least invasive redesign,
4. implement Mode 1 next,
5. implement Mode 3 last,
6. update downstream target-consistency logic,
7. add logs and tests,
8. generate the detailed LaTeX report.

---

## Final deliverables expected from Codex

1. code implementing all four selector modes,
2. one common selector interface,
3. one config system with mode defaults and user overrides,
4. updated logging and plotting fields,
5. unit tests,
6. one comparison helper for closed-loop experiments,
7. one detailed LaTeX report for the four methods.

Do not remove the current selector implementation. Preserve it as Mode 0 so all future comparisons are clean and reversible.
