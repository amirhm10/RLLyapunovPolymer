# Final GART-LMPC Design Notes

GART-LMPC is the final Lyapunov MPC path selected for the Polymer CSTR study. The exposed runner now uses one method only:

- case name: `gartlmpc`
- result folder: `results/GARTLMPC/<timestamp>/gartlmpc`
- plant mode: configured in `GARTLyapunovMPC.py`
- MPC objective: raw setpoint tracking
- Lyapunov mode: hard first-step contraction
- slack use: disabled in the final method

All variables in the controller are in scaled-deviation coordinates unless stated otherwise.

## Final Parameters

The root runner and final experiment helper use:

```python
CASE_NAME = "gartlmpc"
objective = "raw"
lyapunov_mode = "hard"
rho = 0.98
eps = 1.0e-3
dx_s_max_abs = 0.05
dy_s_max_abs = 1.0
input_headroom_frac = 0.01
d_rate_scale = 1.0
adaptive_rate_enabled = False
disable_u_mid_tiebreak = True
disable_x_smoothing = True
disable_y_smoothing = True
```

The final choice keeps the target sequence bounded enough for the moving-target proof bookkeeping, but avoids the conservative jumps seen with `eps = 1e-4` and tighter `dy_s` bounds.

## Model And Certified Disturbance

GART uses the output-disturbance augmented model:

$$
x_{k+1}=Ax_k+Bu_k,\qquad d_{k+1}=d_k,\qquad y_k=Cx_k+d_k.
$$

The observer state is:

$$
\hat z_k=[\hat x_k^\top,\hat d_k^\top]^\top.
$$

The target selector does not use the raw observer disturbance directly. It uses a certified disturbance sequence:

$$
d^c_k=
\Pi_{\mathcal D}
\left[
d^c_{k-1}
+
\operatorname{clip}_{\Delta d_{\max}}
\left(
\alpha_d(\hat d_k-d^c_{k-1})
\right)
\right].
$$

For the final method this is the fixed symmetric bounded-rate certificate:

$$
\|d^c_k-d^c_{k-1}\|_\infty\le \|\Delta d_{\max}\|_\infty.
$$

Adaptive disturbance-rate projection remains in the core implementation for reproducibility, but it is not exposed by the final runner because the disturbance experiments showed mixed behavior and persistent late jumps.

## Target Selection

For the current setpoint $y_{sp,k}$, GART solves a lexicographic equilibrium target problem. Stage 1 chooses the closest reachable output:

$$
J_1^\star =
\min_{x_s,u_s}
\|W_y(Cx_s+d^c_k-y_{sp,k})\|_2^2
$$

subject to the steady-state equation, input bounds with headroom, terminal-input tightening, and the final target-motion bounds:

$$
\|x_s(k)-x_s(k-1)\|_\infty \le 0.05,
\qquad
\|y_s(k)-y_s(k-1)\|_\infty \le 1.0.
$$

The bounds are component-wise in scaled-deviation coordinates.

Stage 2 is only a tie-breaker inside the primary-cost shell:

$$
\|W_y(Cx_s+d^c_k-y_{sp,k})\|_2^2
\le
J_1^\star+\tau_{abs}+\tau_{rel}\max(1,J_1^\star).
$$

In the final runner, the old smoothing-to-previous-target terms and input-midpoint tie-breaker are disabled. The remaining input smoothness reference is the actually applied previous input, so the target selector is consistent with online operation and later RL pretraining trajectories.

## Dynamic Governor

GART first tries the raw reference:

$$
r_k=y_{sp,k}.
$$

The target is accepted only when the target QP solves and the target is usable for LMPC. If the raw reference is rejected, the governor searches:

$$
r(\alpha)=r_{prev}+\alpha(y_{sp,k}-r_{prev}),\qquad \alpha\in[0,1],
$$

using grid candidates followed by bisection. If no positive candidate is accepted, the previous accepted target is held.

The final result logs `solve_success`, `accepted`, `usable_for_lmpc`, and `rejection_reason` separately. In the final controller, only accepted and usable targets can enter LMPC.

## MPC Objective

The final GART-LMPC performance objective tracks the raw setpoint, not the target output:

$$
J_k=
\sum_{i=0}^{N-1}
\left[
\|y_{i+1|k}-y_{sp,k}\|_{Q_{raw}}^2
+
\|\Delta u_{i|k}\|_{R_\Delta}^2
\right].
$$

The target-centered terms are disabled in the final runner:

$$
\eta_y = 0,\qquad \eta_u = 0.
$$

The target is used for certification, not as the performance reference:

$$
V(x_{1|k}-x_{s,k})
\le
\rho V(x_k-x_{s,k})+\epsilon.
$$

The final contraction constants are:

$$
\rho=0.98,\qquad \epsilon=10^{-3}.
$$

## Practical-Stability Bookkeeping

If:

$$
V(x_{k+1}-x_{s,k})
\le
\rho V(x_k-x_{s,k})+\epsilon,
$$

and:

$$
\Delta x_{s,k}=x_{s,k+1}-x_{s,k},
$$

then:

$$
x_{k+1}-x_{s,k+1}
=
(x_{k+1}-x_{s,k})-\Delta x_{s,k}.
$$

For $V(e)=e^\top P e$ and any $\eta>0$:

$$
V(a-b)
\le
(1+\eta)V(a)
+
\left(1+\frac{1}{\eta}\right)\lambda_{\max}(P)\|b\|^2.
$$

Thus:

$$
V(x_{k+1}-x_{s,k+1})
\le
(1+\eta)\rho V(x_k-x_{s,k})
+
(1+\eta)\epsilon
+
c_P\|\Delta x_{s,k}\|^2.
$$

With $\bar\rho=(1+\eta)\rho<1$ and bounded $\|\Delta x_{s,k}\|$, the closed-loop error is practically stable around the moving target sequence.

External tracking decomposes as:

$$
y_k-y_{sp,k}
=
(y_k-y_{s,k})+(y_{s,k}-y_{sp,k}).
$$

This is why the final artifacts still log both plant tracking error and target mismatch.

## Removed Runner Variants

The following exploratory paths are no longer exposed by `GARTLyapunovMPC.py` or `experiments/run_gart_target_selector_study.py`:

- old governed-reference baseline
- target-only synthetic diagnostics
- observer-replay target-only diagnostics
- target ablation matrix
- mixed objective cases
- soft/slack contraction cases
- adaptive disturbance certificate cases
- asymmetric disturbance-rate cases
- no-`dx_s` cases

The reusable core implementations remain available in `Lyapunov/gart_target.py` and `Lyapunov/gart_lmpc.py` for old artifact compatibility and future controlled experiments.
