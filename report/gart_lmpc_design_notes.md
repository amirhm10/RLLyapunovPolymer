# GART-LMPC v0 Design Notes

GART-LMPC is a parallel Lyapunov MPC path for the Polymer CSTR study. It does not replace the existing bounded or governed-reference target selectors. The new target object is designed to make the proof bookkeeping explicit:

$$
y_s \text{ is selected as the closest reachable, contraction-admissible output to } y_{sp}.
$$

The implementation uses scaled-deviation coordinates and the current output-disturbance augmented model:

$$
x_{k+1}=Ax_k+Bu_k,\qquad d_{k+1}=d_k,\qquad y_k=Cx_k+d_k.
$$

The augmented observer state is $\hat z_k=[\hat x_k^\top,\hat d_k^\top]^\top$. GART does not feed the raw disturbance estimate directly into the proof target. Instead it uses a certified disturbance estimate:

$$
d^c_k=
\Pi_{\mathcal D}
\left[
d^c_{k-1}
+
\operatorname{clip}_{\Delta d_{\max}}
\left(
\alpha_d(\hat d^{raw}_k-d^c_{k-1})
\right)
\right].
$$

This guarantees:

$$
\|d^c_k-d^c_{k-1}\|_\infty\le \|\Delta d_{\max}\|_\infty.
$$

## Target Selection

For a command reference $r_k$, GART solves a lexicographic equilibrium target problem. Stage 1 minimizes only the output mismatch:

$$
J_1^\star =
\min_{x_s,u_s}
\|W_y(Cx_s+d^c_k-r_k)\|_2^2
$$

subject to the steady-state equation, input headroom, terminal-input tightening, optional output bounds, and hard target-motion bounds.

Stage 2 is only a tie-breaker inside a near-optimal shell:

$$
\|W_y(Cx_s+d^c_k-r_k)\|_2^2
\le
J_1^\star+\tau_{abs}+\tau_{rel}\max(1,J_1^\star).
$$

The tie-breaker smooths $x_s,u_s,y_s$ against the previous accepted target and weakly favors input midpoint headroom. This avoids recreating the older weighted least-squares selector where setpoint mismatch and smoothing competed in the primary objective.

## Dynamic Governor

The raw reference $r_k=y_{sp,k}$ is tried first. The candidate target is accepted only if:

- the lexicographic target solve succeeds;
- terminal input tightening remains feasible;
- target-motion limits hold;
- the contraction probe passes when enabled.

If the raw reference fails, GART searches:

$$
r(\alpha)=r_{prev}+\alpha(y_{sp,k}-r_{prev}),\qquad \alpha\in[0,1],
$$

using a coarse grid followed by bisection. If no positive $\alpha$ is accepted, the previous target is held.

## MPC Objective

The GART-LMPC step does not switch between tracking $y_{sp}$ and tracking $y_s$. It uses a mixed objective:

$$
J_k=
\sum_{i=0}^{N-1}
\left[
\|y_{i+1|k}-y_{sp,k}\|_{Q_{raw}}^2
+
\eta_y\|y_{i+1|k}-y_{s,k}\|_{Q_s}^2
+
\eta_u\|u_{i|k}-u_{s,k}\|_{R_s}^2
+
\|\Delta u_{i|k}\|_{R_\Delta}^2
\right]
+
\lambda_s s_k.
$$

The Lyapunov constraint remains centered on $x_s$:

$$
V(x_{1|k}-x_{s,k})
\le
\rho V(x_k-x_{s,k})+\epsilon+s_k.
$$

The first implementation supports hard and soft contraction modes. The smoke case uses nominal plant mode with five short episodes to verify the wiring before longer disturbed studies.

## Practical-Stability Bookkeeping

If the controller enforces:

$$
V(x_{k+1}-x_{s,k})
\le
\rho V(x_k-x_{s,k})+\epsilon+s_k,
$$

and the target moves by:

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

Therefore:

$$
V(x_{k+1}-x_{s,k+1})
\le
(1+\eta)\rho V(x_k-x_{s,k})
+
(1+\eta)(\epsilon+s_k)
+
c_P\|\Delta x_{s,k}\|^2.
$$

Choose $\eta$ such that:

$$
\bar\rho=(1+\eta)\rho<1.
$$

If $s_k\le \bar s$ and $\|\Delta x_{s,k}\|\le \bar\Delta_x$, then:

$$
V_{k+1}
\le
\bar\rho V_k+\bar w,
\qquad
\bar w=(1+\eta)(\epsilon+\bar s)+c_P\bar\Delta_x^2.
$$

Thus:

$$
\limsup_{k\to\infty}V_k
\le
\frac{\bar w}{1-\bar\rho}.
$$

External tracking decomposes as:

$$
y_k-y_{sp,k}
=
(y_k-y_{s,k})+(y_{s,k}-y_{sp,k}),
$$

so:

$$
\|y_k-y_{sp,k}\|
\le
L_h\|x_k-x_{s,k}\|
+
\|y_{s,k}-y_{sp,k}\|
+
\text{model/estimator error}.
$$

This is why the implementation logs both $y-y_s$ and $y_s-y_{sp}$ instead of hiding unreachable references behind clean target-centered tracking.
