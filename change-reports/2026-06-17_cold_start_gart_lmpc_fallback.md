# Cold-Start GART-LMPC Fallback Gate

## Summary

Changed the cold-start online TD3 safety-gate runner so unsafe actor actions are replaced by the GART-LMPC fallback controller directly instead of first solving the Section 16 projection QCQP.

## Motivation

The intended fallback behavior for this run is:

$$
u_k =
\begin{cases}
u_k^{RL}, & \text{if the TD3 proposal passes the safety check}, \\
u_k^{GART-LMPC}, & \text{otherwise}.
\end{cases}
$$

The previous root-runner default used `gart_section16_projection`, which inserted a CVXPY projection step:

$$
\min_u \|u-u_k^{RL}\|_W^2
\quad
\text{s.t.}
\quad
V(x_{k+1}(u)-x_{s,k}) \le \rho V(\hat x_k-x_{s,k}) + \epsilon.
$$

That projection can be useful for Section 16 experiments, but it should not be labeled as fallback in the cold-start GART-LMPC fallback run.

## Technical Changes

- Updated `OnlineTD3_ColdStart_SafetyGate.py` to use `PROJECTION_BACKEND = "direct_accept_or_fallback"`.
- Removed the root-runner Section 16 projection configuration from the cold-start safety-gate run.
- Updated the root-runner comments so the saved configuration and source text describe a GART-LMPC fallback gate, not a Section 16 projection gate.

The shared preset already maps the cold-start safety-gate fallback controller to `gart_lmpc`, so the direct accept-or-fallback backend now routes rejected actor proposals to the GART-LMPC solve.

## BC Exploration Check

No BC exploration value was changed in this update. The existing shared defaults are:

- cold-start full-RL exploration start: `0.1`
- cold-start BC exploration: `0.005`
- pretrained BC exploration: `0.002`

So yes, the cold-start BC phase is already reduced relative to the cold-start full-RL exploration start. It remains slightly higher than the pretrained BC value.

## Validation

- `python -c "... py_compile.compile('OnlineTD3_ColdStart_SafetyGate.py', ...) ..."` passed.
- Runtime behavior can be checked from the printed configuration: `projection_backend` should be `direct_accept_or_fallback`.
- In saved Lyapunov diagnostics, rejected unsafe actions should be counted as GART-LMPC fallback events rather than Section 16 projection events.
