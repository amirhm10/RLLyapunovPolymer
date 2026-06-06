# Lyapunov Stability Proof Track And Next Experiments

Date: 2026-06-06

## Objective

This report records the current stability-proof discussion for the direct Lyapunov MPC and direct Lyapunov safety-gate RL work. It also documents the next runner configuration to test:

```python
rho_lyap = 0.99
lyap_eps = 5e-3
n_episodes = 300
```

The main scientific question is whether the current method can support an asymptotic stability proof, or whether the correct claim is practical Lyapunov stability under bounded disturbance-estimation and model errors.

## Current Experiment Configuration

The active 300-episode runners are:

- `DirectLyapunovMPC.py`
- `DirectLyapunovSafetyGateRL_ColdStart.py`
- `DirectLyapunovSafetyGateRL_Pretrained.py`

The direct MPC runner compares:

- governed-reference direct Lyapunov MPC
- offset-free `mpc_only` with governed-reference diagnostics

The two RL runners compare:

- TD3 safety-gate RL with governed-reference direct Lyapunov safety filtering
- offset-free `mpc_only` through the same RL rollout/export path

For all three runners, the Lyapunov diagnostic or hard constraint uses:

$$
V_{k+1|k} \le \rho V_k + \epsilon,
\qquad
\rho = 0.99,
\qquad
\epsilon = 5\times 10^{-3}.
$$

## Evidence From The Latest Short Direct Run

The latest complete short direct run inspected was:

```text
results/directLyap/20260606_014038/
```

It used `rho_lyap = 0.99`, `lyap_eps = 5e-3`, 10 episodes, and the disturbed plant profile.

Performance summary:

| Case | Reward mean | Mean output RMSE | Hard contraction |
|---|---:|---:|---:|
| LMPC | -4.2515 | 0.3734 | 1.0000 |
| `mpc_only` | -4.2413 | 0.3737 | 0.9983 |

Important diagnostic facts:

- `mpc_only` failed the `epsilon = 5e-3` Lyapunov diagnostic at only 14 of 8000 steps.
- LMPC satisfied hard contraction at all 8000 steps.
- The saved disturbance arrays matched exactly between LMPC and `mpc_only`:
  `qi`, `qs`, `ha`, `y_sp`, and `y_sp_steps` all had max absolute difference equal to zero.

This means the latest direct comparison is fair with respect to disturbance and reference schedule. The remaining differences are due to controller action and target/solver behavior.

## Practical Stability Result

With a fixed target, exact model, correct disturbance estimate, and recursive feasibility, the current Lyapunov inequality gives:

$$
V_{k+1} \le \rho V_k + \epsilon.
$$

Unrolling the recursion gives:

$$
V_k \le \rho^k V_0
      + \frac{1-\rho^k}{1-\rho}\epsilon.
$$

Therefore:

$$
\limsup_{k\to\infty} V_k
\le
\frac{\epsilon}{1-\rho}.
$$

For the active setting:

$$
\frac{5\times 10^{-3}}{1-0.99}=0.5.
$$

So the correct proof claim for fixed positive `lyap_eps` is practical stability with an ultimate Lyapunov-value bound of order $O(0.5)$ in the scaled/model Lyapunov coordinates. This should not be written as $o(0.5)$, and it should not be called asymptotic convergence to zero.

## What Would Be Needed For Asymptotic Convergence

For nominal asymptotic convergence to the governed equilibrium, the additive term must vanish:

$$
V_{k+1} \le \rho V_k + \epsilon_k,
\qquad
\epsilon_k \to 0.
$$

If $0 < \rho < 1$ and $\epsilon_k \to 0$, then $V_k \to 0$ under the same fixed-target, feasibility, and model-consistency assumptions.

A proof-oriented algorithm can therefore use a decaying slack:

$$
\epsilon_k = \epsilon_0 \beta^{\tau_k},
\qquad
0 < \beta < 1,
$$

where $\tau_k$ is time since the latest reference or disturbance event.

## Moving Setpoints And Changing Disturbances

When the target changes, the Lyapunov error is measured relative to a moving equilibrium:

$$
e_k = \hat x_k - x_s(k).
$$

Even with perfect control, target motion injects error through:

$$
x_s(k+1)-x_s(k).
$$

For changing setpoints and changing disturbance estimates, the more honest inequality is:

$$
V_{k+1}
\le
\rho V_k
+ c_s\|x_s(k+1)-x_s(k)\|^2
+ c_d\|\hat d_{k+1}-\hat d_k\|^2
+ c_m\|\Delta_{\mathrm{model},k}\|^2
+ \epsilon_{\mathrm{num},k}.
$$

This leads to two possible proof statements:

- If the setpoint, governed target, disturbance estimate, and model error all settle so the additive terms vanish, then asymptotic convergence to the governed equilibrium can be claimed.
- If any of those terms remain persistently nonzero, the correct claim is input-to-state or practical Lyapunov stability, with an ultimate bound determined by the persistent motion and model/estimation error.

For the current repeated setpoint schedule, a global asymptotic claim over the full infinite repeated trajectory is not appropriate. A clean claim is convergence within fixed segments after the setpoint and disturbance estimate settle, plus practical bounded tracking over the full run.

## Epsilon Zero With Rho Near One

The alternative setting:

$$
\epsilon = 0,
\qquad
\rho = 0.9999
$$

is proof-friendly because it removes the additive ultimate bound:

$$
V_k \le 0.9999^k V_0.
$$

However, it has two practical risks:

- The contraction is extremely slow because $\rho$ is very close to one.
- The condition is brittle near the target because even tiny model, observer, or numerical errors can violate strict decrease.

Comparing the two conditions:

$$
0.99V + 0.005 \le 0.9999V
$$

only when:

$$
V \ge 0.505.
$$

Thus `rho = 0.9999`, `epsilon = 0` is looser far from the target but stricter near the target. It may look attractive for proof writing, but it can create unnecessary rejection or fallback near steady operation.

## Recommended Next Algorithmic Step

The next proof-driven controller variant should use adaptive slack:

$$
\epsilon_k =
\epsilon_0\beta^{\tau_k}
+ c_r\|r_k-r_{k-1}\|^2
+ c_s\|x_s(k)-x_s(k-1)\|^2
+ c_d\|\hat d_k-\hat d_{k-1}\|^2.
$$

Suggested initial values:

```python
epsilon_0 = 5e-3
beta = 0.995
rho_lyap = 0.99
```

The purpose is to keep the current robust behavior immediately after setpoint and disturbance movement, while allowing the Lyapunov tolerance to decay toward zero once the governed target and disturbance estimate settle.

## Concrete Next Experiment

Run a 300-episode comparison with the restored baselines:

- direct LMPC with fixed `epsilon = 5e-3`
- cold-start safety-gate RL with fixed `epsilon = 5e-3`
- pretrained safety-gate RL with fixed `epsilon = 5e-3`
- same-run `mpc_only` baselines for all relevant runners

Metrics to inspect:

- hard contraction rate
- diagnostic unsafe rate for `mpc_only`
- fallback or intervention rate for RL gate
- reward and `reward_no_penalty`
- output RMSE and maximum reference error
- target mismatch $y_s-y_{sp}$
- input saturation and input movement
- contraction residual $V_{k+1}-\rho V_k-\epsilon$

If the fixed `5e-3` 300-episode runs remain stable, implement the adaptive epsilon schedule as the next version and compare:

$$
\epsilon = 5\times 10^{-3}
\quad\text{versus}\quad
\epsilon_k \to 0.
$$

## Recommended Paper Claim

For the current fixed-epsilon implementation:

> The controller enforces recursive model-based practical first-step Lyapunov contraction around the governed target, with an ultimate bound determined by the Lyapunov tolerance, target motion, model mismatch, and disturbance-estimation error.

For a future adaptive-epsilon version:

> If the reference, governed target, disturbance estimate, and model error converge, then the vanishing-slack Lyapunov MPC gives asymptotic convergence to the governed equilibrium. If these terms remain bounded but nonzero, the controller provides practical input-to-state Lyapunov stability.

## Open Items

- Implement adaptive `lyap_eps` scheduling in the direct LMPC and safety-gate RL rollout functions.
- Save the per-step effective `lyap_eps_k` in debug arrays.
- Add plots for `lyap_eps_k`, target motion, disturbance-estimate motion, and contraction residual.
- Keep the fixed `5e-3` run as the immediate 300-episode benchmark before changing the algorithm.
