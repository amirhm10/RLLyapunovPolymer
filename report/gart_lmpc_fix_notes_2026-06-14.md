# GART-LMPC Correctness and Runtime Patch

Date: 2026-06-14

## What Was Fixed

1. Separated target solve success from target acceptance.
2. Prevented contraction-failed targets from being used by LMPC.
3. Prevented initial rejected targets from being stored as valid targets.
4. Made GART raw objective the default working candidate.
5. Disabled mixed target-centered objectives by default.
6. Added gated target-centered objective support.
7. Added runtime safety guard and smoke defaults.
8. Disabled recursive result scanning by default.
9. Added observer-replay target-only diagnostics.
10. Added clearer contraction-margin sign conventions.

## Methodology After Patch

The GART target selector now reports three separate concepts:

$$
\text{solve success} \ne \text{accepted target} \ne \text{usable LMPC target}.
$$

The LMPC solver only uses a target if:

$$
\texttt{accepted}=\texttt{usable\_for\_lmpc}=\texttt{True}.
$$

If a target QP solves but fails terminal feasibility or the required contraction probe, the result is:

```text
solve_success = True
accepted = False
usable_for_lmpc = False
success = False
```

The controller then holds the previous input with:

```text
method = gart_target_not_usable_hold_prev
```

## Default Controller

The default GART controller is now:

$$
\text{GART target for Lyapunov centering}
+
\text{raw } y_{sp}\text{ tracking objective}.
$$

The mixed objective remains available only as an experimental option. It is disabled by default because the nominal and disturbance results showed that mixed target-centered terms can pull the optimizer toward a conservative or held target.

## Runtime Safety

The root runner now defaults to a short nominal target-only smoke run:

```text
MODE = nominal
N_TESTS = 1
SET_POINTS_LEN = 20
RUN_TARGET_ONLY = True
RUN_CLOSED_LOOP = False
```

Full runs require explicit confirmation:

```powershell
python GARTLyapunovMPC.py --full --confirm-full --closed-loop --no-target-only --mode nominal --n-tests 5 --set-points-len 400
```

## Diagnostics Added

The reports and saved step records can now include:

- target solve success rate
- target accepted rate
- target usable rate
- rejection reasons
- target exact, good, acceptable, and unreachable rates
- governor active and hold rates
- stage1 and stage2 probe margins
- target-probe margin with positive-good sign
- MPC contraction violation and positive-good margin

## Scientific Conclusion

The previous results do not show that GART failed. They show that target-centered mixed objectives are unsafe when the selected target is far from the requested setpoint. GART raw objective remains the working candidate.
