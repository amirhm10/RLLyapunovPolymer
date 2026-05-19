# RL-MPC Slide Checklist

Use this for technical RL and MPC slides.

## Comparison design

- What is the baseline
- What is the proposed method
- What is held constant
- What metric matters most

## Process-control details

- Are output constraints or input bounds shown when relevant
- Is offset-free behavior discussed explicitly
- Are disturbance estimates, model mismatch, or operating drift described
- Are input-move penalties or actuator movement treated fairly

## Interpretation

- Does reward improvement match tracking improvement
- Is the gain coming from pretraining, reward shaping, replay design, residual action, or another factor
- Is there evidence that the change helps steady-state error, transient response, or both
- Is any claim clearly limited to polymer, `C_2`, nominal, fluctuation, or ramp scenarios
