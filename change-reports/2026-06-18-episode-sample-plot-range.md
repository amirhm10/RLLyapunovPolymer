# Episode Sample Plot Range

## Change

- Removed the fixed 20-block cap from the episode sample plotter in `Lyapunov/safety_debug.py`.
- The `ep_samples` plots now sample one episode from every 10-episode block across the full run length.

## Why

Older online runs used 200 episodes, so the previous hard cap of 20 blocks matched the full run. Newer runs use 300 episodes, which meant the sampled episode plots stopped at the 191-200 block even though later episodes existed.

## Expected Behavior

For a 300-episode run, the sample plot directory should now include sampled blocks through episodes 291-300. The separate last-episode plot remains unchanged.
