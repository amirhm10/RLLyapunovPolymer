# Online TD3 Latest Good Results Report

## Change

- Added `report/online_td3_latest_good_results_2026-06-20.md`.
- Added derived comparison artifacts under `report/figures/2026-06-20_online_td3_latest_good/`.

## Data Used

The report analyzes the latest completed online TD3 runner batch:

- `results/OnlineTD3_ColdStart_NoSafetyGate/20260620_003041`
- `results/OnlineTD3_ColdStart_SafetyGate/20260620_003031`
- `results/OnlineTD3_OFMPCPretrained_NoSafetyGate/20260620_003020`
- `results/OnlineTD3_OFMPCPretrained_SafetyGate/20260620_002952`

## Main Finding

The latest configuration is close to the desired behavior. OF-MPC-pretrained no-gate is still the nominal upper bound, while OF-MPC-pretrained with the GART safety gate has low intervention burden and strong late-episode tracking. The latest batch also reduces diagnostic unsafe or intervention rates relative to the previous same-seed reference.

## Validation

- Recomputed summary, phase, late-100, and delta tables from local result exports.
- Generated compact aggregate figures for reward/tracking, safety rates, and deltas versus the previous reference.
- Checked that the report references local result paths and derived artifacts.
