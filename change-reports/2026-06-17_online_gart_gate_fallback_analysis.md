# Online GART Gate And Fallback Analysis Report

## Summary

Created a data-backed report for the latest completed cold-start and OF-MPC-pretrained online TD3 GART runs. The report analyzes raw setpoint tracking, reward behavior, safety-gate intervention rates, verified GART-LMPC fallback, target-unusable hold-previous events, and no-gate diagnostic unsafe rates.

## Files Changed

- `analysis/online_gart_gate_fallback_analysis.py`
  - Added a reproducible script that reads the latest complete online GART result bundles.
  - Writes compact summary, phase, episode, and gate-detail metrics.
  - Generates report figures for episode tracking/reward, gate activity, last-episode tracking, and gate mode counts.
- `report/online_gart_gate_fallback_analysis_2026-06-17.md`
  - Added the scientific analysis report.
- `report/figures/2026-06-17_online_gart_gate_fallback/`
  - Added source metrics and figures used by the report.

## Main Finding

The GART safety gate is active and mostly rejects TD3 candidates for Lyapunov contraction failures, then replaces them with verified GART-LMPC fallback actions. However, both active-gate runs track worse than their no-gate counterparts on `reward_no_penalty` and output RMSE. The no-gate runs are not certified safe because they show diagnostic unsafe events, but they learn better raw tracking.

## Validation

Ran:

```powershell
python analysis/online_gart_gate_fallback_analysis.py
python -c "import ast, pathlib; files=['analysis/online_gart_gate_fallback_analysis.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8'), filename=f) for f in files]; print('ast parse ok', len(files))"
```

The direct `py_compile` validation path was not used as final validation because this OneDrive-backed workspace denied `.pyc` atomic writes. The no-write AST parse passed.
