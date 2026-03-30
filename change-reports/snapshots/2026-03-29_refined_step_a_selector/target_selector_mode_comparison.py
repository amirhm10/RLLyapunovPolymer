import numpy as np

from Lyapunov.target_selector import TARGET_SELECTOR_MODES


def _metric_from_bundle(bundle, key):
    if bundle is None:
        return None
    summary = bundle.get("summary", {}) if isinstance(bundle, dict) else {}
    return summary.get(key)


def summarize_mode_artifact(selector_mode, bundle=None):
    summary = {} if bundle is None else dict(bundle.get("summary", {}))
    return {
        "selector_mode": str(selector_mode),
        "target_mismatch_inf_max": summary.get("target_mismatch_inf_max"),
        "target_error_inf_max": summary.get("target_error_inf_max"),
        "lyapunov_margin_min": summary.get("lyapunov_margin_min"),
        "n_optimized_correction": summary.get("n_optimized_correction"),
        "n_fallback_mpc_verified": summary.get("n_fallback_mpc_verified"),
        "n_fallback_mpc_unverified": summary.get("n_fallback_mpc_unverified"),
        "n_target_failures": summary.get("n_target_failures"),
        "reward_mean": summary.get("reward_mean"),
        "reward_min": summary.get("reward_min"),
        "mode_counts": summary.get("mode_counts"),
    }


def run_target_selector_mode_comparison(
    run_callable,
    base_kwargs,
    selector_modes=TARGET_SELECTOR_MODES,
    bundle_builder=None,
    bundle_builder_kwargs=None,
):
    """
    Run the same closed-loop configuration across multiple selector modes.

    Parameters
    ----------
    run_callable:
        Function such as ``run_mpc_lyapunov`` or ``run_rl_lyapunov``.
    base_kwargs:
        Keyword arguments shared across all runs.
    selector_modes:
        Iterable of selector-mode strings to compare.
    bundle_builder:
        Optional callable that converts ``results`` into a debug/export bundle.
    bundle_builder_kwargs:
        Extra kwargs forwarded to ``bundle_builder``.
    """
    outputs = {}
    bundle_builder_kwargs = {} if bundle_builder_kwargs is None else dict(bundle_builder_kwargs)

    for selector_mode in selector_modes:
        run_kwargs = dict(base_kwargs)
        run_kwargs["selector_mode"] = str(selector_mode)
        results = run_callable(**run_kwargs)
        bundle = None
        if bundle_builder is not None:
            bundle = bundle_builder(results=results, **bundle_builder_kwargs)
        outputs[str(selector_mode)] = {
            "results": results,
            "bundle": bundle,
            "summary": summarize_mode_artifact(selector_mode, bundle=bundle),
        }
    return outputs


def comparison_records(comparison_outputs):
    records = []
    for selector_mode, payload in comparison_outputs.items():
        summary = dict(payload.get("summary", {}))
        summary["selector_mode"] = str(selector_mode)
        records.append(summary)
    return records
