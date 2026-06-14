from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(1 for row in rows if str(row.get(key, "")).strip().lower() == "true") / len(rows))


def summarize_target_steps(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    target_errors = [v for row in rows if (v := _to_float(row.get("target_error_inf"))) is not None]
    margins = [v for row in rows if (v := _to_float(row.get("contraction_probe_margin_good"))) is not None]
    alphas = [v for row in rows if (v := _to_float(row.get("governor_alpha"))) is not None]
    headroom = [v for row in rows if (v := _to_float(row.get("input_headroom_min"))) is not None]
    return {
        "source": str(path),
        "n_steps": len(rows),
        "target_solve_success_rate": _rate(rows, "target_solve_success"),
        "target_accepted_rate": _rate(rows, "target_accepted"),
        "target_usable_rate": _rate(rows, "target_usable_for_lmpc"),
        "target_good_rate": _rate(rows, "target_good"),
        "target_acceptable_rate": _rate(rows, "target_acceptable"),
        "target_unreachable_rate": _rate(rows, "target_unreachable"),
        "contraction_probe_success_rate": _rate(rows, "contraction_probe_success"),
        "governor_active_rate": _rate(rows, "governor_active"),
        "hold_previous_rate": _rate(rows, "hold_previous"),
        "mean_target_error_inf": _mean(target_errors),
        "mean_probe_margin_good": _mean(margins),
        "mean_governor_alpha": _mean(alphas),
        "mean_input_headroom": _mean(headroom),
    }


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Summarize GART observer-replay target diagnostics.")
    parser.add_argument("steps_csv", help="Path to target_only_steps.csv or an ablation case target_only_steps.csv.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    summary = summarize_target_steps(args.steps_csv)
    text = json.dumps(summary, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return summary


if __name__ == "__main__":
    main()
