from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float:
    if value in {None, ""}:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    vals = []
    for row in rows:
        text = str(row.get(key, "")).strip().lower()
        vals.append(1.0 if text in {"true", "1", "yes"} else 0.0)
    return float(np.mean(vals)) if vals else None


def _numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals = np.array([_to_float(row.get(key)) for row in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": None, "p95": None, "max": None}
    return {"mean": float(np.mean(vals)), "p95": float(np.quantile(vals, 0.95)), "max": float(np.max(vals))}


def summarize_target_only(target_dir: str | Path) -> dict[str, Any]:
    target_dir = Path(target_dir)
    rows = _read_csv(target_dir / "target_only_steps.csv")
    return {
        "target_dir": str(target_dir),
        "n_steps": len(rows),
        "target_success_rate": _mean_bool(rows, "target_success"),
        "governor_active_rate": _mean_bool(rows, "governor_active"),
        "hold_previous_rate": _mean_bool(rows, "hold_previous"),
        "unreachable_rate": _mean_bool(rows, "classified_unreachable"),
        "target_error_inf": _numeric_summary(rows, "target_error_inf"),
        "contraction_probe_margin": _numeric_summary(rows, "contraction_probe_margin"),
        "input_headroom_min": _numeric_summary(rows, "input_headroom_min"),
    }


def summarize_closed_loop(lmpc_dir: str | Path) -> dict[str, Any]:
    lmpc_dir = Path(lmpc_dir)
    comparison = _read_csv(lmpc_dir / "comparison.csv")
    cases = {}
    for row in comparison:
        case = row.get("case_name", "unknown")
        cases[case] = {key: row[key] for key in row}
    return {
        "lmpc_dir": str(lmpc_dir),
        "n_cases": len(cases),
        "cases": cases,
    }


def make_analysis_plots(target_dir: str | Path | None = None, lmpc_dir: str | Path | None = None) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written: list[str] = []
    if target_dir is not None:
        target_dir = Path(target_dir)
        arrays_path = target_dir / "target_only_arrays.npz"
        if arrays_path.exists():
            with np.load(arrays_path) as data:
                y_sp = np.asarray(data["y_sp"], dtype=float)
                y_s = np.asarray(data["y_s"], dtype=float)
                margins = np.asarray(data["contraction_probe_margin"], dtype=float)
            plot_dir = target_dir / "analysis_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            steps = np.arange(y_s.shape[0])
            plt.figure(figsize=(8, 4))
            for idx in range(y_s.shape[1]):
                plt.plot(steps, y_sp[: y_s.shape[0], idx], "--", label=f"y_sp[{idx}]")
                plt.plot(steps, y_s[:, idx], label=f"y_s[{idx}]")
            plt.legend()
            plt.tight_layout()
            path = plot_dir / "target_error_overlay.png"
            plt.savefig(path, dpi=180)
            plt.close()
            written.append(str(path))

            plt.figure(figsize=(8, 3))
            plt.plot(steps, margins)
            plt.axhline(0.0, color="k", linewidth=0.8)
            plt.tight_layout()
            path = plot_dir / "probe_margin.png"
            plt.savefig(path, dpi=180)
            plt.close()
            written.append(str(path))
    if lmpc_dir is not None:
        lmpc_dir = Path(lmpc_dir)
        rows = _read_csv(lmpc_dir / "comparison.csv")
        if rows:
            names = [row.get("case_name", "unknown") for row in rows]
            rmse = [_to_float(row.get("output_rmse_raw_ysp")) for row in rows]
            target = [_to_float(row.get("mean_target_error_inf")) for row in rows]
            plot_dir = lmpc_dir / "analysis_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            x = np.arange(len(names))
            plt.figure(figsize=(9, 4))
            plt.bar(x - 0.18, rmse, width=0.36, label="RMSE to y_sp")
            plt.bar(x + 0.18, target, width=0.36, label="mean target error")
            plt.xticks(x, names, rotation=20, ha="right")
            plt.legend()
            plt.tight_layout()
            path = plot_dir / "closed_loop_comparison.png"
            plt.savefig(path, dpi=180)
            plt.close()
            written.append(str(path))
    return written


def _latest(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GART target-selector and GART-LMPC run artifacts.")
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--lmpc-dir", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    target_dir = Path(args.target_dir) if args.target_dir else None
    lmpc_dir = Path(args.lmpc_dir) if args.lmpc_dir else None
    if args.latest:
        target_dir = target_dir or _latest(repo_root / "results" / "GARTTargetSelectorStudy")
        lmpc_dir = lmpc_dir or _latest(repo_root / "results" / "GARTLMPC")

    summary: dict[str, Any] = {}
    if target_dir is not None:
        summary["target_only"] = summarize_target_only(target_dir)
    if lmpc_dir is not None:
        summary["closed_loop"] = summarize_closed_loop(lmpc_dir)
    if args.plots:
        summary["plots"] = make_analysis_plots(target_dir, lmpc_dir)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2))
    return summary


if __name__ == "__main__":
    main()
