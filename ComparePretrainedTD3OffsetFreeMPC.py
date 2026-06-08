from __future__ import annotations

import argparse
import os
from pprint import pprint

from utils.of_mpc_td3_workflow import (
    ComparisonRunConfig,
    mode_list,
    run_pretrained_of_mpc_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a saved TD3 agent and compare it against offset-free MPC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent-path",
        default=None,
        help=(
            "Checkpoint path. If omitted, the newest generated OF-MPC pretraining "
            "checkpoint is used, falling back to Data/agent_2507171027.pkl."
        ),
    )
    parser.add_argument(
        "--modes",
        choices=["both", "nominal", "disturb"],
        default="both",
        help="Comparison modes to run.",
    )
    parser.add_argument("--n-tests", type=int, default=2)
    parser.add_argument("--set-points-len", type=int, default=400)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use auto to select cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join("results", "PretrainOFMPCComparison"),
        help="Comparison artifact root. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--baseline-cache-dir",
        default=os.path.join("results", "PretrainOFMPCComparison", "baselines"),
        help="Cached OF-MPC baseline root. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--force-baseline-refresh",
        action="store_true",
        help="Regenerate OF-MPC baselines even when cached pickles are available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ComparisonRunConfig(
        agent_path=args.agent_path,
        modes=mode_list(args.modes),
        n_tests=int(args.n_tests),
        set_points_len=int(args.set_points_len),
        seed=int(args.seed),
        device_requested=str(args.device),
        output_root=str(args.output_root),
        baseline_cache_dir=str(args.baseline_cache_dir),
        force_baseline_refresh=bool(args.force_baseline_refresh),
    )

    print("Starting pretrained TD3 versus OF-MPC comparison.")
    print(f"Modes: {', '.join(config.modes)}")
    print(f"Setpoint length: {config.set_points_len}")
    result = run_pretrained_of_mpc_comparison(config)
    print("Comparison complete.")
    print(f"Run directory: {result['run_dir']}")
    print(f"Summary: {result['summary_path']}")
    print(f"Metrics JSON: {result['metrics_json']}")
    print(f"Metrics CSV: {result['metrics_csv']}")
    pprint(result["records"])


if __name__ == "__main__":
    main()
