from __future__ import annotations

import argparse
import os
from pprint import pprint

from utils.of_mpc_td3_workflow import (
    ComparisonRunConfig,
    mode_list,
    run_pretrained_of_mpc_comparison,
)


DEFAULT_ACTOR_LAYER_SIZES = [256, 256, 256]
DEFAULT_CRITIC_LAYER_SIZES = [256, 256, 256]


def parse_layer_sizes(value: str) -> tuple[int, ...]:
    values = [part.strip() for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Layer sizes must contain at least one integer.")
    try:
        layer_sizes = tuple(int(part) for part in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Layer sizes must be comma-separated integers.") from exc
    if any(size <= 0 for size in layer_sizes):
        raise argparse.ArgumentTypeError("Layer sizes must be positive integers.")
    return layer_sizes


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
        "--actor-layers",
        type=parse_layer_sizes,
        default=",".join(str(value) for value in DEFAULT_ACTOR_LAYER_SIZES),
        help="Comma-separated actor hidden layer sizes used to load the checkpoint.",
    )
    parser.add_argument(
        "--critic-layers",
        type=parse_layer_sizes,
        default=",".join(str(value) for value in DEFAULT_CRITIC_LAYER_SIZES),
        help="Comma-separated critic hidden layer sizes used to load the checkpoint.",
    )
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
        actor_layer_sizes=tuple(args.actor_layers),
        critic_layer_sizes=tuple(args.critic_layers),
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
    print(f"Actor layers: {list(config.actor_layer_sizes)}")
    print(f"Critic layers: {list(config.critic_layer_sizes)}")
    result = run_pretrained_of_mpc_comparison(config)
    print("Comparison complete.")
    print(f"Run directory: {result['run_dir']}")
    print(f"Summary: {result['summary_path']}")
    print(f"Metrics JSON: {result['metrics_json']}")
    print(f"Metrics CSV: {result['metrics_csv']}")
    pprint(result["records"])


if __name__ == "__main__":
    main()
