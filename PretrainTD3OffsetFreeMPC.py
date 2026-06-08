from __future__ import annotations

import argparse
import os

from utils.of_mpc_td3_workflow import (
    PretrainingRunConfig,
    run_of_mpc_pretraining,
)


DEFAULT_MPC_SAMPLES = 4_900_000
DEFAULT_STEADY_SAMPLES = 100_000
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_ACTOR_EPOCHS = 1000
DEFAULT_CRITIC_EPOCHS = 500
DEFAULT_PRETRAIN_BATCH_SIZE = 8192

DEFAULT_ACTOR_LAYER_SIZES = [512, 512, 512, 512, 512]
DEFAULT_CRITIC_LAYER_SIZES = [512, 512, 512, 512, 512]


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
        description="Pretrain and save a TD3 agent from offset-free MPC expert samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mpc-samples",
        type=int,
        default=DEFAULT_MPC_SAMPLES,
        help="Number of broad synthetic OF-MPC expert samples.",
    )
    parser.add_argument(
        "--steady-samples",
        type=int,
        default=DEFAULT_STEADY_SAMPLES,
        help="Number of near-steady synthetic expert samples.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of expert samples solved per replay-buffer chunk.",
    )
    parser.add_argument(
        "--actor-epochs",
        type=int,
        default=DEFAULT_ACTOR_EPOCHS,
        help="Behavioral-cloning actor pretraining epochs.",
    )
    parser.add_argument(
        "--critic-epochs",
        type=int,
        default=DEFAULT_CRITIC_EPOCHS,
        help="Critic TD warm-up epochs after actor cloning.",
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=DEFAULT_PRETRAIN_BATCH_SIZE,
        help="DataLoader batch size for actor cloning and critic warm-up.",
    )
    parser.add_argument(
        "--actor-layers",
        type=parse_layer_sizes,
        default=",".join(str(value) for value in DEFAULT_ACTOR_LAYER_SIZES),
        help="Comma-separated actor hidden layer sizes.",
    )
    parser.add_argument(
        "--critic-layers",
        type=parse_layer_sizes,
        default=",".join(str(value) for value in DEFAULT_CRITIC_LAYER_SIZES),
        help="Comma-separated critic hidden layer sizes.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use auto to select cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join("results", "PretrainOFMPC"),
        help="Artifact root. Relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainingRunConfig(
        mpc_samples=int(args.mpc_samples),
        steady_samples=int(args.steady_samples),
        chunk_size=int(args.chunk_size),
        actor_epochs=int(args.actor_epochs),
        critic_epochs=int(args.critic_epochs),
        pretrain_batch_size=int(args.pretrain_batch_size),
        actor_layer_sizes=tuple(args.actor_layers),
        critic_layer_sizes=tuple(args.critic_layers),
        seed=int(args.seed),
        device_requested=str(args.device),
        output_root=str(args.output_root),
    )

    print("Starting OF-MPC TD3 pretraining.")
    print(f"MPC samples: {config.mpc_samples}")
    print(f"Near-steady samples: {config.steady_samples}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Actor epochs: {config.actor_epochs}")
    print(f"Critic epochs: {config.critic_epochs}")
    print(f"Pretrain batch size: {config.pretrain_batch_size}")
    print(f"Actor layers: {list(config.actor_layer_sizes)}")
    print(f"Critic layers: {list(config.critic_layer_sizes)}")

    result = run_of_mpc_pretraining(config)
    print("OF-MPC TD3 pretraining complete.")
    print(f"Run directory: {result['run_dir']}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Config: {result['config_path']}")
    print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
