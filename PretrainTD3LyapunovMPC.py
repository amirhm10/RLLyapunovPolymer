from __future__ import annotations

import argparse
import os

from utils.lmpc_td3_workflow import (
    DEFAULT_ACTOR_EPOCHS,
    DEFAULT_ACTOR_LAYER_SIZES,
    DEFAULT_CANDIDATE_CHUNK_SIZE,
    DEFAULT_CRITIC_EPOCHS,
    DEFAULT_CRITIC_LAYER_SIZES,
    DEFAULT_LMPC_SAMPLES,
    DEFAULT_MAX_ATTEMPT_MULTIPLIER,
    DEFAULT_PRETRAIN_BATCH_SIZE,
    DEFAULT_STEADY_SAMPLES,
    DEFAULT_WORKER_BATCH_SIZE,
    LMPCPretrainingRunConfig,
    run_lmpc_pretraining,
)


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
        description="Pretrain and save a TD3 agent from Direct Lyapunov MPC expert samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lmpc-samples",
        type=int,
        default=DEFAULT_LMPC_SAMPLES,
        help="Number of accepted broad Direct LMPC expert samples.",
    )
    parser.add_argument(
        "--steady-samples",
        type=int,
        default=DEFAULT_STEADY_SAMPLES,
        help="Number of accepted near-steady Direct LMPC expert samples.",
    )
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=DEFAULT_CANDIDATE_CHUNK_SIZE,
        help="Number of candidate states drawn before progress reporting.",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=DEFAULT_WORKER_BATCH_SIZE,
        help="Accepted transitions flushed to replay storage per batch.",
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=float,
        default=DEFAULT_MAX_ATTEMPT_MULTIPLIER,
        help="Maximum candidate attempts per requested accepted label.",
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
        default=os.path.join("results", "PretrainLMPC"),
        help="Artifact root. Relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LMPCPretrainingRunConfig(
        lmpc_samples=int(args.lmpc_samples),
        steady_samples=int(args.steady_samples),
        candidate_chunk_size=int(args.candidate_chunk_size),
        worker_batch_size=int(args.worker_batch_size),
        max_attempt_multiplier=float(args.max_attempt_multiplier),
        actor_epochs=int(args.actor_epochs),
        critic_epochs=int(args.critic_epochs),
        pretrain_batch_size=int(args.pretrain_batch_size),
        actor_layer_sizes=tuple(args.actor_layers),
        critic_layer_sizes=tuple(args.critic_layers),
        seed=int(args.seed),
        device_requested=str(args.device),
        output_root=str(args.output_root),
    )

    print("Starting Direct LMPC TD3 pretraining.")
    print(f"Accepted broad LMPC samples: {config.lmpc_samples}")
    print(f"Accepted near-steady samples: {config.steady_samples}")
    print(f"Candidate chunk size: {config.candidate_chunk_size}")
    print(f"Worker batch size: {config.worker_batch_size}")
    print(f"Max attempt multiplier: {config.max_attempt_multiplier}")
    print(f"Actor epochs: {config.actor_epochs}")
    print(f"Critic epochs: {config.critic_epochs}")
    print(f"Pretrain batch size: {config.pretrain_batch_size}")
    print(f"Actor layers: {list(config.actor_layer_sizes)}")
    print(f"Critic layers: {list(config.critic_layer_sizes)}")

    result = run_lmpc_pretraining(config)
    print("Direct LMPC TD3 pretraining complete.")
    print(f"Run directory: {result['run_dir']}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Config: {result['config_path']}")
    print(f"Summary: {result['summary_path']}")
    print(f"Label diagnostics: {result['label_diagnostics_path']}")


if __name__ == "__main__":
    main()
