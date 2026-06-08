from __future__ import annotations

import argparse
import os

from utils.of_mpc_td3_workflow import (
    DEFAULT_ACTOR_EPOCHS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CRITIC_EPOCHS,
    DEFAULT_MPC_SAMPLES,
    DEFAULT_STEADY_SAMPLES,
    PretrainingRunConfig,
    run_of_mpc_pretraining,
)


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

    result = run_of_mpc_pretraining(config)
    print("OF-MPC TD3 pretraining complete.")
    print(f"Run directory: {result['run_dir']}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Config: {result['config_path']}")
    print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
