import argparse
from typing import Sequence

DATASET_CHOICES: Sequence[str] = (
    "bloodmnist",
    "dermamnist",
    "organsmnist",
    "organcmnist",
    "octmnist",
)


def build_base_parser(
    description: str,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description=description)
    else:
        parser.description = description
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_CHOICES,
        default="bloodmnist",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=("image", "audio"),
        default="image",
        help="Type of backbone to load",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints and logs",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-Forgettable Models CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Import inside the function to avoid circular imports with train/eval modules.
    from preforgettable.train import (
        register_subparser as register_train_subparser,
    )
    from preforgettable.eval import (
        register_subparser as register_eval_subparser,
    )

    register_train_subparser(subparsers)
    register_eval_subparser(subparsers)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return create_parser().parse_args(argv)
