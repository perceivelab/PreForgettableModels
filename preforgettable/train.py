import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict

from cli import build_base_parser
from preforgettable import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_NAME,
    INFERENCE_MODES,
    PreForgettableModel,
    build_medmnist_dataloaders,
    configure_seed,
    evaluate_model,
    get_device,
    kl_divergence_with_uniform,
    save_checkpoint,
    test_model_best_prompt,
)


def register_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "train",
        help="Train a Pre-Forgettable ViT",
    )
    build_base_parser("Training script for Pre-Forgettable ViT", parser)
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace checkpoint to fine-tune",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    parser.set_defaults(func=run)


@dataclass
class TrainingConfig:
    dataset: str
    data_type: str
    seed: int
    save_dir: str
    model_name: str
    batch_size: int
    epochs: int
    lr: float
    gpu_id: int


def train_model(
    model: PreForgettableModel,
    train_loader,
    optimizer,
    device,
    desc: str | None = None,
    progress_position: int = 1,
    *,
    nn_module,
    tqdm_module,
    kl_uniform: Callable,
) -> float:
    model = model.to(device)
    criterion = nn_module.CrossEntropyLoss()
    running_loss = 0.0
    num_batches = 0

    iterator = tqdm_module(
        train_loader,
        total=len(train_loader),
        desc=desc or "Training",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
        position=progress_position,
    )

    model.train()
    for pixel_values1, labels1 in iterator:
        pixel_values1, labels1 = (
            pixel_values1.to(device),
            labels1.to(device),
        )

        logits_both = model(
            pixel_values1, labels1, inference_prompt="mixed_subset"
        )
        loss_both = criterion(logits_both, labels1)

        logits_incorrect_prompt_sub = model(
            pixel_values1, inference_prompt="wrong_subset", label=labels1
        )
        loss_incorrect_prompt_sub = kl_uniform(logits_incorrect_prompt_sub)

        loss = loss_incorrect_prompt_sub + loss_both
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    if num_batches == 0:
        return 0.0
    return running_loss / num_batches


def evaluate_inference_modes(
    model,
    test_loader,
    device,
    progress_position: int = 2,
    *,
    tqdm_module,
    evaluate_model_fn: Callable,
    best_prompt_fn: Callable,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    mode_bar = tqdm_module(
        INFERENCE_MODES,
        desc="Eval prompts",
        leave=False,
        dynamic_ncols=True,
        unit="mode",
        position=progress_position,
    )
    for inf_type in mode_bar:
        if inf_type == "infer_best":
            acc = best_prompt_fn(
                model, test_loader, device, show_progress=False
            )
        else:
            acc = evaluate_model_fn(
                model,
                test_loader,
                device,
                inference_prompt=inf_type,
                desc=f"Testing ({inf_type})",
                show_progress=False,
            )

        results[inf_type] = acc
        mode_bar.set_postfix_str(f"{inf_type}={acc:.4f}")

    return results


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    cfg = TrainingConfig(
        **{
            field: getattr(args, field)
            for field in TrainingConfig.__annotations__
        }
    )
    seed = cfg.seed
    configure_seed(seed)

    device = get_device(cfg.gpu_id)
    os.makedirs(cfg.save_dir, exist_ok=True)

    train_loader, test_loader, dataset_spec = build_medmnist_dataloaders(
        cfg.dataset, cfg.batch_size
    )
    num_classes = dataset_spec.num_classes
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    print(
        f"Dataset: {cfg.dataset} ({num_classes} classes) | "
        f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}"
    )

    model = PreForgettableModel(
        cfg.model_name,
        num_classes,
        data_type=cfg.data_type,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )

    exp_name = f"{cfg.dataset}_seed_{seed}"
    accs = defaultdict(list)
    accs_log_path = os.path.join(cfg.save_dir, f"{exp_name}_accuracies.json")

    print(f"Using device {device}")
    print(
        f"Training for {cfg.epochs} epochs | batch size {cfg.batch_size} | lr {cfg.lr}."
    )
    print(f"Checkpoint saved after every epoch to '{cfg.save_dir}'.")

    epoch_bar = tqdm(
        range(1, cfg.epochs + 1),
        desc="Epochs",
        dynamic_ncols=True,
        unit="epoch",
        position=0,
    )
    for epoch in epoch_bar:
        avg_loss = train_model(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"Train {epoch}/{cfg.epochs}",
            progress_position=1,
            nn_module=nn,
            tqdm_module=tqdm,
            kl_uniform=kl_divergence_with_uniform,
        )

        checkpoint_path = os.path.join(cfg.save_dir, f"{exp_name}_{epoch}.pt")
        save_checkpoint(model, optimizer, checkpoint_path)

        mode_results = evaluate_inference_modes(
            model,
            test_loader,
            device,
            progress_position=2,
            tqdm_module=tqdm,
            evaluate_model_fn=evaluate_model,
            best_prompt_fn=test_model_best_prompt,
        )
        for inf_type, acc in mode_results.items():
            accs[inf_type].append(acc)

        epoch_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            mixed=f"{mode_results['mixed_subset']:.4f}",
            best=f"{mode_results['infer_best']:.4f}",
        )

        summary_modes = ("mixed_subset", "infer_best", "wrong_subset")
        summary = " | ".join(
            f"{mode}:{mode_results[mode]:.4f}" for mode in summary_modes
        )
        tqdm.write(f"[Epoch {epoch:02d}] loss={avg_loss:.4f} | {summary}")

    with open(accs_log_path, "w") as f:
        json.dump(accs, f, indent=4)

    print(f"Accuracies saved to {accs_log_path}")
