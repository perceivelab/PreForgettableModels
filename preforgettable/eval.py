import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Dict

from cli import build_base_parser
from preforgettable import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    PreForgettableModel,
    build_class_index_map,
    build_subset_loader_by_class,
    configure_seed,
    evaluate_model,
    get_device,
    load_checkpoint,
    load_medmnist_datasets,
)


def register_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate forgetting",
    )
    build_base_parser(
        "Evaluation script for Conditional Prompt Tuning ViT",
        parser,
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for subset evaluation",
    )
    parser.set_defaults(func=run)


@dataclass
class EvalConfig:
    dataset: str
    data_type: str
    seed: int
    save_dir: str
    ckpt: str
    batch_size: int
    gpu_id: int


def evaluate_subset_accuracy(
    model,
    dataset,
    class_ids,
    class_index_map,
    batch_size,
    device,
    forget_list,
    *,
    build_subset_loader_fn,
    evaluate_model_fn,
) -> float:
    loader = build_subset_loader_fn(
        dataset,
        class_ids,
        class_index_map,
        batch_size,
    )
    return evaluate_model_fn(
        model,
        loader,
        device,
        inference_prompt="full_bank",
        forget_list=forget_list,
        show_progress=False,
    )


def evaluate_forgetting_for_k(
    model,
    test_dataset,
    class_index_map,
    num_classes,
    k_forget,
    batch_size,
    device,
    *,
    build_subset_loader_fn,
    evaluate_model_fn,
    tqdm_module,
    np_module,
) -> Dict[str, float]:
    combo_count = math.comb(num_classes, k_forget)
    combinations_iter = itertools.combinations(range(num_classes), k_forget)
    forget_accs = []
    retain_accs = []

    progress = tqdm_module(
        combinations_iter,
        total=combo_count,
        desc=f"k={k_forget}",
        dynamic_ncols=True,
        unit="combo",
        leave=False,
    )
    all_classes = tuple(range(num_classes))

    for forget in progress:
        retain = [cls for cls in all_classes if cls not in forget]
        retain_acc = evaluate_subset_accuracy(
            model,
            test_dataset,
            retain,
            class_index_map,
            batch_size,
            device,
            forget_list=forget,
            build_subset_loader_fn=build_subset_loader_fn,
            evaluate_model_fn=evaluate_model_fn,
        )
        retain_accs.append(retain_acc)

        forget_acc = evaluate_subset_accuracy(
            model,
            test_dataset,
            forget,
            class_index_map,
            batch_size,
            device,
            forget_list=forget,
            build_subset_loader_fn=build_subset_loader_fn,
            evaluate_model_fn=evaluate_model_fn,
        )
        forget_accs.append(forget_acc)

        progress.set_postfix(
            retain=f"{np_module.mean(retain_accs):.4f}",
            forget=f"{np_module.mean(forget_accs):.4f}",
        )

    return {
        "forget_mean": float(np_module.mean(forget_accs)),
        "forget_std": float(np_module.std(forget_accs)),
        "retain_mean": float(np_module.mean(retain_accs)),
        "retain_std": float(np_module.std(retain_accs)),
    }


def run(args: argparse.Namespace) -> None:
    import numpy as np
    from tqdm import tqdm

    cfg = EvalConfig(
        **{field: getattr(args, field) for field in EvalConfig.__annotations__}
    )
    configure_seed(cfg.seed)
    device = get_device(cfg.gpu_id)
    os.makedirs(cfg.save_dir, exist_ok=True)

    train_dataset, test_dataset, dataset_spec = load_medmnist_datasets(
        cfg.dataset
    )
    num_classes = dataset_spec.num_classes
    k_targets = {
        min(num_classes, 1),
        min(num_classes, max(1, num_classes // 2)),
        min(num_classes, max(1, num_classes - 1)),
    }
    k_forget_list = sorted(k_targets)
    class_index_map = build_class_index_map(test_dataset)

    print(
        f"Dataset: {cfg.dataset} ({num_classes} classes) | "
        f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}"
    )
    print(f"Using device {device}")
    print(f"Evaluating forgetting for k values: {k_forget_list}")

    model = PreForgettableModel(
        DEFAULT_MODEL_NAME,
        num_classes,
        data_type=cfg.data_type,
    )
    load_checkpoint(model, cfg.ckpt)
    print(f"Checkpoint loaded from '{cfg.ckpt}'")

    exp_name = f"{cfg.dataset}_seed_{cfg.seed}"
    accs = {}
    accs_log_path = os.path.join(
        cfg.save_dir, f"{exp_name}_inference_accuracies.json"
    )

    print("-" * 85)
    for k_forget in k_forget_list:
        metrics = evaluate_forgetting_for_k(
            model,
            test_dataset,
            class_index_map,
            num_classes,
            k_forget,
            cfg.batch_size,
            device,
            build_subset_loader_fn=build_subset_loader_by_class,
            evaluate_model_fn=evaluate_model,
            tqdm_module=tqdm,
            np_module=np,
        )
        accs[str(k_forget)] = metrics

        print(
            f"k={k_forget} | "
            f"forget_mean={metrics['forget_mean']:.4f} | "
            f"forget_std={metrics['forget_std']:.4f} | "
            f"retain_mean={metrics['retain_mean']:.4f} | "
            f"retain_std={metrics['retain_std']:.4f}"
        )
        print("-" * 85)

    with open(accs_log_path, "w") as f:
        json.dump(accs, f, indent=4)

    print(f"Accuracies saved to {accs_log_path}")
