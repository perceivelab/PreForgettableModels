import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def kl_divergence_with_uniform(logits: torch.Tensor) -> torch.Tensor:
    batch_size, num_classes = logits.size()
    uniform_dist = torch.full((batch_size, num_classes), 1 / num_classes).to(
        logits.device
    )
    return F.kl_div(
        F.log_softmax(logits, dim=-1), uniform_dist, reduction="batchmean"
    )


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, filename: str
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    inference_prompt: str = "mixed_subset",
    forget_list: Optional[Sequence[int]] = None,
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> float:
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    iterator = data_loader
    if show_progress:
        iterator = tqdm(
            data_loader, desc=desc or f"Testing with {inference_prompt}"
        )

    with torch.no_grad():
        for pixel_values, labels in iterator:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            logits = model(
                pixel_values,
                inference_prompt=inference_prompt,
                label=labels,
                forget_list=forget_list,
            )
            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0
    return correct / total
