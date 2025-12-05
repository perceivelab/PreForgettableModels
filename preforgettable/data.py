from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm

import medmnist
from medmnist import INFO
from transformers import ViTImageProcessor

from preforgettable.configs import DEFAULT_MODEL_NAME

VIT_IMAGE_PROCESSOR = ViTImageProcessor.from_pretrained(DEFAULT_MODEL_NAME)
VIT_IMAGE_MEAN = VIT_IMAGE_PROCESSOR.image_mean
VIT_IMAGE_STD = VIT_IMAGE_PROCESSOR.image_std
VIT_IMAGE_SIZE = VIT_IMAGE_PROCESSOR.size.get("height", 224)


@dataclass(frozen=True)
class MedMNISTSpec:
    num_classes: int
    train_split: str
    repeat_channels: bool


MEDMNIST_SPECS = {
    "bloodmnist": MedMNISTSpec(
        num_classes=8, train_split="train", repeat_channels=False
    ),
    "dermamnist": MedMNISTSpec(
        num_classes=7, train_split="train", repeat_channels=False
    ),
    "octmnist": MedMNISTSpec(
        num_classes=4,
        train_split="val",  # training split is too large
        repeat_channels=True,
    ),
    "organsmnist": MedMNISTSpec(
        num_classes=11, train_split="train", repeat_channels=True
    ),
    "organcmnist": MedMNISTSpec(
        num_classes=11, train_split="train", repeat_channels=True
    ),
}


class ConvertLabel:
    def __call__(self, label):
        label_tensor = torch.as_tensor(label)
        return label_tensor.squeeze().long()


def _build_medmnist_transform(repeat_channels: bool) -> transforms.Compose:
    ops = [transforms.ToTensor()]
    if repeat_channels:
        ops.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    ops.append(transforms.Normalize(mean=VIT_IMAGE_MEAN, std=VIT_IMAGE_STD))
    return transforms.Compose(ops)


def load_medmnist_datasets(
    dataset_name: str,
    image_size: int = VIT_IMAGE_SIZE,
    download: bool = True,
) -> Tuple[data.Dataset, data.Dataset, MedMNISTSpec]:
    if dataset_name not in MEDMNIST_SPECS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Available: {sorted(MEDMNIST_SPECS)}"
        )

    spec = MEDMNIST_SPECS[dataset_name]
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    transform = _build_medmnist_transform(spec.repeat_channels)
    label_transform = ConvertLabel()

    train_dataset = DataClass(
        split=spec.train_split,
        transform=transform,
        target_transform=label_transform,
        download=download,
        size=image_size,
    )

    test_dataset = DataClass(
        split="test",
        transform=transform,
        target_transform=label_transform,
        download=download,
        size=image_size,
    )

    return train_dataset, test_dataset, spec


def build_class_index_map(dataset: data.Dataset) -> Dict[int, List[int]]:
    index_map: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            class_id = int(label.item())
        else:
            class_id = int(label)
        index_map[class_id].append(idx)
    return index_map


def build_subset_loader_by_class(
    dataset: data.Dataset,
    class_ids: Sequence[int],
    class_index_map: Dict[int, List[int]],
    batch_size: int,
    shuffle: bool = False,
) -> data.DataLoader:
    selected_indices: List[int] = []
    for class_id in class_ids:
        selected_indices.extend(class_index_map.get(class_id, []))

    subset = data.Subset(dataset, selected_indices)
    return data.DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def build_medmnist_dataloaders(
    dataset_name: str,
    batch_size: int,
    image_size: int = VIT_IMAGE_SIZE,
    download: bool = True,
    shuffle_train: bool = True,
) -> Tuple[data.DataLoader, data.DataLoader, MedMNISTSpec]:
    train_dataset, test_dataset, spec = load_medmnist_datasets(
        dataset_name, image_size=image_size, download=download
    )

    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle_train
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, spec


def test_model_best_prompt(
    model, test_loader, device, *, show_progress: bool = True
):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    num_groups = getattr(model, "num_groups", getattr(model, "num_classes", 0))
    if num_groups <= 0:
        raise ValueError("Model must expose num_groups or num_classes > 0")

    iterator = test_loader
    if show_progress:
        iterator = tqdm(test_loader, desc="Testing best prompt")

    with torch.no_grad():
        for pixel_values, labels in iterator:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            logits_per_prompt = [
                model(pixel_values, inference_prompt=f"class_{j}")
                for j in range(num_groups)
            ]

            logits = torch.stack(logits_per_prompt, dim=0)
            logits = logits.max(0).values
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy
