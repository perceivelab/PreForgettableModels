from enum import Enum
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from transformers import ViTModel, ASTModel
from peft import get_peft_model, LoraConfig


class PromptMode(str, Enum):
    NONE = "none"
    FULL_BANK = "full_bank"
    FULL_BANK_SHUFFLED = "full_bank_shuffle"
    CLASS_PROMPT = "class_prompt"
    MATCH_ONLY = "match_only"
    RANDOM_WRONG = "random_wrong"
    ALL_WRONG = "all_wrong"
    RANDOM_WRONG_SHUFFLED = "random_wrong_shuffle"
    WRONG_SUBSET = "wrong_subset"
    MIXED_SUBSET = "mixed_subset"

    @classmethod
    def parse(cls, value: str) -> Tuple["PromptMode", Optional[int]]:
        if isinstance(value, cls):
            return value, None

        if not isinstance(value, str):
            raise TypeError("Inference prompt must be a string or PromptMode")

        normalized = cls._normalize_prompt_value(value)

        if normalized.startswith("class_"):
            class_id = cls._extract_class_identifier(value, normalized)
            return cls.CLASS_PROMPT, class_id

        try:
            return cls(normalized), None
        except ValueError as exc:
            raise ValueError(f"Unsupported inference prompt: {value}") from exc

    @staticmethod
    def _normalize_prompt_value(value: str) -> str:
        return value.strip().lower().replace("-", "_")

    @staticmethod
    def _extract_class_identifier(value: str, normalized: str) -> int:
        try:
            return int(normalized.split("_")[-1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid class prompt identifier: {value}"
            ) from exc


class PreForgettableModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        data_type: str = "image",
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        if data_type == "image":
            backbone = ViTModel.from_pretrained(
                model_name, add_pooling_layer=False
            )
        elif data_type == "audio":
            backbone = ASTModel.from_pretrained(
                model_name, add_pooling_layer=False
            )
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        self.backbone = get_peft_model(
            backbone, self._build_lora_config(lora_config)
        )
        self.hidden_size = self.backbone.config.hidden_size

        self.prompt_dict = nn.ParameterDict(
            {
                str(i): nn.Parameter(torch.randn(1, self.hidden_size))
                for i in range(self.num_classes)
            }
        )

        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def _build_lora_config(
        self, override_config: Optional[LoraConfig]
    ) -> LoraConfig:
        if override_config is not None:
            return override_config
        return LoraConfig(
            inference_mode=False,
            r=4,
            lora_alpha=4,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "value"],
        )

    def _normalize_labels(
        self, label: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if label is None:
            return None
        if isinstance(label, torch.Tensor):
            if label.dim() > 1:
                label = torch.argmax(label, dim=1)
            return label.long()
        raise TypeError(
            "Labels must be tensors when required by the prompt mode"
        )

    def _build_prompt_bank(
        self, batch_size: int, forget_list: Optional[Sequence[int]]
    ) -> Dict[int, torch.Tensor]:
        excluded = set(forget_list or [])
        return {
            class_idx: self.prompt_dict[str(class_idx)]
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            for class_idx in range(self.num_classes)
            if class_idx not in excluded
        }

    def _resolve_prompts(
        self,
        prompt_bank: Dict[int, torch.Tensor],
        mode: PromptMode,
        class_id: Optional[int],
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not prompt_bank or mode == PromptMode.NONE:
            return None

        if mode in (
            PromptMode.FULL_BANK,
            PromptMode.FULL_BANK_SHUFFLED,
        ):
            return self._concatenate_prompt_bank(
                prompt_bank, shuffle=mode == PromptMode.FULL_BANK_SHUFFLED
            )

        if mode == PromptMode.CLASS_PROMPT:
            return self._select_class_prompt(prompt_bank, class_id)

        self._ensure_labels_available(labels, mode)
        return self._select_label_conditioned_prompts(
            prompt_bank, labels, mode
        )

    def _select_class_prompt(
        self,
        prompt_bank: Dict[int, torch.Tensor],
        class_id: Optional[int],
    ) -> Optional[torch.Tensor]:
        if class_id is None:
            raise ValueError(
                "Class prompt requires an identifier, e.g. 'class_3'"
            )
        return prompt_bank.get(class_id)

    def _concatenate_prompt_bank(
        self, prompt_bank: Dict[int, torch.Tensor], shuffle: bool = False
    ) -> Optional[torch.Tensor]:
        ordered = [prompt_bank[g] for g in sorted(prompt_bank.keys())]
        if not ordered:
            return None

        concatenated = torch.cat(ordered, dim=1)
        if not shuffle:
            return concatenated
        return self._shuffle_prompt_tokens(concatenated)

    def _shuffle_prompt_tokens(self, prompts: torch.Tensor) -> torch.Tensor:
        batch_size, total_tokens, feat_dim = prompts.shape
        perm = torch.stack(
            [
                torch.randperm(total_tokens, device=prompts.device)
                for _ in range(batch_size)
            ]
        )
        perm = perm.unsqueeze(-1).expand(-1, -1, feat_dim)
        return torch.gather(prompts, 1, perm)

    def _ensure_labels_available(
        self, labels: Optional[torch.Tensor], mode: PromptMode
    ) -> None:
        if labels is None:
            raise ValueError(
                f"Inference prompt '{mode.value}' requires labels"
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        inference_prompt: Union[str, PromptMode] = "full_bank",
        forget_list: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        embeddings = self.backbone.embeddings(pixel_values)
        batch_size = embeddings.size(0)
        labels = self._normalize_labels(label)
        prompt_bank = self._build_prompt_bank(batch_size, forget_list)

        mode, class_id = PromptMode.parse(inference_prompt)
        prompts = self._resolve_prompts(prompt_bank, mode, class_id, labels)

        extended_embeddings = embeddings
        if prompts is not None:
            extended_embeddings = torch.cat([prompts, embeddings], dim=1)

        outputs = self.backbone.encoder(extended_embeddings)
        cls_representation = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_representation)
        return logits

    def _select_label_conditioned_prompts(
        self,
        prompt_bank: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        mode: PromptMode,
    ) -> Optional[torch.Tensor]:
        batch_prompts = []
        available_class_ids = tuple(sorted(prompt_bank.keys()))
        if not available_class_ids:
            return None

        subset_size = torch.randint(
            0, self.num_classes, (1,), device=labels.device
        ).item()
        insert_position = torch.randint(
            0, subset_size + 1, (1,), device=labels.device
        ).item()

        for sample_idx in range(labels.size(0)):
            sample_prompt = self._resolve_sample_prompt(
                prompt_bank,
                available_class_ids,
                labels[sample_idx].item(),
                sample_idx,
                mode,
                subset_size,
                insert_position,
            )
            if sample_prompt is None:
                return None
            batch_prompts.append(sample_prompt)

        if not batch_prompts:
            return None
        return torch.stack(batch_prompts)

    def _resolve_sample_prompt(
        self,
        prompt_bank: Dict[int, torch.Tensor],
        available_class_ids: Tuple[int, ...],
        correct_class: int,
        sample_idx: int,
        mode: PromptMode,
        subset_size: int,
        insert_position: int,
    ) -> Optional[torch.Tensor]:
        correct_prompt = prompt_bank.get(correct_class)
        candidate_classes = [
            g for g in available_class_ids if g != correct_class
        ]

        shuffled_candidates = candidate_classes
        if candidate_classes and mode in (
            PromptMode.RANDOM_WRONG,
            PromptMode.RANDOM_WRONG_SHUFFLED,
            PromptMode.MIXED_SUBSET,
            PromptMode.WRONG_SUBSET,
        ):
            shuffled_candidates = candidate_classes.copy()
            np.random.shuffle(shuffled_candidates)

        if mode == PromptMode.MATCH_ONLY:
            return self._slice_prompt(correct_prompt, sample_idx)

        if mode in (
            PromptMode.RANDOM_WRONG,
            PromptMode.RANDOM_WRONG_SHUFFLED,
        ):
            if not candidate_classes:
                return None
            chosen_group = shuffled_candidates[0]
            return prompt_bank[chosen_group][sample_idx]

        if mode == PromptMode.ALL_WRONG:
            if not candidate_classes:
                return None
            prompt_list = [
                prompt_bank[g][sample_idx] for g in candidate_classes
            ]
            return torch.cat(prompt_list, dim=0)

        if mode == PromptMode.MIXED_SUBSET:
            if correct_prompt is None:
                return None
            selected = shuffled_candidates[:subset_size]
            prompt_list = [prompt_bank[g][sample_idx] for g in selected]
            prompt_list.insert(insert_position, correct_prompt[sample_idx])
            return torch.cat(prompt_list, dim=0)

        if mode == PromptMode.WRONG_SUBSET:
            if subset_size == 0 or not candidate_classes:
                return None
            selected = shuffled_candidates[:subset_size]
            prompt_list = [prompt_bank[g][sample_idx] for g in selected]
            return torch.cat(prompt_list, dim=0)

        raise ValueError(f"Unsupported inference prompt: {mode}")

    @staticmethod
    def _slice_prompt(
        prompt: Optional[torch.Tensor], sample_idx: int
    ) -> Optional[torch.Tensor]:
        if prompt is None:
            return None
        return prompt[sample_idx]
