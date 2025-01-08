"""Class for translation datasets"""

import torch
import typing

from typing import Any

from src.configs import names

from torch.utils.data import DataLoader, Dataset


_TranslationDataset = typing.TypeVar(
    name="_TranslationDataset", bound="TranslationDataset"
)


class TranslationDataset(Dataset):
    def __init__(
        self: _TranslationDataset,
        src_input_tokens: list[list[int]],
        tgt_input_tokens: list[list[int]],
        tgt_output_tokens: list[int],
    ):
        self.src_input_tokens = src_input_tokens
        self.tgt_input_tokens = tgt_input_tokens
        self.tgt_output_tokens = tgt_output_tokens
        self.max_length = max(
            max(len(en), len(bn)) for en, bn in zip(src_input_tokens, tgt_input_tokens)
        )

    def __len__(self: _TranslationDataset) -> int:
        return len(self.src_input_tokens)

    def __getitem__(
        self: _TranslationDataset, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_input_data = self.src_input_tokens[index]
        tgt_input_data = self.tgt_input_tokens[index]
        tgt_output_data = self.tgt_output_tokens[index]
        return (
            torch.tensor(src_input_data),
            torch.tensor(tgt_input_data),
            torch.tensor(tgt_output_data),
        )

    def get_dataloader(
        self: _TranslationDataset, params: dict[str, Any], shuffle: bool = False
    ) -> None:
        return DataLoader(
            dataset=self, batch_size=params[names.BATCH_SIZE], shuffle=shuffle
        )
