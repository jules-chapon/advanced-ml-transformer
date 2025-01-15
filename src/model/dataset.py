"""Class for translation datasets"""

import numpy as np
import torch
import typing

from typing import Any

from src.configs import constants, names

from torch.utils.data import DataLoader, Dataset


_TranslationDataset = typing.TypeVar(
    name="_TranslationDataset", bound="TranslationDataset"
)


class TranslationDataset(Dataset):
    def __init__(
        self: _TranslationDataset,
        src_tokens: list[list[int]],
        tgt_tokens: list[list[int]],
        params: dict[str, Any],
    ):
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.params = params

    def __len__(self: _TranslationDataset) -> int:
        return len(self.src_tokens)

    def __getitem__(
        self: _TranslationDataset, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_data = self.src_tokens[index]
        tgt_data = self.tgt_tokens[index]
        if len(src_data) >= self.params[names.MAX_LENGTH_SRC]:
            src_data = src_data[: self.params[names.MAX_LENGTH_SRC]]
        else:
            src_data += [constants.PAD_TOKEN_ID] * (
                self.params[names.MAX_LENGTH_SRC] - len(src_data)
            )
        tgt_data = [constants.BOS_TOKEN_ID] + tgt_data + [constants.EOS_TOKEN_ID]
        if len(tgt_data) < self.params[names.MAX_CONTEXT_TGT] + 1:
            tgt_data += [constants.PAD_TOKEN_ID] * (
                self.params[names.MAX_CONTEXT_TGT] + 1 - len(tgt_data)
            )
        else:
            idx = np.random.randint(
                low=0, high=len(tgt_data) - self.params[names.MAX_CONTEXT_TGT]
            )
            tgt_data = tgt_data[idx : idx + self.params[names.MAX_CONTEXT_TGT] + 1]
        return (
            torch.tensor(src_data),
            torch.tensor(tgt_data[:-1]),
            torch.tensor(tgt_data[1:]),
        )

    def get_dataloader(self: _TranslationDataset, shuffle: bool = False) -> None:
        return DataLoader(
            dataset=self,
            batch_size=self.params[names.BATCH_SIZE],
            shuffle=shuffle,
            num_workers=self.params[names.NUM_WORKERS],
        )
