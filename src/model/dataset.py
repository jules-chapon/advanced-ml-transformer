"""Class for translation datasets."""

import numpy as np
import torch
import typing

from typing import Any, Dict, List, Tuple

from src.configs import constants, names

from torch.utils.data import DataLoader, Dataset


_TranslationDataset = typing.TypeVar(
    name="_TranslationDataset", bound="TranslationDataset"
)


class TranslationDataset(Dataset):
    """
    Dataset for translation tasks.

    Args:
        src_tokens (List[List[int]]): List of source token sequences.
        tgt_tokens (List[List[int]]): List of target token sequences.
        params (Dict[str, Any]): Dictionary of parameters.

    Methods:
        __len__(self) -> int: Returns the total number of samples in the dataset.
        __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]: Returns the source, target input, and target output tensors for a given index.
        get_dataloader(self, shuffle: bool = False) -> DataLoader: Returns a DataLoader for the dataset.
    """

    def __init__(
        self: _TranslationDataset,
        src_tokens: List[List[int]],
        tgt_tokens: List[List[int]],
        params: Dict[str, Any],
    ):
        """
        Initialize class instance.

        Args:
            self (_TranslationDataset): Class instance.
            src_tokens (List[List[int]]): Tokens for the source language.
            tgt_tokens (List[List[int]]): Tokens for the target language.
            params (Dict[str, Any]): Parameters of the model.
        """
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens
        self.params = params

    def __len__(self: _TranslationDataset) -> int:
        """
        Defines what returns len(self).

        Args:
            self (_TranslationDataset): Class instance.

        Returns:
            int: Number of rows in the dataset.
        """
        return len(self.src_tokens)

    def __getitem__(
        self: _TranslationDataset, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Defines what returns self[index].

        Args:
            self (_TranslationDataset): Class index.
            index (int): Row index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Source input, target input, target output.
        """
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

    def get_dataloader(self: _TranslationDataset, shuffle: bool = False) -> DataLoader:
        """
        Get a dataloader from the dataset.

        Args:
            self (_TranslationDataset): Class instance.
            shuffle (bool, optional): Whether to shuffle the dataset or not. Defaults to False.

        Returns:
            DataLoader: Dataloader instance.
        """
        return DataLoader(
            dataset=self,
            batch_size=self.params[names.BATCH_SIZE],
            shuffle=shuffle,
            num_workers=self.params[names.NUM_WORKERS],
        )
