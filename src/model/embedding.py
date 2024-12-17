"""Embedding functions"""

import abc
import typing
import numpy as np
import pandas as pd
import torch

from src.configs import ml_config, names

_Embedding = typing.TypeVar("_Embedding", bound="Embedding")

class Embedding(abc.ABC):

    def __init__(self: _Embedding, id_experiment: int | None) -> None:
        """
        Initialize class instance.

        Args:
            self (_Embedding): Class instance.
            id_experiment (int | None): ID of experiment.

        Returns:
            _Embedding: Class instance.
        """
        self.id_experiment = id_experiment

    @abc.abstractmethod
    def tokenize_text(self: _Embedding,  text: str) -> list[str]:
        """
        Transform text into tokens of text.

        Args:
            self (_Embedding): Class instance.
            text (str): Text input.

        Returns:
            list[str]: List of tokens.
        """
        pass

    def detokenize_text(self: _Embedding, tokens: list[str]) -> str:
        """
        Transform tokens into text.

        Args:
            self (_Embedding): Class instance.
            tokens (list[str]): List of tokens.

        Returns:
            str: Text output.
        """
        return "".join(tokens)

    def create_vocabulary(self: _Embedding,  text: str) -> list[str]:
        """
        Create vocabulary from text.

        Args:
            self (_Embedding): Class instance.
            text (str): Complete text dataset.

        Returns:
            list[str]: List of whole vocabulary.
        """
        vocab = sorted(list(set(self.tokenize_text(text=text))))
        return vocab

    @abc.abstractmethod
    def create_embedding(self: _Embedding, vocabulary: list[str]) -> dict[str, int]:
        """
        _summary_

        Args:
            self (_Embedding): Class instance.
            vocabulary (list[str]): List of vocabulary.

        Returns:
            dict[str, int]: Dictionary mapping vocabulary to embedding.
        """
        pass

    def encode(self: _Embedding, text: str, dict_encoding: dict[str, int]) -> torch.Tensor:
        """
        Encode a text into a tensor of embeddings.
        Each embedding corresponds to a token of the text.

        Args:
            self (_Embedding): Class instance.
            text (str): Text input.
            dict_encoding (dict[str, int]): Dictionary mapping tokens to embeddings.

        Returns:
            torch.Tensor: Tensor of embeddings.
        """
        tokens = self.tokenize_text(text=text)
        embeddings = torch.zeros(len(tokens), dtype=torch.int)
        for i, token in enumerate(tokens):
            if token in dict_encoding:
                embeddings[i] = dict_encoding[token]
            else:
                embeddings[i] = ml_config.EXPERIMENTS_CONFIGS[self.id_experiment][names.DEFAULT_UNKNOWN_TOKEN]
        return embeddings

    def decode(self: _Embedding, embeddings: torch.Tensor, dict_encoding: dict[str, int]) -> str:
        """
        Decode a tensor of embeddings into a text.

        Args:
            self (_Embedding): Class instance.
            embeddings (torch.Tensor): Tensor of embeddings.
            dict_encoding (dict[str, int]): Dictionary mapping tokens to embeddings.

        Returns:
            str: Text output.
        """
        dict_decoding = {embedding: vocab for vocab, embedding in dict_encoding.items()}
        list_tokens = []
        for embedding in embeddings:
            if embedding.item() in dict_decoding:
                list_tokens.append(dict_decoding[embedding.item()])
            else:
                list_tokens.append(ml_config.EXPERIMENTS_CONFIGS[self.id_experiment][names.DEFAULT_UNKNOWN_EMBEDDING])
        return self.detokenize_text(tokens=list_tokens)


_CharacterEmbedding = typing.TypeVar("_CharacterEmbedding", bound="CharacterEmbedding")

class CharacterEmbedding(Embedding):


    def __init__(self: _CharacterEmbedding, id_experiment: int) -> _CharacterEmbedding:
        """
        Initialize class instance.

        Args:
            self (_CharacterEmbedding): Class instance.
            id_experiment (int | None): ID of experiment.

        Returns:
            _CharacterEmbedding: Class instance.
        """
        super().__init__(id_experiment)

    def tokenize_text(self: _CharacterEmbedding, text: str) -> list[str]:
        """
        Transform text into tokens of text.

        Args:
            self (_CharacterEmbedding): Class instance.
            text (str): Text input.

        Returns:
            list[str]: List of tokens.
        """
        return [char for char in text]
    
    def create_embedding(self: _CharacterEmbedding, vocabulary: list[str]) -> dict[str, int]:
        """
        _summary_

        Args:
            self (_CharacterEmbedding): Class instance.
            vocabulary (list[str]): List of vocabulary.

        Returns:
            dict[str, int]: Dictionary mapping vocabulary to embedding.
        """
        return {char: i for i, char in enumerate(vocabulary)}



test_text = "Projet de ML en légende"