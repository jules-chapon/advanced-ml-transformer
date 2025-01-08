"""Class for attention"""

import math
import torch
import torch.nn as nn
import typing


_Head = typing.TypeVar("_Head", bound="Head")


class Head(nn.Module):
    def __init__(
        self: _Head,
        head_input_dimension: int,
        head_size: int,
        head_output_dimension: int,
        context_length: int,
        mask: bool = True,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(head_input_dimension, head_size, bias=False)
        self.query = nn.Linear(head_input_dimension, head_size, bias=False)
        self.value = nn.Linear(head_input_dimension, head_output_dimension, bias=False)
        # Some Pytorch way of defining a matrix without trainable parameters
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length), diagonal=1)
        )  # Not trainable parameters
        self.head_size = head_size
        self.mask = mask

    def forward(
        self: _Head, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = q.shape
        # B = batch_size
        # T = context_length
        # I = head_input_dimension
        # H = head_size
        # O = head_output_dimension
        K = self.key(k)  # (B, T, H)
        Q = self.query(q)  # (B, T, H)
        V = self.value(v)  # (B, T, O)
        attention_scores = Q @ K.transpose(1, 2)  # (B, T, H) @ (B, H, T) -> (B, T, T)
        if self.mask:
            masked_attention_scores = attention_scores.masked_fill(
                self.tril[:T, :T] == 0, float("-inf")
            )  # (B, T, T)
        else:
            masked_attention_scores = attention_scores  # (B, T, T)
        attention_weights = torch.softmax(
            masked_attention_scores * self.head_size**-0.5, dim=-1
        )  # (B, T, T)
        context_vectors = attention_weights @ V  # (B, T, T) @ (B, T, O) -> (B, T, O)
        return context_vectors


_MultiHeadAttention = typing.TypeVar("_MultiHeadAttention", bound="MultiHeadAttention")


class MultiHeadAttention(nn.Module):
    def __init__(
        self: _MultiHeadAttention,
        num_heads: int,
        embedding_dimension: int,
        head_size: int,
        head_output_dimension: int,
        context_length: int,
        mask: bool = True,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    head_input_dimension=embedding_dimension,
                    head_size=head_size,
                    head_output_dimension=head_output_dimension,
                    context_length=context_length,
                    mask=mask,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_heads * head_output_dimension, embedding_dimension)

    def forward(
        self: _MultiHeadAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        # q, k, v : (B, T, C)
        # B = batch_size
        # T = context_length
        # C = embedding_dimension
        out = torch.cat([h(q=q, k=k, v=v) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out  # Output : (B, T, C)


_NormLayer = typing.TypeVar("_NormLayer", bound="NormLayer")


class NormLayer(nn.Module):
    def __init__(self: _NormLayer, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps  # To avoid dividing by 0
        self.gamma = nn.Parameter(torch.ones(dimension))  # Parameter to train
        self.beta = nn.Parameter(torch.zeros(dimension))  # Parameter to train

    def forward(self: _NormLayer, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C)
        # B = batch_size
        # T = length
        # C = dimension
        mean = x.mean(dim=-1, keepdim=True)  # Mean : (B, T, C)
        std = x.std(dim=-1, keepdim=True)  # Standard deviation : (B, T, C)
        x_normalized = (x - mean) / (std + self.eps)  # Normalized tensor : (B, T, C)
        return self.gamma * x_normalized + self.beta  # Output : (B, T, C)


_FeedForward = typing.TypeVar("_FeedForward", bound="FeedForward")


class FeedForward(nn.Module):
    def __init__(
        self: _FeedForward,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self: _FeedForward, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, I)
        # B = batch_size
        # T = context_length
        # I = input_dimension
        # O = output_dimension
        return self.net(x)  # Output : (B, T, O)


_PositionalEncoding = typing.TypeVar("_PositionalEncoding", bound="PositionalEncoding")


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self: _PositionalEncoding,
        embedding_dimension: int,
        context_length: int,
    ) -> _PositionalEncoding:
        super().__init__()
        pe = torch.zeros(context_length, embedding_dimension)
        position = torch.arange(
            start=0, end=context_length, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dimension, 2).float()
            * -(math.log(10000.0) / embedding_dimension)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self: _PositionalEncoding, x: torch.Tensor):
        return self.pe[:, : x.size(1)]
