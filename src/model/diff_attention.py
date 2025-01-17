"""Class for attention"""

import math
import torch
import torch.nn as nn
import typing


_DiffHead = typing.TypeVar("_DiffHead", bound="DiffHead")


class DiffHead(nn.Module):
    """
    A single differential head of differential multi-head attention.

    Args:
        head_input_dimension (int): The dimension of the input to the head.
        head_size (int): The size of the head.
        head_output_dimension (int): The dimension of the output of the head.
        context_length (int): The length of the context.
        lambda_init (float): The initial value for the lambda parameter.
        mask (bool, optional): Whether to apply masking. Defaults to True.

    Methods:
        forward(q, k, v, pad_mask): Computes the attention scores and context vectors.
    """

    def __init__(
        self: _DiffHead,
        head_input_dimension: int,
        head_size: int,
        head_output_dimension: int,
        context_length: int,
        lambda_init: float,
        mask: bool = True,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_Head): Class instance.
            head_input_dimension (int): Dimension of the input.
            head_size (int): Size of the attention head.
            head_output_dimension (int): Dimension of the output.
            context_length (int): Length of the context.
            lambda_init (float): Initial value for lambda parameter.
            mask (bool, optional): Whether to use a mask or not. Defaults to True.
        """
        super().__init__()
        self.query = nn.Linear(head_input_dimension, 2 * head_size, bias=False)
        self.key = nn.Linear(head_input_dimension, 2 * head_size, bias=False)
        self.value = nn.Linear(head_input_dimension, head_output_dimension, bias=False)
        self.lambda_q1 = nn.Parameter(torch.randn(head_size))
        self.lambda_k1 = nn.Parameter(torch.randn(head_size))
        self.lambda_q2 = nn.Parameter(torch.randn(head_size))
        self.lambda_k2 = nn.Parameter(torch.randn(head_size))
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length), diagonal=1)
        )  # Non trainable parameters
        self.head_size = head_size
        self.lambda_init = lambda_init
        self.mask = mask

    def forward(
        self: _DiffHead,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method to get the output of the model.

        Args:
            self (_DiffHead): Class instance.
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            pad_mask (torch.Tensor): Padding mask.

        Returns:
            torch.Tensor: Output of the model.
        """
        B, T_q, C = q.shape
        _, T_k, _ = k.shape
        # B = batch_size
        # T_q = context_length of query
        # T_k = context_length of key and value
        # I = head_input_dimension
        # H = head_size
        # O = head_output_dimension
        Q = self.query(q)  # (B, T_q, 2H)
        K = self.key(k)  # (B, T_k, 2H)
        V = self.value(v)  # (B, T, O)
        Q1, Q2 = Q.chunk(2, dim=-1)  # (B, T_q, H), (B, T_q, H)
        K1, K2 = K.chunk(2, dim=-1)  # (B, T_k, H), (B, T_k, H)
        V = self.value(v)  # (B, T_k, O)
        attention_scores_1 = Q1 @ K1.transpose(
            1, 2
        )  # (B, T_q, 2H) @ (B, 2H, T_k) -> (B, T_q, T_k)
        attention_scores_2 = Q2 @ K2.transpose(
            1, 2
        )  # (B, T_q, 2H) @ (B, 2H, T_k) -> (B, T_q, T_k)
        if self.mask:
            masked_attention_scores_1 = attention_scores_1.masked_fill(
                self.tril[:T_q, :T_k] == 0, 1e-9
            )  # (B, T_q, T_k)
            masked_attention_scores_2 = attention_scores_2.masked_fill(
                self.tril[:T_q, :T_k] == 0, 1e-9
            )  # (B, T_q, T_k)
        else:
            masked_attention_scores_1 = attention_scores_1  # (B, T_q, T_k)
            masked_attention_scores_2 = attention_scores_2  # (B, T_q, T_k)
        masked_attention_scores_1 = masked_attention_scores_1.masked_fill(
            pad_mask.unsqueeze(2).expand(-1, -1, T_k) == 1, 1e-9
        )  # (B, T_q, T_k)
        masked_attention_scores_2 = masked_attention_scores_2.masked_fill(
            pad_mask.unsqueeze(2).expand(-1, -1, T_k) == 1, 1e-9
        )  # (B, T_q, T_k)
        lbd = (
            torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
            - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
            + self.lambda_init
        )  # (1)
        attention_weights = torch.softmax(
            masked_attention_scores_1 * self.head_size**-0.5, dim=-1
        ) - lbd * torch.softmax(
            masked_attention_scores_2 * self.head_size**-0.5, dim=-1
        )  # (B, T_q, T_k)
        context_vectors = (
            attention_weights @ V
        )  # (B, T_q, T_k) @ (B, T_k, O) -> (B, T_q, O)
        return context_vectors


_MultiDiffHeadAttention = typing.TypeVar(
    "_MultiDiffHeadAttention", bound="MultiDiffHeadAttention"
)


class MultiDiffHeadAttention(nn.Module):
    """
    Differential multi-head attention.

    Args:
        num_heads (int): Number of heads.
        embedding_dimension (int): Dimension of the embedding (=input).        head_size (int): The size of the head.
        head_output_dimension (int): The dimension of the output of the head.
        context_length (int): The length of the context.
        lambda_init (float): The initial value for the lambda parameter.
        mask (bool, optional): Whether to apply masking. Defaults to True.

    Methods:
        forward(q, k, v, pad_mask): Computes the attention scores and context vectors.
    """

    def __init__(
        self: _MultiDiffHeadAttention,
        num_heads: int,
        embedding_dimension: int,
        head_size: int,
        head_output_dimension: int,
        context_length: int,
        lambda_init: float,
        mask: bool = True,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_MultiDiffHeadAttention): Class instance.
            num_heads (int): Number of heads.
            embedding_dimension (int): Dimension of the embedding (=input).
            head_size (int): Size of the attention head.
            head_output_dimension (int): Dimension of the output.
            context_length (int): Length of the context.
            lambda_init (float): Initial value for the lambda parameter.
            mask (bool, optional): Whether to use a mask or not. Defaults to True.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                DiffHead(
                    head_input_dimension=embedding_dimension,
                    head_size=head_size,
                    head_output_dimension=2 * head_output_dimension,  # To respect paper
                    context_length=context_length,
                    lambda_init=lambda_init,
                    mask=mask,
                )
                for _ in range(num_heads // 2)  # To respect paper
            ]
        )
        self.proj = nn.Linear(num_heads * head_output_dimension, embedding_dimension)
        self.lambda_init = lambda_init

    def forward(
        self: _MultiDiffHeadAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor,
    ):
        """
        Forward method to get the output of the model.

        Args:
            self (_MultiDiffHead): Class instance.
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            pad_mask (torch.Tensor): Padding mask.

        Returns:
            torch.Tensor: Output of the model.
        """
        # q, k, v : (B, T, C)
        # B = batch_size
        # T = context_length
        # C = embedding_dimension
        out = torch.cat(
            [
                (1 - self.lambda_init) * h(q=q, k=k, v=v, pad_mask=pad_mask)
                for h in self.heads
            ],
            dim=-1,
        )
        out = self.proj(out)
        return out  # Output : (B, T, C)


_RMSNormLayer = typing.TypeVar("_RMSNormLayer", bound="RMSNormLayer")


class RMSNormLayer(nn.Module):
    """
    Normalization layer.

    Args:
        dimension (int): The dimension of the input and output tensors.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

    Methods:
        forward(x): Applies layer normalization to the input tensor.
    """

    def __init__(self: _RMSNormLayer, dimension: int, eps: float = 1e-6):
        """
        Initialize class instance.

        Args:
            self (_RMSNormLayer): Class instance.
            dimension (int): Dimension of the input tensor.
            eps (float, optional): To avoid dividing by 0. Defaults to 1e-6.
        """
        super().__init__()
        self.dimension = dimension
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dimension))

    def forward(self: _RMSNormLayer, x: torch.Tensor):
        """
        Forward method to get the output of the layer.

        Args:
            self (_RMSNormLayer): Class instance.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.scale * x_norm


_SwiGLU = typing.TypeVar("_SwiGLU", bound="SwiGLU")


class SwiGLU(nn.Module):
    """
    SwiGLU layer.

    Args:
        dimension (int): The dimension of the input tensor.

    Methods:
        forward(x): Applies a SwiGLU layer to the input tensor.
    """

    def __init__(self: _SwiGLU, dimension: int):
        """
        Initialize class instance.

        Args:
            self (_SwiGLU): Class instance.
            dimension (int): Dimension of the input tensor.
        """
        super(SwiGLU, self).__init__()
        self.W_G = nn.Linear(dimension, 3 * dimension)
        self.W1 = nn.Linear(dimension, 3 * dimension)
        self.W2 = nn.Linear(3 * dimension, dimension)

    def forward(self: _SwiGLU, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to compute the output of the layer.

        Args:
            self (_SwiGLU): Class instance.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # x : (B, T, I)
        # B = batch_size
        # T = context_length
        # I = input_dimension
        swish_activation = (
            x @ self.W_G.weight.T + self.W_G.bias
        )  # (B, T, I) @ (I, 3*I) -> (B, T, 3*I)
        swish_output = swish_activation * torch.sigmoid(
            swish_activation
        )  # (B, T, 3*I) * (B, T, 3*I) -> (B, T, 3*I)
        linear_output = (
            x @ self.W1.weight.T + self.W1.bias
        )  # (B, T, I) @ (I, 3*I) -> (B, T, 3*I)
        gated_output = (
            swish_output * linear_output
        )  # (B, T, 3*I) * (B, T, 3*I) -> (B, T, 3*I)
        output = (
            gated_output @ self.W2.weight.T + self.W2.bias
        )  # (B, T, 3*I) @ (3*I, I) -> (B, T, I)
        return output  # (B, T, I)


_PositionalEncoding = typing.TypeVar("_PositionalEncoding", bound="PositionalEncoding")


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding layer.

    Args:
        embedding_dimension (int): The dimension of the input and output tensors.
        context_length (int): The length of the context.

    Methods:
        forward(x): Applies positional encoding to the input tensor.
    """

    def __init__(
        self: _PositionalEncoding,
        embedding_dimension: int,
        context_length: int,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_PositionalEncoding): Class instance.
            embedding_dimension (int): Dimension of the embedding.
            context_length (int): Length of the context.
        """
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

    def forward(self: _PositionalEncoding, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to compute the output of the embedding.

        Args:
            self (_PositionalEncoding): Class instance.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Embedded tensor.
        """
        return self.pe[:, : x.size(1)]
