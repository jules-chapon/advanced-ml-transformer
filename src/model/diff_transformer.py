"""DiffTransformer class"""

import pandas as pd
import time
import torch
import torch.nn.functional as F
import typing
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple

from src.configs import names, constants

from src.model.diff_attention import (
    MultiDiffHeadAttention,
    RMSNormLayer,
    SwiGLU,
    PositionalEncoding,
)

from src.libs.preprocessing import (
    clean_text,
    tokenize_sentence,
    from_tokens_to_text,
)

from src.libs.evaluation import calculate_metrics


_Encoder = typing.TypeVar(name="_Encoder", bound="Encoder")


class Encoder(torch.nn.Module):
    """
    Class for the Encoder part of the Differential Transformer.

    Methods:
        forward(x, pad_mask): Computes the output of the Encoder.
    """

    def __init__(
        self: _Encoder,
        num_heads: int,
        embedding_dimension: int,
        head_size: int,
        context_length: int,
        lambda_init: float,
        dropout: float,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_Encoder): Class instance.
            num_heads (int): Number of heads.
            embedding_dimension (int): Dimension of the embedding.
            head_size (int): Size of each single head.
            context_length (int): Lenght of the context.
            lambda_init (float): Initiial value of lambda.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attention = MultiDiffHeadAttention(
            num_heads=num_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=head_size,
            context_length=context_length,
            lambda_init=lambda_init,
            mask=False,
        )
        self.norm_layer_1 = RMSNormLayer(dimension=embedding_dimension)
        self.norm_layer_2 = RMSNormLayer(dimension=embedding_dimension)
        self.swish_glu = SwiGLU(dimension=embedding_dimension)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self: _Encoder,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method to get the output of the encoder.

        Args:
            self (_Encoder): Class instance.
            x (torch.Tensor): Input tensor.
            pad_mask (torch.Tensor): Padding mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.norm_layer_1(x)
        out = (
            self.dropout(self.self_attention(q=out, k=out, v=out, pad_mask=pad_mask))
            + out
        )
        out = self.dropout(self.swish_glu(self.norm_layer_2(out))) + out
        return out


_Decoder = typing.TypeVar(name="_Decoder", bound="Decoder")


class Decoder(torch.nn.Module):
    """
    Class for the Decoder part of the Differential Transformer.

    Methods:
        forward(x, encoder_output, pad_mask): Computes the output of the Decoder.
    """

    def __init__(
        self: _Decoder,
        num_heads: int,
        embedding_dimension: int,
        head_size: int,
        context_length: int,
        lambda_init: float,
        dropout: float,
    ) -> None:
        """
        Initialize class instance.

        Args:
            self (_Decoder): Class instance.
            num_heads (int): Number of heads.
            embedding_dimension (int): Dimension of the embedding.
            head_size (int): Size of each single head.
            context_length (int): Lenght of the context.
            lambda_init (float): Initiial value of lambda.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.self_attention = MultiDiffHeadAttention(
            num_heads=num_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=embedding_dimension,
            context_length=context_length,
            lambda_init=lambda_init,
            mask=True,
        )
        self.cross_attention = MultiDiffHeadAttention(
            num_heads=num_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=embedding_dimension,
            context_length=context_length,
            lambda_init=lambda_init,
            mask=False,
        )
        self.norm_layer_1 = RMSNormLayer(dimension=embedding_dimension)
        self.norm_layer_2 = RMSNormLayer(dimension=embedding_dimension)
        self.swish_glu = SwiGLU(dimension=embedding_dimension)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self: _Decoder,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method to get the output of the decoder.

        Args:
            self (_Decoder): Class instance.
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.
            pad_mask (torch.Tensor): Padding mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.norm_layer_1(x)
        out = (
            self.dropout(self.self_attention(q=out, k=out, v=out, pad_mask=pad_mask))
            + out
        )
        out = self.cross_attention(
            q=out, k=encoder_output, v=encoder_output, pad_mask=pad_mask
        )
        out = self.dropout(out) + out
        out = self.dropout(self.swish_glu(self.norm_layer_2(out))) + out
        return out


_DiffTransformer = typing.TypeVar(name="_DiffTransformer", bound="DiffTransformer")


class DiffTransformer(torch.nn.Module):
    """
    Class for Differential Transformer model.

    Methods:
        forward(src_input, tgt_input): get predictions of the model.
        get_padding_mask(input): define the âdding mask of an input.
        train_model(train_dataloader, valid_dataloader): train the model.
        translate(src_vocab, tgt_vocab_reversed, src_text): translate a sentence.
        evaluate(df_test, src_vocab, tgt_vocab_reversed): evaluate the model on the test set.
    """

    def __init__(self: _DiffTransformer, params: Dict[str, Any]) -> None:
        """
        Initialize class instance.

        Args:
            self (_DiffTransformer): Class instance.
            params (Dict[str, Any]): Parameters of the model.
        """
        super().__init__()
        self.params = params
        if (self.params[names.DEVICE] == "cuda") and (torch.cuda.is_available()):
            self.params[names.DEVICE] = "cuda"
        else:
            self.params[names.DEVICE] = "cpu"
        self.encoder_embedding = torch.nn.Embedding(
            self.params[names.SRC_VOCAB_SIZE], self.params[names.EMBEDDING_DIMENSION]
        )
        self.decoder_embedding = torch.nn.Embedding(
            self.params[names.TGT_VOCAB_SIZE], self.params[names.EMBEDDING_DIMENSION]
        )
        self.encoder_positional_encoding = PositionalEncoding(
            embedding_dimension=self.params[names.EMBEDDING_DIMENSION],
            context_length=self.params[names.MAX_LENGTH_SRC],
        )
        self.decoder_positional_encoding = PositionalEncoding(
            embedding_dimension=self.params[names.EMBEDDING_DIMENSION],
            context_length=self.params[names.MAX_CONTEXT_TGT],
        )
        self.encoder_layers = torch.nn.ModuleList(
            [
                Encoder(
                    num_heads=self.params[names.NB_HEADS],
                    embedding_dimension=self.params[names.EMBEDDING_DIMENSION],
                    head_size=self.params[names.HEAD_SIZE],
                    context_length=self.params[names.MAX_LENGTH_SRC],
                    lambda_init=self.params[names.LAMBDA_INIT],
                    dropout=self.params[names.DROPOUT],
                )
                for _ in range(self.params[names.NB_LAYERS])
            ]
        )
        self.decoder_layers = torch.nn.ModuleList(
            [
                Decoder(
                    num_heads=self.params[names.NB_HEADS],
                    embedding_dimension=self.params[names.EMBEDDING_DIMENSION],
                    head_size=self.params[names.HEAD_SIZE],
                    context_length=self.params[names.MAX_CONTEXT_TGT],
                    lambda_init=self.params[names.LAMBDA_INIT],
                    dropout=self.params[names.DROPOUT],
                )
                for _ in range(self.params[names.NB_LAYERS])
            ]
        )
        self.linear = torch.nn.Linear(
            self.params[names.EMBEDDING_DIMENSION],
            self.params[names.TGT_VOCAB_SIZE],
        )
        self.dropout = torch.nn.Dropout(self.params[names.DROPOUT])

    def forward(
        self: _DiffTransformer, src_input: torch.Tensor, tgt_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward method to get the output of the Differential Transformer.

        Args:
            self (_DiffTransformer): Class instance.
            src_input (torch.Tensor): Source input tensor.
            tgt_input (torch.Tensor): Target input tensor.

        Returns:
            torch.Tensor: Logits.
        """
        # Encoder
        src_pad_mask = self.get_padding_mask(src_input)
        src_embedded = self.encoder_embedding(src_input)
        encoder_output = self.dropout(
            src_embedded + self.encoder_positional_encoding(x=src_embedded)
        )
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(x=encoder_output, pad_mask=src_pad_mask)
        # Decoder
        tgt_pad_mask = self.get_padding_mask(input=tgt_input)
        tgt_embedded = self.decoder_embedding(tgt_input)
        decoder_output = self.dropout(
            tgt_embedded + self.decoder_positional_encoding(x=tgt_embedded)
        )
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                x=decoder_output, encoder_output=encoder_output, pad_mask=tgt_pad_mask
            )
        logits = self.linear(decoder_output)
        return logits

    def get_padding_mask(self: _DiffTransformer, input: torch.Tensor) -> torch.Tensor:
        """
        Define the padding mask.

        Args:
            self (_DiffTransformer): Class instance.
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Padding mask.
        """
        pad_mask = input == constants.PAD_TOKEN_ID
        return pad_mask

    def train_model(
        self: _DiffTransformer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> Tuple[List[int], List[int]]:
        """
        Train the model.

        Args:
            self (_DiffTransformer): Class instance.
            train_dataloader (DataLoader): Dataloader for training data.
            valid_dataloader (DataLoader): Dataloader for validation data.

        Returns:
            Tuple[List[int], List[int]]: Train loss history, validation loss history.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.params[names.LEARNING_RATE],
            betas=self.params[names.BETAS],
            eps=self.params[names.EPSILON],
        )
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.to(self.params[names.DEVICE])
        train_loss_history = []
        valid_loss_history = []
        start_training = time.time()
        for epoch in range(self.params[names.NB_EPOCHS]):
            # Training
            self.train()
            train_loss = 0.0
            for src, tgt_input, tgt_output in train_dataloader:
                src, tgt_input, tgt_output = (
                    src.to(self.params[names.DEVICE]),
                    tgt_input.to(self.params[names.DEVICE]),
                    tgt_output.to(self.params[names.DEVICE]),
                )
                optimizer.zero_grad()
                logits = self(src, tgt_input)
                B, T, _ = logits.shape
                loss = criterion(
                    logits.view(B * T, self.params[names.TGT_VOCAB_SIZE]),
                    tgt_output.view(B * T),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
                train_loss += loss.item()
            # Validation
            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for src, tgt_input, tgt_output in valid_dataloader:
                    src, tgt_input, tgt_output = (
                        src.to(self.params[names.DEVICE]),
                        tgt_input.to(self.params[names.DEVICE]),
                        tgt_output.to(self.params[names.DEVICE]),
                    )
                    logits = self(src, tgt_input)
                    B, T, _ = logits.shape
                    loss = criterion(
                        logits.reshape(B * T, self.params[names.TGT_VOCAB_SIZE]),
                        tgt_output.reshape(B * T),
                    )
                    valid_loss += loss.item()
            ###
            train_loss /= len(train_dataloader)
            valid_loss /= len(valid_dataloader)
            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            print(f"Epoch {epoch+1} / {self.params[names.NB_EPOCHS]} -------------")
            print(f"Train loss : {train_loss:.4f}. Valid loss : {valid_loss:.4f}.")
        print(
            f"Trained successfully. It took {time.time() - start_training:.2f} seconds. \n"
        )
        return train_loss_history, valid_loss_history

    def translate(
        self: _DiffTransformer,
        src_vocab: Dict[str, int],
        tgt_vocab_reversed: Dict[str, int],
        src_text: str,
    ) -> str:
        """
        Translate a sentence.

        Args:
            self (_DiffTransformer): Class instance.
            src_vocab (Dict[str, int]): Vocabulary of the source language.
            tgt_vocab_reversed (Dict[str, int]): Reversed vocabulary of the target language.
            src_text (str): Sentence to be translated.

        Returns:
            str: Translation.
        """
        self.eval()
        with torch.no_grad():
            src_tokens = tokenize_sentence(sentence=src_text, vocab=src_vocab)
            if len(src_tokens) < self.params[names.MAX_LENGTH_SRC]:
                src_tokens += [constants.PAD_TOKEN_ID] * (
                    self.params[names.MAX_LENGTH_SRC] - len(src_tokens)
                )
            src_tensor = (
                torch.tensor(src_tokens[: self.params[names.MAX_LENGTH_SRC]])
                .unsqueeze(0)
                .to(self.params[names.DEVICE])
            )  # (1, T)
            tgt_tensor = (
                torch.tensor([constants.BOS_TOKEN_ID])
                .unsqueeze(0)
                .to(self.params[names.DEVICE])
            )  # (1, 1)
            generated_tokens = [constants.BOS_TOKEN_ID]
            for _ in range(self.params[names.MAX_LENGTH_TGT]):
                logits = self(
                    src_tensor, tgt_tensor[:, : self.params[names.MAX_CONTEXT_TGT]]
                )
                logits = logits[:, -1, :]  # (1, C)
                probs = F.softmax(logits, dim=-1)  # (1, C)
                output = torch.multinomial(probs, num_samples=1)  # (1, 1)
                tgt_tensor = torch.cat((tgt_tensor, output), dim=1)  # (1, T + 1)
                predicted_token_id = output.item()
                generated_tokens.append(predicted_token_id)
                if predicted_token_id == constants.EOS_TOKEN_ID:
                    break
        translation = []
        for token in generated_tokens:
            translation.append(tgt_vocab_reversed[token])
        return from_tokens_to_text(tokens=translation)

    def evaluate(
        self: _DiffTransformer,
        df_test: pd.DataFrame,
        src_vocab: Dict[str, int],
        tgt_vocab_reversed: Dict[str, int],
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Evaluate the model on the test set.

        Args:
            self (_Transformer): Class instance.
            df_test (pd.DataFrame): Test set.
            src_vocab (Dict[str, int]): Vocabulary of the source language.
            tgt_vocab_reversed (Dict[str, int]): Reversed vocabulary of the target language.

        Returns:
            Tuple[Dict[str, float], List[str]]: (Metrics, translations of the model).
        """
        start_time = time.time()
        list_translations = []
        list_references = []
        for srs_sent, tgt_sent in zip(
            df_test[self.params[names.SRC_LANGUAGE]],
            df_test[self.params[names.TGT_LANGUAGE]],
        ):
            reference = clean_text(text=tgt_sent)
            translation = self.translate(
                src_vocab=src_vocab,
                tgt_vocab_reversed=tgt_vocab_reversed,
                src_text=srs_sent,
            )
            list_references.append(reference)
            list_translations.append(translation)
        rouge_1, rouge_l = calculate_metrics(
            predicted_sentences=list_translations,
            reference_sentences=list_references,
        )
        metrics = {"rouge_1": rouge_1, "rouge_l": rouge_l}
        print(
            f"Evaluated successfully. It took {(time.time() - start_time):.2f} seconds"
        )
        return metrics, list_translations
