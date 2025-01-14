"""Transformer class"""

import numpy as np
import time
import torch
import torch.nn.functional as F
import typing
from torch.utils.data import DataLoader
from typing import Any

from src.configs import names, constants

from src.model.attention import (
    MultiHeadAttention,
    NormLayer,
    FeedForward,
    PositionalEncoding,
)

from src.libs.preprocessing import (
    tokenize_sentence_inference,
    from_tokens_to_text,
)

from src.libs.evaluation import calculate_metrics

_Encoder = typing.TypeVar(name="_Encoder", bound="Encoder")


class Encoder(torch.nn.Module):
    def __init__(
        self: _Encoder,
        num_heads: int,
        embedding_dimension: int,
        head_size: int,
        context_length: int,
        hidden_dimension: int,
        dropout: float,
    ) -> _Encoder:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=embedding_dimension,
            context_length=context_length,
            mask=False,
        )
        self.norm_layer_1 = NormLayer(dimension=embedding_dimension)
        self.norm_layer_2 = NormLayer(dimension=embedding_dimension)
        self.feed_forward = FeedForward(
            input_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=embedding_dimension,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self: _Encoder, x: torch.Tensor) -> torch.Tensor:
        out = self.self_attention(q=x, k=x, v=x)
        out = self.norm_layer_1(out + self.dropout(input=out))
        out = self.feed_forward(x=out)
        out = self.norm_layer_2(out + self.dropout(input=out))
        return out


_Decoder = typing.TypeVar(name="_Decoder", bound="Decoder")


class Decoder(torch.nn.Module):
    def __init__(
        self: _Decoder,
        num_heads: int,
        embedding_dimension: int,
        head_size: int,
        context_length: int,
        hidden_dimension: int,
        dropout: float,
    ) -> _Decoder:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=embedding_dimension,
            context_length=context_length,
            mask=True,
        )
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_dimension=embedding_dimension,
            head_size=head_size,
            head_output_dimension=embedding_dimension,
            context_length=context_length,
            mask=False,
        )
        self.norm_layer_1 = NormLayer(dimension=embedding_dimension)
        self.norm_layer_2 = NormLayer(dimension=embedding_dimension)
        self.norm_layer_3 = NormLayer(dimension=embedding_dimension)
        self.feed_forward = FeedForward(
            input_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=embedding_dimension,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self: _Decoder, x: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        out = self.self_attention(q=x, k=x, v=x)
        out = self.norm_layer_1(out + self.dropout(input=out))
        out = self.cross_attention(q=out, k=encoder_output, v=encoder_output)
        out = self.norm_layer_2(out + self.dropout(input=out))
        out = self.feed_forward(x=out)
        out = self.norm_layer_3(out + self.dropout(input=out))
        return out


_Transformer = typing.TypeVar(name="_Transformer", bound="Transformer")


class Transformer(torch.nn.Module):
    def __init__(self: _Transformer, params: dict[str, Any]) -> _Transformer:
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
        self.positional_encoding = PositionalEncoding(
            embedding_dimension=self.params[names.EMBEDDING_DIMENSION],
            context_length=self.params[names.MAX_SEQUENCE_LENGTH],
        )
        self.encoder_layers = torch.nn.ModuleList(
            [
                Encoder(
                    num_heads=self.params[names.NB_HEADS],
                    embedding_dimension=self.params[names.EMBEDDING_DIMENSION],
                    head_size=self.params[names.HEAD_SIZE],
                    context_length=self.params[names.MAX_SEQUENCE_LENGTH],
                    hidden_dimension=self.params[names.FEEDFORWARD_DIMENSION],
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
                    context_length=self.params[names.MAX_SEQUENCE_LENGTH],
                    hidden_dimension=self.params[names.FEEDFORWARD_DIMENSION],
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

    def forward(self: _Transformer, src_input: torch.Tensor, tgt_input: torch.Tensor):
        # Encoder
        src_embedded = self.encoder_embedding(src_input)
        encoder_output = self.dropout(
            src_embedded + self.positional_encoding(x=src_embedded)
        )
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(x=encoder_output)
        # Decoder
        tgt_embedded = self.decoder_embedding(tgt_input)
        decoder_output = self.dropout(
            tgt_embedded + self.positional_encoding(x=tgt_embedded)
        )
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                x=decoder_output, encoder_output=encoder_output
            )
        logits = self.linear(decoder_output)
        return logits

    def train_model(
        self: _Transformer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> tuple[list[int], list[int]]:
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
            for src_input, tgt_input, tgt_output in train_dataloader:
                src_input, tgt_input, tgt_output = (
                    src_input.to(self.params[names.DEVICE]),
                    tgt_input.to(self.params[names.DEVICE]),
                    tgt_output.to(self.params[names.DEVICE]),
                )
                optimizer.zero_grad()
                logits = self(src_input, tgt_input)
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
                for src_input, tgt_input, tgt_output in valid_dataloader:
                    src_input, tgt_input, tgt_output = (
                        src_input.to(self.params[names.DEVICE]),
                        tgt_input.to(self.params[names.DEVICE]),
                        tgt_output.to(self.params[names.DEVICE]),
                    )
                    logits = self(src_input, tgt_input)
                    B, T, _ = logits.shape
                    loss = criterion(
                        logits.view(B * T, self.params[names.TGT_VOCAB_SIZE]),
                        tgt_output.view(B * T),
                    )
                    valid_loss += loss.item()
            ###
            train_loss /= len(train_dataloader)
            valid_loss /= len(valid_dataloader)
            train_loss_history.append(train_loss)
            valid_loss_history.append(valid_loss)
            if epoch % (1 + self.params[names.NB_EPOCHS] // 20) == 0:
                print(f"Epoch {epoch+1} / {self.params[names.NB_EPOCHS]} -------------")
                print(f"Train loss : {train_loss:.4f}. Valid loss : {valid_loss:.4f}.")
        print(
            f"Trained successfully. It took {time.time() - start_training:.2f} seconds. \n"
        )
        return train_loss_history, valid_loss_history

    def evaluate(
        self: _Transformer,
        test_dataloader: DataLoader,
        src_vocab: dict[str, int],
        tgt_vocab: dict[str, int],
    ) -> tuple[dict[str, float], list[str], list[str], list[str]]:
        start_time = time.time()
        translations_src = []
        translations_tgt = []
        translations_predictions = []
        src_vocab_reversed = {token_id: token for token, token_id in src_vocab.items()}
        tgt_vocab_reversed = {token_id: token for token, token_id in tgt_vocab.items()}
        self.eval()
        with torch.no_grad():
            for (
                src_input_tensor,
                tgt_input_tensor,
                tgt_output_tensor,
            ) in test_dataloader:
                src_input_tensor, tgt_input_tensor, tgt_output_tensor = (
                    src_input_tensor.to(self.params[names.DEVICE]),
                    tgt_input_tensor.to(self.params[names.DEVICE]),
                    tgt_output_tensor.to(self.params[names.DEVICE]),
                )
                for src_input, tgt_input, tgt_output in zip(
                    src_input_tensor, tgt_input_tensor, tgt_output_tensor
                ):
                    translation_tensor = self.decode(src_input=src_input.unsqueeze(0))
                    translations_src.append(
                        from_tokens_to_text(
                            tokens=[src_vocab_reversed[i.item()] for i in src_input],
                            params=self.params,
                        )
                    )
                    translations_tgt.append(
                        from_tokens_to_text(
                            tokens=[tgt_vocab_reversed[i.item()] for i in tgt_input],
                            params=self.params,
                        )
                    )
                    translations_predictions.append(
                        from_tokens_to_text(
                            tokens=[
                                tgt_vocab_reversed[i.item()] for i in translation_tensor
                            ],
                            params=self.params,
                        )
                    )
        rouge_1, rouge_l = calculate_metrics(
            predicted_sentences=translations_predictions,
            reference_sentences=translations_tgt,
        )
        metrics = {"rouge_1": rouge_1, "rouge_l": rouge_l}
        print(
            f"Evaluated successfully. It took {(time.time() - start_time):.2f} seconds"
        )
        return metrics, translations_src, translations_tgt, translations_predictions

    def evaluate_bis(
        self: _Transformer,
        test_dataloader: DataLoader,
        src_vocab: dict[str, int],
        tgt_vocab: dict[str, int],
    ) -> tuple[dict[str, float], list[str], list[str], list[str]]:
        start_time = time.time()
        translations_src = []
        translations_tgt = []
        translations_predictions = []
        src_vocab_reversed = {token_id: token for token, token_id in src_vocab.items()}
        tgt_vocab_reversed = {token_id: token for token, token_id in tgt_vocab.items()}
        self.eval()
        with torch.no_grad():
            for src_input, tgt_input, tgt_output in test_dataloader:
                src_input, tgt_input, tgt_output = (
                    src_input.to(self.params[names.DEVICE]),
                    tgt_input.to(self.params[names.DEVICE]),
                    tgt_output.to(self.params[names.DEVICE]),
                )
                translation_tensor = self.decode(src_input=src_input)
                src_input_text = np.array(
                    [[src_vocab_reversed[i.item()] for i in row] for row in src_input]
                )
                tgt_output_text = np.array(
                    [[tgt_vocab_reversed[i.item()] for i in row] for row in tgt_input]
                )
                translation_tensor_text = np.array(
                    [
                        [tgt_vocab_reversed[i.item()] for i in row]
                        for row in translation_tensor
                    ]
                )
                for i in range(src_input.shape[0]):
                    translations_src.append(
                        from_tokens_to_text(
                            tokens=src_input_text[i], params=self.params
                        )
                    )
                    translations_tgt.append(
                        from_tokens_to_text(
                            tokens=tgt_output_text[i], params=self.params
                        )
                    )
                    translations_predictions.append(
                        from_tokens_to_text(
                            tokens=translation_tensor_text[i], params=self.params
                        )
                    )
        rouge_1, rouge_l = calculate_metrics(
            predicted_sentences=translations_predictions,
            reference_sentences=translations_tgt,
        )
        metrics = {"rouge_1": rouge_1, "rouge_l": rouge_l}
        print(
            f"Evaluated successfully. It took {(time.time() - start_time):.2f} seconds"
        )
        return metrics, translations_src, translations_tgt, translations_predictions

    def decode(
        self: _Transformer,
        src_input: torch.Tensor,  # (T, C)
    ):
        tgt = torch.tensor(
            [constants.BOS_TOKEN_ID], dtype=torch.long, device=self.params[names.DEVICE]
        ).unsqueeze(0)  # (1, 1)
        # Encoder
        src_embedded = self.encoder_embedding(src_input)
        encoder_output = self.dropout(
            src_embedded + self.positional_encoding(x=src_embedded)
        )
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(x=encoder_output)
        last_predicted_token = constants.BOS_TOKEN_ID
        for _ in range(self.params[names.MAX_SEQUENCE_LENGTH]):
            if last_predicted_token == constants.EOS_TOKEN_ID:
                break
            tgt_embedded = self.decoder_embedding(tgt)
            decoder_output = self.dropout(
                tgt_embedded + self.positional_encoding(x=tgt_embedded)
            )
            for decoder_layer in self.decoder_layers:
                decoder_output = decoder_layer(
                    x=decoder_output, encoder_output=encoder_output
                )
            logits = self.linear(decoder_output)  # (1, T, C)
            logits = logits[:, -1, :]  # (1, C)
            probs = F.softmax(logits, dim=-1)  # (1, C)
            output = torch.multinomial(probs, num_samples=1)  # (1, 1)
            tgt = torch.cat((tgt, output), dim=1)  # (1, T + 1)
            last_predicted_token = output.item()
        return tgt.squeeze()  # (1, T)

    def decode_bis(
        self: _Transformer,
        src_input: torch.Tensor,  # (B, T)
    ):
        tgt = torch.full(
            (src_input.size(0), 1), constants.BOS_TOKEN_ID, dtype=torch.long
        ).to(self.params[names.DEVICE])  # (B, 1)
        # Encoder
        src_embedded = self.encoder_embedding(src_input)
        encoder_output = self.dropout(
            src_embedded + self.positional_encoding(x=src_embedded)
        )
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(x=encoder_output)
        for i in range(self.params[names.MAX_SEQUENCE_LENGTH]):
            tgt_embedded = self.decoder_embedding(
                tgt[:, -self.params[names.MAX_SEQUENCE_LENGTH] :]
            )
            decoder_output = self.dropout(
                tgt_embedded + self.positional_encoding(x=tgt_embedded)
            )
            for decoder_layer in self.decoder_layers:
                decoder_output = decoder_layer(
                    x=decoder_output, encoder_output=encoder_output
                )
            logits = self.linear(decoder_output)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            output = torch.multinomial(probs, num_samples=1)  # (B, 1)
            tgt = torch.cat((tgt, output), dim=1)  # (B, T + 1)
        return tgt[:, 1:]  # (B, T)

    def translate(
        self: _Transformer,
        src_vocab: dict[str, int],
        tgt_vocab_reversed: dict[str, int],
        src_text: str,
    ) -> torch.Tensor:
        src_tensor = (
            torch.tensor(
                tokenize_sentence_inference(
                    sentence=src_text, vocab=src_vocab, params=self.params
                )
            )
            .unsqueeze(0)
            .to(self.params[names.DEVICE])
        )
        self.eval()
        with torch.no_grad():
            translation_tensor = self.decode(src_input=src_tensor)
            translation_indices = translation_tensor.squeeze(0).cpu().tolist()
            if isinstance(translation_tensor, int):
                return ""
            translation_text_tokens = [
                tgt_vocab_reversed[idx] for idx in translation_indices
            ]
            translation = from_tokens_to_text(
                tokens=translation_text_tokens, params=self.params
            )
        return translation
