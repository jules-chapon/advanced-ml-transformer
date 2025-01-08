"""Transformer class"""

import numpy as np
import pandas as pd
import time
import torch
import torch.nn.functional as F
import typing
from torch.utils.data import DataLoader
from typing import Any

from src.configs import ml_config, names, constants

from src.model.attention import (
    MultiHeadAttention,
    NormLayer,
    FeedForward,
    PositionalEncoding,
)

from src.libs.preprocessing import (
    load_data_from_local,
    get_train_valid_test_sets,
    tokenize_datasets,
    tokenize_sentence_inference,
    from_tokens_to_text,
)

from src.libs.evaluation import calculate_metrics

from src.model.dataset import TranslationDataset

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
        self.norm_layer = NormLayer(dimension=embedding_dimension)
        self.feed_forward = FeedForward(
            input_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=embedding_dimension,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self: _Encoder, x: torch.Tensor) -> torch.Tensor:
        out = self.self_attention(q=x, k=x, v=x)
        out = self.norm_layer(out + self.dropout(input=out))
        out = self.feed_forward(x=out)
        out = self.norm_layer(out + self.dropout(input=out))
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
        self.norm_layer = NormLayer(dimension=embedding_dimension)
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
        out = self.norm_layer(out + self.dropout(input=out))
        out = self.cross_attention(q=out, k=encoder_output, v=encoder_output)
        out = self.norm_layer(out + self.dropout(input=out))
        out = self.feed_forward(x=out)
        out = self.norm_layer(out + self.dropout(input=out))
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
    ) -> None:
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
        src_input: torch.Tensor,  # (B, T)
    ):
        tgt = torch.full_like(src_input, constants.BOS_TOKEN_ID)  # (B, T)
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
        return tgt[:, -(i + 2) :]

    def translate(
        self: _Transformer,
        src_vocab: dict[str, int],
        tgt_vocab_reversed: dict[str, int],
        src_texts: list[str] | pd.Series,
    ) -> torch.Tensor:
        list_tensors = []
        for src_text in src_texts:
            list_tensors.append(
                torch.tensor(
                    tokenize_sentence_inference(
                        sentence=src_text, vocab=src_vocab, params=self.params
                    )
                ).unsqueeze(0)
            )
        src_tensor = torch.cat(list_tensors).to(self.params[names.DEVICE])
        self.eval()
        with torch.no_grad():
            translation_tensor = self.decode(src_input=src_tensor)
        translations = []
        for translation_indices in translation_tensor:
            translation_indices = translation_indices.squeeze(0).cpu().tolist()
            translation_text_tokens = [
                tgt_vocab_reversed[idx] for idx in translation_indices
            ]
            translation = from_tokens_to_text(
                tokens=translation_text_tokens, params=self.params
            )
            translations.append(translation)
        return translations


if __name__ == "__main__":
    df = load_data_from_local()
    id_experiment = 10
    params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
    params[names.DEVICE] = "cpu"
    df_train, df_valid, df_test = get_train_valid_test_sets(df=df)
    (
        src_vocab,
        tgt_vocab,
        src_input_tokens_train,
        tgt_input_tokens_train,
        tgt_output_tokens_train,
        src_input_tokens_valid,
        tgt_input_tokens_valid,
        tgt_output_tokens_valid,
        src_input_tokens_test,
        tgt_input_tokens_test,
        tgt_output_tokens_test,
    ) = tokenize_datasets(
        df_train=df_train, df_valid=df_valid, df_test=df_test, params=params
    )
    dataset_train = TranslationDataset(
        src_input_tokens_train, tgt_input_tokens_train, tgt_output_tokens_train
    )
    dataset_valid = TranslationDataset(
        src_input_tokens_valid, tgt_input_tokens_valid, tgt_output_tokens_valid
    )
    dataset_test = TranslationDataset(
        src_input_tokens_test, tgt_input_tokens_test, tgt_output_tokens_test
    )
    dataloader_train = dataset_train.get_dataloader(params=params, shuffle=True)
    dataloader_valid = dataset_valid.get_dataloader(params=params)
    dataloader_test = dataset_test.get_dataloader(params=params)
    for src_input, tgt_input, tgt_output in dataloader_train:
        break
    print(f"Source input shape : {src_input.shape}")
    print(f"Target input shape : {tgt_input.shape}")
    print(f"Target output shape : {tgt_output.shape}")
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"Source vocabulary length : {src_vocab_size}")
    print(f"Target vocabulary length : {tgt_vocab_size}")
    params[names.SRC_VOCAB_SIZE] = src_vocab_size
    params[names.TGT_VOCAB_SIZE] = tgt_vocab_size
    self = Transformer(params=params)
    print(sum(p.numel() for p in self.parameters()) / 1e6, " M parameters")
    logits = self.forward(src_input, tgt_input)
    tgt_vocab_reversed = {token_id: token for token, token_id in tgt_vocab.items()}
    src_texts = df_test["en"][:10]
    translations = self.translate(
        src_vocab=src_vocab,
        tgt_vocab_reversed=tgt_vocab_reversed,
        src_texts=src_texts,
    )
    metrics, tr_src, tr_tgt, tr_preds = self.evaluate(
        dataloader_test, src_vocab, tgt_vocab
    )
