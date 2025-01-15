"""Pipeline for Transformer model"""

import numpy as np
import os
import pandas as pd
import pickle as pkl
import time
import torch
import typing

from torch.utils.data import DataLoader

from src.configs import constants, ml_config, names

from src.model.dataset import TranslationDataset

from src.model.transformer import Transformer

from src.model.diff_transformer import DiffTransformer

from src.model.pipeline import Pipeline

from src.libs.preprocessing import (
    load_data_from_local,
    create_vocabs,
    tokenize_dataframe,
)

_TransformerPipeline = typing.TypeVar(
    name="_TransformerPipeline", bound="TransformerPipeline"
)


class TransformerPipeline(Pipeline):
    def __init__(
        self: _TransformerPipeline, id_experiment: int, iteration: int = 0
    ) -> None:
        super().__init__(id_experiment=id_experiment)
        self.id_experiment
        self.iteration = iteration
        self.params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        self.metrics = {}
        self.training_time = 0.0
        if (self.params[names.DEVICE] == "cuda") and (torch.cuda.is_available()):
            self.params[names.DEVICE] = "cuda"
        else:
            self.params[names.DEVICE] = "cpu"
        self.folder_name = (
            f"{self.params[names.MODEL_TYPE]}_{id_experiment}_{iteration}"
        )
        os.makedirs(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "training"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(constants.OUTPUT_FOLDER, self.folder_name, "test"),
            exist_ok=True,
        )
        print("Pipeline initialized successfully")

    def full_pipeline(
        self: _TransformerPipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        if self.iteration == 0:
            df = pd.concat([df_train, df_valid, df_test])
            self.get_vocabs(df=df)
            self.get_model()
        else:
            self.load()
        train_dataloader, valid_dataloader = self.preprocess_data(
            df_train=df_train, df_valid=df_valid
        )
        start_training_time = time.time()
        train_losses, valid_losses = self.model.train_model(
            train_dataloader=train_dataloader, valid_dataloader=valid_dataloader
        )
        self.training_time = time.time() - start_training_time
        self.save_losses(train_loss=train_losses, valid_loss=valid_losses)
        metrics, translations = self.model.evaluate(
            df_test=df_test,
            src_vocab=self.src_vocab,
            tgt_vocab_reversed=self.tgt_vocab_reversed,
        )
        self.metrics = metrics
        self.save()
        self.save_translations(translations=translations)
        print("Full pipeline ran successfully")

    def learning_pipeline(
        self: _TransformerPipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        raise NotImplementedError

    def testing_pipeline(
        self: _TransformerPipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        raise NotImplementedError

    def get_model(
        self: _TransformerPipeline,
    ) -> None:
        if self.params[names.MODEL_TYPE] == names.TRANSFORMER:
            self.model = Transformer(params=self.params)
        elif self.params[names.MODEL_TYPE] == names.DIFF_TRANSFORMER:
            self.model = DiffTransformer(params=self.params)
        self.model.to(self.params[names.DEVICE])
        print(f"Model {self.params[names.MODEL_TYPE]} loaded successfully")

    def get_vocabs(self: _TransformerPipeline, df: pd.DataFrame) -> None:
        src_vocab, tgt_vocab = create_vocabs(df=df, params=self.params)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_reversed = {idx: word for word, idx in src_vocab.items()}
        self.tgt_vocab_reversed = {idx: word for word, idx in tgt_vocab.items()}
        self.params[names.SRC_VOCAB_SIZE] = len(src_vocab)
        self.params[names.TGT_VOCAB_SIZE] = len(tgt_vocab)
        print("Vocabularies created successfully")

    def preprocess_data(
        self: _TransformerPipeline, df_train: pd.DataFrame, df_valid: pd.DataFrame
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        (
            src_tokens_train,
            tgt_tokens_train,
        ) = tokenize_dataframe(
            df=df_train,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            params=self.params,
        )
        (
            src_tokens_valid,
            tgt_tokens_valid,
        ) = tokenize_dataframe(
            df=df_valid,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            params=self.params,
        )
        train_dataset = TranslationDataset(
            src_tokens=src_tokens_train, tgt_tokens=tgt_tokens_train, params=self.params
        )
        valid_dataset = TranslationDataset(
            src_tokens=src_tokens_valid, tgt_tokens=tgt_tokens_valid, params=self.params
        )
        train_dataloader = train_dataset.get_dataloader(shuffle=True)
        valid_dataloader = valid_dataset.get_dataloader(shuffle=False)
        print("Dataloaders loaded successfully")
        return train_dataloader, valid_dataloader

    def save(self):
        """save model"""
        path = os.path.join(
            constants.OUTPUT_FOLDER, self.folder_name, "training", "pipeline.pkl"
        )
        self.model = self.model.to("cpu")
        self.metrics = move_to_cpu(self.metrics)
        self.training_time = move_to_cpu(self.training_time)
        self.src_vocab = move_to_cpu(self.src_vocab)
        self.src_vocab_reversed = move_to_cpu(self.src_vocab_reversed)
        self.tgt_vocab = move_to_cpu(self.tgt_vocab)
        self.tgt_vocab_reversed = move_to_cpu(self.tgt_vocab_reversed)
        self_to_cpu = move_to_cpu(self)
        with open(path, "wb") as file:
            pkl.dump(self_to_cpu, file)
        print("Model saved successfully")

    def save_losses(self, train_loss: list[float], valid_loss: list[float]) -> None:
        train_loss = move_to_cpu(train_loss)
        valid_loss = move_to_cpu(valid_loss)
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "training", "train_loss.npy"
            ),
            train_loss,
        )
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER, self.folder_name, "training", "valid_loss.npy"
            ),
            valid_loss,
        )
        print("Losses saved successfully")

    def save_translations(self: _TransformerPipeline, translations: list[str]) -> None:
        translations = move_to_cpu(translations)
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER,
                self.folder_name,
                "test",
                "translations.npy",
            ),
            translations,
        )
        print("Translations saved successfully")

    def load(self: _TransformerPipeline) -> _TransformerPipeline:
        path = os.path.join(
            constants.OUTPUT_FOLDER,
            f"{self.params[names.MODEL_TYPE]}_{self.id_experiment}_{self.iteration-1}",
            "training",
            "pipeline.pkl",
        )
        with open(path, "rb") as file:
            pipeline: TransformerPipeline = pkl.load(file)
        self.src_vocab = pipeline.src_vocab
        self.src_vocab_reversed = pipeline.src_vocab_reversed
        self.tgt_vocab = pipeline.tgt_vocab
        self.tgt_vocab_reversed = pipeline.tgt_vocab_reversed
        self.model = pipeline.model.to(self.params[names.DEVICE])
        print(
            f"Model {self.params[names.MODEL_TYPE]} number {self.iteration-1} of experiment {id_experiment} loaded successfully"
        )
        return pipeline


def move_to_cpu(obj):
    if isinstance(obj, torch.nn.Module):
        return obj.to("cpu")
    elif isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {key: move_to_cpu(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(val) for val in obj]
    else:
        return obj


if __name__ == "__main__":
    df_train = load_data_from_local(type="samples")
    df_valid = load_data_from_local(type="samples")
    df_test = load_data_from_local(type="test")
    df = pd.concat([df_train, df_valid, df_test])
    print(df.shape)
    id_experiment = 105
    iteration = 0
    obj = TransformerPipeline(id_experiment=id_experiment, iteration=iteration)
    # obj.get_vocabs(df=pd.concat([df_train, df_valid, df_test]))
    # train_dataloader, valid_dataloader = obj.preprocess_data(
    #     df_train=df_train, df_valid=df_valid
    # )
    # obj.get_model()
    # self = obj.model
    # optimizer = torch.optim.Adam(
    #     self.parameters(),
    #     lr=self.params[names.LEARNING_RATE],
    #     betas=self.params[names.BETAS],
    #     eps=self.params[names.EPSILON],
    # )
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    # for src, tgt_input, tgt_output in train_dataloader:
    #     break
    # logits = self(src, tgt_input)
    # B, T, _ = logits.shape
    # loss = criterion(
    #     logits.view(B * T, self.params[names.TGT_VOCAB_SIZE]),
    #     tgt_output.view(B * T),
    # )
    obj.full_pipeline(df_train=df_train, df_valid=df_valid, df_test=df_test)
