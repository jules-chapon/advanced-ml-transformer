"""Pipeline for Transformer model"""

import gdown
import numpy as np
import os
import pandas as pd
import pickle as pkl
import time
import torch
import typing

from torch.utils.data import DataLoader
from typing import Any

from src.configs import constants, ml_config, names

from src.model.dataset import TranslationDataset

from src.model.transformer import Transformer

from src.model.diff_transformer import DiffTransformer

from src.model.pipeline import Pipeline

from src.libs.preprocessing import (
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
        """
        Initialize class instance.

        Args:
            self (_TransformerPipeline): Class instance.
            id_experiment (int): ID of the experiment.
            iteration (int, optional): Iteration of the experiment. Defaults to 0.
        """
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
        """
        Run the full pipeline to train the model and save the results.

        Args:
            self (_TransformerPipeline): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Validation set.
            df_test (pd.DataFrame): Test set.
        """
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
        """
        Get the model associated to the ID of the experiment.

        Args:
            self (_TransformerPipeline): Class instance.
        """
        if self.params[names.MODEL_TYPE] == names.TRANSFORMER:
            self.model = Transformer(params=self.params)
        elif self.params[names.MODEL_TYPE] == names.DIFF_TRANSFORMER:
            self.model = DiffTransformer(params=self.params)
        self.model.to(self.params[names.DEVICE])
        print(f"Model {self.params[names.MODEL_TYPE]} loaded successfully")

    def get_vocabs(self: _TransformerPipeline, df: pd.DataFrame) -> None:
        """
        Define the vocabularies for both languages.

        Args:
            self (_TransformerPipeline): Class instance.
            df (pd.DataFrame): Full dataset (train, valid and test).
        """
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
    ) -> tuple[DataLoader, DataLoader]:
        """
        Preprocess raw data before training.

        Args:
            self (_TransformerPipeline): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Validation set.

        Returns:
            tuple[DataLoader, DataLoader]: (Train dataloader, validation dataloader).
        """
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

    def save(self: _TransformerPipeline) -> None:
        """
        Save the instance.

        Args:
            self (_TransformerPipeline): Class instance.
        """
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
        """
        Save train and validation losses.

        Args:
            train_loss (list[float]): Train losses.
            valid_loss (list[float]): Validation losses.
        """
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
        """
        Save translations of the test set.

        Args:
            self (_TransformerPipeline): Class instance.
            translations (list[str]): Translations.
        """
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

    def load(self: _TransformerPipeline) -> None:
        """
        Load a pre-trained model.

        Args:
            self (_TransformerPipeline): Class instance.
        """
        if self.params[names.MODEL_TYPE] == names.TRANSFORMER:
            file_id = "1dhsO2XsoBOSzO36_3VrUMdSxFQgNGIlQ"
        elif self.params[names.MODEL_TYPE] == names.DIFF_TRANSFORMER:
            file_id = "1WPMhTzCaIRCOJF6M-Bu8GxCrCqkGzPfR"
        os.makedirs(
            os.path.join(
                constants.OUTPUT_FOLDER,
                f"{self.params[names.MODEL_TYPE]}_{self.id_experiment}_{self.iteration-1}",
                "training",
            ),
            exist_ok=True,
        )
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            os.path.join(
                constants.OUTPUT_FOLDER,
                f"{self.params[names.MODEL_TYPE]}_{self.id_experiment}_{self.iteration-1}",
                "training",
                "pipeline.pkl",
            ),
            quiet=False,
        )
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
            f"Model {self.params[names.MODEL_TYPE]} number {self.iteration-1} of experiment {self.id_experiment} loaded successfully"
        )


def move_to_cpu(obj: Any) -> Any:
    """
    Move an object to CPU.

    Args:
        obj (Any): Object.

    Returns:
        Any: Object moved to CPU.
    """
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
