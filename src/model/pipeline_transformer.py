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
    get_train_valid_test_sets,
    tokenize_datasets,
)

_TransformerPipeline = typing.TypeVar(
    name="_TransformerPipeline", bound="TransformerPipeline"
)


class TransformerPipeline(Pipeline):
    def __init__(self: _TransformerPipeline, id_experiment: int) -> None:
        super().__init__(id_experiment=id_experiment)
        self.params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        self.metrics = {}
        self.training_time = 0.0
        if (self.params[names.DEVICE] == "cuda") and (torch.cuda.is_available()):
            self.params[names.DEVICE] = "cuda"
        else:
            self.params[names.DEVICE] = "cpu"
        self.folder_name = f"{self.params[names.MODEL_TYPE]}_{id_experiment}"
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
        train_dataloader, valid_dataloader, test_dataloader = self.preprocess_data(
            df_train=df_train, df_valid=df_valid, df_test=df_test
        )
        self.get_model()
        start_training_time = time.time()
        train_losses, valid_losses = self.model.train_model(
            train_dataloader=train_dataloader, valid_dataloader=valid_dataloader
        )
        self.training_time = time.time() - start_training_time
        self.save_losses(train_loss=train_losses, valid_loss=valid_losses)
        metrics, translations_src, translations_tgt, translations_predictions = (
            self.model.evaluate(
                test_dataloader=test_dataloader,
                src_vocab=self.src_vocab,
                tgt_vocab=self.tgt_vocab,
            )
        )
        self.metrics = metrics
        self.save()
        self.save_translations(
            translations_src=translations_src,
            translations_tgt=translations_tgt,
            translations_predictions=translations_predictions,
        )
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

    def preprocess_data(
        self: _TransformerPipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
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
            df_train=df_train, df_valid=df_valid, df_test=df_test, params=self.params
        )
        train_dataset = TranslationDataset(
            src_input_tokens_train, tgt_input_tokens_train, tgt_output_tokens_train
        )
        valid_dataset = TranslationDataset(
            src_input_tokens_valid, tgt_input_tokens_valid, tgt_output_tokens_valid
        )
        test_dataset = TranslationDataset(
            src_input_tokens_test, tgt_input_tokens_test, tgt_output_tokens_test
        )
        train_dataloader = train_dataset.get_dataloader(
            params=self.params, shuffle=True
        )
        valid_dataloader = valid_dataset.get_dataloader(params=self.params)
        test_dataloader = test_dataset.get_dataloader(params=self.params)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.params[names.SRC_VOCAB_SIZE] = len(src_vocab)
        self.params[names.TGT_VOCAB_SIZE] = len(tgt_vocab)
        print("Dataloaders loaded successfully")
        return train_dataloader, valid_dataloader, test_dataloader

    def save(self):
        """save model"""
        path = os.path.join(
            constants.OUTPUT_FOLDER, self.folder_name, "training", "pipeline.pkl"
        )
        self.model = self.model.to("cpu")
        self.metrics = move_to_cpu(self.metrics)
        self.training_time = move_to_cpu(self.training_time)
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

    def save_translations(
        self,
        translations_src: list[str],
        translations_tgt: list[str],
        translations_predictions: list[str],
    ) -> None:
        translations_src = move_to_cpu(translations_src)
        translations_tgt = move_to_cpu(translations_tgt)
        translations_predictions = move_to_cpu(translations_predictions)
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER,
                self.folder_name,
                "test",
                "translations_src.npy",
            ),
            translations_src,
        )
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER,
                self.folder_name,
                "test",
                "translations_tgt.npy",
            ),
            translations_tgt,
        )
        np.save(
            os.path.join(
                constants.OUTPUT_FOLDER,
                self.folder_name,
                "test",
                "translations_predictions.npy",
            ),
            translations_predictions,
        )
        print("Translations saved successfully")

    @classmethod
    def load(cls, id_experiment: int | None = None) -> _TransformerPipeline:
        params = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
        path = os.path.join(
            constants.OUTPUT_FOLDER,
            f"{params[names.MODEL_TYPE]}_{id_experiment}",
            "training",
            "pipeline.pkl",
        )
        with open(path, "rb") as file:
            pipeline = pkl.load(file)
        if (pipeline.params[names.DEVICE] == "cuda") and (torch.cuda.is_available()):
            pipeline.device = "cuda"
        else:
            pipeline.device = "cpu"
        pipeline.model = pipeline.model.to(pipeline.device)
        print(
            f"Pipeline of experiment {id_experiment} of model {params[names.MODEL_TYPE]} loaded successfully"
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
    df = load_data_from_local()
    print(len(df))
    df_train, df_valid, df_test = get_train_valid_test_sets(df)
    id_experiment = 0
    pipeline = TransformerPipeline(id_experiment=id_experiment)
    pipeline.full_pipeline(df_train=df_train, df_valid=df_valid, df_test=df_test)
