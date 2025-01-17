"""abstract pipeline"""

import abc
import pandas as pd
import typing


_Pipeline = typing.TypeVar(name="_Pipeline", bound="Pipeline")


class Pipeline(abc.ABC):
    """abstract pipeline object"""

    def __init__(self: _Pipeline, id_experiment: int | None):
        self.id_experiment = id_experiment

    @abc.abstractmethod
    def full_pipeline(
        self: _Pipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        """full pipeline object"""
        raise NotImplementedError

    @abc.abstractmethod
    def learning_pipeline(
        self: _Pipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        """learning pipeline object"""
        raise NotImplementedError

    @abc.abstractmethod
    def testing_pipeline(
        self: _Pipeline,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ):
        """testing pipeline object"""
        raise NotImplementedError
