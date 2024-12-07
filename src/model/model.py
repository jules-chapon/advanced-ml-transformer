"""Class for model"""

import abc
import typing


_Model = typing.TypeVar("_Model", bound="Model")


class Model(abc.ABC):
    """Abstract base class for ML models"""

    def __init__(self, id_experiment: int | None) -> None:
        """
        Initialize class instance.

        Args:
            id_experiment (int | None): ID of the experiment.
        """
        self.id_experiment = id_experiment
