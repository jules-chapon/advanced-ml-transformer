"""Functions to define different experiments"""

from src.configs import ml_config, names

from src.model.model import Model, _Model


def init_model_from_config(id_experiment: int) -> _Model | None:
    """
    Initialize a model for a given experiment.

    Args:
        id_experiment (int): ID of the experiment.

    Returns:
        _LGBMModel | None: Model with the parameters of the given experiment.
    """
    config = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
    if config[names.MODEL_TYPE] == names.MODEL:
        return Model(id_experiment=id_experiment)
    else:
        return None
