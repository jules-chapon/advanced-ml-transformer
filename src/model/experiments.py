"""Functions to define different experiments"""

from src.configs import ml_config, names

from src.model.pipeline_transformer import TransformerPipeline


def init_pipeline_from_config(id_experiment: int) -> TransformerPipeline | None:
    """
    Initialize a pipeline for a given experiment.

    Args:
        id_experiment (int): ID of the experiment.

    Returns:
        TransformerPipeline | None: Pipeline with the parameters of the given experiment.
    """
    config = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
    if config[names.MODEL_TYPE] in [names.TRANSFORMER, names.DIFF_TRANSFORMER]:
        return TransformerPipeline(id_experiment=id_experiment)
    else:
        return None


def load_pipeline_from_config(id_experiment: int) -> TransformerPipeline | None:
    """
    Load a trained for a given experiment.

    Args:
        id_experiment (int): ID of the experiment.

    Returns:
        TransformerPipeline | None: Model with the parameters of the given experiment.
    """
    config = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
    if config[names.MODEL_TYPE] == names.TRANSFORMER:
        return TransformerPipeline.load(id_experiment=id_experiment)
    else:
        return None
