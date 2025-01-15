"""Functions to define different experiments"""

from src.model.pipeline_transformer import TransformerPipeline


def init_pipeline_from_config(
    id_experiment: int, iteration: int
) -> TransformerPipeline | None:
    """
    Initialize a pipeline for a given experiment.

    Args:
        id_experiment (int): ID of the experiment.
        iteration (int): Iteration number.

    Returns:
        TransformerPipeline | None: Pipeline with the parameters of the given experiment.
    """
    return TransformerPipeline(id_experiment=id_experiment, iteration=iteration)
