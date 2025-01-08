"""Parameters for ML models"""

from src.configs import names


###############################################################
#                                                             #
#                     EXPERIMENTS CONFIGS                     #
#                                                             #
###############################################################

EXPERIMENTS_CONFIGS = {
    0: {
        names.MODEL_TYPE: names.TRANSFORMER,
        # TRANSLATION
        names.SRC_LANGUAGE: "en",
        names.TGT_LANGUAGE: "fr",
        # ARCHITECTURE
        names.EMBEDDING_DIMENSION: 64,
        names.MAX_SEQUENCE_LENGTH: 32,
        names.NB_LAYERS: 2,
        names.NB_HEADS: 4,
        names.HEAD_OUTPUT_DIMENSION: 64,
        names.HEAD_SIZE: 16,  # = EMBEDDING_DIMENSION / NB_HEADS
        names.DROPOUT: 0.1,
        names.FEEDFORWARD_DIMENSION: 256,
        names.DEVICE: "cuda",
        # TRAINING
        names.NB_EPOCHS: 2,
        names.LEARNING_RATE: 1e-4,
        names.BATCH_SIZE: 16,
        names.BETAS: (0.9, 0.98),
    },
    1: {},
    # Add more experiments as needed
}
