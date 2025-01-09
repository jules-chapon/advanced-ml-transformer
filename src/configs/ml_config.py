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
        names.TOKENIZATION: names.BASIC,
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
        names.NUM_WORKERS: 0,
        names.BETAS: (0.9, 0.98),
        names.EPSILON: 1e-9,
    },
    1: {
        names.MODEL_TYPE: names.TRANSFORMER,
        # TRANSLATION
        names.SRC_LANGUAGE: "en",
        names.TGT_LANGUAGE: "fr",
        names.TOKENIZATION: names.BASIC,
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
        names.NUM_WORKERS: 0,
        names.BETAS: (0.9, 0.98),
        names.EPSILON: 1e-9,
    },
    2: {
        names.MODEL_TYPE: names.TRANSFORMER,
        # TRANSLATION
        names.SRC_LANGUAGE: "en",
        names.TGT_LANGUAGE: "fr",
        names.TOKENIZATION: names.BASIC,
        # ARCHITECTURE
        names.EMBEDDING_DIMENSION: 48,
        names.MAX_SEQUENCE_LENGTH: 32,
        names.NB_LAYERS: 1,
        names.NB_HEADS: 2,
        names.HEAD_OUTPUT_DIMENSION: 64,
        names.HEAD_SIZE: 32,  # = EMBEDDING_DIMENSION / NB_HEADS
        names.DROPOUT: 0.1,
        names.FEEDFORWARD_DIMENSION: 128,
        names.DEVICE: "cuda",
        # TRAINING
        names.NB_EPOCHS: 50,
        names.LEARNING_RATE: 1e-4,
        names.BATCH_SIZE: 32,
        names.NUM_WORKERS: 4,
        names.BETAS: (0.9, 0.98),
        names.EPSILON: 1e-9,
    },
    3: {
        names.MODEL_TYPE: names.TRANSFORMER,
        # TRANSLATION
        names.SRC_LANGUAGE: "en",
        names.TGT_LANGUAGE: "fr",
        names.TOKENIZATION: names.ADVANCED,
        # ARCHITECTURE
        names.EMBEDDING_DIMENSION: 64,
        names.MAX_SEQUENCE_LENGTH: 32,
        names.NB_LAYERS: 2,
        names.NB_HEADS: 4,
        names.HEAD_OUTPUT_DIMENSION: 64,
        names.HEAD_SIZE: 16,  # = EMBEDDING_DIMENSION / NB_HEADS
        names.DROPOUT: 0.1,
        names.FEEDFORWARD_DIMENSION: 128,
        names.DEVICE: "cuda",
        # TRAINING
        names.NB_EPOCHS: 50,
        names.LEARNING_RATE: 1e-4,
        names.BATCH_SIZE: 32,
        names.NUM_WORKERS: 4,
        names.BETAS: (0.9, 0.98),
        names.EPSILON: 1e-9,
    },
    # Add more experiments as needed
}
