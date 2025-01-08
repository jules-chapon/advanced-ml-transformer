"""Parameters for ML models"""

from src.configs import names


###############################################################
#                                                             #
#                     EXPERIMENTS CONFIGS                     #
#                                                             #
###############################################################

EXPERIMENTS_CONFIGS = {
    0: {
        # EMBEDDING
        names.EMBEDDING_MODE: "word",
        names.SRC_VOCAB_SIZE: 300,
        names.TGT_VOCAB_SIZE: 300,
        names.TO_LOWERCASE: True,
        names.WITH_PUNCTUATION: False,
        # ARCHITECTURE
        names.EMBEDDING_DIMENSION: 64,
        names.MAX_SEQUENCE_LENGTH: 32,
        names.NB_LAYERS: 2,
        names.NB_HEADS: 4,
        names.HEAD_SIZE: 16,  # = EMBEDDING_DIMENSION / NB_HEADS
        names.BLOCK_SIZE: 32,
        names.DROPOUT: 0.2,
        names.FEEDFORWARD_DIMENSION: 256,
        names.DEVICE: "cpu",
        # TRAINING
        names.NB_EPOCHS: 100,
        names.LEARNING_RATE: 1e-4,
        names.BATCH_SIZE: 32,
        names.BETAS: (0.9, 0.98),
        names.EPSILON: 1e-9,
    },
    # Add more experiments as needed
    10: {
        names.MODEL_TYPE: names.TRANSFORMER,
        # TRANSLATION
        "src_language": "en",
        "tgt_language": "fr",
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
}
