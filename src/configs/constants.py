"""Constants"""

###############################################################
#                                                             #
#                             PATHS                           #
#                                                             #
###############################################################

OUTPUT_FOLDER = "output"

REMOTE_TRAINING_FOLDER = "remote_training"


###############################################################
#                                                             #
#                          DATASETS                           #
#                                                             #
###############################################################

### LOCAL

DATA_LARGE_FILENAME = "data/input/en-fr.csv"

DATA_SMALL_FILENAME = "data/input/samples.csv"

### HUGGING FACE

HF_LARGE_FILENAME = "jules-chapon/advanced-ml-transformer-large"

HF_SMALL_FILENAME = "jules-chapon/advanced-ml-transformer-small"

###############################################################
#                                                             #
#                      REMOTE TRAINING                        #
#                                                             #
###############################################################

GIT_USER = "jules-chapon"

GIT_REPO = "advanced-ml-transformer"

NOTEBOOK_ID = "advanced-ml-transformer"

KAGGLE_DATASET_LIST = []


###############################################################
#                                                             #
#                        FIXED VALUES                         #
#                                                             #
###############################################################

RANDOM_SEED = 42

TRAIN_RATIO = 0.90
VALID_RATIO = 0.09


# TOKENS

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unknown>"

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3

DEFAULT_VOCAB = {
    PAD_TOKEN: PAD_TOKEN_ID,
    BOS_TOKEN: BOS_TOKEN_ID,
    EOS_TOKEN: EOS_TOKEN_ID,
    UNK_TOKEN: UNK_TOKEN_ID,
}
