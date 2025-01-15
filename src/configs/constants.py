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

DATA_TRAIN_FILENAME = "data/input/train.csv"
DATA_VALID_FILENAME = "data/input/valid.csv"
DATA_TEST_FILENAME = "data/input/test.csv"
DATA_SAMPLES_FILENAME = "data/input/samples.csv"


### HUGGING FACE

HF_TRAIN_FILENAME = "jules-chapon/advanced-ml-train"
HF_VALID_FILENAME = "jules-chapon/advanced-ml-valid"
HF_TEST_FILENAME = "jules-chapon/advanced-ml-test"
HF_SAMPLES_FILENAME = "jules-chapon/advanced-ml-transformer-small"


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

NB_DATA_TRAIN = 10  # 1 000 000 lignes


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
