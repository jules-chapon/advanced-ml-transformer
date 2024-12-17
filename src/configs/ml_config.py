"""Parameters for ML models"""

from src.configs import constants, names


###############################################################
#                                                             #
#                           CONSTANTS                         #
#                                                             #
###############################################################

NB_OPTUNA_TRIALS = 3

###############################################################
#                                                             #
#                     EXPERIMENTS CONFIGS                     #
#                                                             #
###############################################################

EXPERIMENTS_CONFIGS = {
    0: {
        names.DEFAULT_UNKNOWN_TOKEN: -1,
        names.DEFAULT_UNKNOWN_EMBEDDING: "",
        },
    1: {},
    2: {},
    # Add more experiments as needed
}
