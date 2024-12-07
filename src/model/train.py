"""Functions to launch training from the command line"""

import sys
import argparse
from typing import Optional
import logging

from src.libs.preprocessing import load_data

from src.model.experiments import init_model_from_config


def get_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """
    Create parser to run training from terminal.

    Args:
        parser (Optional[argparse.ArgumentParser], optional): Parser. Defaults to None.

    Returns:
        argparse.ArgumentParser: Parser with the new arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    # Experiment ID
    parser.add_argument(
        "-e", "--exp", nargs="+", type=int, required=True, help="Experiment id"
    )

    # Local flag
    parser.add_argument(
        "--local_data", action="store_true", help="Load data from local filesystem"
    )
    # Learning flag
    parser.add_argument(
        "--learning", action="store_true", help="Whether to launch learning or not"
    )
    # Testing flag
    parser.add_argument(
        "--testing", action="store_true", help="Whether to launch testing or not"
    )
    # Full flag (learning and testing)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Whether to launch learning and testing or not",
    )
    return parser


def train_main(argv):
    """
    Launch training from terminal.

    Args:
        argv (_type_): Parser arguments.
    """
    parser = get_parser()
    args = parser.parse_args(argv)
    # Load data
    df_learning = load_data(is_train=True, is_local=args.local_data)
    df_testing = load_data(is_train=False, is_local=args.local_data)
    for exp in args.exp:
        model = init_model_from_config(exp)
        logging.info(f"Training experiment { exp }")
        if args.full:
            model.full_pipeline(df_learning=df_learning, df_testing=df_testing)
        elif args.learning:
            model.learning_pipeline(df_learning=df_learning)
        elif args.testing:
            model.testing_pipeline(df_learning=df_learning, df_testing=df_testing)


if __name__ == "__main__":
    train_main(sys.argv[1:])
