"""Functions to launch training from the command line"""

import sys
import argparse
from typing import Optional

from src.libs.preprocessing import load_data, get_train_valid_test_sets

from src.model.experiments import init_pipeline_from_config, load_pipeline_from_config


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
    # Load a pretrained pipeline
    parser.add_argument("--load", action="store_true", help="Load pretrained pipeline")
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
    df = load_data(local=args.local_data, small=False)
    print("Data loaded successfully")
    df_train, df_valid, df_test = get_train_valid_test_sets(df=df)
    for exp in args.exp:
        print(f"Experiment {exp}")
        if args.load:
            pipeline = load_pipeline_from_config(exp)
        else:
            pipeline = init_pipeline_from_config(exp)
        print("Pipeline loaded successfully")
        if args.full:
            pipeline.full_pipeline(
                df_train=df_train, df_valid=df_valid, df_test=df_test
            )
        elif args.learning:
            pipeline.learning_pipeline(
                df_train=df_train, df_valid=df_valid, df_test=df_test
            )
        elif args.testing:
            pipeline.testing_pipeline(
                df_train=df_train, df_valid=df_valid, df_test=df_test
            )


if __name__ == "__main__":
    train_main(sys.argv[1:])
