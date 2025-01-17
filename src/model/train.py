"""Functions to launch training from the command line"""

import argparse
import sys

from typing import Optional

from src.libs.preprocessing import load_data

from src.model.experiments import init_pipeline_from_config


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
    # Iteration of the experiment
    parser.add_argument(
        "-i",
        "--iteration",
        nargs="+",
        type=int,
        required=True,
        help="Iteration of the exepriment",
    )  # Local flag
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


def train_main(argv: argparse.ArgumentParser) -> None:
    """
    Launch training from terminal.

    Args:
        argv (argparse.ArgumentParser): Parser arguments.
    """
    parser = get_parser()
    args = parser.parse_args(argv)
    # Load data
    df_train = load_data(local=args.local_data, type="train")
    print("Training set loaded successfully")
    df_valid = load_data(local=args.local_data, type="valid")
    print("Validation set loaded successfully")
    df_test = load_data(local=args.local_data, type="test")
    print("Test set loaded successfully")
    for exp, iter in zip(args.exp, args.iteration):
        print(f"Experiment {exp} - Iteration {iter}")
        pipeline = init_pipeline_from_config(
            id_experiment=int(exp), iteration=int(iter)
        )
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
