"""Functions to initiate remote training"""

import kaggle
import json
import argparse
import os
import shutil
import sys
import subprocess
from remote_training.__kaggle_login import kaggle_users
from remote_training import config
from src.configs import constants
from src.model.train import get_parser
from typing import Optional


def get_git_branch_name() -> str:
    """
    Get the current Git branch name.

    Returns:
        str: Current Git branch name.
    """
    try:
        branch_name = (
            subprocess.check_output(["git", "branch", "--show-current"])
            .strip()
            .decode()
        )
        return branch_name
    except subprocess.CalledProcessError:
        return "Error: Could not determine the Git branch name."


def init_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """
    Add arguments for the remote training parser.

    Args:
        parser (Optional[argparse.ArgumentParser], optional): Parser. Defaults to None.

    Returns:
        argparse.ArgumentParser: Parser for remote training.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Train a model on Kaggle using a script"
        )
    # Notebook flag
    parser.add_argument(
        "-n",
        "--notebook_id",
        type=str,
        help="Notebook name in kaggle",
        default=config.NOTEBOOK_ID,
    )
    # User flag
    parser.add_argument(
        "-u", "--user", type=str, help="Kaggle user", choices=list(kaggle_users.keys())
    )
    # Branch flag
    parser.add_argument(
        "--branch", type=str, help="Git branch name", default=get_git_branch_name()
    )
    # CPU flag
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    # Push flag
    parser.add_argument("-p", "--push", action="store_true", help="Push")
    # Download flag
    parser.add_argument(
        "-d", "--download", action="store_true", help="Download results"
    )
    return parser


def prepare_notebook(
    output_nb_path: str,
    exp: int,
    branch: str,
    git_user: str = None,
    git_repo: str = None,
    template_nb_path: str = os.path.join(
        config.REMOTE_TRAINING_FOLDER, "remote_training.ipynb"
    ),
    output_dir: str = constants.OUTPUT_FOLDER,
    dataset_files: Optional[list] = None,
) -> None:
    """
    Prepare the Kaggle notebook.

    Args:
        output_nb_path (str): Output path.
        exp (int): Experiment.
        branch (str): Git branch.
        git_user (str, optional): Git username. Defaults to None.
        git_repo (str, optional): Git repository. Defaults to None.
        template_nb_path (str, optional): Jupyter notebook template.
            Defaults to os.path.join( config.REMOTE_TRAINING_FOLDER, "remote_training.ipynb" ).
        output_dir (str, optional): Output directory. Defaults to constants.OUTPUT_FOLDER.
        dataset_files (Optional[list], optional): Kaglle datasets. Defaults to None.
    """
    assert git_user is not None, "Please provide a git username for the repo"
    assert git_repo is not None, "Please provide a git repo name for the repo"
    expressions = [
        ("exp", f"{exp}"),
        ("branch", f"'{branch}'"),
        ("git_user", f"'{git_user}'"),
        ("git_repo", f"'{git_repo}'"),
        ("output_dir", "None" if output_dir is None else f"'{output_dir}'"),
        ("dataset_files", "None" if dataset_files is None else f"{dataset_files}"),
    ]
    with open(template_nb_path) as f:
        template_nb = f.readlines()
        # Replace expressions in the notebook template
        # Use parser arguments to replace them
        for line_idx, li in enumerate(template_nb):
            for expr, expr_replace in expressions:
                if f"!!!{expr}!!!" in li:
                    template_nb[line_idx] = template_nb[line_idx].replace(
                        f"!!!{expr}!!!", expr_replace
                    )
        template_nb = "".join(template_nb)
    with open(output_nb_path, "w") as w:
        w.write(template_nb)


def define_config(
    args: argparse.ArgumentParser, kaggle_user: dict, exp_str: str, notebook_id: str
) -> dict:
    """
    Define the configuration of the notebook.

    Args:
        args (argparse.ArgumentParser): Parser arguments.
        kaggle_user (dict): Kaggle user.
        exp_str (str): Experiment (string format).
        notebook_id (str): ID of the notebook.

    Returns:
        dict: Configuration of the notebook.
    """
    config = {
        "id": f"{kaggle_user['username']}/{exp_str}",
        "title": notebook_id.lower(),
        "code_file": f"{notebook_id}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if not args.cpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "full_pipeline": "ture" if args.full else "false",
        "learning_pipeline": "ture" if args.learning else "false",
        "testing_pipeline": "ture" if args.testing else "false",
        "dataset_sources": config.KAGGLE_DATASET_LIST,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    return config


def main(argv):
    """
    Launch remote training.

    Args:
        argv (_type_): Parser arguments.
    """
    # Get parser arguments
    parser = init_parser()
    get_parser(parser)
    args = parser.parse_args(argv)
    notebook_id = args.notebook_id
    exp_str = "_".join(f"{exp:05d}" for exp in args.exp)
    kaggle_user = kaggle_users[args.user]
    uname_kaggle = kaggle_user["username"]
    kaggle.api._load_config(kaggle_user)
    # If needed, download results
    if args.download:
        tmp_dir = f"__tmp_{exp_str}"
        os.makedirs(tmp_dir, exist_ok=True)
        kaggle.api.kernels_output_cli(
            f"{kaggle_user['username']}/{notebook_id}", path=str(tmp_dir)
        )
        subprocess.run(
            ["tar", "-xzf", f"{tmp_dir}/output.tgz", constants.OUTPUT_FOLDER]
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    # Launch notebook
    kernel_root = f"__nb_{uname_kaggle}"
    kernel_path = f"{kernel_root}/{exp_str}"
    os.makedirs(kernel_path, exist_ok=True)
    branch = args.branch
    config = define_config(args, kaggle_user, exp_str, notebook_id)
    prepare_notebook(
        f"{kernel_path}/{notebook_id}" + ".ipynb",
        args.exp,
        branch,
        git_user=config.GIT_USER,
        git_repo=config.GIT_REPO,
    )
    assert os.path.exists(f"{kernel_path}/{notebook_id}" + ".ipynb")
    with open(f"{kernel_path}/kernel-metadata.json", "w") as f:
        json.dump(config, f, indent=4)
    if args.push:
        kaggle.api.kernels_push_cli(str(kernel_path))


if __name__ == "__main__":
    main(sys.argv[1:])
