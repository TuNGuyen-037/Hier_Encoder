import argparse
import subprocess
import sys

from models import ALL_MODELS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    return parser.parse_args()


def run_train(model_name):
    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--model",
            model_name,
        ],
        check=True,
    )


def run_test(model_name):
    subprocess.run(
        [
            sys.executable,
            "scripts/test.py",
            "--model",
            model_name,
        ],
        check=True,
    )


def main():
    args = parse_args()

    if args.mode == "train":
        if args.model == "all":
            for model_name in ALL_MODELS:
                run_train(model_name)
        else:
            run_train(args.model)

    elif args.mode == "test":
        if args.model == "all":
            for model_name in ALL_MODELS:
                run_test(model_name)
        else:
            run_test(args.model)


if __name__ == "__main__":
    main()
