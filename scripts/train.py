# scripts/train.py

import argparse

from utils import set_seed, load_config
from datasets import get_dataloaders
from trainers import Trainer
from models import ALL_MODELS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or 'all'",
    )

    return parser.parse_args()


def train_single_model(model_name):
    cfg = load_config(model_name)

    seed = cfg["project"]["seed"]
    deterministic = cfg["reproducibility"]["deterministic"]

    set_seed(seed, deterministic)

    (
        train_loader,
        val_loader,
        _,
        dataset,
    ) = get_dataloaders()

    trainer = Trainer(
        model_name=model_name,
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.fit()


def main():
    args = parse_args()

    if args.model == "all":
        model_list = ALL_MODELS
    else:
        if args.model not in ALL_MODELS:
            raise ValueError(
                f"Unknown model: {args.model}"
            )

        model_list = [args.model]

    for model_name in model_list:
        train_single_model(model_name)


if __name__ == "__main__":
    main()
