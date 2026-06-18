# scripts/test.py

import argparse

from utils import (
    set_seed,
    load_config,
    get_device,
    load_checkpoint,
)
from datasets import get_dataloaders
from models import MODEL_REGISTRY
from evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.model)

    seed = cfg["project"]["seed"]
    deterministic = cfg["reproducibility"]["deterministic"]

    set_seed(seed, deterministic)

    device = get_device(
        cfg["train"]["device"]
    )

    (
        _,
        _,
        test_loader,
        dataset,
    ) = get_dataloaders()

    model = MODEL_REGISTRY[args.model](
        dataset.num_items
    ).to(device)

    checkpoint_path = (
        cfg["logging"]["checkpoint_dir"]
        + f"/best_{args.model}.pt"
    )

    load_checkpoint(
        model,
        checkpoint_path,
        map_location=device,
    )

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        model_name=args.model,
        top_k=cfg["fairness"]["top_k"],
    )

    print(results)


if __name__ == "__main__":
    main()
