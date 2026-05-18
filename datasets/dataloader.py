import yaml
from torch.utils.data import DataLoader

from datasets.amz2023 import Amazon2023Dataset


def get_train_loader(config_path="configs/base.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open("configs/fairness.yaml", "r", encoding="utf-8") as f:
        fairness = yaml.safe_load(f)["fairness"]

    batch_size = fairness["batch_size"]
    num_workers = cfg["train"]["num_workers"]

    dataset = Amazon2023Dataset(config_path=config_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, dataset
