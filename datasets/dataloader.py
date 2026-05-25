# datasets/dataloader.py
from torch.utils.data import DataLoader

from utils import load_config
from datasets.amz2023 import Amazon2023Dataset


def get_dataloaders():
    cfg = load_config()

    fairness = cfg["fairness"]
    train_cfg = cfg["train"]

    batch_size = fairness["batch_size"]
    num_workers = train_cfg["num_workers"]

    dataset = Amazon2023Dataset()

    train_loader = DataLoader(
        dataset.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        dataset,
    )
