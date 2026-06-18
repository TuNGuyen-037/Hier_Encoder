# utils/io.py
from pathlib import Path
import torch


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, path):
    ensure_dir(Path(path).parent)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model
