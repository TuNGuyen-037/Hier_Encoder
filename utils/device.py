# utils/decice.py

import torch


def get_device(device_cfg="auto"):
    if device_cfg == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(device_cfg)
