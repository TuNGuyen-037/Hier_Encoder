# utils/decice.py
import torch


def get_device(device_cfg="auto"):
    """
    Tự động chọn cấu hình phần cứng tối ưu (CUDA / CPU).
    """
    if device_cfg == "auto" or device_cfg is None:
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(device_cfg)
