# utils/seed.py
import random
import numpy as np
import torch


def set_seed(seed=42, deterministic=False):
    """
    Khóa seed hệ thống để đồng bộ kết quả giữa các lần chạy Benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Bật benchmark để tối ưu tốc độ tính toán ma trận tuần hoàn (RNN/Transformer) nếu không yêu cầu chặt chẽ
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
