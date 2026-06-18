# utils/io.py
from pathlib import Path
import torch


def ensure_dir(path):
    """
    Đảm bảo thư mục lưu trữ tồn tại, tự động tạo nếu chưa có.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, path, optimizer=None, epoch=None, best_metric=None):
    """
    Lưu trọng số mô hình kèm trạng thái huấn luyện để checkpoint toàn diện.
    """
    ensure_dir(Path(path).parent)
    
    checkpoint = {
        "model_state_dict": model.state_dict()
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric

    torch.save(checkpoint, path)


def load_checkpoint(model, path, optimizer=None, map_location=None):
    """
    Tải trọng số mô hình từ file checkpoint một cách an toàn.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {path}")
        
    checkpoint = torch.load(path, map_location=map_location)
    
    # Hỗ trợ cả định dạng lưu cũ (chỉ có state_dict) và định dạng lưu mới (dict bọc)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    return model
