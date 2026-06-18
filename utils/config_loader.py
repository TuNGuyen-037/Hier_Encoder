# utils/config_loader.py
from pathlib import Path
import yaml


def load_yaml(path):
    """
    Load YAML file an toàn với UTF-8.
    """
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, override):
    """
    Hàm gộp đệ quy dictionary để tránh ghi đè mất cấu trúc con.
    """
    merged = base.copy()

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(model_name=None):
    """
    Tải và gộp hệ thống cấu hình benchmark:
    base.yaml + fairness.yaml + [model_name].yaml
    """
    config_dir = Path("configs")

    base_cfg = load_yaml(config_dir / "base.yaml")
    fairness_cfg = load_yaml(config_dir / "fairness.yaml")

    # Gộp cấu hình nền tảng và giao thức so sánh công bằng
    final_cfg = deep_merge(base_cfg, fairness_cfg)

    if model_name:
        # Xử lý strip để tránh lỗi nếu chuỗi truyền vào chứa khoảng trắng thừa
        clean_model_name = model_name.strip()
        model_file = config_dir / "model" / f"{clean_model_name}.yaml"
        
        model_cfg = load_yaml(model_file)
        final_cfg = deep_merge(final_cfg, model_cfg)

    return final_cfg
