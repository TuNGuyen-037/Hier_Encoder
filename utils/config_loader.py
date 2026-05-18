from pathlib import Path
import yaml


def load_yaml(path):
    """
    Load YAML file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base, override):
    """
    Recursive dict merge.
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
    Load final config:
    base + fairness + optional model config
    """
    config_dir = Path("configs")

    base_cfg = load_yaml(config_dir / "base.yaml")
    fairness_cfg = load_yaml(config_dir / "fairness.yaml")

    final_cfg = deep_merge(base_cfg, fairness_cfg)

    if model_name:
        model_cfg = load_yaml(config_dir / "model" / f"{model_name}.yaml")
        final_cfg = deep_merge(final_cfg, model_cfg)

    return final_cfg
