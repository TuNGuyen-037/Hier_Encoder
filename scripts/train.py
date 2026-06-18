# scripts/train.py
import argparse

from utils import set_seed, load_config
from datasets import get_dataloaders
from trainers import Trainer
from models import ALL_MODELS

HIERARCHICAL_MODELS = {"hier_gru", "hier_sasrec", "hier_nextitnet"}


def parse_args():
    parser = argparse.ArgumentParser(description="Script huấn luyện các mô hình Sequential Recommendation")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Tên mô hình cụ thể (ví dụ: 'hier_sasrec') hoặc 'all' để huấn luyện toàn bộ.",
    )

    return parser.parse_args()


def train_single_model(model_name, dataset, train_loader, val_loader):
    cfg = load_config(model_name)

    # Đồng bộ lấy seed từ cấu hình reproducibility hệ thống
    repro_cfg = cfg.get("reproducibility", {})
    seed = repro_cfg.get("seeds", [42])[0]
    deterministic = repro_cfg.get("deterministic", False)

    set_seed(seed, deterministic)

    # Khởi tạo đối tượng quản lý huấn luyện
    trainer = Trainer(
        model_name=model_name,
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.fit()


def main():
    args = parse_args()
    model_input = args.model.strip().lower()

    if model_input == "all":
        model_list = list(ALL_MODELS.keys())
    else:
        if model_input not in ALL_MODELS:
            raise ValueError(f"Không tìm thấy mô hình: {model_input} trong MODEL_REGISTRY")
        model_list = [model_input]

    # ĐỒNG BỘ HIỆU NĂNG: Chỉ gọi một lần duy nhất để tránh đọc lại ổ đĩa nhiều lần gây thắt nút cổ chai RAM
    print("⏳ Đang tải toàn bộ dữ liệu Datasets và khởi tạo Dataloaders vào bộ nhớ...")
    train_loader, val_loader, _, dataset = get_dataloaders()

    for model_name in model_list:
        train_single_model(
            model_name=model_name,
            dataset=dataset,
            train_loader=train_loader,
            val_loader=val_loader
        )


if __name__ == "__main__":
    main()
