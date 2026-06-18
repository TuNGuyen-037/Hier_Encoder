# scripts/test.py
import argparse

from utils import (
    set_seed,
    load_config,
    get_device,
    load_checkpoint,
)
from datasets import get_dataloaders
from models import MODEL_REGISTRY
from evaluation import evaluate_model

HIERARCHICAL_MODELS = {"hier_gru", "hier_sasrec", "hier_nextitnet"}


def parse_args():
    parser = argparse.ArgumentParser(description="Script kiểm thử hiệu năng mô hình trên tập Test")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Tên mô hình cần tải checkpoint để đánh giá.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model.strip().lower()

    cfg = load_config(model_name)

    repro_cfg = cfg.get("reproducibility", {})
    seed = repro_cfg.get("seeds", [42])[0]
    deterministic = repro_cfg.get("deterministic", False)

    set_seed(seed, deterministic)

    device = get_device(cfg["train"]["device"])

    print(f"⏳ Đang nạp tập dữ liệu kiểm thử (Test Dataset) cho mô hình {model_name}...")
    _, _, test_loader, dataset = get_dataloaders()

    # ĐỒNG BỘ CẤU TRÚC: Kiểm tra điều kiện phân cấp để truyền tham số mạng nơ-ron chính xác
    is_hierarchical = model_name in HIERARCHICAL_MODELS

    if is_hierarchical:
        model = MODEL_REGISTRY[model_name](
            num_items=dataset.num_items,
            num_categories_per_level=dataset.num_categories_per_level
        ).to(device)
    else:
        model = MODEL_REGISTRY[model_name](
            num_items=dataset.num_items
        ).to(device)

    checkpoint_path = (
        cfg["logging"]["checkpoint_dir"]
        + f"/best_{model_name}.pt"
    )

    print(f"📂 Đang nạp file trọng số tối ưu từ: {checkpoint_path}")
    load_checkpoint(
        model,
        checkpoint_path,
        map_location=device,
    )

    print(f"🔬 Đang chạy quy trình tính toán ma trận kiểm thử (Evaluation) trên GPU/CPU...")
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        model_name=model_name,
        top_k=cfg["fairness"]["top_k"],
        hierarchical=is_hierarchical # Đồng bộ cờ định hướng Unpack Batch
    )

    # Format hiển thị tường minh
    print(f"\n📊 KẾT QUẢ KIỂM THỬ TRÊN TẬP TEST CHO MÔ HÌNH: {model_name}")
    for k, v in results.items():
        print(f"👉 {k.upper()}: {v:.4f}")


if __name__ == "__main__":
    main()
