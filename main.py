# main.py
import argparse
import subprocess
import sys

from models import ALL_MODELS
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hệ thống điều hướng Benchmark Mô hình Gợi ý Tuần tự (Sequential Recommendation)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Chế độ chạy: 'train' để huấn luyện, 'test' để đánh giá kết quả.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Tên mô hình cụ thể (vídụ: 'hier_sasrec', 'sasrec') hoặc 'all' để chạy toàn bộ baseline.",
    )

    return parser.parse_args()


def run_train(model_name):
    print(f"\n========================================================")
    print(f"🚀 KÍCH HOẠT TIẾN TRÌNH HUẤN LUYỆN BACKBONE: {model_name}")
    print(f"========================================================")
    try:
        subprocess.run(
            [
                sys.executable,
                "scripts/train.py",
                "--model",
                model_name,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Tiến trình huấn luyện {model_name} gặp sự cố. Lỗi: {e}")
        sys.exit(1)


def run_test(model_name):
    print(f"\n========================================================")
    print(f"📊 KÍCH HOẠT TIẾN TRÌNH ĐÁNH GIÁ (TESTING): {model_name}")
    print(f"========================================================")
    try:
        subprocess.run(
            [
                sys.executable,
                "scripts/test.py",
                "--model",
                model_name,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Tiến trình kiểm thử {model_name} gặp sự cố. Lỗi: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    
    # Chuẩn hóa chuỗi đầu vào loại bỏ khoảng trắng thừa phát sinh
    model_input = args.model.strip().lower()

    # Khóa cứng seed nền tảng ngay tại entry point để đồng bộ tài nguyên sinh số ngẫu nhiên
    # (Đọc cấu hình reproducibility từ file base config mặc định)
    try:
        default_cfg = load_config()
        seed = default_cfg.get("reproducibility", {}).get("seeds", [42])[0]
        deterministic = default_cfg.get("reproducibility", {}).get("deterministic", False)
        set_seed(seed=seed, deterministic=deterministic)
    except Exception:
        # Fallback an toàn nếu chưa cấu hình xong file yaml
        set_seed(seed=42, deterministic=False)

    # Kiểm tra tính hợp lệ của tham số mô hình truyền vào trước khi chạy tiến trình nặng
    if model_input != "all" and model_input not in ALL_MODELS:
        print(f"❌ Lỗi: Mô hình '{model_input}' không nằm trong hệ thống MODEL_REGISTRY.")
        print(f"💡 Các mô hình hợp lệ hiện tại bao gồm: {list(ALL_MODELS.keys())}")
        sys.exit(1)

    # Điều hướng thực thi luồng pipeline
    if args.mode == "train":
        if model_input == "all":
            print(f"🔥 Bắt đầu chạy Benchmark Huấn Luyện cho TOÀN BỘ ({len(ALL_MODELS)}) mô hình...")
            for model_name in ALL_MODELS:
                run_train(model_name)
        else:
            run_train(model_input)

    elif args.mode == "test":
        if model_input == "all":
            print(f"🔬 Bắt đầu đánh giá kiểm thử kiểm tra hiệu năng hệ thống trên TOÀN BỘ mô hình...")
            for model_name in ALL_MODELS:
                run_test(model_name)
        else:
            run_test(model_input)


if __name__ == "__main__":
    main()
