# datasets/preprocess.py
import gzip
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import load_config


def read_jsonl_gz(path, columns, max_rows=None):
    data = []
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu thô tại: {path}")

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows and idx >= max_rows:
                break
            try:
                obj = json.loads(line)
                data.append({col: obj.get(col) for col in columns})
            except Exception:
                continue

    return pd.DataFrame(data)


def kcore_filter(df, k=5):
    while True:
        before = len(df)
        user_count = df["user_id"].value_counts()
        item_count = df["parent_asin"].value_counts()

        df = df[df["user_id"].isin(user_count[user_count >= k].index)]
        df = df[df["parent_asin"].isin(item_count[item_count >= k].index)]

        if len(df) == before:
            break
    return df


def encode_ids(df):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user"] = user_encoder.fit_transform(df["user_id"])
    df["item"] = item_encoder.fit_transform(df["parent_asin"]) + 1  # 0 dành cho padding

    return df, user_encoder, item_encoder


def time_to_bucket(delta, buckets):
    for idx, threshold in enumerate(buckets):
        if delta < threshold:
            return idx
    return len(buckets)


def build_sequences(df, time_buckets):
    train_data = []
    val_data = {}
    test_data = {}

    df = df.sort_values(["user", "timestamp"])

    for user_id, df_user in df.groupby("user"):
        items = df_user["item"].tolist()
        times = df_user["timestamp"].tolist()

        if len(items) < 3:
            continue

        delta = [0]
        for i in range(1, len(times)):
            delta.append(times[i] - times[i - 1])

        time_ids = [time_to_bucket(d, time_buckets) + 1 for d in delta]

        test_data[user_id] = (items[:-1], time_ids[:-1], items[-1])
        val_data[user_id] = (items[:-2], time_ids[:-2], items[-2])

        train_seq = items[:-2]
        train_time = time_ids[:-2]

        for i in range(1, len(train_seq)):
            train_data.append((train_seq[:i], train_time[:i], train_seq[i]))

    return train_data, val_data, test_data


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_and_save():
    cfg = load_config()

    dataset_cfg = cfg["dataset"]
    fairness_cfg = cfg["fairness"]

    raw_dir = Path(dataset_cfg["raw_dir"])
    processed_dir = Path(dataset_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    review_path = raw_dir / dataset_cfg["review_file"]
    meta_path = raw_dir / dataset_cfg.get("meta_file", "meta_Cell_Phones_and_Accessories.jsonl.gz")

    max_rows = dataset_cfg["max_rows"]
    time_buckets = dataset_cfg["time_buckets"]
    k_core = fairness_cfg["k_core"]

    print("Step 1: Đang đọc và làm sạch dữ liệu tương tác (Review Json)...")
    df = read_jsonl_gz(review_path, ["user_id", "parent_asin", "timestamp"], max_rows=max_rows)
    df = df.dropna()
    df = kcore_filter(df, k=k_core)

    df, user_encoder, item_encoder = encode_ids(df)

    print("Step 2: Đang xây dựng chuỗi tuần tự Train/Val/Test...")
    train_data, val_data, test_data = build_sequences(df, time_buckets)

    # ĐỒNG BỘ PHÂN CẤP METADATA: Khai phá cây phân loại (Categories Taxonomy) từ file Meta
    print("Step 3: Đang đọc Metadata sản phẩm để xây dựng cấu trúc phân cấp (Hierarchical Taxonomy)...")
    try:
        # Đọc thông tin parent_asin và nhánh cây danh mục tương ứng
        df_meta = read_jsonl_gz(meta_path, ["parent_asin", "categories"], max_rows=max_rows)
        df_meta = df_meta[df_meta["parent_asin"].isin(item_encoder.classes_)]
        
        # Tạo LabelEncoder cho từng tầng danh mục (Ví dụ hệ thống Amazon thường có tối đa 3-4 tầng danh mục sâu)
        cat_level_encoders = [LabelEncoder(), LabelEncoder(), LabelEncoder()]
        item_taxonomy = {}
        graph_edges = []

        # Tạo từ điển map từ parent_asin sang mã encoded_item_id
        item_map = dict(zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_) + 1))

        for _, row in df_meta.iterrows():
            asin = row["parent_asin"]
            cats = row["categories"] if isinstance(row["categories"], list) else ["Unknown", "Unknown", "Unknown"]
            
            # Pad hoặc cắt bớt danh mục để cố định số tầng phân cấp = 3
            if len(cats) < 3:
                cats = cats + ["Unknown"] * (3 - len(cats))
            cats = cats[:3]

            # Lưu tạm chuỗi text để fit LabelEncoder sau
            item_taxonomy[item_map[asin]] = cats

        # Fit chuyển đổi chuỗi text danh mục thành số ID (Integer)
        for level in range(3):
            level_cats = [cats[level] for cats in item_taxonomy.values()]
            cat_level_encoders[level].fit(level_cats)
            
        for item_id, cats in item_taxonomy.items():
            encoded_path = [
                cat_level_encoders[level].transform([cats[level]])[0] + 1 # 0 để cho vị trí pad
                for level in range(3)
            ]
            item_taxonomy[item_id] = encoded_path
            
            # Tạo cạnh đồ thị liên kết giữa sản phẩm và danh mục cấp thấp nhất (Level 3) phục vụ cho mô hình đồ thị nếu có
            graph_edges.append((item_id, encoded_path[-1]))

    except Exception as e:
        print(f"⚠️ Cảnh báo: Không thể bóc tách file Meta ({e}). Khởi tạo taxonomy fallback mặc định.")
        # Fallback tạo nhãn phân cấp ngẫu nhiên cho các item nếu file meta lỗi để hệ thống không bị crash
        item_taxonomy = {i: [1, 1, 1] for i in range(1, len(item_encoder.classes_) + 1)}
        graph_edges = [(i, 1) for i in range(1, len(item_encoder.classes_) + 1)]

    # Lưu trữ tất cả tài nguyên đã tiền xử lý xuống đĩa cứng dưới dạng file Pickle (.pkl)
    print("Step 4: Đang xuất dữ liệu ra file Pickle phục vụ huấn luyện...")
    save_pickle(train_data, processed_dir / "train.pkl")
    save_pickle(val_data, processed_dir / "val.pkl")
    save_pickle(test_data, processed_dir / "test.pkl")
    save_pickle(user_encoder, processed_dir / "user_encoder.pkl")
    save_pickle(item_encoder, processed_dir / "item_encoder.pkl")
    save_pickle(item_taxonomy, processed_dir / "item_taxonomy.pkl")
    save_pickle(graph_edges, processed_dir / "graph_edges.pkl")
