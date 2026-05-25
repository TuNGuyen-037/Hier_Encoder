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

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows and idx >= max_rows:
                break

            try:
                obj = json.loads(line)
                data.append(
                    {col: obj.get(col) for col in columns}
                )
            except Exception:
                continue

    return pd.DataFrame(data)


def kcore_filter(df, k=5):
    while True:
        before = len(df)

        user_count = df["user_id"].value_counts()
        item_count = df["parent_asin"].value_counts()

        df = df[
            df["user_id"].isin(
                user_count[user_count >= k].index
            )
        ]

        df = df[
            df["parent_asin"].isin(
                item_count[item_count >= k].index
            )
        ]

        if len(df) == before:
            break

    return df


def encode_ids(df):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user"] = user_encoder.fit_transform(df["user_id"])
    df["item"] = item_encoder.fit_transform(
        df["parent_asin"]
    ) + 1

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

        time_ids = [
            time_to_bucket(d, time_buckets) + 1
            for d in delta
        ]

        test_data[user_id] = (
            items[:-1],
            time_ids[:-1],
            items[-1],
        )

        val_data[user_id] = (
            items[:-2],
            time_ids[:-2],
            items[-2],
        )

        train_seq = items[:-2]
        train_time = time_ids[:-2]

        for i in range(1, len(train_seq)):
            train_data.append(
                (
                    train_seq[:i],
                    train_time[:i],
                    train_seq[i],
                )
            )

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

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_path = raw_dir / dataset_cfg["review_file"]

    max_rows = dataset_cfg["max_rows"]
    time_buckets = dataset_cfg["time_buckets"]

    k_core = fairness_cfg["k_core"]

    df = read_jsonl_gz(
        review_path,
        ["user_id", "parent_asin", "timestamp"],
        max_rows=max_rows,
    )

    df = df.dropna()
    df = kcore_filter(df, k=k_core)

    df, user_encoder, item_encoder = encode_ids(df)

    train_data, val_data, test_data = build_sequences(
        df,
        time_buckets,
    )

    save_pickle(train_data, processed_dir / "train.pkl")
    save_pickle(val_data, processed_dir / "val.pkl")
    save_pickle(test_data, processed_dir / "test.pkl")

    save_pickle(
        user_encoder,
        processed_dir / "user_encoder.pkl",
    )

    save_pickle(
        item_encoder,
        processed_dir / "item_encoder.pkl",
    )
