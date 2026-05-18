import gzip
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_jsonl_gz(path, columns, max_rows=None):
    """
    Read compressed Amazon JSONL.GZ file.
    """
    data = []

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
    """
    Iterative k-core filtering.
    """
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
    """
    Encode user/item IDs.
    """
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user"] = user_encoder.fit_transform(df["user_id"])
    df["item"] = item_encoder.fit_transform(df["parent_asin"]) + 1

    return df, user_encoder, item_encoder


def time_to_bucket(delta, buckets):
    """
    Map time delta to bucket.
    """
    for idx, threshold in enumerate(buckets):
        if delta < threshold:
            return idx

    return len(buckets)


def build_sequences(df, time_buckets):
    """
    Build train / val / test sequential recommendation data.
    """
    train_data = []
    val_data = {}
    test_data = {}

    df = df.sort_values(["user", "timestamp"])

    for user_id, df_user in df.groupby("user"):
        items = df_user["item"].tolist()
        times = df_user["timestamp"].tolist()

        if len(items) < 3:
            continue

        delta = [0] + [
            times[i] - times[i - 1]
            for i in range(1, len(times))
        ]

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
