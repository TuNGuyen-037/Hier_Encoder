# datasets/amz2023.py
import torch
from torch.utils.data import Dataset

from utils import load_config
from datasets.preprocess import (
    read_jsonl_gz,
    kcore_filter,
    encode_ids,
    build_sequences,
)


class SequentialDataset(Dataset):
    def __init__(self, data, max_seq_len):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def pad_sequence(self, seq):
        seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        return [0] * pad_len + seq

    def __getitem__(self, idx):
        seq, time_seq, target = self.data[idx]

        seq = self.pad_sequence(seq)
        time_seq = self.pad_sequence(time_seq)

        return (
            torch.LongTensor(seq),
            torch.LongTensor(time_seq),
            torch.LongTensor([target]),
        )


class Amazon2023Dataset:
    def __init__(self):
        cfg = load_config()

        dataset_cfg = cfg["dataset"]
        fairness_cfg = cfg["fairness"]

        review_path = dataset_cfg["review_path"]
        max_rows = dataset_cfg["max_rows"]
        time_buckets = dataset_cfg["time_buckets"]

        k_core = fairness_cfg["k_core"]
        max_seq_len = fairness_cfg["max_seq_len"]

        self.max_seq_len = max_seq_len

        df = read_jsonl_gz(
            review_path,
            ["user_id", "parent_asin", "timestamp"],
            max_rows=max_rows,
        )

        df = df.dropna()
        df = kcore_filter(df, k=k_core)

        df, self.user_encoder, self.item_encoder = encode_ids(df)

        train_data, val_data, test_data = build_sequences(
            df,
            time_buckets,
        )

        self.train_dataset = SequentialDataset(
            train_data,
            max_seq_len,
        )

        self.val_dataset = SequentialDataset(
            list(val_data.values()),
            max_seq_len,
        )

        self.test_dataset = SequentialDataset(
            list(test_data.values()),
            max_seq_len,
        )

        self.num_items = df["item"].max() + 1
        self.num_users = df["user"].max() + 1
