import yaml
import torch
from torch.utils.data import Dataset

from datasets.preprocess import (
    read_jsonl_gz,
    kcore_filter,
    encode_ids,
    build_sequences,
)


class Amazon2023Dataset(Dataset):
    def __init__(self, config_path="configs/base.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        with open("configs/fairness.yaml", "r", encoding="utf-8") as f:
            fairness = yaml.safe_load(f)["fairness"]

        dataset_cfg = cfg["dataset"]

        review_path = dataset_cfg["review_path"]
        max_rows = dataset_cfg["max_rows"]

        k_core = fairness["k_core"]
        max_seq_len = fairness["max_seq_len"]
        time_buckets = dataset_cfg["time_buckets"]

        self.max_seq_len = max_seq_len

        df = read_jsonl_gz(
            review_path,
            ["user_id", "parent_asin", "timestamp"],
            max_rows=max_rows,
        )

        df = df.dropna()
        df = kcore_filter(df, k=k_core)

        df, self.user_encoder, self.item_encoder = encode_ids(df)

        self.train_data, self.val_data, self.test_data = build_sequences(
            df,
            time_buckets,
        )

        self.num_items = df["item"].max() + 1

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        seq, time_seq, target = self.train_data[idx]

        seq = seq[-self.max_seq_len:]
        time_seq = time_seq[-self.max_seq_len:]

        pad_len = self.max_seq_len - len(seq)

        seq = [0] * pad_len + seq
        time_seq = [0] * pad_len + time_seq

        return (
            torch.LongTensor(seq),
            torch.LongTensor(time_seq),
            torch.LongTensor([target]),
        )
