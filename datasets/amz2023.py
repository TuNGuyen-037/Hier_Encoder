# datasets/amz2023.py
import torch
from pathlib import Path
from torch.utils.data import Dataset

from utils import load_config
from datasets.preprocess import (
    load_pickle,
    preprocess_and_save,
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

        processed_dir = Path(
            dataset_cfg["processed_dir"]
        )

        train_path = processed_dir / "train.pkl"

        if not train_path.exists():
            preprocess_and_save()

        train_data = load_pickle(
            processed_dir / "train.pkl"
        )

        val_data = load_pickle(
            processed_dir / "val.pkl"
        )

        test_data = load_pickle(
            processed_dir / "test.pkl"
        )

        self.user_encoder = load_pickle(
            processed_dir / "user_encoder.pkl"
        )

        self.item_encoder = load_pickle(
            processed_dir / "item_encoder.pkl"
        )

        self.graph_edges = load_pickle(
            processed_dir / "graph_edges.pkl"
        )
        
        self.item_taxonomy = load_pickle(
            processed_dir / "item_taxonomy.pkl"
        )

        
        max_seq_len = fairness_cfg["max_seq_len"]

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

        self.num_users = len(
            self.user_encoder.classes_
        )

        self.num_items = (
            len(self.item_encoder.classes_) + 1
        )
