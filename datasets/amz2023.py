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
    def __init__(self, data, max_seq_len, item_taxonomy=None):
        self.data = data
        self.max_seq_len = max_seq_len
        self.item_taxonomy = item_taxonomy

    def __len__(self):
        return len(self.data)

    def pad_sequence(self, seq, pad_value=0):
        seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        return [pad_value] * pad_len + seq

    def pad_category_paths(self, paths):
        # paths có dạng list of lists: [seq_len, num_levels]
        paths = paths[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(paths)
        
        # Nếu không có path nào (bị pad), gán nhãn gốc là 0 cho tất cả các level phân cấp
        num_levels = len(paths[0]) if len(paths) > 0 else 1
        pad_path = [0] * num_levels
        
        return [pad_path] * pad_len + paths

    def __getitem__(self, idx):
        # Hỗ trợ cả định dạng dữ liệu Train (list) và Val/Test (dict.values)
        if isinstance(self.data, list):
            seq, time_seq, target = self.data[idx]
        else:
            seq, time_seq, target = self.data[idx]

        # Thực hiện ép độ dài chuỗi (Padding / Truncating)
        padded_seq = self.pad_sequence(seq, pad_value=0)
        padded_time = self.pad_sequence(time_seq, pad_value=0)

        # ĐỒNG BỘ PHÂN CẤP: Nếu có từ điển taxonomy, sinh ra ma trận category levels tương ứng với item sequence
        if self.item_taxonomy is not None:
            category_paths = []
            for item_id in padded_seq:
                if item_id == 0: # Vị trí bị padded
                    # Giả định hệ thống phân cấp có số tầng tương ứng với độ dài vector phân cấp trong taxonomy
                    num_levels = len(list(self.item_taxonomy.values())[0]) if self.item_taxonomy else 3
                    category_paths.append([0] * num_levels)
                else:
                    # Lấy đường dẫn phân cấp (Ví dụ: [Cấp_1_ID, Cấp_2_ID, Cấp_3_ID]) của item từ taxonomy
                    # Sử dụng phương thức .get() đề phòng itemencoder không khớp
                    category_paths.append(self.item_taxonomy.get(item_id, [0, 0, 0]))
            
            return (
                torch.LongTensor(padded_seq),
                torch.LongTensor(padded_time),
                torch.LongTensor(category_paths), # Trả thêm phần tử thứ 4 cho mô hình Hierarchical
                torch.LongTensor([target]),
            )

        # Fallback cho chế độ chạy baseline thông thường (3 phần tử)
        return (
            torch.LongTensor(padded_seq),
            torch.LongTensor(padded_time),
            torch.LongTensor([target]),
        )


class Amazon2023Dataset:
    def __init__(self):
        cfg = load_config()

        dataset_cfg = cfg["dataset"]
        fairness_cfg = cfg["fairness"]

        processed_dir = Path(dataset_cfg["processed_dir"])
        train_path = processed_dir / "train.pkl"

        if not train_path.exists():
            preprocess_and_save()

        train_data = load_pickle(processed_dir / "train.pkl")
        val_data = load_pickle(processed_dir / "val.pkl")
        test_data = load_pickle(processed_dir / "test.pkl")

        self.user_encoder = load_pickle(processed_dir / "user_encoder.pkl")
        self.item_encoder = load_pickle(processed_dir / "item_encoder.pkl")
        self.graph_edges = load_pickle(processed_dir / "graph_edges.pkl")
        self.item_taxonomy = load_pickle(processed_dir / "item_taxonomy.pkl")
        
        max_seq_len = fairness_cfg["max_seq_len"]

        # ĐỒNG BỘ BIẾN PHÂN CẤP: Tính số lượng thực thể của từng tầng phân cấp (Category Levels)
        # item_taxonomy định dạng: { item_encoded_id: [cat_level_1_id, cat_level_2_id, ...] }
        if self.item_taxonomy:
            import numpy as np
            taxonomy_matrix = np.array(list(self.item_taxonomy.values()))
            # Thêm 1 vào max ID của mỗi cột vì nhãn ID chạy từ 1 (0 dành cho padding)
            self.num_categories_per_level = (taxonomy_matrix.max(axis=0) + 1).tolist()
        else:
            # Fallback an toàn nếu file taxonomy trống
            self.num_categories_per_level = [100, 50, 20] 

        # Đẩy từ điển Taxonomy vào Dataset để bóc tách động trong __getitem__
        self.train_dataset = SequentialDataset(
            train_data,
            max_seq_len,
            item_taxonomy=self.item_taxonomy
        )

        self.val_dataset = SequentialDataset(
            list(val_data.values()),
            max_seq_len,
            item_taxonomy=self.item_taxonomy
        )

        self.test_dataset = SequentialDataset(
            list(test_data.values()),
            max_seq_len,
            item_taxonomy=self.item_taxonomy
        )

        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_) + 1
