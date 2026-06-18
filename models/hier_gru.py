# models/hier_gru.py
import torch
import torch.nn as nn

from utils import load_config
from .hier_encoder import HierarchicalCategoryEncoder


class HierGRU(nn.Module):
    def __init__(self, num_items, num_categories_per_level):
        super().__init__()

        cfg = load_config("hier_gru")

        fairness = cfg["fairness"]
        model_cfg = cfg["model"]

        embedding_dim = fairness["embedding_dim"]
        n_layers = model_cfg["n_layers"]
        dropout = model_cfg["dropout"]
        
        # Nhúng Item gốc
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0,
        )

        self.time_embedding = nn.Embedding(
            16,
            embedding_dim,
            padding_idx=0,
        )
        
        # Khối Encoder Phân cấp Danh mục theo đúng báo cáo ĐATN
        self.hier_encoder = HierarchicalCategoryEncoder(
            num_categories_per_level=num_categories_per_level,
            embedding_dim=embedding_dim,
            max_depth=3
        )

        # Khối Fusion Module (Gating Mechanism) kết hợp Item-Time và Hierarchical Category
        self.fusion_gate = nn.Linear(embedding_dim * 2, embedding_dim)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            embedding_dim,
            num_items,
        )

    def forward(self, seq, time_seq, category_paths):
        """
        seq: [B, L] -> Item Sequence
        time_seq: [B, L] -> Time Step Sequence
        category_paths: [B, L, max_depth] -> Danh mục phân cấp tương ứng của từng Item trong chuỗi
        """
        item_emb = self.item_embedding(seq)
        time_emb = self.time_embedding(time_seq)
        
        # 1. Trích xuất thuộc tính phân cấp danh mục từ cây Taxonomy
        hier_emb = self.hier_encoder(category_paths)

        # Base item representation
        base_features = item_emb + time_emb

        # 2. Định hình khối Fusion Module trong cấu trúc HierGNN
        combined = torch.cat([base_features, hier_emb], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        x = gate * base_features + (1.0 - gate) * hier_emb

        # Forward qua Sequential Backbone (GRU)
        _, hidden = self.gru(x)
        hidden = hidden[-1]
        hidden = self.dropout(hidden)

        logits = self.output(hidden)
        return logits
