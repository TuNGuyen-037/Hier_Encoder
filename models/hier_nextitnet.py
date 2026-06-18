# models/hier_nextitnet.py
import torch
import torch.nn as nn

from utils import load_config
from .nextitnet import ResidualBlock
from .hier_encoder import HierarchicalCategoryEncoder


class HierNextItNet(nn.Module):
    def __init__(self, num_items, num_categories_per_level):
        super().__init__()

        cfg = load_config("hier_nextitnet")

        fairness = cfg["fairness"]
        model_cfg = cfg["model"]

        embedding_dim = fairness["embedding_dim"]

        kernel_size = model_cfg["kernel_size"]
        dilations = model_cfg["dilations"]
        n_blocks = model_cfg["n_blocks"]
        dropout = model_cfg["dropout"]

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
        
        # Khối Encoder Phân cấp Danh mục
        self.hier_encoder = HierarchicalCategoryEncoder(
            num_categories_per_level=num_categories_per_level,
            embedding_dim=embedding_dim,
            max_depth=3
        )
        
        # Khối Fusion Gate
        self.fusion_gate = nn.Linear(embedding_dim * 2, embedding_dim)

        blocks = []
        for _ in range(n_blocks):
            for dilation in dilations:
                blocks.append(
                    ResidualBlock(
                        embedding_dim,
                        kernel_size,
                        dilation,
                        dropout,
                    )
                )

        self.network = nn.Sequential(*blocks)

        self.output = nn.Linear(
            embedding_dim,
            num_items,
        )

    def forward(self, seq, time_seq, category_paths):
        """
        seq: [B, L]
        time_seq: [B, L]
        category_paths: [B, L, max_depth]
        """
        item_emb = self.item_embedding(seq)
        time_emb = self.time_embedding(time_seq)
        
        # Trích xuất biểu diễn đặc trưng phân cấp danh mục
        hier_emb = self.hier_encoder(category_paths)

        base_features = item_emb + time_emb
        
        # Khối Fusion Module tích hợp đặc trưng phân cấp danh mục cây
        combined = torch.cat([base_features, hier_emb], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        x = gate * base_features + (1.0 - gate) * hier_emb

        # Chuyển đổi chiều để đưa vào Conv1D mạng NextItNet [B, d, L]
        x = x.transpose(1, 2)

        x = self.network(x)

        x = x.transpose(1, 2)

        hidden = x[:, -1, :]

        logits = self.output(hidden)

        return logits
