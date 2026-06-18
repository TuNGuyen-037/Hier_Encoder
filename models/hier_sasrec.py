# models/hier_sasrec.py
import torch
import torch.nn as nn

from utils import load_config
from .hier_encoder import HierarchicalCategoryEncoder


class HierSASRec(nn.Module):
    def __init__(self, num_items, num_categories_per_level):
        super().__init__()

        cfg = load_config("hier_sasrec")

        fairness = cfg["fairness"]
        model_cfg = cfg["model"]

        embedding_dim = fairness["embedding_dim"]
        max_seq_len = fairness["max_seq_len"]

        n_heads = model_cfg["n_heads"]
        n_layers = model_cfg["n_layers"]
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

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim,
        )
        
        # Khối Encoder Phân cấp Danh mục
        self.hier_encoder = HierarchicalCategoryEncoder(
            num_categories_per_level=num_categories_per_level,
            embedding_dim=embedding_dim,
            max_depth=3
        )
        
        # Khối Fusion Gate
        self.fusion_gate = nn.Linear(embedding_dim * 2, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            embedding_dim,
            num_items,
        )

    def causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        )
        return mask.bool()

    def forward(self, seq, time_seq, category_paths):
        """
        seq: [B, L]
        time_seq: [B, L]
        category_paths: [B, L, max_depth]
        """
        batch_size, seq_len = seq.size()

        positions = torch.arange(
            seq_len,
            device=seq.device,
        ).unsqueeze(0).expand(batch_size, -1)

        item_emb = self.item_embedding(seq)
        time_emb = self.time_embedding(time_seq)
        pos_emb = self.position_embedding(positions)
        
        # Trích xuất biểu diễn đặc trưng cây phân cấp
        hier_emb = self.hier_encoder(category_paths)

        base_features = item_emb + time_emb + pos_emb
        base_features = self.dropout(base_features)

        # Khối Fusion Module tích hợp thông tin danh mục phân cấp trước Transformer Encoder
        combined = torch.cat([base_features, hier_emb], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        x = gate * base_features + (1.0 - gate) * hier_emb

        mask = self.causal_mask(seq_len, seq.device)

        x = self.encoder(
            x,
            mask=mask,
        )

        hidden = x[:, -1, :]
        logits = self.output(hidden)

        return logits
