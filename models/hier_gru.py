# models/hier_gru.py
import torch
import torch.nn as nn

from utils import load_config


class HierGRU(nn.Module):
    def __init__(self, num_items):
        super().__init__()

        cfg = load_config("hier_gru")

        fairness = cfg["fairness"]
        model_cfg = cfg["model"]

        embedding_dim = fairness["embedding_dim"]
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

        self.short_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

        self.long_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1),
        )

        self.gate = nn.Linear(
            embedding_dim * 2,
            embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            embedding_dim,
            num_items,
        )

    def attention_pool(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * x).sum(dim=1)
        return pooled

    def forward(self, seq, time_seq):
        """
        seq: [B, L]
        time_seq: [B, L]
        """

        item_emb = self.item_embedding(seq)
        time_emb = self.time_embedding(time_seq)

        x = item_emb + time_emb

        short_input = x[:, -10:, :]

        _, short_hidden = self.short_gru(short_input)
        short_hidden = short_hidden[-1]

        long_output, _ = self.long_gru(x)
        long_hidden = self.attention_pool(long_output)

        fusion = torch.cat(
            [short_hidden, long_hidden],
            dim=-1,
        )

        gate = torch.sigmoid(self.gate(fusion))

        hidden = gate * short_hidden + (1.0 - gate) * long_hidden

        hidden = self.dropout(hidden)

        logits = self.output(hidden)

        return logits
