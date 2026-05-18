import torch
import torch.nn as nn

from utils import load_config


class GRU4Rec(nn.Module):
    def __init__(self, num_items):
        super().__init__()

        cfg = load_config("gru4rec")

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

    def forward(self, seq):
        """
        seq: [B, L]
        """

        x = self.item_embedding(seq)

        _, hidden = self.gru(x)

        hidden = hidden[-1]

        hidden = self.dropout(hidden)

        logits = self.output(hidden)

        return logits
