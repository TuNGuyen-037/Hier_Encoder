# models/sasrec.py
import torch
import torch.nn as nn

from utils import load_config


class SASRec(nn.Module):
    def __init__(self, num_items):
        super().__init__()

        cfg = load_config("sasrec")

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

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim,
        )

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

    def forward(self, seq):
        """
        seq: [B, L]
        """

        batch_size, seq_len = seq.size()

        positions = torch.arange(
            seq_len,
            device=seq.device,
        ).unsqueeze(0).expand(batch_size, -1)

        item_emb = self.item_embedding(seq)
        pos_emb = self.position_embedding(positions)

        x = item_emb + pos_emb
        x = self.dropout(x)

        mask = self.causal_mask(seq_len, seq.device)

        x = self.encoder(
            x,
            mask=mask,
        )

        hidden = x[:, -1, :]

        logits = self.output(hidden)

        return logits
