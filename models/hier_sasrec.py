# models/hier_sasrec.py
import torch
import torch.nn as nn

from utils import load_config


class HierSASRec(nn.Module):
    def __init__(self, num_items):
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.short_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.long_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
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

    def causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        )
        return mask.bool()

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

        batch_size, seq_len = seq.size()

        positions = torch.arange(
            seq_len,
            device=seq.device,
        ).unsqueeze(0).expand(batch_size, -1)

        item_emb = self.item_embedding(seq)
        time_emb = self.time_embedding(time_seq)
        pos_emb = self.position_embedding(positions)

        x = item_emb + time_emb + pos_emb
        x = self.dropout(x)

        mask = self.causal_mask(seq_len, seq.device)

        short_input = x[:, -10:, :]
        short_mask = self.causal_mask(short_input.size(1), seq.device)

        short_out = self.short_encoder(
            short_input,
            mask=short_mask,
        )

        short_hidden = short_out[:, -1, :]

        long_out = self.long_encoder(
            x,
            mask=mask,
        )

        long_hidden = self.attention_pool(long_out)

        fusion = torch.cat(
            [short_hidden, long_hidden],
            dim=-1,
        )

        gate = torch.sigmoid(self.gate(fusion))

        hidden = gate * short_hidden + (1.0 - gate) * long_hidden

        logits = self.output(hidden)

        return logits
