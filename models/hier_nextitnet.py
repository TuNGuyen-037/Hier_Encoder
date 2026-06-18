# models/hier_nextitnet.py
import torch
import torch.nn as nn

from utils import load_config


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def crop(self, x, target_len):
        return x[:, :, -target_len:]

    def forward(self, x):
        residual = x
        target_len = x.size(-1)

        out = self.conv1(x)
        out = self.crop(out, target_len)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.crop(out, target_len)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual


class HierNextItNet(nn.Module):
    def __init__(self, num_items):
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

        short_blocks = []
        long_blocks = []

        for dilation in dilations:
            short_blocks.append(
                ResidualBlock(
                    embedding_dim,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )

        for _ in range(n_blocks):
            for dilation in dilations:
                long_blocks.append(
                    ResidualBlock(
                        embedding_dim,
                        kernel_size,
                        dilation,
                        dropout,
                    )
                )

        self.short_network = nn.Sequential(*short_blocks)
        self.long_network = nn.Sequential(*long_blocks)

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1),
        )

        self.gate = nn.Linear(
            embedding_dim * 2,
            embedding_dim,
        )

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
        short_input = short_input.transpose(1, 2)

        short_out = self.short_network(short_input)
        short_out = short_out.transpose(1, 2)

        short_hidden = short_out[:, -1, :]

        long_input = x.transpose(1, 2)

        long_out = self.long_network(long_input)
        long_out = long_out.transpose(1, 2)

        long_hidden = self.attention_pool(long_out)

        fusion = torch.cat(
            [short_hidden, long_hidden],
            dim=-1,
        )

        gate = torch.sigmoid(self.gate(fusion))

        hidden = gate * short_hidden + (1.0 - gate) * long_hidden

        logits = self.output(hidden)

        return logits
