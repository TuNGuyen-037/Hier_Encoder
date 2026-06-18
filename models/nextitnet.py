# models/nextitnet.py
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


class NextItNet(nn.Module):
    def __init__(self, num_items):
        super().__init__()

        cfg = load_config("nextitnet")

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

    def forward(self, seq):
        """
        seq: [B, L]
        """

        x = self.item_embedding(seq)

        x = x.transpose(1, 2)

        x = self.network(x)

        x = x.transpose(1, 2)

        hidden = x[:, -1, :]

        logits = self.output(hidden)

        return logits
