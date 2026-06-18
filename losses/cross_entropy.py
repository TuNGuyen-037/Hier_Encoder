# losses/cross_entropy.py
import torch.nn as nn


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        logits: [B, num_items]
        targets: [B] or [B, 1]
        """
        if targets.dim() > 1:
            targets = targets.squeeze(-1)

        return self.loss_fn(logits, targets)
