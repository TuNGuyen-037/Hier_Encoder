import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        """
        pos_scores: [B]
        neg_scores: [B]
        """

        loss = -torch.log(
            torch.sigmoid(pos_scores - neg_scores) + 1e-8
        )

        return loss.mean()
