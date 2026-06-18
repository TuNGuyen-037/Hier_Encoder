# evaluation/metrics.py
import torch


def hit_rate(predictions, targets, k=10):
    """
    HR@K
    predictions: [B, K]
    targets: [B]
    """

    hits = (predictions[:, :k] == targets.unsqueeze(1)).any(dim=1)

    return hits.float().mean().item()


def ndcg(predictions, targets, k=10):
    """
    NDCG@K
    predictions: [B, K]
    targets: [B]
    """

    predictions = predictions[:, :k]

    batch_size = predictions.size(0)

    score = 0.0

    for i in range(batch_size):
        target = targets[i]

        pred = predictions[i]

        match = (pred == target).nonzero(as_tuple=False)

        if len(match) > 0:
            rank = match[0].item()
            score += 1.0 / torch.log2(
                torch.tensor(rank + 2.0)
            ).item()

    return score / batch_size
