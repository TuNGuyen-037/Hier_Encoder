# evaluation/evakuator.py
import torch

from evaluation.metrics import hit_rate, ndcg


HIERARCHICAL_MODELS = {
    "hier_gru",
    "hier_sasrec",
    "hier_nextitnet",
}


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    model_name,
    top_k=None,
):
    if top_k is None:
        top_k = [5, 10, 20]

    model.eval()

    hierarchical = model_name in HIERARCHICAL_MODELS

    metric_results = {}

    for k in top_k:
        metric_results[f"hr@{k}"] = []
        metric_results[f"ndcg@{k}"] = []

    for batch in dataloader:
        seq, time_seq, target = batch

        seq = seq.to(device)
        time_seq = time_seq.to(device)
        target = target.squeeze(-1).to(device)

        if hierarchical:
            logits = model(seq, time_seq)
        else:
            logits = model(seq)

        _, predictions = torch.topk(
            logits,
            k=max(top_k),
            dim=1,
        )

        for k in top_k:
            hr = hit_rate(predictions, target, k)
            ndcg_score = ndcg(predictions, target, k)

            metric_results[f"hr@{k}"].append(hr)
            metric_results[f"ndcg@{k}"].append(ndcg_score)

    final_results = {}

    for key, values in metric_results.items():
        final_results[key] = sum(values) / len(values)

    return final_results
