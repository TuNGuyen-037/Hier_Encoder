# evaluation/evaluator.py
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
    hierarchical=None,  # Nhận cờ truyền từ trainer
):
    if top_k is None:
        top_k = [5, 10, 20]

    model.eval()

    # Nếu trainer không truyền vào thì tự động check theo registry
    if hierarchical is None:
        hierarchical = model_name in HIERARCHICAL_MODELS

    metric_results = {}

    for k in top_k:
        metric_results[f"hr@{k}"] = []
        metric_results[f"ndcg@{k}"] = []

    for batch in dataloader:
        # ĐỒNG BỘ UNPACK BATCH: Kiểm tra số lượng phần tử trả ra tương thích với mô hình phân cấp
        if hierarchical and len(batch) == 4:
            seq, time_seq, category_paths, target = batch
            category_paths = category_paths.to(device)
        else:
            if len(batch) == 4:
                seq, time_seq, _, target = batch
            else:
                seq, time_seq, target = batch
            category_paths = None

        seq = seq.to(device)
        time_seq = time_seq.to(device)
        target = target.squeeze(-1).to(device)

        # Đẩy ma trận phân cấp danh mục vào hàm forward pass nếu là mô hình lai
        if hierarchical:
            logits = model(seq, time_seq, category_paths)
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
