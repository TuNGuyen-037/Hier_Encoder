# evaluation/metrics.py
import torch


def hit_rate(predictions, targets, k=10):
    """
    HR@K (Vectorized Version)
    predictions: [B, Max_K]
    targets: [B]
    """
    # Lấy top K phần tử dự đoán
    topk_preds = predictions[:, :k] # [B, K]
    
    # Kiểm tra xem target có nằm trong top K hay không
    hits = (topk_preds == targets.unsqueeze(1)).any(dim=1) # [B]

    return hits.float().mean().item()


def ndcg(predictions, targets, k=10):
    """
    NDCG@K (Vectorized Version - Tối ưu hóa bỏ vòng lặp Python)
    predictions: [B, Max_K]
    targets: [B]
    """
    topk_preds = predictions[:, :k] # [B, K]
    
    # Tạo ma trận so sánh boolean kích thước [B, K]
    matches = (topk_preds == targets.unsqueeze(1)) # [B, K]
    
    # Lấy ra chỉ số rank (0 đến K-1) cho các vị trí trúng thưởng (hit)
    # Nếu không hit, giá trị mặc định sẽ không ảnh hưởng đến tổng điểm nhờ phép nhân nhân tử phía sau
    ranks = torch.arange(k, device=predictions.device).unsqueeze(0).expand_as(matches) # [B, K]
    
    # Công thức tính Discounted Gain: 1 / log2(rank + 2)
    discount = 1.0 / torch.log2(ranks.float() + 2.0) # [B, K]
    
    # Chỉ tính discount tại những vị trí phần tử khớp target (matches == True)
    idcg_scores = (matches.float() * discount).sum(dim=1) # [B]
    
    # Do trong bài toán Sequential Recommendation với 1 Target duy nhất (Leave-one-out), 
    # giá trị Ideal DCG (IDCG) luôn luôn bằng 1 / log2(0 + 2) = 1.0
    # Vì vậy NDCG chính bằng giá trị DCG thu được.
    return idcg_scores.mean().item()
