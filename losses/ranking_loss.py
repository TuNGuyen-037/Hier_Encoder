# losses/ranking_loss.py
import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        Đồng bộ hóa interface với SequenceCrossEntropyLoss để Trainer gọi mượt mà.
        logits: [B, num_items] - Điểm số dự đoán cho toàn bộ các item trong hệ thống
        targets: [B] hoặc [B, 1] - ID của các Item tích cực (Ground Truth)
        """
        if targets.dim() > 1:
            targets = targets.squeeze(-1) # Đảm bảo kích thước [B]

        batch_size, num_items = logits.size()

        # 1. Trích xuất điểm số của các item tích cực (Positive Scores)
        # Sử dụng gather để lấy logits tại vị trí của targets tương ứng
        pos_scores = logits.gather(dim=1, index=targets.unsqueeze(1)).squeeze(-1) # [B]

        # 2. Lấy mẫu ngẫu nhiên các item tiêu cực (Negative Sampling)
        # Tạo ngẫu nhiên các ID từ 1 đến num_items - 1 (tránh padding_idx = 0 nếu có)
        neg_items = torch.randint(
            low=1, 
            high=num_items, 
            size=(batch_size,), 
            device=logits.device
        )

        # Để đảm bảo tính chính xác, nếu neg_item tình cờ trùng với target, ta dịch chuyển nó đi 1 đơn vị
        # (Phép toán vectorized giúp tránh dùng vòng lặp)
        collision_mask = (neg_items == targets)
        neg_items = torch.where(
            collision_mask, 
            (neg_items % (num_items - 1)) + 1, 
            neg_items
        )

        # 3. Trích xuất điểm số của các item tiêu cực (Negative Scores)
        neg_scores = logits.gather(dim=1, index=neg_items.unsqueeze(1)).squeeze(-1) # [B]

        # 4. Tính toán BPR Loss theo đúng công thức lý thuyết đồ án
        loss = -torch.log(
            torch.sigmoid(pos_scores - neg_scores) + 1e-8
        )

        return loss.mean()
