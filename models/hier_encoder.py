# models/hier_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCategoryEncoder(nn.Module):
    def __init__(self, num_categories_per_level, embedding_dim, max_depth=3):
        """
        Khối mã hóa phân cấp danh mục Taxonomy theo thiết kế đồ án HierGNN 2026.
        Args:
            num_categories_per_level (list): Số lượng danh mục của từng cấp, ví dụ: [num_L1, num_L2, num_L3]
            embedding_dim (int): Kích thước vector nhúng d (đồng bộ với embedding_dim của Item)
            max_depth (int): Độ sâu phân cấp tối ưu theo thực nghiệm đồ án (thường = 3)
        """
        super().__init__()
        self.max_depth = max_depth
        
        # Tạo không gian nhúng riêng biệt cho từng cấp độ danh mục (L1 -> L3)
        self.category_embeddings = nn.ModuleList([
            nn.Embedding(num_cats, embedding_dim, padding_idx=0)
            for num_cats in num_categories_per_level[:max_depth]
        ])
        
        # Mạng Attention học trọng số động của các tầng danh mục
        self.attn_weights = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, category_paths):
        """
        Args:
            category_paths: Ma trận thực thể danh mục [B, L, max_depth]
        Returns:
            hier_emb: Vector đặc trưng phân cấp [B, L, embedding_dim]
        """
        level_embs = []
        for i in range(self.max_depth):
            cat_ids = category_paths[:, :, i] # [B, L]
            cat_emb = self.category_embeddings[i](cat_ids) # [B, L, d]
            level_embs.append(cat_emb.unsqueeze(2)) # [B, L, 1, d]
            
        # Xếp chồng các tầng danh mục phân cấp: [B, L, max_depth, d]
        stacked_taxonomy = torch.cat(level_embs, dim=2)
        
        # Tính toán Attention đa tầng danh mục
        scores = self.attn_weights(stacked_taxonomy).squeeze(-1) # [B, L, max_depth]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # [B, L, max_depth, 1]
        
        # Tổng hợp thông tin đại diện phân cấp danh mục
        hier_emb = torch.sum(stacked_taxonomy * weights, dim=2) # [B, L, d]
        return hier_emb
