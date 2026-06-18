# trainers/trainer.py
import torch
from tqdm import tqdm

from utils import load_config, get_device, save_checkpoint
from models import MODEL_REGISTRY
from losses import SequenceCrossEntropyLoss, BPRLoss
from evaluation import evaluate_model
from results import save_result


HIERARCHICAL_MODELS = {
    "hier_gru",
    "hier_sasrec",
    "hier_nextitnet",
}


class Trainer:
    def __init__(
        self,
        model_name,
        dataset,
        train_loader,
        val_loader,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cfg = load_config(model_name)

        fairness = self.cfg["fairness"]
        train_cfg = self.cfg["train"]
        logging_cfg = self.cfg["logging"]

        self.device = get_device(train_cfg["device"])

        self.hierarchical = (
            model_name in HIERARCHICAL_MODELS
        )

        if self.hierarchical:
            self.model = MODEL_REGISTRY[model_name](
                num_items=dataset.num_items,
                num_categories_per_level=dataset.num_categories_per_level
            ).to(self.device)
        else:
            self.model = MODEL_REGISTRY[model_name](
                num_items=dataset.num_items
            ).to(self.device)

        loss_type = train_cfg.get("loss_type", "ce").lower()
        if loss_type == "bpr":
            self.loss_fn = BPRLoss()
        else:
            self.loss_fn = SequenceCrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=fairness["learning_rate"],
            weight_decay=fairness["weight_decay"],
        )

        self.epochs = fairness["epochs"]
        self.grad_clip = fairness["grad_clip"]
        self.top_k = fairness["top_k"]

        self.early_stopping_patience = fairness[
            "early_stopping_patience"
        ]

        self.early_stopping_metric = fairness[
            "early_stopping_metric"
        ]

        self.best_metric = float("-inf")
        self.patience_counter = 0

        self.best_model_path = (
            logging_cfg["checkpoint_dir"]
            + f"/best_{model_name}.pt"
        )

    def forward_pass(self, seq, time_seq, category_paths=None):
        if self.hierarchical:
            return self.model(seq, time_seq, category_paths)

        return self.model(seq)

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training {self.model_name}",
            leave=False
        )

        for batch in progress_bar:
            if self.hierarchical and len(batch) == 4:
                seq, time_seq, category_paths, target = batch
                category_paths = category_paths.to(self.device)
            else:
                seq, time_seq, target = batch
                category_paths = None

            seq = seq.to(self.device)
            time_seq = time_seq.to(self.device)
            target = target.squeeze(-1).to(self.device)

            self.optimizer.zero_grad()

            logits = self.forward_pass(seq, time_seq, category_paths)

            loss = self.loss_fn(logits, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )

            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return total_loss / len(self.train_loader)

    def validate(self):
        return evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            model_name=self.model_name,
            top_k=self.top_k,
            hierarchical=self.hierarchical
        )

    # ĐÃ VÁ LỖI: Nhận thêm biến `epoch` từ vòng lặp để truyền vào save_checkpoint một cách hợp lệ
    def check_early_stopping(self, results, epoch):
        metric_value = results[
            self.early_stopping_metric
        ]

        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.patience_counter = 0

            # Biến epoch và self.best_metric giờ đã được định nghĩa tường minh để lưu trữ dữ liệu
            save_checkpoint(
                self.model,
                self.best_model_path,
                optimizer=self.optimizer,
                epoch=epoch,
                best_metric=self.best_metric
            )

            save_result(
                self.model_name,
                results,
            )

            return False

        self.patience_counter += 1

        return (
            self.patience_counter
            >= self.early_stopping_patience
        )

    def fit(self):
        print(f"\n--- Khởi chạy Huấn luyện Mô hình: {self.model_name} ---")
        print(f"Hàm mục tiêu (Loss Function): {self.loss_fn.__class__.__name__}")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch()

            results = self.validate()

            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in results.items()])
            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Metrics -> {metrics_str}"
            )

            # ĐÃ VÁ LỖI: Truyền thêm giá trị epoch hiện tại vào hàm kiểm tra
            if self.check_early_stopping(results, epoch):
                print(
                    f"[Early Stopping] Kích hoạt tại Epoch {epoch}. "
                    f"Kết quả {self.early_stopping_metric} tốt nhất: {self.best_metric:.4f}"
                )
                break

        print(
            f"Hoàn thành Benchmark cho mô hình: {self.model_name}\n"
        )
