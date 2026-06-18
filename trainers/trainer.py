# trainers/trainer.py
import torch
from tqdm import tqdm

from utils import load_config, get_device, save_checkpoint
from models import MODEL_REGISTRY
from losses import SequenceCrossEntropyLoss
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

        # ĐỒNG BỘ KHỞI TẠO: Kiểm tra nếu là mô hình Hierarchical thì truyền thêm danh sách category levels từ dataset
        if self.hierarchical:
            self.model = MODEL_REGISTRY[model_name](
                num_items=dataset.num_items,
                num_categories_per_level=dataset.num_categories_per_level
            ).to(self.device)
        else:
            self.model = MODEL_REGISTRY[model_name](
                num_items=dataset.num_items
            ).to(self.device)

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

    # ĐỒNG BỘ ROUTING FORWARD: Đón nhận thêm tham số category_paths cho khối Hierarchical Category Encoder
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
        )

        for batch in progress_bar:
            # ĐỒNG BỘ UNPACK BATCH: Kiểm tra số lượng phần tử trả ra từ DataLoader để bóc tách category_paths
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

            # Đẩy ma trận phân cấp vào luồng forward pass
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

    # ĐỒNG BỘ ĐÁNH GIÁ (VALIDATION): Đảm bảo module evaluate_model nhận diện cấu trúc forward mới
    def validate(self):
        return evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            model_name=self.model_name,
            top_k=self.top_k,
            hierarchical=self.hierarchical # Truyền cờ này để hàm evaluate bóc tách batch tương tự lúc train
        )

    def check_early_stopping(self, results):
        metric_value = results[
            self.early_stopping_metric
        ]

        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.patience_counter = 0

            save_checkpoint(
                self.model,
                self.best_model_path,
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
        print(f"\nTraining model: {self.model_name}")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch()

            results = self.validate()

            print(
                f"Epoch {epoch} | "
                f"Loss: {train_loss:.4f} | "
                f"Metrics: {results}"
            )

            if self.check_early_stopping(results):
                print(
                    f"Early stopping triggered for "
                    f"{self.model_name}"
                )
                break

        print(
            f"Finished training {self.model_name}"
        )
