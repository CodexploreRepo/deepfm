"""Training loop with early stopping, checkpointing, and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepfm.config import ExperimentConfig
from deepfm.data.dataset import TabularDataset
from deepfm.data.schema import DatasetSchema
from deepfm.models.base import BaseCTRModel
from deepfm.training.metrics import (
    RankingEvaluator,
    compute_auc,
    compute_logloss,
)
from deepfm.utils import get_logger, save_checkpoint


class Trainer:
    """Trains a CTR model with early stopping and ranking evaluation.

    Args:
        model: A BaseCTRModel instance.
        schema: Dataset schema.
        config: Full experiment configuration.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        test_ds: Test dataset.
        adapter: Optional dataset adapter for dynamic negative re-sampling.
        device: Torch device string.
    """

    def __init__(
        self,
        model: BaseCTRModel,
        schema: DatasetSchema,
        config: ExperimentConfig,
        train_ds: TabularDataset,
        val_ds: TabularDataset,
        test_ds: TabularDataset,
        adapter: object | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.schema = schema
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.adapter = adapter
        self.device = device

        self.logger = get_logger("deepfm.trainer")
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.ranking_evaluator = RankingEvaluator(ks=[5, 10, 20])

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.config.training
        if tc.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=tc.lr)
        elif tc.optimizer == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=tc.lr)
        elif tc.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=tc.lr, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {tc.optimizer}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        tc = self.config.training
        if tc.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=2
            )
        elif tc.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {tc.scheduler}")

    def train(self) -> dict[str, float]:
        """Full training loop with early stopping.

        Returns:
            Best validation metrics dict.
        """
        tc = self.config.training
        best_metric = -float("inf")
        best_epoch = 0
        patience_counter = 0
        best_metrics: dict[str, float] = {}
        epoch = 0

        for epoch in range(1, tc.num_epochs + 1):
            # Dynamic negative re-sampling
            if self.adapter is not None and epoch > 1:
                self.train_ds = self.adapter.resample_train()

            train_loss = self._train_epoch(epoch)

            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_ds, "val")
            current_metric = val_metrics.get(
                tc.metric, val_metrics.get("auc", 0.0)
            )

            self.logger.info(
                f"Epoch {epoch}/{tc.num_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_auc={val_metrics.get('auc', 0):.4f}  "
                f"val_logloss={val_metrics.get('logloss', 0):.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()

            # Early stopping + checkpointing
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                best_metrics = val_metrics
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_metric": best_metric,
                    },
                    self.output_dir / "best_model.pt",
                )
                self.logger.info(
                    f"  -> New best {tc.metric}={current_metric:.4f}, saved checkpoint"
                )
            else:
                patience_counter += 1
                if patience_counter >= tc.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {tc.early_stopping_patience} epochs)"
                    )
                    break

        # Final evaluation on test set
        self.logger.info("--- Final evaluation on test set ---")
        test_metrics = self.evaluate(self.test_ds, "test")
        for k, v in test_metrics.items():
            self.logger.info(f"  test_{k} = {v:.4f}")

        self._save_results(best_metrics, test_metrics, best_epoch, epoch)

        return best_metrics

    def _save_results(
        self,
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        best_epoch: int,
        total_epochs: int,
    ) -> None:
        import dataclasses
        from datetime import datetime

        from deepfm.utils.io import save_results

        results = {
            "run_id": self.output_dir.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": dataclasses.asdict(self.config),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "training_info": {
                "best_epoch": best_epoch,
                "total_epochs": total_epochs,
            },
        }
        save_results(results, self.output_dir / "results.json")
        self.logger.info(f"Results saved to {self.output_dir / 'results.json'}")

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        tc = self.config.training

        loader = DataLoader(
            self.train_ds,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=0,
        )

        total_loss = 0.0
        num_batches = 0

        for batch_features, batch_labels in loader:
            # Move to device
            batch_features = {
                k: v.to(self.device) for k, v in batch_features.items()
            }
            batch_labels = batch_labels.to(self.device)

            # Forward
            logits = self.model(batch_features).squeeze(1)  # (B,)
            loss = self.criterion(logits, batch_labels)

            # L2 regularization
            if self.config.feature.embedding_l2_reg > 0:
                loss = loss + self.model.get_l2_reg_loss()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if tc.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), tc.gradient_clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(
        self, dataset: TabularDataset, split_name: str = "eval"
    ) -> dict[str, float]:
        """Evaluate classification and ranking metrics on a dataset.

        Args:
            dataset: Dataset to evaluate.
            split_name: Name for logging.

        Returns:
            Dict with auc, logloss, and HR@K/NDCG@K metrics.
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch_features, batch_labels in loader:
                batch_features = {
                    k: v.to(self.device) for k, v in batch_features.items()
                }
                probs = (
                    self.model.predict(batch_features).squeeze(1).cpu().numpy()
                )
                all_scores.append(probs)
                all_labels.append(batch_labels.numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        metrics: dict[str, float] = {}

        # Classification metrics
        try:
            metrics["auc"] = compute_auc(labels, scores)
        except ValueError:
            metrics["auc"] = 0.0
        metrics["logloss"] = compute_logloss(labels, scores)

        # Ranking metrics (leave-one-out: 1 pos + N neg per user)
        ranking_metrics = self._compute_ranking_metrics(dataset, scores, labels)
        metrics.update(ranking_metrics)

        return metrics

    def _compute_ranking_metrics(
        self,
        dataset: TabularDataset,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute ranking metrics by grouping scores per user."""
        user_ids = dataset.features.get("user_id")
        if user_ids is None:
            return {}

        # Group scores and labels by user
        user_scores: dict[int, list[float]] = {}
        user_labels_map: dict[int, list[float]] = {}

        for i, uid in enumerate(user_ids):
            uid_int = int(uid)
            if uid_int not in user_scores:
                user_scores[uid_int] = []
                user_labels_map[uid_int] = []
            user_scores[uid_int].append(scores[i])
            user_labels_map[uid_int].append(labels[i])

        # Only evaluate users with both positive and negative samples
        eval_scores: list[np.ndarray] = []
        eval_labels: list[np.ndarray] = []
        for uid in user_scores:
            s = np.array(user_scores[uid])
            ul = np.array(user_labels_map[uid])
            if ul.sum() > 0 and ul.sum() < len(ul):
                eval_scores.append(s)
                eval_labels.append(ul)

        if not eval_scores:
            return {}

        return self.ranking_evaluator.evaluate(eval_scores, eval_labels)
