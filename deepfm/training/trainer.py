from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepfm.config import ExperimentConfig
from deepfm.data.dataset import EvalRankingDataset, NegativeSamplingDataset
from deepfm.training.metrics import MetricCalculator, compute_ranking_metrics
from deepfm.utils.io import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with early stopping, LR scheduling, and ranking evaluation.

    Supports dynamic negative re-sampling per epoch and leave-one-out
    ranking evaluation with HR@K and NDCG@K.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_dataset: NegativeSamplingDataset,
        val_dataset: EvalRankingDataset,
        test_dataset: Optional[EvalRankingDataset] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.config = config
        self.tcfg = config.training
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Early stopping state
        self.best_metric = float("-inf") if self.tcfg.early_stopping_mode == "max" else float("inf")
        self.patience_counter = 0
        self.best_epoch = 0

    def _build_optimizer(self):
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        opt_cls = optimizers[self.tcfg.optimizer]
        kwargs = {"lr": self.tcfg.learning_rate, "weight_decay": self.tcfg.weight_decay}
        if self.tcfg.optimizer == "sgd":
            kwargs["momentum"] = 0.9
        return opt_cls(self.model.parameters(), **kwargs)

    def _build_scheduler(self):
        if self.tcfg.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.tcfg.early_stopping_mode,
                patience=self.tcfg.scheduler_patience,
                factor=self.tcfg.scheduler_factor,
            )
        elif self.tcfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.tcfg.epochs
            )
        return None

    def _move_batch(self, batch):
        """Move batch tensors to device."""
        result = {
            "sparse": batch["sparse"].to(self.device),
            "dense": batch["dense"].to(self.device),
            "label": batch["label"].to(self.device),
        }
        if "sequences" in batch:
            result["sequences"] = {
                k: v.to(self.device) for k, v in batch["sequences"].items()
            }
        else:
            result["sequences"] = None
        return result

    def fit(self) -> Dict[str, float]:
        """Main training loop."""
        for epoch in range(1, self.tcfg.epochs + 1):
            t0 = time.time()

            # Re-sample negatives each epoch
            self.train_dataset.resample_negatives()

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.tcfg.batch_size,
                shuffle=True,
                num_workers=self.tcfg.num_workers,
                pin_memory=self.tcfg.pin_memory,
            )

            # Train
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._evaluate_ranking(self.val_dataset)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch}/{self.tcfg.epochs} [{elapsed:.1f}s] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_auc={val_metrics.get('auc', 0):.4f} "
                f"val_hr@10={val_metrics.get('hr@10', 0):.4f} "
                f"val_ndcg@10={val_metrics.get('ndcg@10', 0):.4f} "
                f"lr={lr:.2e}"
            )

            # LR scheduling
            monitor = val_metrics.get(self.tcfg.early_stopping_metric, 0)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(monitor)
            elif self.scheduler is not None:
                self.scheduler.step()

            # Early stopping
            improved = (
                self.tcfg.early_stopping_mode == "max" and monitor > self.best_metric
            ) or (self.tcfg.early_stopping_mode == "min" and monitor < self.best_metric)

            if improved:
                self.best_metric = monitor
                self.best_epoch = epoch
                self.patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics, self.config.output_dir
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.tcfg.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}"
                    )
                    break

        # Load best model and evaluate on test
        load_checkpoint(self.model, self.config.output_dir)
        self.model.to(self.device)

        if self.test_dataset:
            test_metrics = self._evaluate_ranking(self.test_dataset)
            logger.info(f"Test metrics: {test_metrics}")
            return test_metrics
        return val_metrics

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = self._move_batch(batch)

            self.optimizer.zero_grad()

            logit = self.model(batch["sparse"], batch["dense"], batch["sequences"])
            loss = self.criterion(logit.squeeze(-1), batch["label"])
            loss = loss + self.model.get_l2_reg_loss()

            loss.backward()

            if self.tcfg.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.gradient_clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def _evaluate_ranking(self, dataset: EvalRankingDataset) -> Dict[str, float]:
        """Evaluate with ranking metrics (HR@K, NDCG@K) + classification metrics."""
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.tcfg.batch_size * 2,
            shuffle=False,
            num_workers=self.tcfg.num_workers,
            pin_memory=self.tcfg.pin_memory,
        )

        all_scores = []
        all_labels = []

        for batch in loader:
            batch = self._move_batch(batch)
            logit = self.model(batch["sparse"], batch["dense"], batch["sequences"])
            probs = torch.sigmoid(logit.squeeze(-1))
            all_scores.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        # Classification metrics
        metric_calc = MetricCalculator()
        metric_calc.update(scores, labels)
        metrics = metric_calc.compute()

        # Ranking metrics
        ranking = compute_ranking_metrics(
            scores,
            num_users=dataset.num_users,
            candidates_per_user=dataset.candidates_per_user,
        )
        metrics.update(ranking)

        return metrics

    @torch.no_grad()
    def evaluate(self, dataset: EvalRankingDataset) -> Dict[str, float]:
        """Public evaluation method."""
        return self._evaluate_ranking(dataset)
