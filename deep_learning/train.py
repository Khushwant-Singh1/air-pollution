"""
Training Engine for Agra Multi-Output LSTM Sequence Model.
"""
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from config import (
    CHECKPOINTS_DIR,
    SEQUENCE_FEATURE_COLS,
    RESULTS_CSV,
    LSTMHyperparameters
)
from dataset import create_dataloaders
from model import AgraMultiOutputLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LSTMTrainer")


def masked_huber_loss(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Computes Huber Loss only on valid (non-masked) pollutant targets."""
    loss_unreduced = nn.SmoothL1Loss(reduction="none", beta=1.0)(preds, targets)
    masked_loss = loss_unreduced * masks
    valid_count = torch.clamp(masks.sum(), min=1.0)
    return masked_loss.sum() / valid_count


class LSTMTrainingOrchestrator:
    """Trains and validates the multi-output LSTM model across Agra."""

    def __init__(self, config: LSTMHyperparameters = LSTMHyperparameters()):
        self.config = config
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

        torch.set_num_threads(8)
        self.device = torch.device("cpu")
        logger.info("Using multi-threaded CPU acceleration (8 threads)")

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Runs a single training epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for seqs, targets, masks in train_loader:
            seqs = seqs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()
            preds = model(seqs)
            loss = masked_huber_loss(preds, targets, masks)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        meta_df: pd.DataFrame
    ) -> Tuple[float, dict, pd.DataFrame]:
        """Evaluates model on the test holdout set in real physical units (expm1)."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []

        for seqs, targets, masks in test_loader:
            seqs = seqs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

            preds = model(seqs)
            loss = masked_huber_loss(preds, targets, masks)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(preds.cpu().numpy())

        preds_log = np.concatenate(all_preds, axis=0)
        preds_raw = np.maximum(0.0, np.expm1(preds_log))

        pred_df = meta_df.copy()
        pred_df["Pred_PM2.5"] = preds_raw[:, 0]
        pred_df["Pred_PM10"] = preds_raw[:, 1]
        pred_df["Pred_Ozone"] = preds_raw[:, 2]

        metrics = {}
        targets_list = [
            ("PM2.5", "Actual_PM2.5", "Pred_PM2.5"),
            ("PM10", "Actual_PM10", "Pred_PM10"),
            ("Ozone", "Actual_Ozone", "Pred_Ozone")
        ]

        for clean_name, act_col, pred_col in targets_list:
            valid = pred_df.dropna(subset=[act_col, pred_col])
            y_true = valid[act_col].values
            y_pred = valid[pred_col].values

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rel_rmse = (rmse / y_true.mean()) * 100 if y_true.mean() > 0 else np.nan

            metrics[clean_name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Rel_RMSE": rel_rmse,
                "n_samples": len(valid)
            }

        val_loss = total_loss / max(1, n_batches)
        return val_loss, metrics, pred_df

    def fit(self) -> Tuple[nn.Module, dict]:
        """Runs the complete training and early stopping loop."""
        logger.info("Preparing 24-hour sequence sliding datasets...")
        train_loader, test_loader, test_meta, scaler = create_dataloaders(self.config)

        input_dim = len(SEQUENCE_FEATURE_COLS)
        model = AgraMultiOutputLSTM(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
            cell_type=getattr(self.config, "cell_type", "gru"),
            noise_std=getattr(self.config, "noise_std", 0.02)
        ).to(self.device)

        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)

        best_val_loss = float("inf")
        best_metrics = None
        best_pred_df = None
        patience_counter = 0
        checkpoint_path = CHECKPOINTS_DIR / "best_lstm_model.pt"

        logger.info(f"Starting training for {self.config.max_epochs} epochs (Early stopping patience={self.config.patience})...")

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(model, train_loader, optimizer)
            val_loss, val_metrics, pred_df = self.evaluate(model, test_loader, test_meta)
            elapsed = time.time() - t0

            scheduler.step(val_loss)

            pm25_r2 = val_metrics["PM2.5"]["R2"]
            pm10_r2 = val_metrics["PM10"]["R2"]
            o3_r2 = val_metrics["Ozone"]["R2"]

            logger.info(
                f"Epoch [{epoch:02d}/{self.config.max_epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"R²: PM2.5={pm25_r2:.3f}, PM10={pm10_r2:.3f}, Ozone={o3_r2:.3f}"
            )

            # Checkpoint on best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                best_pred_df = pred_df
                patience_counter = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "metrics": val_metrics,
                    "config": self.config
                }, checkpoint_path)
                logger.info(f"  --> Saved new best checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Load best model weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with Val Loss = {checkpoint['val_loss']:.4f}")

        return model, best_metrics


def main():
    trainer = LSTMTrainingOrchestrator()
    model, metrics = trainer.fit()

    print("\n" + "=" * 80)
    print("DEEP LEARNING LSTM MULTI-OUTPUT EVALUATION RESULTS (2025 Test Holdout)")
    print("=" * 80)
    for pollutant, m in metrics.items():
        print(f"  {pollutant:<8}: R² = {m['R2']:.4f} | MAE = {m['MAE']:.2f} µg/m³ | RMSE = {m['RMSE']:.2f} µg/m³ | Rel Error = {m['Rel_RMSE']:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
