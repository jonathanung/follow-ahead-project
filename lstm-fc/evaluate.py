"""Comprehensive evaluation of trained LSTM-FC model."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lstm_fc.config import DataConfig, ModelConfig
from lstm_fc.data.dataset import TrajectoryDataset
from lstm_fc.model import LSTMModel2D


def evaluate(model_path: str, data_path: str = "data_3d_h36m.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", ModelConfig())
    model = LSTMModel2D(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    data_config = DataConfig(data_path=data_path)
    test_dataset = TrajectoryDataset(data_config, split="test")

    test_x = test_dataset.samples.to(device)
    test_y = test_dataset.targets.to(device)

    print(f"Model: {model_path}")
    print(f"Config: hidden_size={model_config.hidden_size}, "
          f"num_layers={model_config.num_layers}, "
          f"dropout={getattr(model_config, 'dropout', 0.0)}")
    print(f"Test samples: {len(test_x)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Full test set evaluation in batches
    all_preds = []
    batch_size = 2048
    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            batch = test_x[i : i + batch_size]
            preds = model(batch)
            all_preds.append(preds)

    all_preds = torch.cat(all_preds, dim=0)

    # Metrics
    mse = nn.MSELoss()(all_preds, test_y).item()
    mae = torch.abs(all_preds - test_y).mean().item()

    pred_classes = all_preds.argmax(dim=1)
    true_classes = test_y.argmax(dim=1)
    accuracy = (pred_classes == true_classes).float().mean().item()

    print(f"{'='*50}")
    print(f"OVERALL METRICS")
    print(f"{'='*50}")
    print(f"MSE Loss:     {mse:.6f}")
    print(f"MAE:          {mae:.6f}")
    print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()

    # Per-class metrics
    class_names = ["left", "straight", "right"]
    print(f"{'='*50}")
    print(f"PER-CLASS METRICS")
    print(f"{'='*50}")

    for c, name in enumerate(class_names):
        mask = true_classes == c
        n_samples = mask.sum().item()
        if n_samples > 0:
            class_acc = (pred_classes[mask] == c).float().mean().item()
            class_mse = nn.MSELoss()(all_preds[mask], test_y[mask]).item()
            # Precision and recall
            pred_mask = pred_classes == c
            tp = ((pred_classes == c) & (true_classes == c)).sum().item()
            precision = tp / max(pred_mask.sum().item(), 1)
            recall = tp / max(n_samples, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            print(f"  {name:10s}: Acc={class_acc:.4f} MSE={class_mse:.6f} "
                  f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} (n={n_samples})")

    # Confusion matrix
    print(f"\n{'='*50}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*50}")
    print(f"{'':15s}  Pred Left  Pred Str  Pred Right")
    for c, name in enumerate(class_names):
        mask = true_classes == c
        row = []
        for pc in range(3):
            count = ((pred_classes == pc) & mask).sum().item()
            row.append(count)
        total = sum(row)
        print(f"  True {name:8s}: {row[0]:8d}  {row[1]:8d}  {row[2]:8d}  (total={total})")

    # Probability distribution analysis
    print(f"\n{'='*50}")
    print(f"PREDICTION DISTRIBUTION")
    print(f"{'='*50}")
    mean_pred = all_preds.mean(dim=0).cpu().numpy()
    std_pred = all_preds.std(dim=0).cpu().numpy()
    mean_true = test_y.mean(dim=0).cpu().numpy()
    for c, name in enumerate(class_names):
        print(f"  {name:10s}: pred={mean_pred[c]:.4f}+/-{std_pred[c]:.4f}  "
              f"true={mean_true[c]:.4f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Test MSE:  {mse:.6f}")
    print(f"Test Acc:  {accuracy*100:.2f}%")
    print(f"Model:     LSTM(h={model_config.hidden_size}, L={model_config.num_layers}) + Dropout({getattr(model_config, 'dropout', 0.0)}) + FC")
    print(f"Params:    {sum(p.numel() for p in model.parameters()):,}")

    return {
        "mse": mse,
        "mae": mae,
        "accuracy": accuracy,
        "model_path": model_path,
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/final_best/model_final.pt"
    evaluate(model_path)
