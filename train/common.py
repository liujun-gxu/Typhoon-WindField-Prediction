"""Shared, relative-path training implementation for all forecast horizons."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing.dataset import load_split
from data_processing.normalization import denormalize_y, fit_train_minmax, normalize_x, normalize_y
from evaluation.metrics import all_metrics
from models import FullFieldSTLNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def mse(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in data_loader:
            prediction, target = model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True)
            total += torch.sum((prediction - target) ** 2).item()
            count += target.numel()
    return total / count


def predict(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for x, _ in data_loader:
            values.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(values, axis=0)


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train = load_split(args.data_dir, args.horizon, "train")
    x_val, y_val = load_split(args.data_dir, args.horizon, "val")
    x_test, y_test = load_split(args.data_dir, args.horizon, "test")
    channel_min, channel_max = fit_train_minmax(x_train, y_train)
    train_loader = loader(normalize_x(x_train, channel_min, channel_max), normalize_y(y_train, channel_min, channel_max), args.batch_size, True)
    val_loader = loader(normalize_x(x_val, channel_min, channel_max), normalize_y(y_val, channel_min, channel_max), args.batch_size, False)
    test_loader = loader(normalize_x(x_test, channel_min, channel_max), normalize_y(y_test, channel_min, channel_max), args.batch_size, False)

    result_dir = args.output_dir / f"{args.horizon}h" / f"seed_{args.seed}"
    checkpoint_path = result_dir / "best_model.pt"
    if result_dir.exists() and any(result_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing run: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    model = FullFieldSTLNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_val, best_epoch, stale, history = float("inf"), 0, 0, []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.numel()
            count += y.numel()
        train_mse, val_mse = total / count, mse(model, val_loader, device)
        scheduler.step(val_mse)
        is_best = val_mse < best_val - 1e-12
        if is_best:
            best_val, best_epoch, stale = val_mse, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "horizon_h": args.horizon, "seed": args.seed, "best_val_mse_normalized": best_val, "channel_min": channel_min, "channel_max": channel_max}, checkpoint_path)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_mse_normalized": train_mse, "val_mse_normalized": val_mse, "learning_rate": optimizer.param_groups[0]["lr"], "is_best": int(is_best)})
        if stale >= args.early_stopping_patience:
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    prediction = denormalize_y(predict(model, test_loader, device), channel_min, channel_max).astype(np.float32)
    metrics = all_metrics(prediction, y_test)
    metrics.update({"horizon_h": args.horizon, "seed": args.seed, "best_epoch": best_epoch, "actual_epochs": len(history), "best_val_mse_normalized": best_val, "test_samples": len(y_test), "units": "m/s"})
    with (result_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(result_dir / "test_predictions.npz", prediction=prediction, target=y_test.astype(np.float32))
    (result_dir / "normalization.json").write_text(json.dumps({"fit_split": "train only", "fit_sources": ["X_train", "Y_train"], "channel_min": channel_min.tolist(), "channel_max": channel_max.tolist()}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def build_parser(horizon: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train fixed-t0 STL-Net for {horizon} h local wind-field prediction.")
    parser.set_defaults(horizon=horizon)
    parser.add_argument("--data-dir", type=Path, default=Path("data/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser
