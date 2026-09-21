"""Evaluate a saved prediction NPZ containing ``prediction`` and ``target`` arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import all_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_npz", type=Path)
    args = parser.parse_args()
    with np.load(args.prediction_npz, allow_pickle=False) as data:
        metrics = all_metrics(data["prediction"], data["target"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
