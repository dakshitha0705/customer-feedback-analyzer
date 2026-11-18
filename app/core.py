"""Customer Feedback Analyzer: dataset, GPT-2 category tokens, fine-tuning and inference (CPU only).

CLI:  python -m app.core train   |   python -m app.core eval
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "feedback.csv"
MODEL_DIR = ROOT / "models" / "feedback-distilgpt2"
RESULTS_DIR = ROOT / "results"


# --- Milestone 1: Define 3-5 feedback categories and create a small dataset of example comments with labels.
LABELS = ["product_issue", "delivery_shipping", "staff_praise", "pricing_value"]


def load_dataset(path: Path | str = DATA_PATH) -> list[dict]:
    """Rows of {"text", "label"} from the (synthetic) CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{"text": r["text"].strip(), "label": r["label"].strip()} for r in csv.DictReader(f)]


def split_dataset(rows: list[dict], test_frac: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Stratified split: the same fraction of every label goes to the test set."""
    rng = random.Random(seed)
    train, test = [], []
    for label in LABELS:
        group = [r for r in rows if r["label"] == label]
        rng.shuffle(group)
        n_test = round(len(group) * test_frac)
        test += group[:n_test]
        train += group[n_test:]
    rng.shuffle(train)
    return train, test
