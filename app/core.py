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


# --- Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.
LABEL_TOKENS = [f"<|{label}|>" for label in LABELS]
PAD_TOKEN = "<|pad|>"
MAX_LENGTH = 64


def format_example(text: str, label: str | None = None) -> str:
    """Training text: 'Comment: {text}\nCategory: <|label|>' (prompt only when label is None)."""
    prompt = f"Comment: {text}\nCategory:"
    return prompt if label is None else f"{prompt} <|{label}|>"


def extend_tokenizer_and_model(tokenizer, model=None) -> int:
    """Add one special token per category plus a pad token; resize the model's embeddings. Returns #tokens added."""
    added = tokenizer.add_special_tokens({"pad_token": PAD_TOKEN, "additional_special_tokens": LABEL_TOKENS})
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    return added


def label_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.convert_tokens_to_ids(t) for t in LABEL_TOKENS]
    assert all(isinstance(i, int) and i != tokenizer.unk_token_id for i in ids), "category tokens missing"
    return ids


def prompt_ids(tokenizer, text: str, max_length: int = MAX_LENGTH) -> list[int]:
    """Token ids of the prompt, left-truncated in the comment so 'Category:' + label always fit."""
    ids = tokenizer(format_example(text), add_special_tokens=False)["input_ids"]
    return ids[-(max_length - 1):]


def load_base(model_name: str = "distilgpt2"):
    """Download distilgpt2, add the category tokens, print the new vocab size."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    old = len(tokenizer)
    added = extend_tokenizer_and_model(tokenizer, model)
    print(f"vocab {old} -> {len(tokenizer)} (+{added}: {LABEL_TOKENS + [PAD_TOKEN]})")
    return tokenizer, model
