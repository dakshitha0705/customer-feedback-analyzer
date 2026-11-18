"""Tests for app/core.py. No downloads: GPT-2 tests use a tiny random config and a word-level tokenizer built here."""
from collections import Counter

import pytest

from app import core


# Milestone 1: categories + dataset
def test_dataset_has_four_balanced_classes():
    rows = core.load_dataset()
    counts = Counter(r["label"] for r in rows)
    assert set(counts) == set(core.LABELS) and len(core.LABELS) == 4
    assert set(counts.values()) == {30}
    train, test = core.split_dataset(rows)
    assert len(train) == 96 and len(test) == 24
    assert set(Counter(r["label"] for r in test).values()) == {6}
    assert not {r["text"] for r in train} & {r["text"] for r in test}
