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


# Milestone 2: category tokens (tiny offline tokenizer + random GPT-2)
def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    texts = [core.format_example(r["text"]) for r in core.load_dataset()]
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")


def tiny_model(vocab_size):
    from transformers import GPT2Config, GPT2LMHeadModel

    import torch
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=vocab_size, n_positions=64, n_embd=16, n_layer=1, n_head=2))


@pytest.fixture(scope="module")
def tiny():
    tok = tiny_tokenizer()
    model = tiny_model(len(tok))
    core.extend_tokenizer_and_model(tok, model)
    return tok, model


def test_category_tokens_are_single_tokens(tiny):
    tok, model = tiny
    for token, tid in zip(core.LABEL_TOKENS, core.label_token_ids(tok)):
        assert tok(f"Category: {token}", add_special_tokens=False)["input_ids"][-1] == tid
        assert tok.tokenize(token) == [token]
    assert tok.pad_token == core.PAD_TOKEN
    assert model.get_input_embeddings().weight.shape[0] == len(tok)
