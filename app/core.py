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
    init_label_embeddings(tokenizer, model)
    print(f"vocab {old} -> {len(tokenizer)} (+{added}: {LABEL_TOKENS + [PAD_TOKEN]})")
    return tokenizer, model


def init_label_embeddings(tokenizer, model) -> None:
    """Start each category token at the mean embedding of its words (e.g. ' product issue').

    With the default mean-init all 4 tokens start identical and training stalls at loss ln(4) (chance);
    GPT-2 ties input and output embeddings, so this also gives each label a distinct output direction.
    """
    import torch

    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for label, tid in zip(LABELS, label_token_ids(tokenizer)):
            word_ids = tokenizer(" " + label.replace("_", " "), add_special_tokens=False)["input_ids"]
            emb[tid] = emb[word_ids].mean(dim=0)


# --- Milestone 3: Fine-tune the model using the Hugging Face Trainer.
def encode_example(tokenizer, text: str, label: str, max_length: int = MAX_LENGTH) -> dict:
    """Prompt + category token, right-padded. Loss only on the category token: every other label is -100."""
    ids = prompt_ids(tokenizer, text, max_length) + [tokenizer.convert_tokens_to_ids(f"<|{label}|>")]
    n, pad = len(ids), max_length - len(ids)
    return {"input_ids": ids + [tokenizer.pad_token_id] * pad,
            "attention_mask": [1] * n + [0] * pad,
            "labels": [-100] * (n - 1) + [ids[-1]] + [-100] * pad}


def train(epochs: int = 3, batch_size: int = 4, lr: float = 1e-4, seed: int = 42) -> dict:
    """Fine-tune distilgpt2 on the train split, save to models/, write results/train_log.json."""
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    tokenizer, model = load_base()
    train_rows, _ = split_dataset(load_dataset())
    dataset = [encode_example(tokenizer, r["text"], r["label"]) for r in train_rows]
    args = TrainingArguments(output_dir=str(ROOT / "models" / "trainer_tmp"), num_train_epochs=epochs,
                             per_device_train_batch_size=batch_size, learning_rate=lr, logging_steps=1,
                             save_strategy="no", report_to="none", use_cpu=True, seed=seed, disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    start = time.time()
    trainer.train()
    seconds = round(time.time() - start, 1)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    losses = [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h]
    log = {"base_model": "distilgpt2", "train_examples": len(dataset), "epochs": epochs, "batch_size": batch_size,
           "learning_rate": lr, "max_length": MAX_LENGTH, "vocab_size": len(tokenizer), "steps": len(losses),
           "first_loss": losses[0], "final_loss": losses[-1], "train_seconds": seconds, "loss_per_step": losses}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"trained {len(losses)} steps in {seconds}s, loss {losses[0]} -> {losses[-1]}")
    return log


# --- Milestone 4: Implement an inference function to classify new feedback.
_LOADED: dict = {}


def load_model(model_dir: Path | str = MODEL_DIR):
    """(tokenizer, model) from models/, or None if not trained yet. Cached per directory."""
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        return None
    if model_dir not in _LOADED:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _LOADED[model_dir] = (AutoTokenizer.from_pretrained(model_dir),
                              AutoModelForCausalLM.from_pretrained(model_dir).eval())
    return _LOADED[model_dir]


def classify(text: str, tokenizer=None, model=None) -> dict:
    """Compare next-token logits of the 4 category tokens only -> {"label", "confidence", "probs"}."""
    import torch

    if model is None:
        loaded = load_model()
        if loaded is None:
            raise FileNotFoundError("No trained model: run `python -m app.core train`")
        tokenizer, model = loaded
    ids = torch.tensor([prompt_ids(tokenizer, text)])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1, label_token_ids(tokenizer)]
    probs = torch.softmax(logits.float(), dim=-1).tolist()
    best = max(range(len(LABELS)), key=probs.__getitem__)
    return {"label": LABELS[best], "confidence": probs[best], "probs": dict(zip(LABELS, probs))}


def metrics(gold: list[str], pred: list[str]) -> dict:
    """Accuracy, macro-F1 and confusion matrix (rows = gold, columns = predicted)."""
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS) - tp
        fn = sum(confusion[label].values()) - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s), "confusion": confusion}


def evaluate() -> dict:
    """Score the fine-tuned model and a majority-class baseline on the test split; write results/eval.md."""
    train_rows, test_rows = split_dataset(load_dataset())
    gold = [r["label"] for r in test_rows]
    outs = [classify(r["text"]) for r in test_rows]
    model_m = metrics(gold, [o["label"] for o in outs])
    majority = Counter(r["label"] for r in train_rows).most_common(1)[0][0]
    base_m = metrics(gold, [majority] * len(gold))
    errors = sorted([(r, o) for r, o in zip(test_rows, outs) if o["label"] != r["label"]],
                    key=lambda x: -x[1]["confidence"])[:3]
    lines = [f"# Evaluation (test split, n={len(gold)}, synthetic data)", "",
             "| Model | Accuracy | Macro-F1 |", "|---|---|---|",
             f"| Fine-tuned distilgpt2 | {model_m['accuracy']:.3f} | {model_m['macro_f1']:.3f} |",
             f"| Majority baseline ({majority}) | {base_m['accuracy']:.3f} | {base_m['macro_f1']:.3f} |", "",
             "## Confusion matrix (rows = true, columns = predicted)", "",
             "| true \\ pred | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
    lines += [f"| {g} | " + " | ".join(str(model_m["confusion"][g][p]) for p in LABELS) + " |" for g in LABELS]
    lines += ["", "## Error examples", ""]
    lines += [f"- \"{r['text']}\" — true `{r['label']}`, predicted `{o['label']}` ({o['confidence']:.2f})"
              for r, o in errors]
    if len(errors) < 3:
        lines.append(f"- Only {len(errors)} test error(s); lowest-confidence correct predictions instead:")
        hard = sorted([(r, o) for r, o in zip(test_rows, outs) if o["label"] == r["label"]],
                      key=lambda x: x[1]["confidence"])[:3 - len(errors)]
        lines += [f"  - \"{r['text']}\" — `{o['label']}` ({o['confidence']:.2f})" for r, o in hard]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:6]))
    return {"model": model_m, "baseline": base_m}

