# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/q4mxjuh77r/badge.svg)](https://ask.thili.ai/b/q4mxjuh77r)

A Streamlit app that sorts customer comments into four categories with a fine-tuned **distilgpt2**:
`product_issue`, `delivery_shipping`, `staff_praise`, `pricing_value`. Runs on CPU, with no API key and no `.env`.

**How it works:** each category is a new special token (`<|product_issue|>` and so on). The model is trained on
`Comment: {text}\nCategory: <|label|>`, with loss only on the category token. To classify, it compares the next-token
logits of those 4 tokens and returns the label with its softmax probability. The category embeddings start from the
mean of their words' embeddings, because with identical starting points training stalled at chance (loss ln 4).

## Setup and run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # CPU-only torch
python -m app.core train            # fine-tune (~2-3 min CPU) -> models/, results/train_log.json
python -m app.core eval             # test-set metrics -> results/eval.md
streamlit run app/main.py
pytest -q                           # offline: tiny random GPT-2 + word-level tokenizer
```

## Results (24-comment held-out test split)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Fine-tuned distilgpt2 (3 epochs, batch 4, lr 1e-4) | 0.917 | 0.919 |
| Majority-class baseline | 0.250 | 0.100 |

The confusion matrix and error examples are in [results/eval.md](results/eval.md). The training log is in
[results/train_log.json](results/train_log.json).

## Data note

[data/feedback.csv](data/feedback.csv) holds **120 synthetic comments** (30 per category) that were written for this
demo, split 80/20 with stratification. They are short and clean, so expect lower accuracy on real customer text.
