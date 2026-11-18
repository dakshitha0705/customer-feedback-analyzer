# Evaluation (test split, n=24, synthetic data)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Fine-tuned distilgpt2 | 0.917 | 0.919 |
| Majority baseline (delivery_shipping) | 0.250 | 0.100 |

## Confusion matrix (rows = true, columns = predicted)

| true \ pred | product_issue | delivery_shipping | staff_praise | pricing_value |
|---|---|---|---|---|
| product_issue | 6 | 0 | 0 | 0 |
| delivery_shipping | 0 | 6 | 0 | 0 |
| staff_praise | 0 | 1 | 5 | 0 |
| pricing_value | 0 | 1 | 0 | 5 |

## Error examples

- "Priya in returns made the process painless." — true `staff_praise`, predicted `delivery_shipping` (0.90)
- "Paying extra for gift wrapping feels like a rip off." — true `pricing_value`, predicted `delivery_shipping` (0.55)
- Only 2 test error(s); lowest-confidence correct predictions instead:
  - "Packaging was torn open when it got here." — `delivery_shipping` (0.86)
