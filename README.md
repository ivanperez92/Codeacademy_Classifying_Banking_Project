# Classifying Banking Intent from Customer Queries

Portfolio project (Codecademy AI Engineer path): end-to-end intent classification on **Banking77** — 77 banking intents — comparing a classical baseline against a parameter-efficient transformer.

## Results (test set, 3,080 queries)

| Model | Accuracy | Macro F1 | Notes |
| --- | --- | --- | --- |
| MLP on TF-IDF (12 epochs) | 0.8448 | 0.8449 | Converged baseline |
| RoBERTa-base + LoRA r=16 (7 epochs, GPU) | 0.7802 | 0.7582 | Improved from 0.576 (3-epoch run); still below baseline |

Honest read: the transformer keeps improving with more epochs (0.576 → 0.780) but needs a longer schedule to beat the baseline. Details + per-intent analysis in Sections 8–9 of the notebook.

## Structure

- `datasets/` — `banking77_train.csv` (10,003 rows), `banking77_test.csv` (3,080 rows); columns `text`, `category` (77 intents).
- `banking_classification_BERT.ipynb` — full work: CRISP-DM framework, EDA, MLP, LoRA-RoBERTa, comparison, conclusions.
- `requirements.txt` — upstream Codecademy pins (torch 2.4.1, transformers 4.34.1, ...).

## Quickstart

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook banking_classification_BERT.ipynb
```

> GPU note: Sections 2–5 run on CPU. Section 7 (LoRA fine-tuning) needs a GPU — open the notebook in Colab (T4), clone this repo for `datasets/`, `!pip install -U "torchao>=0.16" transformers peft`, then Run All.

## Privacy

Customer queries may contain emails, phones, or card numbers. EDA includes a detect-and-hash PII stub; enforce masking before any logging or training in production.
