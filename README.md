# Classifying Banking Intents with BERT

Goal: fine-tune a BERT classifier on the Banking77 dataset (77 intents) following the Codecademy AI Engineer portfolio project.

## Structure

- `datasets/` — `banking77_train.csv` (10,003 rows), `banking77_test.csv` (3,080 rows); columns: `text`, `category` (77 intents).
- `banking_classification_BERT.ipynb` — starter notebook (blank, 1 empty cell).
- `requirements.txt` — upstream Codecademy pins (torch 2.4.1, transformers 4.34.1, ...).

## Quickstart

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook banking_classification_BERT.ipynb
```

> Note: full install is heavy (torch + transformers). Install only when ready to train.
