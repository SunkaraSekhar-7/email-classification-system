# Intelligent Email Classification & Response Generation System

Classifies customer-support emails into a support queue and a priority level, then suggests a reply.
Works on English and German emails.

## What it does
1. **Classify** the email into one of 10 queues (Billing and Payments, Technical Support, IT Support, ...).
2. **Predict priority** (low / medium / high).
3. **Suggest a reply**
   - default: finds the most similar past ticket in the same queue and reuses its reply (no API key needed);
   - optional `--llm`: asks Claude to write a reply, using similar past replies as examples.

## Dataset
Customer support tickets (multi-language ticket dataset, ~28.6k tickets after removing duplicates).
Columns used: `subject`, `body`, `answer`, `queue`, `priority`.  
The dataset is included at `data/tickets.csv` (customer support tickets, ~28.6k rows).
(the file `aa_dataset-tickets-multi-lang-5-2-50-version.csv` from the archive, renamed).

## Method
- Text = subject + body, lowercased, links and line-break artifacts removed.
- TF-IDF (word 1-2 grams) + Linear SVM (class-balanced, probability-calibrated).
- 80/20 stratified train/test split.

## Results (held-out test set, 5,718 tickets)
| Task | Accuracy | Macro F1 |
|---|---|---|
| Queue (10 classes) | 0.600 | 0.599 |
| Priority (3 classes) | 0.630 | 0.606 |

Confusion matrices are in `reports/`. Most errors are between queues with similar meaning
(Technical Support, IT Support, Product Support, Customer Service), so these labels overlap in the data.
For comparison, Logistic Regression scored 0.59 and a word+character SVM scored 0.595 on the same split.

## How to run
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/train.py               # trains models, writes metrics + plots
python src/predict.py "Payment failed. My card was charged twice, please refund."
```

Optional LLM replies:
```bash
export ANTHROPIC_API_KEY=your_key   # Windows: set ANTHROPIC_API_KEY=your_key
python src/predict.py --llm "Payment failed. My card was charged twice, please refund."
```

## Project structure
```
data/          dataset (not committed)
src/
  preprocess.py   loading, cleaning, splitting
  train.py        training + evaluation
  respond.py      reply generation (retrieval / LLM)
  predict.py      command-line demo
models/        trained models (not committed)
reports/       metrics.json and confusion matrices
```

## Limitations and future work
- Queue accuracy is about 60% because several queues overlap in meaning; merging similar queues would raise it.
- Retrieved replies contain anonymization placeholders such as `<name>` and `<tel_num>` that must be filled in.
- Next steps: try a multilingual transformer model (e.g. XLM-R), and add the other dataset files after de-duplicating.
