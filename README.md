# Customer Review Sentiment Analyzer

A Python-based machine learning project that analyzes product or service reviews and classifies them as **Positive**, **Neutral**, or **Negative** using Logistic Regression with TF-IDF vectorization augmented by VADER sentiment scores.

## Features
- Classifies reviews into 3 sentiment classes: Positive, Neutral, Negative
- Displays prediction confidence percentage
- Augments TF-IDF features with VADER sentiment scores for stronger signal
- Strips URLs, mentions, punctuation, and stop words for cleaner input
- Trained on a balanced 31K dataset with trigram support
- Web UI built with Streamlit

## Project Structure

```
├── ReviewSentimentAnalyzer.py   # Training script — builds and saves the model
├── app.py                       # Streamlit web app
├── sentiment_model.joblib       # Pre-trained model (ready to use)
├── sentiment_dataset.csv        # Balanced 31K training dataset
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8 or higher

```bash
pip install -r requirements.txt
```

## Usage

### Run the web app
```bash
streamlit run Predict_Review.py
```

### Retrain the model (optional)
Only needed if you want to retrain from scratch. Requires `sentiment_dataset.csv` in the same directory:
```bash
py ReviewSentimentAnalyzer.py
```
This will print the classification report, display the confusion matrix, and overwrite `sentiment_model.joblib`.

## Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression (`class_weight='balanced'`) |
| Vectorizer | TF-IDF (unigrams + trigrams, sublinear TF) |
| Max Features | 10,000 |
| Extra Features | VADER scores (neg, neu, pos, compound) |
| Train/Test Split | 75% / 25% (stratified) |
| Dataset | Balanced — ~10K samples per class |
| Overall Accuracy | 68.37% |

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.67 | 0.68 | 0.67 | 2,276 |
| Neutral  | 0.63 | 0.61 | 0.62 | 2,912 |
| Positive | 0.75 | 0.77 | 0.76 | 2,620 |
| **Macro avg** | **0.68** | **0.69** | **0.68** | 7,808 |

## Design Decisions

**Balanced dataset:** Uses a pre-balanced dataset (~10K examples per class) and passes `class_weight='balanced'` to the classifier to treat all three classes equally, preventing the model from being biased toward more common sentiment labels.

**VADER feature augmentation:** VADER produces four explicit sentiment scores per review (`neg`, `neu`, `pos`, `compound`) which are concatenated with the TF-IDF matrix before training. This gives the model pre-computed sentiment signal it would otherwise have to infer purely from word co-occurrence counts. Crucially, VADER runs on the original uncleaned text to preserve casing and punctuation (e.g. `"GREAT!!!"`) that it uses internally to detect emphasis.

**Sublinear TF scaling:** `sublinear_tf=True` applies log normalization to term frequencies, dampening the dominance of very frequent words that survive stop word removal.

**Trigrams:** `ngram_range=(1,3)` allows the model to learn longer phrases like `"not that great"` or `"works as expected"`, capturing more contextual meaning than bigrams alone.

**Joblib serialization:** The model artifact is saved with `joblib` for better compatibility and compression than `pickle`.