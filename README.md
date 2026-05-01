# Customer Review Sentiment Analyzer

A Python-based machine learning project that analyzes Amazon product reviews and classifies them as **Positive**, **Neutral**, or **Negative** using Logistic Regression with TF-IDF vectorization.

## Features
- Classifies reviews into 3 sentiment classes: Positive, Neutral, Negative
- Displays prediction confidence percentage
- Removes stopwords and non-alphabetic characters for cleaner input
- Trained on Amazon product reviews dataset with bigram support
- Desktop GUI built with Tkinter
- Web UI alternative built with Streamlit

## Project Structure

```
├── ReviewSentimentAnalyzer.py   # Training script — builds and saves the model
├── Predict_Review.py            # Desktop GUI app (Tkinter)
├── Predict_Review_Streamlit.py  # Web UI app (Streamlit)
├── sentiment_model.pkl          # Pre-trained model (ready to use)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8 or higher

```bash
pip install -r requirements.txt
```

## Usage

### Option 1 — Desktop GUI (Tkinter)
```bash
py Predict_Review.py
```

### Option 2 — Web UI (Streamlit)
```bash
streamlit run Predict_Review_Streamlit.py
```

### Retrain the model (optional)
Only needed if you want to retrain from scratch. Requires the dataset CSV in the same directory:
```bash
py ReviewSentimentAnalyzer.py
```
This will print the classification report, total accuracy, and display the confusion matrix, then overwrite `sentiment_model.pkl`.

## Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF (unigrams + bigrams) |
| Max Features | 5000 |
| Train/Test Split | 75% / 25% |
| Overall Accuracy | ~71% |

## Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 0.73 | 0.83 | 0.78 |
| Neutral | 0.45 | 0.24 | 0.31 |
| Positive | 0.76 | 0.82 | 0.79 |

> **Note:** Neutral class performance is lower due to dataset imbalance — Amazon star ratings naturally produce fewer 3-star (neutral) reviews compared to positive and negative ones. 3-class labeled sentiment datasets are less common than binary alternatives, which is a known limitation of this dataset.