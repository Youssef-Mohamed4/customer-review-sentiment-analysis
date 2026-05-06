import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import joblib
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from scipy.sparse import hstack, csr_matrix

# Suppress background warnings for a cleaner terminal output
warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# --- 1. DATA PREPARATION ---
# Load the pre-balanced 31K dataset
data = pd.read_csv('sentiment_dataset.csv')
data.dropna(subset=['text', 'label'], inplace=True)

# --- 2. NLP PIPELINE ---
stp_words = set(stopwords.words('english'))

def clean_review(review):
    # Standardize text: lowercase, strip URLs, mentions, and punctuation
    review = str(review).lower()
    review = re.sub(r'http\S+|www\S+|https\S+', '', review, flags=re.MULTILINE)
    review = re.sub(r'\@\w+|\#', '', review)
    review = re.sub(r'[^\w\s]', '', review)

    # Rebuild the string excluding stop words
    return " ".join(word for word in review.split() if word not in stp_words)

data['clean_text'] = data['text'].apply(clean_review)

# --- 3. VADER FEATURE EXTRACTION ---
# VADER produces 4 explicit sentiment scores (neg, neu, pos, compound) per review.
# These are concatenated with TF-IDF to give the model pre-computed sentiment signal
# it would otherwise have to infer purely from word co-occurrence counts.
sia = SentimentIntensityAnalyzer()

def get_vader_features(text):
    scores = sia.polarity_scores(str(text))
    return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

# Run VADER on the ORIGINAL (uncleaned) text to preserve punctuation and casing,
# which VADER uses internally to detect emphasis (e.g. "GREAT!!!")
vader_features = np.array(data['text'].apply(get_vader_features).tolist())

# --- 4. VECTORIZATION ---
# sublinear_tf=True applies log normalization to term frequencies, dampening the
# dominance of very common words that survive stop word removal.
# Trigrams (1,3) capture longer phrases like "not that great" or "worked as expected".
# 10,000 features gives the model a richer vocabulary to work with.
TF_IDF = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), sublinear_tf=True)
X_tfidf = TF_IDF.fit_transform(data['clean_text'])

# Combine TF-IDF sparse matrix with dense VADER scores into a single feature matrix
X = hstack([X_tfidf, csr_matrix(vader_features)])
y = data['label']

# stratify=y ensures the 1:1:1 class ratio is maintained in both training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- 5. MODEL TRAINING ---
# class_weight='balanced' forces the algorithm to penalize errors equally across all 3 classes
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(x_train, y_train)

# --- 6. EVALUATION & EXPORT ---
pred = model.predict(x_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, pred, target_names=['Negative', 'Neutral', 'Positive']))

acc = accuracy_score(y_test, pred)
print(f"\nOVERALL ACCURACY: {acc * 100:.2f}%")

# Generate visual confusion matrix
cm = confusion_matrix(y_test, pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])
cm_display.plot(cmap='Blues')
plt.title('Logistic Regression (TF-IDF + VADER)')
plt.show()

# Package the trained model, vectorizer, and label mapping into a single compressed Joblib file
classes_dict = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
joblib.dump({'model': model, 'vectorizer': TF_IDF, 'classes': classes_dict}, 'sentiment_model.joblib')