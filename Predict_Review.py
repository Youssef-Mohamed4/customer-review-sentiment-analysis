import streamlit as st
import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from scipy.sparse import hstack, csr_matrix
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# @st.cache_resource keeps the model in RAM so the web page doesn't reload it on every button click
@st.cache_resource
def load_model_data():
    return joblib.load('sentiment_model.joblib')

# Safely attempt to load the packaged model
try:
    model_data = load_model_data()
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    classes = model_data['classes']
except FileNotFoundError:
    st.error("Model file not found. Please run main.py first.")
    st.stop()

# --- NLP SETUP ---
# Must exactly match the logic used during the model's training phase
stp_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def clean_review(review):
    review = str(review).lower()
    review = re.sub(r'http\S+|www\S+|https\S+', '', review, flags=re.MULTILINE)
    review = re.sub(r'\@\w+|\#', '', review)
    review = re.sub(r'[^\w\s]', '', review)
    return " ".join(word for word in review.split() if word not in stp_words)

def get_vader_features(text):
    scores = sia.polarity_scores(str(text))
    return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

# --- STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠")

st.title("🧠 Review Sentiment Analyzer")
st.markdown("Enter a product or service review below to analyze its sentiment.")

# User Input Widget
user_input = st.text_area("Review Text", height=150, placeholder="Type your review here...")

# Inference Trigger
if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():

        # 1. Clean text and extract TF-IDF features
        cleaned_text = clean_review(user_input)
        X_tfidf = vectorizer.transform([cleaned_text])

        # 2. Extract VADER features from the original (uncleaned) text,
        #    preserving punctuation and casing that VADER relies on
        vader_feats = np.array(get_vader_features(user_input)).reshape(1, -1)

        # 3. Combine into the same feature matrix shape used during training
        review_vec = hstack([X_tfidf, csr_matrix(vader_feats)])

        # 4. Extract prediction and max probability (confidence)
        prediction = model.predict(review_vec)[0]
        confidence = model.predict_proba(review_vec).max()
        sentiment = classes[prediction]

        # 5. Dynamic UI rendering based on the result
        st.subheader("Result:")
        if sentiment == 'Positive':
            st.success(f"**{sentiment.upper()}** (Confidence: {confidence:.1%})")
        elif sentiment == 'Negative':
            st.error(f"**{sentiment.upper()}** (Confidence: {confidence:.1%})")
        else:
            st.warning(f"**{sentiment.upper()}** (Confidence: {confidence:.1%})")

    else:
        st.warning("Please enter some text to analyze.")