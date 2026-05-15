import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk

# Download NLTK data (first run only)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model_data = joblib.load('sentiment_model.joblib')
    return model_data['model'], model_data['vectorizer'], model_data['classes']

model, vectorizer, classes = load_model()
sia = SentimentIntensityAnalyzer()
stp_words = set(stopwords.words('english'))

def clean_review(review):
    review = str(review).lower()
    review = re.sub(r'http\S+|www\S+|https\S+', '', review, flags=re.MULTILINE)
    review = re.sub(r'\@\w+|\#', '', review)
    review = re.sub(r'[^\w\s]', '', review)
    return " ".join(word for word in review.split() if word not in stp_words)

def get_vader_features(text):
    scores = sia.polarity_scores(str(text))
    return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

st.title("📊 Sentiment Analysis")
st.markdown("**TF-IDF + VADER + Logistic Regression**")

text = st.text_area("Enter your review or text:", height=200, placeholder="Type here...")

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if text.strip():
        with st.spinner("Analyzing..."):
            # Preprocess
            clean_text = clean_review(text)
            vader_feats = np.array(get_vader_features(text)).reshape(1, -1)
            
            # Vectorize
            tfidf_vec = vectorizer.transform([clean_text])
            X_input = hstack([tfidf_vec, csr_matrix(vader_feats)])
            
            # Predict
            pred = model.predict(X_input)[0]
            probs = model.predict_proba(X_input)[0]
            
            sentiment = classes[pred]
            
            # Display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if sentiment == "Positive":
                    st.success("Positive 😊")
                elif sentiment == "Negative":
                    st.error("Negative 😞")
                else:
                    st.warning("Neutral 😐")
            
            with col2:
                st.subheader("Confidence")
                for label, prob in zip(["Negative", "Neutral", "Positive"], probs):
                    st.progress(float(prob), text=f"{label}: {prob:.1%}")

    else:
        st.warning("Please enter some text!")