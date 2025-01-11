import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Define paths
model_path = r"C:\Users\isabe\OneDrive\Desktop\FakeNewsDetectioWithSentimentAnalysis\FakeNewsDetectioWithSentimentAnalysis\Best_Logistic_Regression_Model.pkl"
vectorizer_path = r"C:\Users\isabe\OneDrive\Desktop\FakeNewsDetectioWithSentimentAnalysis\FakeNewsDetectioWithSentimentAnalysis\Vectorizer.pkl"
scaler_path = r"C:\Users\isabe\OneDrive\Desktop\FakeNewsDetectioWithSentimentAnalysis\FakeNewsDetectioWithSentimentAnalysis\Scaler.pkl"

# Check if files exist
if not os.path.isfile(model_path):
    st.error(f"Model file not found at: {model_path}")
if not os.path.isfile(vectorizer_path):
    st.error(f"Vectorizer file not found at: {vectorizer_path}")
if not os.path.isfile(scaler_path):
    st.error(f"Scaler file not found at: {scaler_path}")

# Load the saved model, vectorizer, and scaler
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Function definitions for cleaning and processing tweets
def clean_text(text):
    """Clean the input text by removing unwanted characters."""
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove @ mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove links
    text = re.sub(r'\#', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def get_sentiment(content):
    """Calculates sentiment polarity of the tweet."""
    return TextBlob(content).sentiment.polarity

def lemmatize_text(content):
    """Lemmatizes the text to reduce words to their base form."""
    lemmatizer = WordNetLemmatizer()
    lemmatized_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    lemmatized_content = [
        lemmatizer.lemmatize(word) for word in lemmatized_content
        if word not in stopwords.words('english')
    ]
    return ' '.join(lemmatized_content)

# Streamlit app
st.title("Twitter Sentiment Analysis for Fake News Detection")

# User input
user_input = st.text_area("Enter a tweet:")

if st.button("Analyze"):
    if user_input:
        # Clean and process the input
        cleaned_tweet = clean_text(user_input)
        sentiment_score = get_sentiment(cleaned_tweet)
        lemmatized_tweet = lemmatize_text(cleaned_tweet)

        # Prepare the features for prediction
        tweet_vector = vectorizer.transform([lemmatized_tweet])
        tweet_features = np.hstack((tweet_vector.toarray(), [[sentiment_score, len(cleaned_tweet.split())]]))
        scaled_features = scaler.transform(tweet_features)

        # Predict sentiment
        prediction = model.predict(scaled_features)

        sentiment_label = "Fake News" if prediction[0] == 1 else "Real News"
        st.success(f"Sentiment Score: {sentiment_score:.2f}")
        st.write(f"Prediction: {sentiment_label}")
    else:
        st.warning("Please enter a tweet for analysis.")