import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download the necessary NLTK resources (if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define paths to the model, vectorizer, and scaler
model_path = "Best_Logistic_Regression_Model.pkl"
vectorizer_path = "Vectorizer.pkl"
scaler_path = "Scaler.pkl"

# Load the model, vectorizer, and scaler
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    vectorizer = joblib.load(vectorizer_path)
    st.success("Vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")

try:
    scaler = joblib.load(scaler_path)
    st.success("Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading scaler: {e}")

# Streamlit app title
st.title("Twitter Sentiment Analysis for Fake News Detection")

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

# User input
user_input = st.text_area("Enter a tweet:")

if st.button("Analyze"):
    if model and vectorizer and scaler and user_input:
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
        st.warning("Please ensure all models are loaded and enter a tweet for analysis.")