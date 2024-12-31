import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Loading the pre-trained model and vectorizer
classifier = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
train_data = pd.read_csv('processed_train_data.csv')

print(train_data['processed_text'].isna().sum())

train_data['processed_text'] = train_data['processed_text'].fillna('')

train_data = train_data.dropna(subset=['processed_text'])

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
vectorizer.fit(train_data['processed_text'])


# Function to predict sentiment
def predict_sentiment(text):
    if not text.strip():
        return "Invalid Input"

    text_tfidf = vectorizer.transform([text])
    
    sentiment = classifier.predict(text_tfidf)
    return str(sentiment[0]) 


# Streamlit UI
st.title("Sentiment Prediction App")
st.write("Enter a sentence to predict its sentiment.")

# User input
user_input = st.text_area("Enter your text below:", height=150)

sentiment_map = {
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive"
        }

if st.button("Predict Sentiment"):
    if user_input:
        predicted_sentiment = predict_sentiment(user_input)
        sentiment_label = sentiment_map.get(predicted_sentiment, "Unknown Sentiment")
        print(sentiment_label)
        st.write(f"### Predicted Sentiment: {sentiment_label}")
    else:
        st.write("Please enter a valid text.")