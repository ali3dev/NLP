# streamlit_app.py
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')

# Load the saved Logistic Regression model
model = pickle.load(open("Updated_sentiment_model_logreg.pkl", "rb"))

# Load the fitted TF-IDF vectorizer (this should be the vectorizer used during training)
with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

# Function to preprocess user input
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    return text

# Streamlit App Interface
st.title("Sentiment Analysis App")
st.write("Enter a movie review, and the model will predict whether it's positive or negative.")

# Text input from the user
user_input = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    # Preprocess the input
    clean_input = preprocess_text(user_input)
    
    # Convert the text input into a format the model can understand (using the same TF-IDF used for training)
    input_features = tfidf.transform([clean_input]).toarray()
    
    # Predict the sentiment
    prediction = model.predict(input_features)
    
    # Display the result
    if prediction == 1:
        st.write("The review is **Positive**!")
    else:
        st.write("The review is **Negative**.")
