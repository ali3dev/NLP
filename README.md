# IMDB Movie Review Sentiment Analysis

## Overview
This project performs sentiment analysis on a dataset of 50,000 IMDB movie reviews. The goal is to classify reviews as either **positive** or **negative** based on the textual data. The project uses Natural Language Processing (NLP) techniques, including **TF-IDF** vectorization and a **Logistic Regression** model.

## Project Structure
- `Colab.ipynb`: Jupyter notebook for data preprocessing and model training.
- `model.pkl`: Trained Logistic Regression model.
- `app.py`: Streamlit web app for user interaction.

## Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Data
The dataset contains 50k movie reviews from IMDB, labeled as either positive or negative.

## Model
The model is trained using a **TF-IDF vectorizer** and a **Logistic Regression** classifier, achieving over **85% accuracy**.

## Usage
1. Input a movie review in the app.
2. The model predicts if the sentiment is positive or negative.
