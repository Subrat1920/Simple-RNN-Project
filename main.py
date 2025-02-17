import streamlit
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}

model = load_model('simple_rnn.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500, padding='pre')
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.50 else 'Negative'
    return sentiment, prediction[0][0]


import streamlit as st

st.title('IMDB Movie Review Seniment Analysis')
st.write('Express your feeling about the movie, is paragraph (10-20 wods)')
review = st.text_area('Movie Review')

if st.button("Classify"):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiments = 'Positive' if prediction[0][0] > 0.50 else 'Negative'
    st.write(f'Sentiment: {sentiments}')
    st.write(f"Prediction Score: {np.round(prediction[0][0], 2)}")

else:
    st.write("Please enter movie seniment")