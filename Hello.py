import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('lingSpam.csv')  # Replace 'lingSpam.csv' with your dataset file path
X = data['Body']  # Email text
y = data['Label']  # Spam or ham label

# Preprocess the text
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len)

# Encode labels
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Build the RNN model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_pad, y_enc, batch_size=32, epochs=5)

# Function to preprocess input text
def preprocess_text(text):
    # Tokenize the text
    seq = tokenizer.texts_to_sequences([text])
    # Pad sequences
    padded_seq = pad_sequences(seq, maxlen=max_len)
    return padded_seq

# Define the Streamlit UI
def main():
    st.title("Spam Classification App")

    # Take input text from the user
    input_text = st.text_area("Enter the text to classify:", "")

    # Classify button
    if st.button("Classify"):
        # Preprocess the input text
        preprocessed_text = preprocess_text(input_text)

        # Predict using the trained model
        prediction = (model.predict(preprocessed_text) > 0.5).astype("int32")

        # Interpret the prediction
        if prediction == 1:
            st.write("The input text is classified as spam.")
        else:
            st.write("The input text is classified as ham (non-spam).")

if __name__ == "__main__":
    main()
