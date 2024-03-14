# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Load the model
model = load_model('spam_classifier_model1.h5')

# Function to preprocess text
def preprocess_text(text):
    max_len = 100
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(text)
    X_seq = tokenizer.texts_to_sequences(text)
    X_pad = pad_sequences(X_seq, maxlen=max_len)
    return X_pad

# Function to predict
def predict(text):
    preprocessed_text = preprocess_text([text])
    prediction = model.predict(preprocessed_text)[0][0]
    return prediction

# Function to convert prediction to label
def prediction_to_label(prediction):
    if prediction > 0.5:
        return 'Spam'
    else:
        return 'Ham'

# Streamlit UI
def main():
    st.title('Email Spam Classifier')

    # Input text box for user input
    user_input = st.text_area("Enter email text:", "")

    # Button to predict
    if st.button('Predict'):
        if user_input:
            prediction = predict(user_input)
            label = prediction_to_label(prediction)
            st.write(f"Prediction: {label} (Probability: {prediction:.2f})")

# Run the app
if __name__ == '__main__':
    main()


