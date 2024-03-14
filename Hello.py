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
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('spam_classifier_model')

# Define the maximum length for padding sequences
max_len = 100

# Function to preprocess input text
def preprocess_text(text):
    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    # Pad sequences
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

# Streamlit application
def main():
    st.title("Email Spam Classification")
    st.write("Enter the email text below:")

    # Get user input
    user_input = st.text_input("Email text", "")

    if st.button("Classify"):
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        # Make prediction
        prediction = model.predict(processed_text)
        # Convert prediction to human-readable label
        if prediction > 0.5:
            st.write("This email is predicted to be spam.")
        else:
            st.write("This email is predicted to be ham.")

if __name__ == "__main__":
    main()
