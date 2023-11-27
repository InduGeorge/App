import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# Load the saved model for tumor detection
loaded_tumor_model = tf.keras.models.load_model("tumor_detection_model.h5")

# Load the IMDb sentiment analysis models
loaded_dnn_sentiment_model = load_model("imdb_sentiment_model.h5")
loaded_lstm_sentiment_model = load_model("lstm_model.h5")

# Load the backpropagation model
loaded_backpropagation_model = tf.keras.models.load_model("backpropagation_model.h5")

# Load the perceptron model
loaded_perceptron_model = tf.keras.models.load_model("perceptron_model.h5")

# Load the RNN model
loaded_rnn_model = tf.keras.models.load_model("trained_rnn_model.h5")

def preprocess_input(text, num_words=10000, max_len=500):
    word_index = tf.keras.datasets.imdb.get_word_index()
    words = text.split()
    sequence = [word_index[word] if word_index.get(word) and word_index[word] < num_words else 2 for word in words]
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_len)
    return padded_sequence

def predict_sentiment(model, input_text):
    # Preprocess input text
    processed_input = preprocess_input(input_text)

    # Make prediction using the loaded model
    prediction = model.predict(processed_input)

    # Return the predicted sentiment
    return "Positive" if prediction > 0.5 else "Negative"

def make_prediction_cnn(uploaded_file, model):
    # Read the image file as bytes
    content = uploaded_file.read()
    # Convert bytes to numpy array
    img_array = np.frombuffer(content, np.uint8)
    # Decode the image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # Resize the image
    img = cv2.resize(img, (128, 128))
    # Normalize the image
    img = img / 255.0
    # Expand dimensions to match the model's expected input shape
    img = np.expand_dims(img, axis=0)
    # Make prediction
    res = model.predict(img)
    if res > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"

def make_sentiment_prediction(model, user_input):
    # Replace this with the actual prediction logic for sentiment analysis
    # prediction = model.predict(user_input)
    prediction = predict_sentiment(model, user_input)
    return prediction

def main():
    st.title("Model Selection")

    task_options = ["Tumor Detection", "Sentiment Analysis"]
    selected_task = st.selectbox("Choose a task:", task_options)

    if selected_task == "Sentiment Analysis":
        sentiment_analysis()
    elif selected_task == "Tumor Detection":
        tumor_detection()

def sentiment_analysis():
    st.header("Sentiment Analysis Model Selection")
    model_options = ["DNN", "LSTM", "Backpropagation", "Perceptron", "RNN"]  # Add RNN to the options
    selected_model = st.selectbox("Choose a model:", model_options)
    
    st.write("Provide input data for Sentiment Analysis:")
    user_input = st.text_input("Enter text for analysis:")

    if st.button("Predict"):
        if selected_model == "DNN":
            prediction = make_sentiment_prediction(loaded_dnn_sentiment_model, user_input)
        elif selected_model == "LSTM":
            prediction = make_sentiment_prediction(loaded_lstm_sentiment_model, user_input)
        elif selected_model == "Backpropagation":
            prediction = make_sentiment_prediction(loaded_backpropagation_model, user_input)
        elif selected_model == "Perceptron":
            prediction = make_sentiment_prediction(loaded_perceptron_model, user_input)
        elif selected_model == "RNN":
            prediction = make_sentiment_prediction(loaded_rnn_model, user_input)

        st.write(f"Prediction: {prediction}")

def tumor_detection():
    st.header("Tumor Detection Model Selection")
    model_options = ["CNN"]
    selected_model = st.selectbox("Choose a model:", model_options)
    
    st.write("Provide input data for Tumor Detection:")
    uploaded_file = st.file_uploader("Choose an image for prediction", type=["jpg", "png"])

    if uploaded_file is not None:
        if selected_model == "CNN":
            prediction = make_prediction_cnn(uploaded_file, loaded_tumor_model)

            # Display the result
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
