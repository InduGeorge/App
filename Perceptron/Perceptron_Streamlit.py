import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import imdb


# Load the IMDB dataset
num_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Preprocess the data
max_len = 500
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Create a perceptron model
perceptron_model = Sequential()
perceptron_model.add(Dense(1, input_dim=max_len, activation='sigmoid'))

# Compile the model
perceptron_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
perceptron_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Evaluate the model
loss, accuracy = perceptron_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the trained model
perceptron_model.save("perceptron_model.h5")
print("Perceptron model saved as perceptron_model.h5")
