import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# Load the IMDB dataset
num_words = 10000  # Only keep the top 10,000 most frequent words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

# Preprocess the data
max_len = 500  # Limit each review to 500 words
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=max_len)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=max_len)

# Build the DNN model
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=32, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
model.fit(train_data, train_labels, epochs=5, batch_size=32, validation_split=0.2)

# Save the trained model
model.save("imdb_sentiment_model.h5")
print("Model saved as imdb_sentiment_model.h5")
