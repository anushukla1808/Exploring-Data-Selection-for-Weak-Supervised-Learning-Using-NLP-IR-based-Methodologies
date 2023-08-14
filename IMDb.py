#Import necessary libraries
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Set the random seed for reproducibility
np.random.seed(7)

# Load and pad the dataset once
top_words = 5000
(X_original, y_original), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500

X_original_padded = pad_sequences(X_original, maxlen=max_review_length)
X_test_padded = pad_sequences(X_test, maxlen=max_review_length)
embedding_vector_length = 32

def run_experiment(X_train, y_train, X_test, y_test, corruption_ratio=0.0):
    if corruption_ratio:
        num_corrupted = int(corruption_ratio * len(y_train))
        corrupted_indices = np.random.choice(len(y_train), size=num_corrupted, replace=False)
        y_train[corrupted_indices] = 1 - y_train[corrupted_indices]

    # Build the model
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    earlystop = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[earlystop])

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    return accuracy, precision, recall, f1_score

# Evaluate on clean data
accuracy, precision, recall, f1_score = run_experiment(X_original_padded, y_original.copy(), X_test_padded, y_test)
print(f"Corruption Ratio: 0, Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

# For each corruption ratio
for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
    accuracy, precision, recall, f1_score = run_experiment(X_original_padded.copy(), y_original.copy(), X_test_padded, y_test, r)
    print(f"Corruption Ratio: {r}, Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")
