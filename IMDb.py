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

#Set the random seed for reproducibility
np.random.seed(7)

# Load the dataset
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Truncate and pad input sequences
max_review_length = 500
X_train = pad_sequences(X_train, maxlen=max_review_length)  
X_test = pad_sequences(X_test, maxlen=max_review_length)  

# Specify the embedding vector length
embedding_vector_length = 32

def run_experiment(X_train, y_train, X_test, y_test, corruption_ratio):
    # Corrupt the labels
    num_corrupted = int(corruption_ratio * len(y_train))
    corrupted_indices = np.random.choice(len(y_train), size=num_corrupted, replace=False)
    y_train_corrupted = y_train.copy()
    y_train_corrupted[corrupted_indices] = 1 - y_train[corrupted_indices]

    # Build the model
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    earlystop = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, y_train_corrupted, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[earlystop])

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)  # convert probabilities to class labels
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    return accuracy, precision, recall, f1_score

# Define the corruption ratios
ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

# Run the experiment for each corruption ratio
for r in ratios:
    accuracy, precision, recall, f1_score = run_experiment(X_train, y_train, X_test, y_test, r)
    print(f"Corruption Ratio: {r}, Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

