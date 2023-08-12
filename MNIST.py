# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Set random seed for reproducibility
np.random.seed(0)

# Load MNIST dataset
(data, labels), (x_test, y_test) = mnist.load_data()

data.shape
labels.shape
x_test.shape
y_test.shape

def run_experiment(data, labels, x_test, y_test, corruption_ratio):
    # Preprocess the data
    x_train = data.reshape(data.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Normalize inputs from 0-255 to 0-1
    x_train /= 255
    x_test /= 255

    num_classes = 10  # There are 10 classes (0-9) in MNIST

    # Corrupt labels
    y_train = labels.copy()
    if corruption_ratio > 0:
        indices = np.random.choice(np.arange(y_train.shape[0]), size=int(y_train.shape[0]*corruption_ratio), replace=False)
        for i in indices:
            # Generate an array with all classes except the true class
            possible_labels = np.delete(np.arange(num_classes), y_train[i])

            # Select an index at random
            incorrect_label = np.random.choice(possible_labels)
            y_train[i] = incorrect_label

    # One hot encode outputs
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Build the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=0)

    # Predict the labels for the test data
    y_pred = np.argmax(model.predict(x_test), axis=-1)

    y_true = np.argmax(y_test, axis = 1)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = np.mean(y_true == y_pred)

    return accuracy, precision, recall, f1

ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
# Run the experiment for each corruption ratio
for r in ratios:
    accuracy, precision, recall, f1 = run_experiment(data, labels, x_test, y_test, r)
    print(f"Corruption Ratio: {r}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
# Run the experiment for each corruption ratio
for r in ratios:
    accuracy, precision, recall, f1 = run_experiment(data, labels, x_test, y_test, r)
    print(f"Corruption Ratio: {r}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
