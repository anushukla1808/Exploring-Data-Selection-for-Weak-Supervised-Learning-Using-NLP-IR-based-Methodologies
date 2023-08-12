#Import libraries
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

#Preprocess the dataset
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Function to corrupt data
def corrupt_data(images, corruption_ratio):
    corrupted = images.copy()
    num_corrupt_pixels = int(corruption_ratio * 28 * 28)
    for img in corrupted:
        idx = np.random.choice(28*28, num_corrupt_pixels, replace=False)
        img.reshape(-1)[idx] = np.random.rand(num_corrupt_pixels)
    return corrupted

for ratio in [0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    corrupted_train_images = corrupt_data(train_images, ratio)
    
    # Define the CNN model
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
    
    # Compile the model with just accuracy as a metric
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Train the model and store the history
    model.fit(corrupted_train_images, train_labels, epochs=5, validation_split=0.2)
    
    # Get model predictions on the original test data
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics using scikit-learn
    acc = accuracy_score(test_labels, predicted_classes)
    precision = precision_score(test_labels, predicted_classes, average='macro')
    recall = recall_score(test_labels, predicted_classes, average='macro')
    f1 = f1_score(test_labels, predicted_classes, average='macro')
    
    print(f"Corruption Ratio: {ratio}")
    print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
