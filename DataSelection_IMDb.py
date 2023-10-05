#Import Libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from itertools import product

#Downloading data 
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip -d glove.6B

# Load the GloVe embeddings file
glove_file = "glove.6B/glove.6B.100d.txt"
word2vec_output_file = "glove.6B/glove.6B.100d.txt.word2vec"
glove2word2vec(glove_file, word2vec_output_file)
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Step 1: Load and preprocess the IMDb dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data = pad_sequences(train_data, maxlen=100)
test_data = pad_sequences(test_data, maxlen=100)

# Prepare the embedding matrix
vocab_size = len(imdb.get_word_index()) + 1
embedding_matrix = np.zeros((vocab_size, 100))
for word, index in imdb.get_word_index().items():
    if word in model.key_to_index:
        embedding_matrix[index] = model[word]

# Define the neural network model
def create_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=100, trainable=False),
        LSTM(units=128),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
nn_model = create_model()
nn_model.fit(train_data, train_labels, epochs=5, batch_size=128, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = nn_model.evaluate(test_data, test_labels, verbose=0)

# Get the predictions and compute precision, recall, and F1-score
y_pred = (nn_model.predict(test_data) > 0.5).astype('int32')
precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, y_pred, average='binary')

def precompute_neighbors(X_train, k_max):
    num_points = X_train.shape[0]

    neighbors_dict = {}

    for i in range(num_points):
        if i % 2000 == 0:
            print(f"Precomputing neighbors for point {i}/{num_points}")

        # Compute the cosine similarities between the current point and all other points
        similarities = cosine_similarity([X_train[i]], X_train)[0]

        # Get the indices of the k_max nearest neighbors (excluding the point itself)
        k_nearest_indices = np.argsort(similarities)[-k_max-1:-1]

        # Store the indices and their corresponding similarity scores in the dictionary
        neighbors_dict[i] = list(zip(k_nearest_indices, similarities[k_nearest_indices]))

    return neighbors_dict

def compute_informativeness_scores(X_train, y_train, neighbors_dict, k, axioms):
    num_points = X_train.shape[0]
    scores = np.zeros(num_points)

    for i in neighbors_dict.keys():
        if i % 2000 == 0:
            print(f"Processing point {i}/{num_points}")

        k_nearest_indices = [x[0] for x in neighbors_dict[i][:k]]
        neighbors_labels = y_train[k_nearest_indices]
        neighbors = X_train[k_nearest_indices]

        score1, score2, score3 = 0.0, 0.0, 0.0

        # Axiom 1: Entropy of class priors
        if 1 in axioms:
            class_priors = [np.sum(neighbors_labels == cat) / len(neighbors_labels) for cat in np.unique(y_train)]
            score1 = entropy(class_priors)

        # Axiom 2: Average centroid similarity
        if 2 in axioms:
            centroids = [neighbors[neighbors_labels == cat].mean(axis=0) for cat in np.unique(y_train) if np.any(neighbors_labels == cat)]
            if centroids:
                centroid_similarities = cosine_similarity(centroids)
                score2 = centroid_similarities.mean()

        # Axiom 3: Ratio of periphery points to core points
        if 3 in axioms:
            centroid = neighbors.mean(axis=0).reshape(1, -1)
            similarities_to_centroid = cosine_similarity(centroid, neighbors)
            core_points = np.sum(similarities_to_centroid >= np.median(similarities_to_centroid))
            periphery_points = len(neighbors) - core_points
            score3 = periphery_points / (core_points + 1e-5)

        # Calculate the final score based on the included axioms
        valid_scores = [score for score in [score1, score2, score3] if score != 0.0]
        scores[i] = np.mean(valid_scores) if valid_scores else 0.0

    return scores

# Introduce noise function
def introduce_noise(labels, noise_ratio):
    noisy_labels = labels.copy()
    num_noisy_labels = int(noise_ratio * len(labels))
    noisy_label_indices = random.sample(range(len(labels)), num_noisy_labels)
    for idx in noisy_label_indices:
        noisy_labels[idx] = 1 - noisy_labels[idx]  # Flip the label
    return noisy_labels

noise_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
k_values = [3, 5, 7, 9]
axiom_combinations = [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
percentiles = [0.25, 0.50, 0.75, 0.90]
results = {}


k_max = 9  # Maximum value of k for which we will compute the neighbors
neighbors_dict = precompute_neighbors(train_data, k_max)

for noise_ratio in noise_ratios:
    print(f"Introducing noise with ratio: {noise_ratio}")
    noisy_train_labels = introduce_noise(train_labels, noise_ratio)

    scores_dict = {}
    for axioms in axiom_combinations:
        for k in k_values:
            print(f"Computing scores for k={k} using axioms: {axioms}")
            scores = compute_informativeness_scores(train_data, noisy_train_labels, neighbors_dict, k, axioms)
            scores_dict[(k, tuple(axioms))] = scores

for noise_ratio in noise_ratios:
    print(f"Introducing noise with ratio: {noise_ratio}")

    # Introduce noise into the training labels
    noisy_train_labels = introduce_noise(train_labels, noise_ratio)


    for p, k in product(percentiles, k_values):
      for axioms in axiom_combinations:
        print(f"Processing for percentile: {p}, k: {k}, axioms: {axioms}")
        scores = scores_dict[(k, tuple(axioms))]

        # Normalize scores to [0, 1]
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        # Select the top p percentile of points
        sorted_indices = np.argsort(normalized_scores)
        top_indices = sorted_indices[:int(p * len(sorted_indices))]

        # Get the most informative data points
        X_train_selected = train_data[top_indices]
        y_train_selected = noisy_train_labels[top_indices]

        # Train and evaluate the model using the selected data points
        nn_model = create_model()
        nn_model.fit(X_train_selected, y_train_selected, epochs=5, batch_size=128, validation_split=0.2)

        # Evaluate the model on the test set
        test_acc = nn_model.evaluate(test_data, test_labels, verbose=2)[1]

        # Get the predictions and print classification report
        y_pred = (nn_model.predict(test_data) > 0.5).astype('int32')

        # Get precision, recall, f1-score
        precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, y_pred, average='binary')

        print(classification_report(test_labels, y_pred))

        # Store the results
        key = (p, k, tuple(axioms))  # Creating a tuple key to store the results
        results[key] = {'Test Accuracy': test_acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1_score}

# Step 9: Print the results
for key, value in results.items():
    p, k, axioms = key
    print(f"Results for percentile: {p}, k: {k}, axioms: {axioms}")
    print(f"Test Accuracy: {value['Test Accuracy']:.4f}")
    print(f"Precision: {value['Precision']:.4f}")
    print(f"Recall: {value['Recall']:.4f}")
    print(f"F1-score: {value['F1-score']:.4f}")
    print("-" * 100)

