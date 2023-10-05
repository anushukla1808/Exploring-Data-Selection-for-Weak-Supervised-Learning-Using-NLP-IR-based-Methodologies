import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from itertools import product
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Load and Preprocess the Dataset
(trImages, trLabels), (tImages, tLabels) = mnist.load_data()
trImages, tImages = trImages / 255.0, tImages / 255.0

# Training Parameters
batchSize = 128
nEpochs = 5

# Define, Compile and Train the Models
for numUnits_L1, numUnits_L2 in [(10, None), (128, 32)]:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=np.shape(trImages[0]),name='Images'))

    if numUnits_L2:
        model.add(tf.keras.layers.Dense(units=numUnits_L1, activation=tf.nn.relu, use_bias=True, name='Dense-Relu1'))
        model.add(tf.keras.layers.Dense(units=numUnits_L2, activation=tf.nn.relu, use_bias=True, name='Dense-Relu2'))
        model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, use_bias=True, name='Logistic'))
    else:
        model.add(tf.keras.layers.Dense(units=numUnits_L1, activation=tf.nn.softmax, use_bias=True, name='Logistic'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

    # Train the model
    trHistory = model.fit(x=trImages, y=trLabels, batch_size=batchSize, epochs=nEpochs, shuffle=False, validation_split=0.0)

def precompute_neighbors(X_train, k_max):
    num_points = X_train.shape[0]
    X_train_flat = X_train.reshape((X_train.shape[0], -1))

    neighbors_dict = {}

    for i in range(num_points):
        if i % 2000 == 0:
            print(f"Precomputing neighbors for point {i}/{num_points}")

        # Compute the cosine similarities between the current point and all other points
        similarities = cosine_similarity([X_train_flat[i]], X_train_flat)[0]

        # Get the indices of the k_max nearest neighbors (excluding the point itself)
        k_nearest_indices = np.argsort(similarities)[-k_max-1:-1]

        # Store the indices and their corresponding similarity scores in the dictionary
        neighbors_dict[i] = list(zip(k_nearest_indices, similarities[k_nearest_indices]))

    return neighbors_dict

# Compute the informativeness scores using the precomputed neighbors dictionary
def compute_informativeness_scores(X_train, y_train, neighbors_dict, k, axioms):
    num_points = X_train.shape[0]
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    scores = np.zeros(num_points)

    for i in neighbors_dict.keys():
        if i % 2000 == 0:
            print(f"Processing point {i}/{num_points}")

        k_nearest_indices = [x[0] for x in neighbors_dict[i][:k]]

        neighbors_labels = y_train[k_nearest_indices]

        neighbors = X_train_flat[k_nearest_indices]

        score1, score2, score3 = 0.0, 0.0, 0.0

        # Axiom 1: Entropy of class priors
        if 1 in axioms:
            class_priors = [np.sum(neighbors_labels == cat) / len(neighbors_labels) for cat in np.unique(y_train)]
            score1 = entropy(class_priors)

        # Axiom 2: Average centroid similarity
        if 2 in axioms:
            centroids = [neighbors[neighbors_labels == cat].mean(axis=0) for cat in np.unique(y_train) if neighbors[neighbors_labels == cat].size > 0]
            if centroids:
                centroid_similarities = cosine_similarity(centroids)
                score2 = centroid_similarities.mean()
            else:
                score2 = 0.0

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

k_max = 9
# Add the noise ratios loop here, before starting the computation of neighbors and scores
noise_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
axiom_combinations = [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
k_values = [3, 5, 7, 9]
# Define percentiles for data selection
percentiles = [0.25, 0.50, 0.75, 0.90]

for noise_ratio in noise_ratios:
    print(f"Processing for noise ratio: {noise_ratio}")

    # Create a copy of the original labels
    noisy_trLabels = trLabels.copy()

    # Randomly select a proportion of the data points to introduce noise
    num_noisy_points = int(noise_ratio * len(trLabels))
    noisy_indices = np.random.choice(len(trLabels), num_noisy_points, replace=False)

    # Assign random labels to the selected data points
    random_labels = np.random.choice(10, num_noisy_points)
    noisy_trLabels[noisy_indices] = random_labels

    # Precompute the neighbors with the noisy labels
    neighbors_dict = precompute_neighbors(trImages, k_max)

    # Compute informativeness scores for different values of k with the noisy labels
    scores_dict = {}
    for axioms in axiom_combinations:
        for k in k_values:
            print(f"Computing scores for k={k} using axioms: {axioms}")
            scores = compute_informativeness_scores(trImages, noisy_trLabels, neighbors_dict, k, axioms)
            scores_dict[(k, tuple(axioms))] = scores

def create_model(numUnits_L1, numUnits_L2=None):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=np.shape(trImages[0]), name='Images'))

    if numUnits_L2:
        model.add(tf.keras.layers.Dense(units=numUnits_L1, activation=tf.nn.relu, use_bias=True, name='Dense-Relu1'))
        model.add(tf.keras.layers.Dense(units=numUnits_L2, activation=tf.nn.relu, use_bias=True, name='Dense-Relu2'))
        model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, use_bias=True, name='Logistic'))
    else:
        model.add(tf.keras.layers.Dense(units=numUnits_L1, activation=tf.nn.softmax, use_bias=True, name='Logistic'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the models using the most informative data points
structured_results = {}
for noise in noise_ratios:
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
           X_train_selected = trImages[top_indices]
           y_train_selected = noisy_trLabels[top_indices]

           for numUnits_L1, numUnits_L2 in [(10, None), (128, 32)]:
               model = create_model(numUnits_L1, numUnits_L2)

               # Train the model using the selected data points
               model.fit(x=X_train_selected, y=y_train_selected, batch_size=batchSize, epochs=nEpochs, shuffle=False, validation_split=0.1)

               # Evaluate the model on the test set
               test_acc = model.evaluate(tImages, tLabels, verbose=2)[1]

               # Get the predictions and print classification report
               y_pred = model.predict(tImages)
               y_pred_classes = np.argmax(y_pred, axis=1)

               # Get precision, recall, f1-score
               precision, recall, f1_score, _ = precision_recall_fscore_support(tLabels, y_pred_classes, average='weighted')

               print(classification_report(tLabels, y_pred_classes))

               # Store the results
               key = (noise_ratio, p, k, tuple(axioms), numUnits_L1, numUnits_L2)  # Adding noise_ratio to the key
               structured_results[key] = {'Test Accuracy': test_acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1_score}

# Print the results
for key, value in structured_results.items():
  noise_ratio, p, k, axioms, numUnits_L1, numUnits_L2 = key
  print(f"Results for noise ratio: {noise_ratio}, percentile: {p}, k: {k}, axioms: {axioms}, L1 units: {numUnits_L1}, L2 units: {numUnits_L2}")
  print(f"Test Accuracy: {value['Test Accuracy']:.4f}")
  print(f"Precision: {value['Precision']:.4f}")
  print(f"Recall: {value['Recall']:.4f}")
  print(f"F1-score: {value['F1-score']:.4f}")
  print("-" * 100)


