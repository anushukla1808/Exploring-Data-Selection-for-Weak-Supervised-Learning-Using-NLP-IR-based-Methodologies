# Exploring-Data-Selection-for-Weak-Supervised-Learning-Using-NLP-IR-based-Methodologies
The project is a part of my Masters degree. It primarily focuses on data selection for weak supervised learning. In deep learning sometimes, the available data may be noisy. This drawback can make it challenging to train the program effectively and may affect the results. The idea of QPP (Query Performance Prediction), NLP/IR technique, will be used to estimate the quality of the neighborhood of a data instance. It is useful when both the data instances and the labels are noisy as it provides an estimate of the reliability of a training data instance. The goal is to train a classifier on clean instances to achieve better results, drawing an analogy between a query and a data instance, and the top-k documents and the neighborhood of other points around a data instance.

# Weakly Supervised Learning with Data Selection

This repository provides the implementation of a weakly supervised learning approach with a data selection strategy using informativeness scores. It explores the model's performance under various conditions, particularly focusing on the introduction of artificial label noise and selection of subsets of training data based on their informativeness. The code is implemented using TensorFlow and executed on two datasets: MNIST and IMDb.

## Overview

1. **MNIST**: A dataset of 28x28 grayscale images of handwritten digits (0 through 9) and their respective labels. The task is to categorize each image under one of the 10 digit classes.
   
2. **IMDb**: A dataset consisting of movie reviews labeled as either positive (1) or negative (0) based on the sentiment expressed. The task is to classify the sentiment of each review.

The code performs the following primary operations:

- **Model Training**: Train a neural network model under standard conditions.
- **Noise Introduction**: Introduce artificial label noise to explore the model's robustness.
- **Data Selection**: Employ a data selection strategy based on precomputed informativeness scores to determine valuable training instances.
- **Evaluation**: Assess the model's performance under varied noise and data selection scenarios.

## Repository Structure

- `DataSelection_MNIST.py`: Implementation on the MNIST dataset.
- `DataSelection_IMDb.py`: Implementation on the IMDb dataset.
- `glove.6B/`: Directory containing the GloVe word embeddings (required for IMDb implementation).

## Dependencies

- TensorFlow
- NumPy
- Scikit-learn
- Gensim

## Usage

### MNIST Implementation

To execute the MNIST implementation, run:

```bash
python DataSelection_MNIST.py
```

### IMDb Implementation

First, ensure that the GloVe embeddings are available in the `glove.6B/` directory. Then, run:

```bash
python DataSelection_IMDb.py
```

## Implementation Details

### Model Training

- **MNIST**: Utilizes a simple neural network model.
- **IMDb**: Employs a model with an embedding layer initialized with GloVe embeddings, followed by an LSTM layer.

### Data Selection Strategy

Involves computing informativeness scores for each training instance based on:

- **Axiom 1**: Entropy of class priors in the neighborhood.
- **Axiom 2**: Average similarity between class centroids.
- **Axiom 3**: Ratio of periphery points to core points in the neighborhood.

### Experiments

For both datasets, the following experiments are performed:

1. **Label Noise**: Introduce various levels of label noise to the training data.
2. **Data Selection**: Utilize different percentiles of data based on informativeness scores for training.
3. **Evaluation**: Analyze the model's performance using metrics like accuracy, precision, recall, and F1-score under varied settings.

## Results and Analysis

The results provide insights into the model's learning efficacy under different levels of label noise and data selection strategies. Comprehensive results and analyses can be found in the respective code files (`DataSelection_MNIST.py` and `DataSelection_IMDb.py`).

## License

This project is open source, under the [MIT License](LICENSE).


