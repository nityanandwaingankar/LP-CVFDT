import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from dwm import DynamicWeightedMajority


def generate_rotating_hyperplane_with_drift(
    n_samples, n_features, noise=0.1, drift_rate=0.01, random_state=None
):
    """
    Generates synthetic data for a rotating hyperplane with concept drift.

    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features for the data.
        noise (float): Noise level (proportion of labels flipped).
        drift_rate (float): Rate at which concept drift occurs (controls weight change).
        random_state (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: Feature matrix (X).
        np.ndarray: Labels (y).
    """
    rng = np.random.default_rng(random_state)

    # Initialize weights and bias for the hyperplane
    weights = rng.uniform(-1, 1, size=n_features)
    bias = 0.0  # Hyperplane threshold

    # Store the generated data
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        # Generate a random feature vector
        x = rng.uniform(0, 1, size=n_features)

        # Compute the label based on the current hyperplane
        decision_boundary = np.dot(weights, x)
        label = 1 if decision_boundary >= bias else 0

        # Introduce noise by flipping some labels
        if rng.uniform(0, 1) < noise:
            label = 1 - label

        # Store the sample and label
        X[i] = x
        y[i] = label

        # Introduce drift: Slowly update weights
        drift = rng.uniform(-drift_rate, drift_rate, size=n_features)
        weights += drift  # Incrementally adjust weights

    return X, y


# Parameters
n_samples = 1000
n_features = 5
noise = 0.1
drift_rate = 0.02
random_state = 47

# Generate synthetic data with concept drift
X, y = generate_rotating_hyperplane_with_drift(
    n_samples=n_samples,
    n_features=n_features,
    noise=noise,
    drift_rate=drift_rate,
    random_state=random_state,
)

X = X.reshape(-1, X.shape[-1])  # Flatten to (time_steps * n_samples, n_features)
y = y.flatten()
print("Flattened Feature Matrix Shape:", X[0:5])  # (time_steps * n_samples, n_features)
print("Flattened Labels Shape:", y[0:5])


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

import time

param_grid = {
    "num_classes": [2],
    "beta": [0.2],
    "theta": [0.4],
    "p": [8],
    "create_classifier": [lambda: DecisionTreeClassifier(max_depth=3)],
    "num_features": [5],
    "window_size": [60],
}

best_accuracy = 0
best_params = {}

for params in ParameterGrid(param_grid):

    tree = DynamicWeightedMajority(**params)

    train_start_time = time.time()

    # Update the tree with the training data
    for i in range(len(X_train)):
        tree.update(X_train[i], Y_train[i], i)

    train_end_time = time.time()  # End training time
    training_time = train_end_time - train_start_time
    # Evaluate the tree's accuracy on the test set
    correct_predictions = 0
    total_predictions = len(X_test)

    for i in range(total_predictions):
        prediction = tree.predict(X_test[i])  # Get the prediction for the test sample
        if prediction == Y_test[i]:  # Compare with the actual label
            correct_predictions += 1

    # Calculate accuracy for the current parameter combination
    accuracy = correct_predictions / total_predictions
    # Print the best parameters and the corresponding accuracy
    accuracy = correct_predictions / len(X_test)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print(
    f"Lazy decision trees: using parameter grid:: \nBest Parameters: {best_params}\n Best Accuracy: {best_accuracy * 100:.2f}% execution time: {training_time}"
)
