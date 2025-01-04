# synthetic_data.py
import random
import numpy as np
import pandas as pd
from lazy_decision_tree import LazyDecisionTree
from sklearn.model_selection import ParameterGrid


def generate_synthetic_data(n_samples=1000, n_features=5, drift_point=None):
    """
    Generate synthetic data with an optional drift.
    :param n_samples: Number of data points to generate.
    :param n_features: Number of features.
    :param drift_point: Point at which concept drift occurs.
    :return: Generated features and labels.
    """
    X, y = [], []
    for i in range(n_samples):
        # Generate data points before drift
        if drift_point and i >= drift_point:
            # Change in data distribution (example: mean shift)
            features = [random.gauss(5, 1) for _ in range(n_features)]
        else:
            features = [random.gauss(0, 1) for _ in range(n_features)]

        # Generate label (e.g., sum threshold classification)
        label = 1 if sum(features) > n_features * 2.5 else 0
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


# example_usage.py
# from cvfdt import CVFDT
# from synthetic_data import generate_synthetic_data

# Initialize the CVFDT model

# Generate synthetic data with a concept drift at sample 1000
X, y = generate_synthetic_data(n_samples=6000, n_features=10, drift_point=4000)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

param_grid = {
    "min_samples_split": [_ for _ in range(1, 51, 5)],
    "max_depth": [_ for _ in range(1, 100, 5)],
    "grace_period": [_ for _ in range(1, 21, 2)],
    "n_features": [1, 2, 3, 4, 5, 6, X.shape[1]],
}

# Initialize variables to track the best parameters and accuracy
best_accuracy = 0
best_params = None

# Iterate over all parameter combinations in the grid
for params in ParameterGrid(param_grid):
    # Create a new decision tree with the current parameters
    tree = LazyDecisionTree(**params)

    # Update the tree with the training data
    for i in range(len(X_train)):
        tree.update(X_train[i], Y_train[i])

    # tree.print_tree()
    # print("************\n")
    # Evaluate the tree's accuracy on the test set
    correct_predictions = 0
    total_predictions = len(X_test)

    for i in range(total_predictions):
        prediction = tree.predict(X_test[i])  # Get the prediction for the test sample
        if prediction == Y_test[i]:  # Compare with the actual label
            correct_predictions += 1

    # Calculate accuracy for the current parameter combination
    accuracy = correct_predictions / total_predictions

    # Update the best parameters and accuracy if the current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params


# Print the best parameters and the corresponding accuracy
print(
    f"Lazy decision trees: using parameter grid:: \nBest Parameters: {best_params}\n Best Accuracy: {best_accuracy * 100:.2f}%"
)
