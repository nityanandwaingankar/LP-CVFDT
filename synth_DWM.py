# synthetic_data.py
import random
import numpy as np
import pandas as pd
from lazy_decision_tree import LazyDecisionTree
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier


class Expert:
    """
    Represents an individual expert in the DWM algorithm.
    Each expert can be modeled using a base learner (e.g., a simple classifier).
    """

    def __init__(self, create_classifier):
        self.classifier = create_classifier()
        # print(f"Expert classifier is {self.classifier}")

    def classify(self, x):
        """Classify an input using the expert's classifier."""

        x = np.array(x).reshape(1, -1)  # Ensure input is 2D
        return self.classifier.predict(x)[0]

    def train(self, x, y):
        """Update the expert's classifier with a new training example."""
        # x = np.array(x).reshape(1, -1)  # Ensure input is 2D
        self.classifier.fit(x, y)


from collections import deque
import numpy as np
import random


class DynamicWeightedMajority:
    def __init__(
        self, num_classes, beta, theta, p, create_classifier, num_features, window_size
    ):
        """
        Initialize the DWM algorithm with a sliding window.
        :param num_classes: Number of classes (e.g., 2 for binary classification).
        :param beta: Factor for decreasing weights, 0 ≤ β < 1.
        :param theta: Threshold for deleting experts.
        :param p: Period between expert updates.
        :param create_classifier: Function to create a new base learner for each expert.
        :param num_features: Number of features in the input data.
        :param window_size: Size of the sliding window.
        """
        self.num_classes = num_classes
        self.beta = beta
        self.theta = theta
        self.p = p
        self.num_features = num_features
        self.create_classifier = create_classifier
        self.experts = []
        self.weights = []
        self.window_size = window_size

        # Initialize sliding window
        self.window_X = deque(maxlen=window_size)
        self.window_y = deque(maxlen=window_size)

        # Initialize the first expert with meaningful dummy data
        dummy_x = np.zeros((10, num_features))  # 10 samples of zeroed features
        dummy_y = np.zeros(10)  # 10 samples with label 0
        initial_expert = Expert(create_classifier)
        initial_expert.train(dummy_x, dummy_y)
        self.experts.append(initial_expert)
        self.weights.append(1.0)

    def predict(self, x):
        """
        Make a prediction by aggregating the weighted votes of all experts.
        :param x: Feature vector.
        :return: Predicted class label.
        """
        sigma = np.zeros(self.num_classes)
        for weight, expert in zip(self.weights, self.experts):
            prediction = expert.classify(x)
            sigma[int(prediction)] += weight
        return np.argmax(sigma)

    def update(self, x, y, iteration):
        """
        Update the DWM model with a new example.
        :param x: Feature vector.
        :param y: True label.
        :param iteration: Current iteration number.
        """
        # Add the new sample to the sliding window
        self.window_X.append(x)
        self.window_y.append(y)

        # Step 1: Compute weighted predictions and update expert weights
        sigma = np.zeros(self.num_classes)
        for idx, (weight, expert) in enumerate(zip(self.weights, self.experts)):
            prediction = expert.classify(x)
            if iteration % self.p == 0 and prediction != y:
                # Decrease weight for incorrect predictions
                self.weights[idx] *= self.beta
            sigma[int(prediction)] += weight

        # Step 2: Make a global prediction
        global_prediction = np.argmax(sigma)

        # Step 3: Periodic expert maintenance
        if iteration % self.p == 0:
            # Normalize weights
            total_weight = sum(self.weights)
            if total_weight > 0:
                self.weights = [w / total_weight for w in self.weights]

            # Remove experts with weight below threshold
            remaining_experts = []
            remaining_weights = []
            for weight, expert in zip(self.weights, self.experts):
                if weight >= self.theta:
                    remaining_experts.append(expert)
                    remaining_weights.append(weight)
            self.experts = remaining_experts
            self.weights = remaining_weights

            # Add a new expert if the global prediction was incorrect
            if global_prediction != y:
                new_expert = Expert(self.create_classifier)
                # Train the new expert on the entire dataset seen so far
                new_expert.train(np.array(self.window_X), np.array(self.window_y))
                self.experts.append(new_expert)
                self.weights.append(1.0)

        # Step 4: Retrain all experts on the entire dataset seen so far
        for expert in self.experts:
            expert.train(np.array(self.window_X), np.array(self.window_y))

        # Step 4: Retrain all experts on the sliding window
        # for expert in self.experts:
        #     for xi, yi in zip(self.window_X, self.window_y):
        #         expert.train(xi, yi)


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


# Generate synthetic data with a concept drift at sample 1000
X, y = generate_synthetic_data(n_samples=6000, n_features=10, drift_point=4000)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# param_grid = {
#     "min_samples_split": [_ for _ in range(1, 51, 5)],
#     "max_depth": [_ for _ in range(1, 100, 5)],
#     "grace_period": [_ for _ in range(1, 21, 2)],
#     "n_features": [1, 2, 3, 4, 5, 6, X.shape[1]],
# }
window_size = 50  # Sliding window size
param_grid = {
    "num_classes": [2],
    "beta": [0.2, 0.3, 0.4, 0.5, 0.8],
    "theta": [0.2, 0, 4, 0.5],
    "p": [5, 2, 3, 4, 6],
    "create_classifier": [lambda: DecisionTreeClassifier(max_depth=3)],
    "num_features": [X.shape[1]],
    "window_size": [window_size],
}

# Initialize variables to track the best parameters and accuracy
best_accuracy = 0
best_params = None

# Iterate over all parameter combinations in the grid
# Train and evaluate DWM with sliding window
best_accuracy = 0
best_params = None

for params in ParameterGrid(param_grid):
    dwm = DynamicWeightedMajority(**params)
    for i in range(len(X_train)):
        dwm.update(X_train[i], Y_train[i], i)

    # Evaluate
    correct_predictions = 0
    for i in range(len(X_test)):
        if dwm.predict(X_test[i]) == Y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(X_test)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print(f"Best Parameters: {best_params}\nBest Accuracy: {best_accuracy * 100:.2f}%")

# Print the best parameters and the corresponding accuracy
print(
    f"Lazy decision trees: using parameter grid:: \nBest Parameters: {best_params}\n Best Accuracy: {best_accuracy * 100:.2f}%"
)
