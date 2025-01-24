import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# create_classifier=lambda: GaussianNB()
from sklearn.neighbors import KNeighborsClassifier

# create_classifier=lambda: KNeighborsClassifier(n_neighbors=5)


from sklearn.ensemble import RandomForestClassifier

# create_classifier = lambda: RandomForestClassifier(n_estimators=10, max_depth=5)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid


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


#######################################################
from ucimlrepo import fetch_ucirepo

# Fetch dataset with ID 70 (MONKS dataset)
monk_s_problems = fetch_ucirepo(id=70)

# Separate dataset into features (X) and targets (Y)
X = monk_s_problems.data.features  # Feature data as a pandas DataFrame
Y = monk_s_problems.data.targets  # Target data as a pandas DataFrame

print("Updating tree with streaming data...\n")

# Initialize arrays to hold processed data
new_X = np.empty((0, X.shape[1]))  # Empty 2D array to store features
new_Y = np.array([])  # Empty 1D array to store target values

# Process the feature data row by row and convert to NumPy arrays
for x1, x2 in X.iterrows():
    new_X = np.vstack([new_X, x2.to_numpy()])  # Add new row to the feature array

# Process the target data row by row and convert to a 1D array
for y1, y2 in Y.iterrows():
    new_Y = np.append(new_Y, y2[0])
# print("unique y", len(np.unique(new_Y)))

X = new_X
y = new_Y
################################################################


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Parameters
window_size = 50  # Sliding window size
param_grid = {
    "num_classes": [2],
    "beta": [0.8],
    "theta": [0.2],
    "p": [5],
    "create_classifier": [lambda: DecisionTreeClassifier(max_depth=3)],
    "num_features": [X.shape[1]],
    "window_size": [window_size],
}

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
