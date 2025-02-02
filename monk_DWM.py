import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from dwm import DynamicWeightedMajority

# create_classifier=lambda: GaussianNB()
from sklearn.neighbors import KNeighborsClassifier

# create_classifier=lambda: KNeighborsClassifier(n_neighbors=5)
from sklearn.ensemble import RandomForestClassifier

# create_classifier = lambda: RandomForestClassifier(n_estimators=10, max_depth=5)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
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


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

import time

param_grid = {
    "num_classes": [2],
    "beta": [0.8],
    "theta": [0.2],
    "p": [5],
    "create_classifier": [lambda: DecisionTreeClassifier(max_depth=3)],
    "num_features": [6],
    "window_size": [50],
}

best_accuracy = 0

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
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print(
    f"Lazy decision trees: using parameter grid:: \nBest Parameters: {param_grid}\n Best Accuracy: {accuracy * 100:.2f}% execution time: {training_time}"
)
