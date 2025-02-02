from lazy_decision_tree import LazyDecisionTree
from sklearn.model_selection import ParameterGrid
import numpy as np

# Importing the MONKS dataset using the UCIMLRepo library
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
    new_Y = np.append(new_Y, y2[0])  # Append the target value to the array


# Split the dataset into training and testing sets using scikit-learn
from sklearn.model_selection import train_test_split

# Perform a stratified split to maintain class balance in training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    new_X, new_Y, test_size=0.3, random_state=42, stratify=Y
)

import time

param_grid = {
    "min_samples_split": [1],
    "max_depth": [41],
    "grace_period": [1],
    "n_features": [1],
    "seed": [6],
}
best_accuracy = 0
for params in ParameterGrid(param_grid):

    tree = LazyDecisionTree(**params)

    train_start_time = time.time()

    # Update the tree with the training data
    for i in range(len(X_train)):
        tree.update(X_train[i], Y_train[i])

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
    f"Lazy decision trees: using parameter grid:: \nBest Parameters: {best_params}\n Best Accuracy: {best_accuracy * 100:.2f}% execution time: {training_time}"
)
