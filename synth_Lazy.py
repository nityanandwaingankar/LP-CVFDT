# synthetic_data.py
import random
import numpy as np
import pandas as pd
from lazy_decision_tree import LazyDecisionTree
from sklearn.model_selection import ParameterGrid
import time
import matplotlib.pyplot as plt


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
    "min_samples_split": [1],
    "max_depth": [16],
    "grace_period": [1],
    "n_features": [5],
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
)  # Simulate streaming process and track accuracy over time


def simulate_streaming(X_train, Y_train, X_test, Y_test, model, sleep_time=0.001):
    """
    Trains LDT first on training data, then simulates streaming using test data.
    Tracks accuracy over time and processing time per instance.

    Returns:
        accuracy_over_time: List of accuracy values.
        processing_times: List of time taken per prediction.
    """
    # Phase 1: Train the model with batch training on X_train
    train_start_time = time.time()
    for x, y in zip(X_train, Y_train):
        model.update(x, y)
    train_end_time = time.time()
    training_time = train_end_time - train_start_time
    print("Training phase completed. Starting streaming simulation...\n")

    # Phase 2: Simulate streaming with pruning occurring only at prediction
    correct_predictions = 0
    total_predictions = 0
    accuracy_over_time = []
    processing_times = []

    for i, (x, y) in enumerate(zip(X_test, Y_test)):
        start_time = time.time()
        prediction = model.predict(x)  # Prediction triggers pruning
        end_time = time.time()

        # Update performance metrics
        if prediction == y:
            correct_predictions += 1
        total_predictions += 1
        accuracy_over_time.append(correct_predictions / total_predictions)
        processing_times.append((end_time - start_time) * 1000)  # Convert to ms

        time.sleep(sleep_time)  # Simulate real-time streaming delay

        # Print updates every 500 records
        if i % 500 == 0 and i > 0:
            print(
                f"Processed {i}/{len(X_test)} records - Current Accuracy: {accuracy_over_time[-1] * 100:.2f}%"
            )

    print("Streaming simulation completed.\n")
    return accuracy_over_time, processing_times, training_time


# Generate dataset
X, y = generate_synthetic_data(n_samples=10000, n_features=5, drift_point=8000)

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
for params in ParameterGrid(param_grid):

    # Initialize Lazy Decision Tree
    ldt = LazyDecisionTree(**params)

# Simulate streaming and collect accuracy/time stats
accuracy_over_time, processing_time, training_time = simulate_streaming(
    X_train, Y_train, X_test, Y_test, ldt
)

# Plot accuracy trend over time
plt.figure(figsize=(10, 5))
plt.plot(
    range(len(accuracy_over_time)),
    accuracy_over_time,
    label="LDT Accuracy",
    color="blue",
)
plt.axvline(
    x=3000 * 0.33, color="red", linestyle="--", label="Concept Drift Point"
)  # Show drift point
plt.xlabel("Number of Streamed Records")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Time (Lazy Decision Tree)")
plt.legend()
plt.show()

# Print final results
print(f"Final Accuracy: {accuracy_over_time[-1] * 100:.2f}%")
print(f"Training Time: {training_time:.2f} seconds")

print(f"Average Processing Time per Record: {np.mean(processing_time):.7f} ms")
