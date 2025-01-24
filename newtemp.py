from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Fetch dataset with ID 70 (MONKS dataset)
monk_s_problems = fetch_ucirepo(id=70)

# Separate dataset into features (X) and targets (Y)
X = monk_s_problems.data.features  # Feature data as a pandas DataFrame
Y = monk_s_problems.data.targets  # Target data as a pandas DataFrame

# Initialize arrays to hold processed data
new_X = np.empty((0, X.shape[1]))  # Empty 2D array to store features
new_Y = np.array([])  # Empty 1D array to store target values

# Process the feature data row by row and convert to NumPy arrays
for _, row in X.iterrows():
    new_X = np.vstack([new_X, row.to_numpy()])  # Add new row to the feature array

# Process the target data row by row and convert to a 1D array
for _, row in Y.iterrows():
    new_Y = np.append(new_Y, row.iloc[0])  # Use iloc for proper indexing

X = new_X
y = new_Y

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize the classifier
classifier = DecisionTreeClassifier(max_depth=3)

# Train the classifier with streaming data
for i in range(len(X_train)):
    # Ensure the input is 2D for a single sample
    classifier.fit(X_train[i].reshape(1, -1), [Y_train[i]])

# Evaluate the classifier
correct_predictions = 0
total_predictions = len(X_test)

for i in range(len(X_test)):
    # Ensure the test input is 2D
    if classifier.predict(X_test[i].reshape(1, -1)) == Y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")
