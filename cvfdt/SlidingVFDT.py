import numpy as np
from collections import deque


class Node:
    id_counter = 0

    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.id = Node.id_counter
        Node.id_counter += 1
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.stats = {"class_counts": {}, "total_samples": 0}

    def is_leaf(self):
        return self.value is not None

    def split(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.left = Node()
        self.right = Node()

    def prune(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = max(self.stats["class_counts"], key=self.stats["class_counts"].get)


class SlidingVFDT:
    def __init__(self, delta=0.01, window_size=1000):
        """
        Initialize the CVFDT algorithm.
        :param delta: Confidence parameter for Hoeffding bound.
        :param window_size: Size of the sliding window.
        """
        self.delta = delta
        self.window_size = window_size
        self.tree = Node()  # Initialize with a root node
        self.sliding_window = deque(maxlen=window_size)

    def predict(self, X):
        """
        Predict the label for a single instance.
        :param X: Feature array.
        :return: Predicted label.
        """
        current_node = self.tree
        while not current_node.is_leaf():
            feature = current_node.feature
            threshold = current_node.threshold
            if feature is None or threshold is None:
                break  # In case of incomplete split setup
            current_node = (
                current_node.left if X[feature] <= threshold else current_node.right
            )
        # Return the most frequent label if statistics are available
        if current_node.stats["class_counts"]:
            return max(
                current_node.stats["class_counts"],
                key=current_node.stats["class_counts"].get,
            )
        else:
            return None  # No prediction if stats are empty

    def hoeffding_bound(self, n):
        """
        Calculate Hoeffding bound.
        :param n: Number of observations.
        :return: Hoeffding bound value.
        """
        return np.sqrt(np.log(1 / self.delta) / (2 * n))

    def update(self, X, y):
        """
        Update the decision tree with new instances.
        :param X: Features.
        :param y: Labels.
        """
        self.sliding_window.append((X, y))
        # Perform updates on the tree using the sliding window
        self._update_tree()

    def _update_tree(self):
        """
        Update the decision tree structure based on the current sliding window.
        This includes recalculating statistics, checking Hoeffding bounds,
        and adjusting nodes (splitting or pruning) dynamically.
        """
        # Track statistics for each node
        node_statistics = {}

        for X, y in self.sliding_window:
            current_node = self.tree
            while not current_node.is_leaf():
                feature = current_node.feature
                threshold = current_node.threshold
                if feature is None or threshold is None:
                    break  # Avoid comparing None with numeric values
                current_node = (
                    current_node.left if X[feature] <= threshold else current_node.right
                )
            current_node.stats["total_samples"] += 1
            current_node.stats["class_counts"][y] = (
                current_node.stats["class_counts"].get(y, 0) + 1
            )

        # Adjust the tree based on Hoeffding bounds
        for node_id, stats in node_statistics.items():
            node = self.tree.find_node_by_id(node_id)
            hoeffding_bound = self.hoeffding_bound(stats["total_samples"])
            # Add logic for splitting or pruning based on statistics
