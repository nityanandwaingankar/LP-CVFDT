import numpy as np
import random


class LazyPrunedCVFDT:
    def __init__(self, min_samples_split=10, delta=0.01):
        # Initialize the tree with an empty dictionary and set parameters
        self.tree = {}
        self.min_samples_split = min_samples_split
        self.delta = delta

    def update_tree(self, instance, label):
        # If the tree is empty, create the root node
        if not self.tree:
            self.tree = self.create_node()
        node = self.tree
        #node["children"] = [self.create_node(), self.create_node()]

        # Traverse the tree until a leaf node is reached
        while not self.is_leaf(node):
            feature, threshold = node["split"]
            node = node["children"][0] if instance[feature] <= threshold else node["children"][1]
        
        # Update the statistics of the leaf node
        self.update_node_stats(node, instance, label)
        
        # Check if the node should be split
        if self.should_split(node):
            self.split_node(node)

    def create_node(self):
        # Create a new node with default values
        return {
            "is_leaf": True,
            "samples": [],
            "class_counts": {},
            "feature_summaries": None,
            "children": None,
            "split": None
        }

    def update_node_stats(self, node, instance, label):
        # Initialize feature summaries if not already done
        if node["feature_summaries"] is None:
            node["feature_summaries"] = [{} for _ in range(len(instance))]
        
        # Update samples and class counts
        node["samples"].append((instance, label))
        if label not in node["class_counts"]:
            node["class_counts"][label] = 0
        node["class_counts"][label] += 1
        
        # Update feature summaries
        for i, value in enumerate(instance):
            if value not in node["feature_summaries"][i]:
                node["feature_summaries"][i][value] = 0
            node["feature_summaries"][i][value] += 1

    def should_split(self, node):
        # Check if the node has enough samples to consider splitting
        if len(node["samples"]) < self.min_samples_split:
            return False
        
        # Compute the current entropy of the node
        current_entropy = self.compute_entropy(node["class_counts"])
        
        # Evaluate potential splits for each feature
        for feature_idx in range(len(node["samples"][0][0])):
            if self.evaluate_split(node, feature_idx, current_entropy):
                return True
        return False

    def evaluate_split(self, node, feature_idx, current_entropy):
        # Get unique values of the feature
        feature_values = [sample[0][feature_idx] for sample in node["samples"]]
        thresholds = set(feature_values)
        
        # Evaluate each threshold
        for threshold in thresholds:
            left_split, right_split = self.split_data(node["samples"], feature_idx, threshold)
            if not left_split or not right_split:
                continue
            weighted_entropy = self.compute_weighted_entropy(left_split, right_split, len(node["samples"]))
            if current_entropy - weighted_entropy > self.delta:
                node["split"] = (feature_idx, threshold)
                return True
        return False

    def split_data(self, samples, feature_idx, threshold):
        # Split the data based on the threshold
        left_split = [sample for sample in samples if sample[0][feature_idx] <= threshold]
        right_split = [sample for sample in samples if sample[0][feature_idx] > threshold]
        return left_split, right_split

    def compute_weighted_entropy(self, left_split, right_split, total_samples):
        # Compute the weighted entropy of the splits
        left_counts = self.compute_class_counts(left_split)
        right_counts = self.compute_class_counts(right_split)
        left_entropy = self.compute_entropy(left_counts)
        right_entropy = self.compute_entropy(right_counts)
        return (len(left_split) / total_samples * left_entropy +
                len(right_split) / total_samples * right_entropy)

    def split_node(self, node):
        # Find the best split for the node
        best_gain, best_feature, best_threshold, best_splits = self.find_best_split(node)
        if best_gain <= 0:
            return
        
        # Create child nodes and assign the splits
        left_node, right_node = self.create_node(), self.create_node()
        left_node["samples"], right_node["samples"] = best_splits
        node["split"], node["children"], node["is_leaf"] = (best_feature, best_threshold), [left_node, right_node], False

    def find_best_split(self, node):
        # Initialize variables to track the best split
        best_gain, best_feature, best_threshold, best_splits = -float("inf"), None, None, None
        total_samples, current_entropy = len(node["samples"]), self.compute_entropy(node["class_counts"])
        
        # Evaluate splits for each feature
        for feature_idx in range(len(node["samples"][0][0])):
            feature_values = [sample[0][feature_idx] for sample in node["samples"]]
            thresholds = set(feature_values)
            for threshold in thresholds:
                left_split, right_split = self.split_data(node["samples"], feature_idx, threshold)
                gain = self.compute_information_gain(left_split, right_split, total_samples, current_entropy)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold, best_splits = gain, feature_idx, threshold, (left_split, right_split)
        return best_gain, best_feature, best_threshold, best_splits

    def compute_information_gain(self, left_split, right_split, total_samples, current_entropy):
        # Compute the information gain of a split
        weighted_entropy = self.compute_weighted_entropy(left_split, right_split, total_samples)
        return current_entropy - weighted_entropy

    def predict(self, instance):
        # Traverse the tree to make a prediction
        node, path = self.tree, []
        while not self.is_leaf(node):
            path.append(node)
            feature, threshold = node["split"]
            node = node["children"][0] if instance[feature] <= threshold else node["children"][1]
        
        # Prune nodes along the path if necessary
        for n in path:
            if self.should_prune(n):
                self.prune_node(n)
        return self.get_majority_class(node)

    def should_prune(self, node):
        # Check if the node should be pruned
        if len(node["samples"]) < self.min_samples_split:
            return True
        return self.compute_relevance(node) < self.delta

    def prune_node(self, node):
        # Prune the node by making it a leaf
        node["is_leaf"] = True
        node.pop("children", None)
        node["class"] = self.get_majority_class(node)

    def is_leaf(self, node):
        # Check if the node is a leaf
        return node["is_leaf"]

    def compute_relevance(self, node):
        # Compute the relevance of the node
        classes, counts = np.unique(node["samples"], return_counts=True)
        entropy = -np.sum((counts / np.sum(counts)) * np.log2(counts / np.sum(counts)))
        return entropy

    def get_majority_class(self, node):
        # Get the majority class of the node
        labels = [label for _, label in node["samples"]]
        classes, counts = np.unique(labels, return_counts=True)
        return classes[np.argmax(counts)]

    def compute_entropy(self, class_counts):
        # Compute the entropy of the class counts
        total = sum(class_counts.values())
        entropy = 0
        for count in class_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        return entropy

    def compute_class_counts(self, samples):
        # Compute the class counts of the samples
        counts = {}
        for _, label in samples:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

def simulate_stream(n_samples):
    # Simulate a stream of data
    for _ in range(n_samples):
        feature1 = random.uniform(0, 10)
        feature2 = random.uniform(0, 10)
        label = 1 if feature1 + feature2 > 10 else 0
        yield np.array([feature1, feature2]), label

# Create a LazyPrunedCVFDT instance and simulate a data stream
tree = LazyPrunedCVFDT(min_samples_split=20, delta=0.05)
stream = simulate_stream(1000)

# Update the tree with the data stream
for instance, label in stream:
    tree.update_tree(instance, label)
    print(f"added instance{instance}")

# Simulate another data stream and make predictions
# stream = simulate_stream(1000)
# for instance, label in stream:
#     prediction = tree.predict(instance)
#     print(f"Instance: {instance}, True Label: {label}, Prediction: {prediction}")
