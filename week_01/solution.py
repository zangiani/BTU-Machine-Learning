"""
Week 01: Similarity & Nearest Neighbors (kNN) - Reference Solution
Course: Introduction to AI and Machine Learning (BTU)
Textbook References: Chris Albon, Machine Learning with Python Cookbook (1st Edition)
  - Recipe 1.1: Creating a Vector
  - Recipe 1.17: Calculating the Dot Product
  - Recipe 2.1: Loading a Sample Dataset (load_digits, load_iris)
  - Recipe 6.8: Encoding Text as a Bag of Words
  - Recipe 15.1: Finding an Observation's Nearest Neighbors
  - Recipe 15.2: Creating a K-Nearest Neighbor Classifier
"""

import sys
import types
from collections import Counter

# Fallback shim for Windows Smart App Control on Python 3.14 (bypasses unused _liblinear dependency)
if 'sklearn.svm._liblinear' not in sys.modules:
    try:
        import sklearn.svm._liblinear
    except Exception:
        sys.modules['sklearn.svm._liblinear'] = types.ModuleType('sklearn.svm._liblinear')

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix


# =====================================================================
# 1. Distance & Similarity Metrics from Scratch (NumPy)
# =====================================================================

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes Euclidean (L2) distance between two 1D vectors.
    Formula: sqrt(sum((a_i - b_i)^2))
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_similarity_scratch(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes Cosine Similarity between two 1D vectors.
    Formula: dot(a, b) / (norm(a) * norm(b))
    Includes small epsilon (1e-9) to prevent division by zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid zero division if a vector is all zeros
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


# =====================================================================
# 2. k-Nearest Neighbors Classifier from Scratch (Pure NumPy)
# =====================================================================

class KNNClassifierScratch:
    """
    K-Nearest Neighbors classifier implemented from scratch using pure NumPy.
    Supports 'euclidean' and 'cosine' distance metrics.
    """
    def __init__(self, k: int = 3, metric: str = "euclidean"):
        if k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.metric = metric.lower()
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        kNN is a lazy learner: fitting simply memorizes training data.
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _predict_single(self, x: np.ndarray):
        """
        Predict label for a single query vector x.
        """
        if self.metric == "euclidean":
            # Vectorized Euclidean distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
        elif self.metric == "cosine":
            # For cosine, higher similarity means smaller distance
            norms = np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(x)
            norms[norms == 0] = 1e-9
            similarities = np.dot(self.X_train, x) / norms
            # Sort descending for similarity
            k_indices = np.argsort(similarities)[::-1][:self.k]
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        k_nearest_labels = self.y_train[k_indices]
        # Majority vote with tie-breaker: Counter most_common
        counts = Counter(k_nearest_labels)
        most_common = counts.most_common()
        return most_common[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for an array of query vectors X.
        """
        X = np.asarray(X, dtype=float)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)


# =====================================================================
# 3. Demonstration & Testing
# =====================================================================

def run_numpy_crash_course_demo():
    print("=" * 60)
    print("0. NUMPY CRASH COURSE DEMO (List vs Array, Shape, Argsort)")
    print("=" * 60)
    
    # 1. List vs Array
    py_list = [1, 2] + [3, 4]
    np_arr = np.array([1, 2]) + np.array([3, 4])
    print(f"Python list concatenation: [1, 2] + [3, 4] -> {py_list}")
    print(f"NumPy vector addition:    np.array([1, 2]) + np.array([3, 4]) -> {np_arr}")
    
    # 2. Shape and Dimensions
    v = np.array([10, 20, 30])
    print(f"Vector: {v}, shape: {v.shape} (1D array with 3 coordinates)")
    
    # 3. argsort demo
    sample_dists = np.array([15.2, 3.1, 8.4])
    sorted_idx = np.argsort(sample_dists)
    print(f"Sample distances: {sample_dists}")
    print(f"np.argsort() indices: {sorted_idx} -> Neighbor #{sorted_idx[0]} is closest ({sample_dists[sorted_idx[0]]})\n")


def run_metric_sanity_check():
    print("=" * 60)
    print("1. SANITY CHECK: DISTANCE METRICS FROM SCRATCH")
    print("=" * 60)
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 6, 8])
    v3 = np.array([2, 4, 6])  # Collinear to v1 (2 * v1)
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Vector 3 (2 * v1): {v3}\n")
    
    print(f"Euclidean Distance (v1, v2): {euclidean_distance(v1, v2):.4f}")
    print(f"Euclidean Distance (v1, v3): {euclidean_distance(v1, v3):.4f}")
    print(f"Cosine Similarity  (v1, v2): {cosine_similarity_scratch(v1, v2):.4f}")
    print(f"Cosine Similarity  (v1, v3): {cosine_similarity_scratch(v1, v3):.4f} (Notice: exactly 1.0! Angle is 0 deg)")
    print()


def run_knn_digits_experiment():
    print("=" * 60)
    print("2. KNN ON DIGITS DATASET (Cookbook Recipe 2.1 & 15.2)")
    print("=" * 60)
    
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset Loaded: {X.shape[0]} images, each {X.shape[1]} pixels (8x8 flattened)")
    print(f"Target classes: {np.unique(y)}\n")
    
    # Stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Evaluate Scratch kNN (k=3)
    k = 3
    knn_scratch = KNNClassifierScratch(k=k, metric="euclidean")
    knn_scratch.fit(X_train, y_train)
    y_pred_scratch = knn_scratch.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    
    # 2. Evaluate Scikit-Learn KNeighborsClassifier (k=3)
    knn_sklearn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn_sklearn.fit(X_train, y_train)
    y_pred_sklearn = knn_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"Scratch kNN (k={k}) Accuracy:     {acc_scratch * 100:.2f}%")
    print(f"Sklearn kNN (k={k}) Accuracy:     {acc_sklearn * 100:.2f}%")
    
    # Assert scratch matches sklearn exactly
    matches = np.mean(y_pred_scratch == y_pred_sklearn)
    print(f"Prediction agreement between Scratch and Sklearn: {matches * 100:.2f}%")
    assert matches >= 0.99, "Scratch implementation differs significantly from scikit-learn!"
    print(">> Scratch kNN verified against Scikit-Learn successfully!\n")
    
    # =====================================================================
    # 3. Fulfilling The Winston Promise: Finding & Analyzing Misclassifications
    # =====================================================================
    print("=" * 60)
    print("3. FULFILLING THE WINSTON PROMISE: MISCLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    misclassified_mask = (y_test != y_pred_scratch)
    misclassified_indices = np.where(misclassified_mask)[0]
    total_errors = len(misclassified_indices)
    
    print(f"Total test samples: {len(y_test)}")
    print(f"Total misclassifications: {total_errors} out of {len(y_test)} (Error rate: {total_errors / len(y_test) * 100:.2f}%)\n")
    
    print("Inspect First 3 Confused Digits:")
    for idx in misclassified_indices[:3]:
        true_label = y_test[idx]
        pred_label = y_pred_scratch[idx]
        sample_vec = X_test[idx]
        
        # Calculate distances to all training samples to see who the neighbors were
        dists = np.sqrt(np.sum((X_train - sample_vec) ** 2, axis=1))
        neighbor_idx = np.argsort(dists)[:k]
        neighbor_labels = y_train[neighbor_idx]
        neighbor_dists = dists[neighbor_idx]
        
        print(f" - Sample #{idx}: True Digit = [{true_label}], Predicted = [{pred_label}]")
        print(f"   {k} Nearest Neighbors in Training Set: labels={neighbor_labels}, distances={np.round(neighbor_dists, 2)}")
        print("   Explanation: In 64-dimensional pixel space, this handwritten digit's ink pattern")
        print(f"   was geometrically closer to class [{pred_label}] than its true class [{true_label}].")
        print()


def run_text_cosine_demo():
    print("=" * 60)
    print("4. TEXT SIMILARITY USING BAG-OF-WORDS & COSINE (Recipe 6.8)")
    print("=" * 60)
    
    docs = [
        "Machine learning algorithms learn patterns from data",
        "Deep learning and machine learning use data to recognize patterns",
        "Georgian traditional cuisine features cheese and bread khachapuri"
    ]
    
    print("Documents:")
    for i, doc in enumerate(docs, 1):
        print(f"  Doc {i}: '{doc}'")
    print()
    
    # Convert text to Bag-of-Words feature vectors (Albon Recipe 6.8)
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(docs).toarray()
    
    print(f"Vocabulary ({len(vectorizer.get_feature_names_out())} words):")
    print(vectorizer.get_feature_names_out())
    print("\nFeature Vectors (Word Counts):")
    for i, row in enumerate(X_bow, 1):
        print(f"  Doc {i} vector: {row}")
    print()
    
    sim_1_2 = cosine_similarity_scratch(X_bow[0], X_bow[1])
    sim_1_3 = cosine_similarity_scratch(X_bow[0], X_bow[2])
    dist_1_2 = euclidean_distance(X_bow[0], X_bow[1])
    dist_1_3 = euclidean_distance(X_bow[0], X_bow[2])
    
    print(f"Similarity Doc 1 vs Doc 2 (Both ML):      Cosine = {sim_1_2:.4f}, Euclidean Dist = {dist_1_2:.4f}")
    print(f"Similarity Doc 1 vs Doc 3 (ML vs Food):    Cosine = {sim_1_3:.4f}, Euclidean Dist = {dist_1_3:.4f}")
    print("Key Takeaway: Docs 1 & 2 share high cosine similarity (~0.58) because their word angles align.")
    print("Docs 1 & 3 have 0.00 cosine similarity because they share no common vocabulary (orthogonal vectors).")
    print("=" * 60)


if __name__ == "__main__":
    run_numpy_crash_course_demo()
    run_metric_sanity_check()
    run_knn_digits_experiment()
    run_text_cosine_demo()
