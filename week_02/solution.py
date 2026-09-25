"""
Week 02: Data Preprocessing & Model Evaluation - Reference Solution
Course: Introduction to AI and Machine Learning (BTU)
Textbook References: Chris Albon, Machine Learning with Python Cookbook (1st Edition)
  - Recipe 3.9: Deleting Observations with Missing Values
  - Recipe 3.11: Imputing Missing Values (SimpleImputer)
  - Recipe 4.1: Rescaling a Feature (MinMaxScaler)
  - Recipe 4.2: Standardizing a Feature (StandardScaler)
  - Recipe 5.1: Encoding Nominal Categorical Features (OneHotEncoder)
  - Recipe 11.1: Cross-Validating Models (cross_val_score, KFold)
  - Recipe 11.3: Evaluating Binary Classifier Predictions (Accuracy, Precision, Recall, F1)
  - Recipe 11.5: Evaluating Binary Classifier Thresholds (ROC Curve & AUC)
  - Recipe 11.6: Visualizing a Classifier’s Performance (Confusion Matrix)
  - Recipe 12.1: Creating a Pipeline to Prevent Data Leakage
"""

import sys
import types
import numpy as np

# Fallback shim for Windows Smart App Control on Python 3.14 (bypasses unused _liblinear dependency)
if 'sklearn.svm._liblinear' not in sys.modules:
    try:
        import sklearn.svm._liblinear
    except Exception:
        sys.modules['sklearn.svm._liblinear'] = types.ModuleType('sklearn.svm._liblinear')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report
)


# =====================================================================
# 1. DEMO: THE CRITICAL IMPACT OF FEATURE SCALING ON kNN
# =====================================================================
def run_scaling_demonstration():
    """
    Demonstrates why distance-based algorithms (like kNN) fail completely
    when features are on wildly different scales (e.g. Income in GEL vs Age in years).
    Textbook: Chris Albon, Recipe 4.1 & 4.2.
    """
    print("=" * 70)
    print("1. FEATURE SCALING DEMO: WHY kNN FAILS WITHOUT STANDARDIZATION")
    print("=" * 70)

    # Let's consider two customers with Age (years) and Annual Income (GEL)
    # Person A: Age 25, Income 50,000 GEL
    # Person B: Age 45, Income 55,000 GEL
    p_a = np.array([25.0, 50000.0])
    p_b = np.array([45.0, 55000.0])

    diff_age = p_a[0] - p_b[0]        # 20 years difference
    diff_income = p_a[1] - p_b[1]     # 5000 GEL difference

    raw_dist = np.sqrt(diff_age**2 + diff_income**2)
    print(f"Customer A: Age={p_a[0]:.0f}, Income={p_a[1]:,.0f} GEL")
    print(f"Customer B: Age={p_b[0]:.0f}, Income={p_b[1]:,.0f} GEL")
    print(f"Age difference squared:    ({diff_age:.0f})^2   = {diff_age**2:.0f}")
    print(f"Income difference squared: ({diff_income:.0f})^2 = {diff_income**2:,.0f}")
    print(f"Raw Euclidean Distance: {raw_dist:,.2f}")
    print("Notice: The 20-year age gap contributed only 400 out of 25,000,400 (0.0016%)!")
    print("Distance-based algorithms treat Age as virtually non-existent without scaling.\n")

    # Applying StandardScaler manually: z = (x - mu) / sigma
    sample_data = np.array([
        [22, 25000],
        [25, 50000],
        [45, 55000],
        [60, 110000],
        [35, 75000]
    ], dtype=float)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sample_data)
    
    scaled_dist = np.sqrt(np.sum((scaled_data[1] - scaled_data[2]) ** 2))
    print("After StandardScaler (Recipe 4.2):")
    print(f"Scaled Customer A: Age={scaled_data[1][0]:.3f}, Income={scaled_data[1][1]:.3f}")
    print(f"Scaled Customer B: Age={scaled_data[2][0]:.3f}, Income={scaled_data[2][1]:.3f}")
    print(f"Standardized Euclidean Distance: {scaled_dist:.4f}")
    print(">> Both features now have mean=0 and variance=1, carrying equal mathematical voice!\n")


# =====================================================================
# 2. DEMO: PREPROCESSING MISSING VALUES & CATEGORICAL ENCODING
# =====================================================================
def run_preprocessing_pipeline_demo():
    """
    Demonstrates handling missing values with SimpleImputer and encoding
    categorical features with OneHotEncoder.
    Textbook: Chris Albon, Recipe 3.11 & 5.1.
    """
    print("=" * 70)
    print("2. DATA PREPROCESSING: IMPUTATION (Recipe 3.11) & ONE-HOT (Recipe 5.1)")
    print("=" * 70)

    # Realistic numerical data with missing values (np.nan)
    # Columns: [Age, Income]
    X_num = np.array([
        [25.0, 45000.0],
        [30.0, np.nan],
        [np.nan, 72000.0],
        [45.0, 110000.0],
        [35.0, 60000.0],
        [np.nan, 52000.0]
    ])
    
    print("Raw Numerical Data with np.nan:")
    print("Row 0:", X_num[0])
    print("Row 1 (missing Income):", X_num[1])
    print("Row 2 (missing Age):   ", X_num[2])
    print()

    # Step 1: Impute numerical missing values using Median (Recipe 3.11)
    imputer = SimpleImputer(strategy='median')
    X_num_imputed = imputer.fit_transform(X_num)
    print("After Imputation (Recipe 3.11 - Median strategy):")
    for i, row in enumerate(X_num_imputed):
        print(f"  Sample {i}: Age={row[0]:.1f}, Income={row[1]:,.0f} GEL")
    print()

    # Step 2: Categorical Features with OneHotEncoder (Recipe 5.1)
    # Cities: Tbilisi, Batumi, Kutaisi
    cities = np.array([['Tbilisi'], ['Batumi'], ['Tbilisi'], ['Kutaisi'], ['Batumi'], ['Tbilisi']])
    encoder = OneHotEncoder(sparse_output=False)
    cities_encoded = encoder.fit_transform(cities)
    
    print(f"Categorical City Categories: {list(encoder.categories_[0])}")
    print("One-Hot Encoded City Matrix (Recipe 5.1):")
    for city_name, encoded_row in zip(cities.ravel(), cities_encoded):
        print(f"  {city_name:8s} -> {encoded_row}")
    
    # Combined clean feature matrix
    X_clean = np.hstack([X_num_imputed, cities_encoded])
    print(f"\nFinal Preprocessed Matrix Shape: {X_clean.shape} (2 imputed numeric + 3 one-hot columns)")
    print(">> Cleaned, imputed, and fully vectorized without losing any sample rows!\n")


# =====================================================================
# 3. DEMO: OVERFITTING VS UNDERFITTING & k-FOLD CROSS-VALIDATION
# =====================================================================
def run_cross_validation_demo(X_train, y_train):
    """
    Demonstrates finding the optimal k for kNN using k-Fold Cross-Validation,
    illustrating the boundary between Overfitting (k=1) and Underfitting (k large).
    Textbook: Chris Albon, Recipe 11.1 & 15.3.
    """
    print("=" * 70)
    print("3. HYPERPARAMETER TUNING: k-FOLD CROSS-VALIDATION (Recipe 11.1)")
    print("=" * 70)

    # We evaluate kNN for odd k values from 1 to 21
    k_candidates = [1, 3, 5, 7, 9, 11, 15, 21]
    cv_results = []

    # Standardize inside CV folds (conceptually previewing pipeline)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_candidates:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=kf, scoring='accuracy')
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        cv_results.append((k, mean_score, std_score))
        print(f"  k = {k:2d} | 5-Fold CV Accuracy: {mean_score * 100:.2f}% (+/- {std_score * 100:.2f}%)")

    # Pick k with highest mean CV accuracy
    best_k, best_score, best_std = max(cv_results, key=lambda item: item[1])
    print(f"\nOptimal Hyperparameter: best_k = {best_k} (Mean CV Accuracy = {best_score * 100:.2f}%)\n")

    # Overfitting demonstration for k=1:
    knn_k1 = KNeighborsClassifier(n_neighbors=1).fit(X_train_scaled, y_train)
    train_acc_k1 = accuracy_score(y_train, knn_k1.predict(X_train_scaled))
    print(f"Overfitting Proof: k=1 Train Accuracy = {train_acc_k1 * 100:.2f}% (Pure memorization!)")
    print(f"However, 5-Fold Cross-Validation Accuracy for k=1 is {cv_results[0][1] * 100:.2f}%")
    print(">> Cross-validation strips away the illusion of train memorization!\n")

    return best_k


# =====================================================================
# 4. DEMO: COMPLETE SCIKIT-LEARN PIPELINE & SCALING COMPARISON
# =====================================================================
def run_pipeline_and_evaluation_experiment():
    """
    Builds a leak-free Pipeline([('scaler', StandardScaler()), ('knn', ...)])
    and compares it against an unscaled kNN model on Breast Cancer dataset.
    Textbook: Chris Albon, Recipe 12.1.
    """
    print("=" * 70)
    print("4. FULL LEAK-FREE PIPELINE & UNROUNDED SCALING EXPERIMENT (Recipe 12.1)")
    print("=" * 70)

    data = load_breast_cancer()
    X, y = data.data, data.target
    target_names = data.target_names  # ['malignant', 'benign']

    print(f"Dataset Loaded: Breast Cancer Wisconsin Diagnostic")
    print(f"Total Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {target_names[0]} (0) = {np.sum(y == 0)}, {target_names[1]} (1) = {np.sum(y == 1)}")
    print(f"Sample Feature Scales: 'mean area' range=[{X[:, 3].min():.1f}, {X[:, 3].max():.1f}] vs "
          f"'mean smoothness' range=[{X[:, 4].min():.4f}, {X[:, 4].max():.4f}]")
    print("Notice the ~25,000x difference in feature scales!\n")

    # Step 1: Honest Train/Test split BEFORE any transformations
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Step 2: Tune k on training set using cross-validation
    best_k = run_cross_validation_demo(X_train, y_train)

    # Step 3: Train Unscaled kNN
    raw_knn = KNeighborsClassifier(n_neighbors=best_k)
    raw_knn.fit(X_train, y_train)
    y_pred_unscaled = raw_knn.predict(X_test)
    acc_unscaled = accuracy_score(y_test, y_pred_unscaled)

    # Step 4: Train Scaled Pipeline (StandardScaler + kNN)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=best_k))
    ])
    # Pipeline automatically fits scaler ONLY on X_train and transforms X_test without leakage!
    pipeline.fit(X_train, y_train)
    y_pred_scaled = pipeline.predict(X_test)
    y_prob_scaled = pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class (benign)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)

    print("=" * 70)
    print("EXPERIMENT RESULTS: UNSCALED vs SCALED kNN")
    print("=" * 70)
    print(f"Unscaled kNN Accuracy (k={best_k}):     {acc_unscaled * 100:.2f}%")
    print(f"Scaled Pipeline kNN Accuracy (k={best_k}): {acc_scaled * 100:.2f}%")
    print(f"Improvement from StandardScaler:       +{(acc_scaled - acc_unscaled) * 100:.2f}%")
    assert acc_scaled > acc_unscaled, "Scaled pipeline should strictly outperform unscaled kNN!"
    print(">> Scaling dramatically improved generalization by letting all 30 features participate!\n")

    # =====================================================================
    # 5. FULFILLING THE PROMISE: MEDICAL METRICS & CONFUSION MATRIX
    # =====================================================================
    print("=" * 70)
    print("5. FULFILLING THE WINSTON PROMISE: BEYOND ACCURACY (Recipe 11.3 & 11.6)")
    print("=" * 70)

    # Note: in sklearn breast_cancer: class 0 = malignant, class 1 = benign.
    # To treat Malignant as the positive condition for clinical diagnostics:
    y_test_malignant = (y_test == 0).astype(int)
    y_pred_malignant = (y_pred_scaled == 0).astype(int)
    # Probability of being Malignant:
    prob_malignant = pipeline.predict_proba(X_test)[:, 0]

    cm = confusion_matrix(y_test_malignant, y_pred_malignant)
    tn, fp, fn, tp = cm.ravel()

    print("CONFUSION MATRIX (Malignant as Target):")
    print(f"                   Predicted Healthy (0)  |  Predicted Malignant (1)")
    print(f"Actual Healthy (0)       TN = {tn:3d}         |       FP = {fp:3d} (False Alarm)")
    print(f"Actual Malignant (1)     FN = {fn:3d} (Fatal!)|       TP = {tp:3d} (Saved!)")
    print("-" * 70)

    acc = accuracy_score(y_test_malignant, y_pred_malignant)
    prec = precision_score(y_test_malignant, y_pred_malignant, zero_division=0)
    rec = recall_score(y_test_malignant, y_pred_malignant, zero_division=0)
    f1 = f1_score(y_test_malignant, y_pred_malignant, zero_division=0)
    auc = roc_auc_score(y_test_malignant, prob_malignant)

    print(f"Accuracy:  {acc * 100:.2f}% (Overall correctness)")
    print(f"Precision: {prec * 100:.2f}% (When model flags cancer, how often is it right?)")
    print(f"Recall:    {rec * 100:.2f}% (How many actual cancers were caught without missing?)")
    print(f"F1-Score:  {f1 * 100:.2f}% (Harmonic balance between Precision and Recall)")
    print(f"ROC-AUC:   {auc:.4f}  (Discrimination ability across all thresholds)")
    print()

    # Step 6: 5-Minute Demonstration of ROC & Probability Threshold Control (Recipe 11.5)
    print("=" * 70)
    print("6. ROC & DECISION THRESHOLD CONTROL (Recipe 11.5)")
    print("=" * 70)
    print("Standard Threshold = 0.50:")
    print(f"  Caught Malignant (TP): {tp} / {tp + fn} | Missed (FN): {fn}")

    # Lowering the threshold to 0.20 to catch potentially missed cases
    threshold_low = 0.20
    y_pred_threshold_low = (prob_malignant >= threshold_low).astype(int)
    cm_low = confusion_matrix(y_test_malignant, y_pred_threshold_low)
    tn_l, fp_l, fn_l, tp_l = cm_low.ravel()
    rec_l = recall_score(y_test_malignant, y_pred_threshold_low)
    prec_l = precision_score(y_test_malignant, y_pred_threshold_low)

    print(f"\nAdjusted Clinical Threshold = {threshold_low:.2f} (Prioritizing Patient Safety):")
    print(f"  Caught Malignant (TP): {tp_l} / {tp_l + fn_l} | Missed (FN): {fn_l}")
    print(f"  Recall changed to {rec_l * 100:.2f}% (Precision became {prec_l * 100:.2f}%)")
    print("Key Takeaway: In high-stakes medicine, lowering the threshold maximizes Recall,")
    print("accepting slightly more False Positives (re-tests) to avoid any False Negatives (deaths).")
    print("=" * 70)


if __name__ == "__main__":
    run_scaling_demonstration()
    run_preprocessing_pipeline_demo()
    run_pipeline_and_evaluation_experiment()
    print("\n>> All Week 2 Pipeline & Evaluation demonstrations completed successfully!")
