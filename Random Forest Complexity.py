import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from memory_profiler import memory_usage

# Random Forest Pipeline with Metrics
def random_forest_pipeline():
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()
    memory_before = memory_usage()[0]

    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Output metrics
    print("\nMetrics for Random Forest:")
    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])
    print("Number of trees:", rf.n_estimators)
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Complexity class
    complexity_class = "O(n_trees * n_samples * log(n_samples))"
    complexity_name = "Log-linear Time (O(n * log n))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

# Run the pipeline
if __name__ == "__main__":
    random_forest_pipeline()