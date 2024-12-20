import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from memory_profiler import memory_usage

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.feature_stats = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = X_cls.shape[0] / n_samples
            self.feature_stats[cls] = {
                "mean": np.mean(X_cls, axis=0),
                "var": np.var(X_cls, axis=0)
            }

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        posteriors = []

        for cls in self.classes:
            prior = np.log(self.class_priors[cls])
            likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.feature_stats[cls]["var"]) +
                (x - self.feature_stats[cls]["mean"])**2 / (self.feature_stats[cls]["var"])
            )
            posteriors.append(prior + likelihood)

        return self.classes[np.argmax(posteriors)]

# Naive Bayes Pipeline with Metrics
def naive_bayes_pipeline():
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()
    memory_before = memory_usage()[0]

    # Train Naive Bayes Classifier
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Predict and evaluate
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Output metrics
    print("\nMetrics for Naive Bayes:")
    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Complexity class
    complexity_class = "O(n_samples * n_features)"
    complexity_name = "Linear Time (O(n))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

# Run the pipeline
if __name__ == "__main__":
    naive_bayes_pipeline()
