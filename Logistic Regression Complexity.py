import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from memory_profiler import memory_usage

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression Implementation
def logistic_regression(X, y, lr=0.01, max_iter=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    num_operations = 0

    for i in range(max_iter):
        # Linear model
        linear_model = np.dot(X, weights) + bias
        num_operations += n_samples * n_features

        # Apply sigmoid
        predictions = sigmoid(linear_model)
        num_operations += n_samples

        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)
        num_operations += n_samples * n_features + n_samples

        # Update weights
        weights -= lr * dw
        bias -= lr * db
        num_operations += n_features + 1

    return weights, bias, num_operations

# Predict Function
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    return [1 if p > 0.5 else 0 for p in predictions]

# Logistic Regression Pipeline with Metrics
def logistic_regression_pipeline():
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()
    memory_before = memory_usage()[0]

    # Train logistic regression
    weights, bias, num_operations = logistic_regression(X_train, y_train, lr=0.01, max_iter=1000)

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Predict and evaluate
    y_pred = predict(X_test, weights, bias)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Output metrics
    print("\nMetrics for Logistic Regression:")
    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Complexity class
    complexity_class = "O(n_samples * n_features * max_iter)"
    complexity_name = "Linear Time (O(n))" if len(X) * X.shape[1] * 1000 < 1e6 else "Quadratic Time (O(n^2))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    # Plot weights
    plt.bar(range(len(weights)), weights)
    plt.xlabel("Feature Index")
    plt.ylabel("Weight Value")
    plt.title("Logistic Regression Weights")
    plt.grid()
    plt.show()

# Run the pipeline
if __name__ == "__main__":
    logistic_regression_pipeline()
