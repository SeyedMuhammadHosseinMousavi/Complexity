import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import time
from memory_profiler import memory_usage

# CNN for Iris Classification

def cnn_iris_classification():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for Conv1D

    encoder = OneHotEncoder(sparse_output=False)  # Updated for scikit-learn >= 1.0
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the CNN model
    model = Sequential([
        Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Metrics tracking
    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0  # Initialize operation counter

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
    num_operations += X_train.shape[0] * 50 * X_train.shape[1] * 16  # Approximate computation count

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Output metrics
    print("\nMetrics for CNN on Iris Dataset:")
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Number of classes:", y.shape[1])
    print("Test Accuracy:", test_accuracy)
    print("Test Loss:", test_loss)
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    complexity_class = "O(n_samples * epochs * n_features * n_filters)"
    complexity_name = "Linear Time (O(n))" if X.shape[0] * 50 * X.shape[1] < 1e6 else "Quadratic Time (O(n^2))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    # Plot training history
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('CNN Training and Validation Accuracy')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    cnn_iris_classification()
