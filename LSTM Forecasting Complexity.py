import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time
from memory_profiler import memory_usage

# Generate synthetic time series data
def generate_time_series(n_samples, n_timestamps):
    np.random.seed(42)
    x = np.linspace(0, 50, n_samples)
    y = np.sin(x) + 0.5 * np.random.normal(size=n_samples)
    data = np.array([y[i:i+n_timestamps] for i in range(len(y) - n_timestamps)])
    targets = y[n_timestamps:]
    return data, targets

# LSTM for Time Series Forecasting
def lstm_forecasting():
    # Generate synthetic data
    n_samples = 1000
    n_timestamps = 20
    data, targets = generate_time_series(n_samples, n_timestamps)

    # Scale data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    targets = scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

    # Reshape for LSTM input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define the LSTM model
    model = Sequential([
        LSTM(100, activation='relu', input_shape=(n_timestamps, 1)),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Metrics tracking
    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0  # Initialize operation counter

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0, validation_split=0.2)
    num_operations += X_train.shape[0] * 50 * X_train.shape[1] * 50  # Approximate computation count

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)

    # Output metrics
    print("\nMetrics for LSTM Forecasting:")
    print("Number of samples:", n_samples)
    print("Number of timestamps:", n_timestamps)
    print("Test Loss (MSE):", test_loss)
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    complexity_class = "O(n_samples * epochs * n_timestamps * n_units)"
    complexity_name = "Linear Time (O(n))" if n_samples * 50 * n_timestamps < 1e6 else "Quadratic Time (O(n^2))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    # Plot predictions vs actual values
    plt.plot(y_test, label='Actual Values')
    plt.plot(predictions.flatten(), label='Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.title('LSTM Forecasting Results')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    lstm_forecasting()
