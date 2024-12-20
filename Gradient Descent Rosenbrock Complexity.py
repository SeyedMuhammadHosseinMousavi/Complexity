import numpy as np
import time
from memory_profiler import memory_usage

# Rosenbrock Function
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Gradient Descent Implementation
def gradient_descent_rosenbrock(learning_rate, dimensions, max_iter):
    start_time = time.time()
    memory_before = memory_usage()[0]

    # Initialize position
    position = np.random.uniform(-2, 2, dimensions)  # Smaller initialization range

    # Gradient computation function
    def compute_gradient(x):
        grad = np.zeros_like(x)
        grad[:-1] = -400 * (x[1:] - x[:-1]**2) * x[:-1] - 2 * (1 - x[:-1])
        grad[1:] += 200 * (x[1:] - x[:-1]**2)
        return grad

    num_operations = 0
    for iter in range(max_iter):
        # Compute gradient
        gradient = compute_gradient(position)

        # Clip gradient to prevent exploding updates
        gradient = np.clip(gradient, -10, 10)
        num_operations += 2 * dimensions  # Gradient computation

        # Update position
        position -= learning_rate * gradient
        num_operations += dimensions  # Position update

        # Compute cost
        cost = rosenbrock(position)
        num_operations += dimensions  # Cost computation

        print(f"Iteration {iter + 1}, Cost: {cost}")

        # Stop if cost diverges or becomes NaN
        if np.isnan(cost) or cost > 1e6:
            print("Algorithm diverged. Stopping early.")
            break

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Output metrics for complexity
    print("\nFinal Results:")
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory usage (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)
    print("Final position:", position)
    print("Final cost (Rosenbrock minimum):", cost)

    # Print complexity class
    complexity_class = "O(d * max_iter)"
    complexity_name = "Linear (O(n))" if dimensions < max_iter else "Linearithmic (O(n * log n))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

# Parameters for Gradient Descent
learning_rate = 0.0001  # Reduced learning rate
dimensions = 10
max_iter = 100

gradient_descent_rosenbrock(learning_rate, dimensions, max_iter)
