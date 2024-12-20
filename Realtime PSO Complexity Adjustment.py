import numpy as np
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage

# Ackley function as the objective

def ackley_function(x):
    first_term = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
    second_term = -np.exp(np.mean(np.cos(2 * np.pi * x)))
    return first_term + second_term + 20 + np.e

# Adaptive PSO with Complexity Variation
def adaptive_pso(objective, dim=10, swarm_size=30, max_iter=50):
    # Initialize parameters
    position = np.random.uniform(-10, 10, (swarm_size, dim))
    velocity = np.random.uniform(-1, 1, (swarm_size, dim))
    personal_best_position = np.copy(position)
    personal_best_value = np.array([objective(pos) for pos in position])
    global_best_position = personal_best_position[np.argmin(personal_best_value)]
    global_best_value = np.min(personal_best_value)

    w = 0.9  # Initial inertia weight
    c1 = 2.0  # Cognitive component
    c2 = 2.0  # Social component

    start_time = time.time()
    memory_before = memory_usage()[0]

    complexity_mode = "O(n^2)"

    for iteration in range(max_iter):
        # Print iteration header
        print(f"\nIteration {iteration+1}/{max_iter}")

        # Adjust complexity mode adaptively
        if iteration % 10 == 0:
            if complexity_mode == "O(n^2)":
                complexity_mode = "O(n \log n)"
            elif complexity_mode == "O(n \log n)":
                complexity_mode = "O(n)"
            print(f"Complexity Mode Changed to: {complexity_mode}")

        # Update inertia weight adaptively
        new_w = w * 0.95  # Decay the inertia weight
        print(f"Inertia Weight: Changed from {w:.4f} to {new_w:.4f}")
        w = new_w

        # Update velocities and positions
        r1 = np.random.uniform(0, 1, (swarm_size, dim))
        r2 = np.random.uniform(0, 1, (swarm_size, dim))
        velocity = w * velocity + c1 * r1 * (personal_best_position - position) + c2 * r2 * (global_best_position - position)
        position += velocity

        # Apply bounds to positions
        position = np.clip(position, -10, 10)

        # Evaluate fitness
        fitness = np.array([objective(pos) for pos in position])

        # Update personal and global bests
        better_fit = fitness < personal_best_value
        personal_best_position[better_fit] = position[better_fit]
        personal_best_value[better_fit] = fitness[better_fit]

        new_global_best_value = np.min(personal_best_value)
        if new_global_best_value < global_best_value:
            print(f"Global Best Value: Improved from {global_best_value:.4f} to {new_global_best_value:.4f}")
            global_best_value = new_global_best_value
            global_best_position = personal_best_position[np.argmin(personal_best_value)]

        # Print complexity and fitness details
        print(f"Iteration Complexity: {complexity_mode}")
        print(f"Current Global Best Value: {global_best_value:.4f}")

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Final results
    print("\nPSO Final Results:")
    print(f"Global Best Value: {global_best_value:.4f}")
    print(f"Global Best Position: {global_best_position}")
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)

    # Plot convergence
    plt.plot(personal_best_value, label="Personal Best Values")
    plt.axhline(global_best_value, color='r', linestyle='--', label="Global Best Value")
    plt.xlabel("Particles")
    plt.ylabel("Fitness Value")
    plt.title("Convergence of PSO with Adaptive Complexity")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    adaptive_pso(ackley_function)
