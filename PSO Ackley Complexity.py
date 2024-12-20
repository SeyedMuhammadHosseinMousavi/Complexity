import numpy as np
import time
from memory_profiler import memory_usage

# Ackley Function
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# Particle Swarm Optimization (PSO) implementation
def pso_ackley(num_particles, dimensions, max_iter):
    start_time = time.time()
    memory_before = memory_usage()[0]

    # Initialize particles
    positions = np.random.uniform(-5, 5, (num_particles, dimensions))
    velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
    personal_best_positions = np.copy(positions)
    personal_best_scores = np.array([ackley(p) for p in positions])

    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # PSO parameters
    w = 0.5  # inertia
    c1 = 2.0  # cognitive component
    c2 = 2.0  # social component

    # Iterations
    num_operations = 0  # To count basic operations for complexity
    for iter in range(max_iter):
        print(f"Iteration {iter + 1}")
        for i in range(num_particles):
            r1, r2 = np.random.rand(dimensions), np.random.rand(dimensions)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - positions[i]) +
                             c2 * r2 * (global_best_position - positions[i]))
            positions[i] += velocities[i]
            num_operations += 2 * dimensions  # Position and velocity updates

            # Evaluate fitness
            score = ackley(positions[i])
            num_operations += dimensions  # Fitness evaluation

            # Update personal and global bests
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

        # Print cost for this iteration
        print(f"Global best score at iteration {iter + 1}: {global_best_score}")

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Output metrics for complexity
    print("\nFinal Results:")
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory usage (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)
    print("Best position:", global_best_position)
    print("Best score (Ackley minimum):", global_best_score)

    # Print complexity class
    complexity_class = "O(n * d * max_iter)"
    complexity_name = "Quadratic (O(n^2))" if max_iter * num_particles * dimensions > 10000 else "Linearithmic (O(n * log n))" if dimensions < max_iter else "Linear (O(n))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

# Parameters for PSO
num_particles = 30
dimensions = 10
max_iter = 100

pso_ackley(num_particles, dimensions, max_iter)
