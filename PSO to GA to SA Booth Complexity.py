import numpy as np
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage

# Booth function as the objective
def booth_function(x):
    if len(x) != 2:
        raise ValueError("Booth function is defined for 2-dimensional inputs.")
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

# Particle Swarm Optimization (PSO)
def pso(objective, dim=2, swarm_size=30, max_iter=30):
    position = np.random.uniform(-10, 10, (swarm_size, dim))
    velocity = np.random.uniform(-1, 1, (swarm_size, dim))
    personal_best_position = np.copy(position)
    personal_best_value = np.array([objective(pos) for pos in position])
    global_best_position = personal_best_position[np.argmin(personal_best_value)]
    global_best_value = np.min(personal_best_value)

    w = 0.9  # Inertia weight
    c1, c2 = 2.0, 2.0  # Cognitive and social components

    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0

    for iteration in range(max_iter):
        print(f"\nPSO Iteration {iteration + 1}/{max_iter}")
        r1 = np.random.uniform(0, 1, (swarm_size, dim))
        r2 = np.random.uniform(0, 1, (swarm_size, dim))
        velocity = w * velocity + c1 * r1 * (personal_best_position - position) + c2 * r2 * (global_best_position - position)
        position += velocity
        position = np.clip(position, -10, 10)

        fitness = np.array([objective(pos) for pos in position])
        num_operations += swarm_size * dim

        better_fit = fitness < personal_best_value
        personal_best_position[better_fit] = position[better_fit]
        personal_best_value[better_fit] = fitness[better_fit]

        new_global_best_value = np.min(personal_best_value)
        if new_global_best_value < global_best_value:
            global_best_value = new_global_best_value
            global_best_position = personal_best_position[np.argmin(personal_best_value)]

        print(f"Current Global Best Value: {global_best_value:.4f}")
        print(f"Iteration Complexity: O(n_particles * dim)")

    memory_after = memory_usage()[0]
    end_time = time.time()

    print("\nPSO Metrics:")
    print("Time taken (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    complexity_class = "O(n)"
    complexity_name = "Linear"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    return global_best_position, global_best_value

# Genetic Algorithm (GA)
def ga(objective, dim=2, population_size=30, max_iter=30):
    population = np.random.uniform(-10, 10, (population_size, dim))
    fitness = np.array([objective(ind) for ind in population])

    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0

    for iteration in range(max_iter):
        print(f"\nGA Iteration {iteration + 1}/{max_iter}")
        selected_indices = np.random.choice(range(population_size), size=(population_size // 2) * 2, replace=False)
        selected = population[selected_indices]

        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                crossover_point = np.random.randint(1, dim)
                child1 = np.concatenate((selected[i][:crossover_point], selected[i + 1][crossover_point:]))
                child2 = np.concatenate((selected[i + 1][:crossover_point], selected[i][crossover_point:]))
                offspring.append(child1)
                offspring.append(child2)

        offspring = np.array(offspring)
        mutation = np.random.uniform(-0.1, 0.1, offspring.shape)
        offspring += mutation
        offspring = np.clip(offspring, -10, 10)

        population = np.vstack((selected, offspring))
        fitness = np.array([objective(ind) for ind in population])
        num_operations += len(population) * dim

        best_idx = np.argmin(fitness)
        print(f"Current Best Fitness: {fitness[best_idx]:.4f}")
        print(f"Iteration Complexity: O(population_size * dim)")

    memory_after = memory_usage()[0]
    end_time = time.time()

    print("\nGA Metrics:")
    print("Time taken (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    complexity_class = "O(n log n)"
    complexity_name = "Linearithmic"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

# Simulated Annealing (SA)
def sa(objective, dim=2, max_iter=30, initial_temp=100, cooling_rate=0.95):
    current_solution = np.random.uniform(-10, 10, dim)
    current_value = objective(current_solution)
    best_solution = np.copy(current_solution)
    best_value = current_value

    temperature = initial_temp

    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0

    for iteration in range(max_iter):
        print(f"\nSA Iteration {iteration + 1}/{max_iter}")
        new_solution = current_solution + np.random.uniform(-1, 1, dim)
        new_solution = np.clip(new_solution, -10, 10)
        new_value = objective(new_solution)
        num_operations += 1

        if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / temperature):
            current_solution = new_solution
            current_value = new_value

        if new_value < best_value:
            best_solution = new_solution
            best_value = new_value

        temperature *= cooling_rate

        print(f"Current Best Value: {best_value:.4f}")
        print(f"Iteration Complexity: O(1)")

    memory_after = memory_usage()[0]
    end_time = time.time()

    print("\nSA Metrics:")
    print("Time taken (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    complexity_class = "O(log n)"
    complexity_name = "Logarithmic"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    return best_solution, best_value

# Full Hybrid Optimization Pipeline
def hybrid_optimization(objective):
    print("Starting PSO...")
    best_position, best_value = pso(objective)
    print(f"\nPSO Result: Best Value = {best_value:.4f}")

    print("\nSwitching to GA...")
    best_position, best_value = ga(objective)
    print(f"\nGA Result: Best Value = {best_value:.4f}")

    print("\nSwitching to SA...")
    best_position, best_value = sa(objective)
    print(f"\nSA Result: Best Value = {best_value:.4f}")

    print(f"\nFinal Best Value: {best_value:.4f}")
    print(f"Best Position: {best_position}")

if __name__ == "__main__":
    hybrid_optimization(booth_function)
