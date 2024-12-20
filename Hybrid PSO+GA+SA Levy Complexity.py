import numpy as np
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage

# Levy function as the objective
def levy_function(x):
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0])) ** 2
    term2 = np.sum(((w[:-1] - 1) ** 2) * (1 + 10 * (np.sin(np.pi * w[:-1] + 1)) ** 2))
    term3 = ((w[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)
    return term1 + term2 + term3

# Hybrid PSO-GA-SA

def hybrid_algorithm(objective, dim=2, swarm_size=30, population_size=30, max_iter=30, initial_temp=100, cooling_rate=0.95):
    # Initialize PSO variables
    position = np.random.uniform(-10, 10, (swarm_size, dim))
    velocity = np.random.uniform(-1, 1, (swarm_size, dim))
    personal_best_position = np.copy(position)
    personal_best_value = np.array([objective(pos) for pos in position])
    global_best_position = personal_best_position[np.argmin(personal_best_value)]
    global_best_value = np.min(personal_best_value)

    # Initialize GA variables
    population = np.random.uniform(-10, 10, (population_size, dim))
    fitness = np.array([objective(ind) for ind in population])

    # Initialize SA variables
    current_solution = np.random.uniform(-10, 10, dim)
    current_value = objective(current_solution)
    best_solution = np.copy(current_solution)
    best_value = current_value
    temperature = initial_temp

    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0

    for iteration in range(max_iter):
        print(f"\nHybrid Iteration {iteration + 1}/{max_iter}")

        # PSO Step
        r1 = np.random.uniform(0, 1, (swarm_size, dim))
        r2 = np.random.uniform(0, 1, (swarm_size, dim))
        velocity = 0.9 * velocity + 2.0 * r1 * (personal_best_position - position) + 2.0 * r2 * (global_best_position - position)
        position += velocity
        position = np.clip(position, -10, 10)
        fitness_pso = np.array([objective(pos) for pos in position])
        num_operations += swarm_size * dim

        better_fit = fitness_pso < personal_best_value
        personal_best_position[better_fit] = position[better_fit]
        personal_best_value[better_fit] = fitness_pso[better_fit]

        new_global_best_value = np.min(personal_best_value)
        if new_global_best_value < global_best_value:
            global_best_value = new_global_best_value
            global_best_position = personal_best_position[np.argmin(personal_best_value)]

        print(f"PSO Step Best Value: {global_best_value:.4f}")

        # GA Step
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
        fitness_ga = np.array([objective(ind) for ind in population])
        num_operations += len(population) * dim

        best_idx = np.argmin(fitness_ga)
        print(f"GA Step Best Value: {fitness_ga[best_idx]:.4f}")

        # SA Step
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

        print(f"SA Step Best Value: {best_value:.4f}")

    memory_after = memory_usage()[0]
    end_time = time.time()

    print("\nHybrid Algorithm Metrics:")
    print("Time taken (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    complexity_class = "O(n^2)"
    complexity_name = "Quadratic"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    return best_solution, best_value

if __name__ == "__main__":
    best_solution, best_value = hybrid_algorithm(levy_function, dim=10, swarm_size=50, population_size=50, max_iter=50)
    print(f"\nFinal Best Solution: {best_solution}")
    print(f"Final Best Value: {best_value:.4f}")
