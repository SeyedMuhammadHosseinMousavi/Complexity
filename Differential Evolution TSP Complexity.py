import numpy as np
import time
from memory_profiler import memory_usage

# Distance function for TSP
def calculate_distance(cities, route):
    distance = 0
    for i in range(len(route)):
        from_city = cities[route[i]]
        to_city = cities[route[(i + 1) % len(route)]]  # Wrap around to the first city
        distance += np.linalg.norm(from_city - to_city)
    return distance

# Differential Evolution for TSP
def differential_evolution_tsp(cities, population_size, max_iter, mutation_factor, crossover_prob):
    start_time = time.time()
    memory_before = memory_usage()[0]

    num_cities = len(cities)

    # Initialize population (random permutations of city indices)
    population = [np.random.permutation(num_cities) for _ in range(population_size)]

    num_operations = 0
    for iter in range(max_iter):
        new_population = []
        for i in range(population_size):
            # Mutation: Select three random individuals and generate a mutant vector
            indices = np.random.choice(population_size, 3, replace=False)
            a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
            mutant = np.copy(a)
            swap_indices = np.random.choice(num_cities, 2, replace=False)
            mutant[swap_indices] = mutant[swap_indices[::-1]]  # Swap two cities
            num_operations += num_cities  # Mutation operations

            # Crossover: Perform uniform crossover
            trial = np.copy(population[i])
            for j in range(num_cities):
                if np.random.rand() < crossover_prob:
                    trial[j] = mutant[j]
            num_operations += num_cities  # Crossover operations

            # Selection: Keep the better individual
            if calculate_distance(cities, trial) < calculate_distance(cities, population[i]):
                new_population.append(trial)
            else:
                new_population.append(population[i])
            num_operations += num_cities  # Distance evaluations

        population = new_population

        # Get the best route in the current population
        best_route = min(population, key=lambda route: calculate_distance(cities, route))
        best_distance = calculate_distance(cities, best_route)
        print(f"Iteration {iter + 1}, Best Distance: {best_distance}")

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Output metrics for complexity
    print("\nFinal Results:")
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory usage (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)
    print("Best route:", best_route)
    print("Best distance:", best_distance)

    # Print complexity class
    complexity_class = "O(population_size * max_iter * num_cities)"
    complexity_name = "Quadratic (O(n^2))" if population_size * max_iter * num_cities > 10000 else "Linearithmic (O(n * log n))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

# Parameters for TSP
cities = np.random.rand(10, 2)  # 10 cities with random coordinates in 2D space
population_size = 20
max_iter = 100
mutation_factor = 0.8
crossover_prob = 0.9

differential_evolution_tsp(cities, population_size, max_iter, mutation_factor, crossover_prob)
