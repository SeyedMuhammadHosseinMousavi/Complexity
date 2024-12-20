import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin_min
import time
from memory_profiler import memory_usage

# Genetic Algorithm for Clustering

def initialize_population(n_clusters, n_features, population_size):
    return [np.random.uniform(0, 1, (n_clusters, n_features)) for _ in range(population_size)]

def fitness_function(centroids, data):
    labels, _ = pairwise_distances_argmin_min(data, centroids)
    distances = np.linalg.norm(data - centroids[labels], axis=1)
    return np.sum(distances)

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def mutate(centroids, mutation_rate):
    for centroid in centroids:
        if np.random.rand() < mutation_rate:
            centroid += np.random.normal(0, 0.1, size=centroid.shape)
    return centroids

def genetic_algorithm_clustering(data, n_clusters, population_size, max_iter, mutation_rate):
    n_features = data.shape[1]
    population = initialize_population(n_clusters, n_features, population_size)

    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0

    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    for iteration in range(max_iter):
        fitness_scores = [fitness_function(centroids, data) for centroids in population]
        num_operations += len(population) * len(data)  # Fitness evaluation

        # Update best solution
        best_idx = np.argmin(fitness_scores)
        if fitness_scores[best_idx] < best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_solution = population[best_idx]

        # Selection
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices[:population_size // 2]]

        # Crossover and Mutation
        new_population = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[(i + 1) % len(population)]
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population.extend(new_population)
        num_operations += len(population) * n_clusters  # Crossover and mutation operations

        fitness_history.append(best_fitness)
        print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness:.2f}")

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Final results
    print("\nFinal Results:")
    print("Best Fitness:", best_fitness)
    print("Best Centroids:\n", best_solution)
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    complexity_class = "O(population_size * max_iter * data_size)"
    complexity_name = "Linearithmic (O(n * log n))" if population_size * max_iter > 100 else "Quadratic (O(n^2))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    # Plot convergence
    plt.plot(fitness_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm Convergence")
    plt.grid()
    plt.show()

# Load Iris dataset
if __name__ == "__main__":
    iris = load_iris()
    data = iris.data

    n_clusters = 3
    population_size = 10
    max_iter = 50
    mutation_rate = 0.1

    genetic_algorithm_clustering(data, n_clusters, population_size, max_iter, mutation_rate)
