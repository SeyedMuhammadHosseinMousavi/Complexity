import numpy as np
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage

# Define the system model
def make_model():
    return {
        "PD": 1500,  # Power demand
        "Plants": {
            "Pmin": np.array([100, 80, 50, 60, 40]),
            "Pmax": np.array([500, 400, 300, 250, 200]),
            "alpha": np.array([300, 280, 260, 240, 220]),
            "beta": np.array([8, 7.5, 7, 6.5, 6]),
            "gamma": np.array([0.03, 0.028, 0.027, 0.026, 0.025]),
        },
        "nPlant": 5,  # Number of plants
    }

# Parse function to map x to actual power values
def parse(x, model):
    Pmin = model["Plants"]["Pmin"]
    Pmax = model["Plants"]["Pmax"]
    P = Pmin + (Pmax - Pmin) * x
    return P

# Define the cost function
def cost_function(x, model):
    P = parse(x, model)
    alpha = model["Plants"]["alpha"]
    beta = model["Plants"]["beta"]
    gamma = model["Plants"]["gamma"]

    # Calculate cost
    cost = np.sum(alpha + beta * P + gamma * P ** 2)

    # Power balance constraint
    P_total = np.sum(P)
    PD = model["PD"]
    power_loss = 0.05 * P_total  # Simplified power loss model
    power_balance_violation = max(0, PD - (P_total - power_loss))

    penalty = 10  # Penalty for constraint violation
    z = cost + penalty * power_balance_violation

    return z, {
        "P": P,
        "Cost": cost,
        "PowerLoss": power_loss,
        "PowerBalanceViolation": power_balance_violation,
    }

# Define fuzzy logic adjustment
def fuzzy_adjustment(iteration, max_iter, violation):
    if violation > 0.1:
        penalty = 20  # Increase penalty for high violations
    else:
        penalty = 10

    if iteration / max_iter < 0.5:
        r = 0.2  # Larger neighborhood radius in early iterations
    else:
        r = 0.1  # Smaller radius for fine-tuning

    return penalty, r

# Bee dance function
def bee_dance(position, r):
    nVar = len(position)
    k = np.random.randint(0, nVar)
    new_position = position.copy()
    new_position[k] += np.random.uniform(-r, r)
    new_position = np.clip(new_position, 0, 1)  # Ensure within bounds
    return new_position

# Bees Algorithm implementation with metrics
def bees_algorithm(model):
    # Parameters
    max_iter = 20
    n_scout_bees = 7
    n_elite_sites = 3
    n_selected_sites = 4
    n_elite_bees = 5
    n_selected_bees = 3
    rdamp = 0.7

    num_operations = 0  # Count operations
    start_time = time.time()
    memory_before = memory_usage()[0]

    # Initialize scout bees
    bees = [{"position": np.random.uniform(0, 1, model["nPlant"]), "cost": None} for _ in range(n_scout_bees)]
    for bee in bees:
        bee["cost"], bee["details"] = cost_function(bee["position"], model)
        num_operations += model["nPlant"]  # Cost calculation

    # Sort by cost
    bees = sorted(bees, key=lambda b: b["cost"])
    best_costs = []

    # Main loop
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")

        # Adjust fuzzy parameters
        penalty, r = fuzzy_adjustment(iteration, max_iter, bees[0]["details"]["PowerBalanceViolation"])

        # Elite sites
        for i in range(n_elite_sites):
            for _ in range(n_elite_bees):
                new_position = bee_dance(bees[i]["position"], r)
                new_cost, new_details = cost_function(new_position, model)
                num_operations += model["nPlant"]  # Cost calculation
                if new_cost < bees[i]["cost"]:
                    bees[i] = {"position": new_position, "cost": new_cost, "details": new_details}

        # Selected non-elite sites
        for i in range(n_elite_sites, n_selected_sites):
            for _ in range(n_selected_bees):
                new_position = bee_dance(bees[i]["position"], r)
                new_cost, new_details = cost_function(new_position, model)
                num_operations += model["nPlant"]  # Cost calculation
                if new_cost < bees[i]["cost"]:
                    bees[i] = {"position": new_position, "cost": new_cost, "details": new_details}

        # Non-selected sites
        for i in range(n_selected_sites, n_scout_bees):
            new_position = np.random.uniform(0, 1, model["nPlant"])
            new_cost, new_details = cost_function(new_position, model)
            num_operations += model["nPlant"]  # Cost calculation
            bees[i] = {"position": new_position, "cost": new_cost, "details": new_details}

        # Sort by cost
        bees = sorted(bees, key=lambda b: b["cost"])

        # Store the best cost
        best_costs.append(bees[0]["cost"])
        print(f"Best cost at iteration {iteration + 1}: {bees[0]['cost']:.2f}")

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Final results
    best_solution = bees[0]
    print("\nFinal Results:")
    print(f"Cost: {best_solution['cost']:.2f}")
    print(f"Power Distribution: {best_solution['details']['P']}")
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    complexity_class = "O(n_scout_bees * max_iter * nPlant)"
    complexity_name = "Linearithmic (O(n * log n))" if max_iter * n_scout_bees > 100 else "Quadratic (O(n^2))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    # Plot the results
    plt.plot(best_costs, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("Convergence of Bees Algorithm with Fuzzy Logic")
    plt.grid()
    plt.show()

# Run the algorithm
if __name__ == "__main__":
    model = make_model()
    bees_algorithm(model)
