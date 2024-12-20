import numpy as np
import time
from memory_profiler import memory_usage

# Linear Search Implementation
def linear_search(arr, target):
    """
    Perform a linear search on the array to find the target element.

    Args:
        arr (list): The array to search.
        target: The element to find.

    Returns:
        int: The index of the target element if found, otherwise -1.
    """
    for index, value in enumerate(arr):
        if value == target:
            return index  # Found the target
    return -1  # Target not found

# Linear Search with Metrics
def linear_search_with_metrics(arr, target):
    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0

    # Perform the search
    result = -1
    for index, value in enumerate(arr):
        num_operations += 1  # Counting each comparison
        if value == target:
            result = index
            break

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Output metrics
    print("\nMetrics for Linear Search:")
    print("Array size:", len(arr))
    print("Target element:", target)
    print("Result index:", result)
    print("Number of operations performed:", num_operations)
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)

    # Complexity class
    complexity_class = "O(n)"
    complexity_name = "Linear Time (O(n))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    return result

# Example Usage
if __name__ == "__main__":
    # Generate a random array of integers
    array_size = 1000
    example_array = np.random.randint(0, 1000, array_size)

    # Define the target element
    target = np.random.choice(example_array)

    # Perform linear search with metrics
    result_index = linear_search_with_metrics(example_array, target)

    if result_index != -1:
        print(f"\nElement {target} found at index {result_index}.")
    else:
        print(f"\nElement {target} not found in the array.")
