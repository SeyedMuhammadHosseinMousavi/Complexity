import numpy as np
import time
from memory_profiler import memory_usage

# Quick Sort Implementation with operation counting
num_operations = 0  # Global variable to count operations
def quick_sort(arr):
    global num_operations
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        num_operations += len(arr)  # Counting comparisons
        return quick_sort(left) + middle + quick_sort(right)

# Function to measure performance and complexity
def quick_sort_analysis(arr):
    global num_operations
    num_operations = 0  # Reset operation counter
    start_time = time.time()
    memory_before = memory_usage()[0]

    # Perform Quick Sort
    sorted_arr = quick_sort(arr)

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Output metrics
    print("Original array:", arr)
    print("Sorted array:", sorted_arr)
    print("Time taken (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)
    print("Number of operations performed:", num_operations)

    # Complexity class
    n = len(arr)
    complexity_class = "O(n * log n)" if n > 1 else "O(1)"
    complexity_name = "Linearithmic (O(n * log n))" if n > 1 else "Constant (O(1))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

# Example array
example_array = [3, 6, 8, 10, 1, 2, 1]
quick_sort_analysis(example_array)
