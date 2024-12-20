import numpy as np
import time
from memory_profiler import memory_usage

# Check if a queen can be placed safely
def is_safe(board, row, col, n):
    # Check column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper right diagonal
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

# Solve N-Queens using backtracking
def solve_n_queens(board, row, n, solutions):
    if row >= n:
        solutions.append(np.copy(board))
        return

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            solve_n_queens(board, row + 1, n, solutions)
            board[row][col] = 0  # Backtrack

# N-Queens pipeline with metrics
def n_queens_pipeline(n):
    start_time = time.time()
    memory_before = memory_usage()[0]
    num_operations = 0

    board = np.zeros((n, n), dtype=int)
    solutions = []

    # Solve the problem
    solve_n_queens(board, 0, n, solutions)

    num_operations += len(solutions) * n * n  # Approximate operation count

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Output metrics
    print("\nMetrics for N-Queens Problem:")
    print("Board size:", n)
    print("Number of solutions found:", len(solutions))
    print("Number of operations performed:", num_operations)
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)

    # Complexity class
    complexity_class = "O(n!)"
    complexity_name = "Factorial Time (O(n!))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    return solutions

# Example Usage
if __name__ == "__main__":
    n = 8  # Change this value for different board sizes
    solutions = n_queens_pipeline(n)

    print("\nExample Solution:")
    if solutions:
        print(solutions[0])
    else:
        print("No solutions found.")
