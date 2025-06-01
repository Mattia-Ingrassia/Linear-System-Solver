import matplotlib.pyplot as plt
import os
import numpy as np

from solvers.base_solver import SolverMethod

def create_confront_plot(data, matrix_name, output_dir):
    """
    Create comparative plots for different solvers across tolerances.
    
    Parameters
    ----------
    data : list
        List of dict containing the results for each tolerance
    matrix_name : str
        Name of the matrix being analyzed
    output_dir : str
        output path to save the generated plots
    """

    # Create images directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results_iterations = {}
    results_time = {}
    results_errors = {}

    for method in SolverMethod:
        results_iterations[method.value] = []
        results_time[method.value] = []
        results_errors[method.value] = []

    tolerances = []
    
    for tolerance_data in data:
        tolerance = tolerance_data["tolerance"]
        tolerances.append(tolerance)
        
        for single_solution in tolerance_data["solutions"]:
            results_iterations[single_solution["solver_name"]].append(single_solution["iterations"])
            results_time[single_solution["solver_name"]].append(single_solution["time_spent"])
            results_errors[single_solution["solver_name"]].append(single_solution["relative_error"])

    markers = ['o', 's', '^', 'd']
    
    # Iterations plot

    plt.figure(figsize=(10, 6))

    marker_index = 0
    for method in results_iterations.keys():
        plt.plot(tolerances, results_iterations[method], marker=markers[marker_index], label=method)
        marker_index += 1

    # Format the plot
    plt.gca().invert_xaxis()  # Invert x-axis to see tolerances in descending order
    plt.xlabel('Tolerance (log scale)')
    plt.ylabel('Number of Iterations')
    plt.title(f'Comparison of Methods by Iterations - {matrix_name}')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{matrix_name}_iterations_comparison.png"))



    # Time plot

    plt.figure(figsize=(10, 6))

    marker_index = 0
    for method in results_time.keys():
        plt.plot(tolerances, results_time[method], marker=markers[marker_index], label=method)
        marker_index += 1

    # Format the plot
    plt.gca().invert_xaxis()  # Invert x-axis to see tolerances in descending order
    plt.xlabel('Tolerance (log scale)')
    plt.ylabel('Execution time (seconds)')
    plt.title(f'Comparison of Methods by Execution Time - {matrix_name}')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{matrix_name}_time_comparison.png"))

    # Errors plot

    plt.figure(figsize=(10, 6))

    marker_index = 0
    for method in results_errors.keys():
        plt.plot(tolerances, results_errors[method], marker=markers[marker_index], label=method)
        marker_index += 1
    
    # Format the plot
    plt.gca().invert_xaxis()  # Invert x-axis to see tolerances in descending order
    plt.xlabel('Tolerance (log scale)')
    plt.ylabel('Relative Error')
    plt.title(f'Comparison of Methods by Relative Error - {matrix_name}')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{matrix_name}_error_comparison.png"))

def create_sparsity_plot(matrix, matrix_name, output_dir):
    """
    Create and save a sparsity pattern visualization for a matrix.
    
    Parameters
    ----------
    matrix : ndarray
        The matrix to visualize
    matrix_name : str
        Name of the matrix
    output_dir : str
        Directory to save the generated plot
    """

    # Count non-zero elements and calculate density percentage
    not_zero_counter = np.count_nonzero(matrix)
    matrix_density = not_zero_counter / (matrix.shape[0] * matrix.shape[1]) * 100
    
    # Change marker size for readability
    markersize = max(0.05, min(1.0, 100 / (matrix.shape[0])))

    plt.figure(figsize=(10, 10))
        
    plt.title(f"Distribuzione elementi di {matrix_name}\nSize: {matrix.shape[0]}x{matrix.shape[1]}, Non-zeros: {not_zero_counter} ({matrix_density:.2f}%)")
    plt.spy(matrix, markersize= markersize)


    plt.savefig(os.path.join(output_dir, f"{matrix_name}_sparsity_pattern.png"))
    plt.close()
