import click
import numpy as np
import os
import json

from scipy.io import mmread

from solvers.base_solver import SolverMethod
from solvers.jacobi import JacobiSolver
from solvers.gauss_seidel import GaussSeidelSolver
from solvers.gradient import GradientSolver
from solvers.conjugate_gradient import ConjugateGradientSolver

from plot_generator import create_confront_plot, create_sparsity_plot

DEFAULT_TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]
DEFAULT_RESULTS_DIR  = "results"

@click.command()
@click.option('--a_path', type=click.Path(exists=True), required=True, help="File containing the A matrix in .mtx format.")
@click.option('--b_path', type=click.Path(exists=True), default = None, help="File containing the b vector, representing rhs vector.\n.npy, .txt and .csv files are supported.")
@click.option('--x_path', type=click.Path(exists=True), default = None, help="File containing the correct_x vector, representing the right solution of the system.\n.npy, .txt and .csv files are supported.")
@click.option('--tol', type=float, multiple=True, help="Array of scalars representing the tolerance.")
@click.option('--verbose', type=bool, default = True, help="True if you want comments on execution, False otherwise.")

def run_benchmark(a_path, b_path, x_path, tol, verbose):
    """Run benchmarks for different solver methods with given tolerances."""

    # Get the name of the matrix
    matrix_name = os.path.splitext(os.path.basename(a_path))[0]
    matrix_results_dir = os.path.join(DEFAULT_RESULTS_DIR, matrix_name)
    os.makedirs(matrix_results_dir, exist_ok=True)

    matrix_images_dir = os.path.join(matrix_results_dir, "images")
    os.makedirs(matrix_images_dir, exist_ok=True)

    A = load_matrix(a_path)
    
    # If the solution is not given it is initialized as an array of ones
    if x_path is None:
        x = np.ones(len(A))
    else:
        x = load_vector(x_path)
        
    # If RHS is not given it gets calculated
    if b_path is None:
        b = np.dot(A, x)
    else:
        b = load_vector(b_path)
    
    # If the tolerance is not given the default array is used
    if len(tol) == 0:
        tolerances = DEFAULT_TOLERANCES
    else:
        tolerances = list(tol)
    
    solvers = {
        SolverMethod.JACOBI.value: JacobiSolver(),
        SolverMethod.GAUSS_SEIDEL.value: GaussSeidelSolver(),
        SolverMethod.GRADIENT.value: GradientSolver(),
        SolverMethod.CONJUGATE_GRADIENT.value: ConjugateGradientSolver()
    }

    solvers_color = {
        SolverMethod.JACOBI.value: "green",
        SolverMethod.GAUSS_SEIDEL.value: "yellow",
        SolverMethod.GRADIENT.value: "magenta",
        SolverMethod.CONJUGATE_GRADIENT.value: "cyan"
    }


    total_data = []

    for tolerance in tolerances:
        if verbose:
            click.echo(f"\nRunning solvers with tolerance {tolerance}:")
        
        solutions = []
        
        for name, solver in solvers.items():
            
            result = solver.solve(A, b, tolerance, x, verbose=verbose)
     
            solutions.append(result)
            
            if verbose:
                click.secho(f"Results for {name} - tolerance {tolerance}:", fg=solvers_color[name])
                click.secho(f"\tConverged: {result["converged"]}", fg=solvers_color[name])
                click.secho(f"\tRelative Error: {result["relative_error"]}, Iterations: {result["iterations"]}, Time spent:{result["time_spent"]}s", fg=solvers_color[name])

        tolerance_result = {
            "tolerance": tolerance,
            "solutions": solutions
        }
        total_data.append(tolerance_result)

    with open(os.path.join(matrix_results_dir, "results.json"), "w") as f:
        json.dump(total_data, f, indent=4)

    # Generate plots after all calculations
    create_confront_plot(total_data, matrix_name, matrix_images_dir)
    create_sparsity_plot(A, matrix_name, matrix_results_dir)



def load_matrix(path):
    """Load matrix data from .mtx file"""
    if path.endswith('.mtx'):
        # We manage the matrix as a dense matrix
        A = mmread(path)
        A = A.toarray() #type: ignore
        return A
    else:
        raise ValueError("La matrice deve essere in formato .mtx")
 
def load_vector(path):
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.txt'):
        return np.loadtxt(path)
    elif path.endswith('.csv'):
        return np.loadtxt(path, delimiter=',')
    else:
        raise ValueError("Format file for vectors is not supported.(supported filetypes: .npy, .txt, .csv)")


if __name__ == "__main__":
    run_benchmark()