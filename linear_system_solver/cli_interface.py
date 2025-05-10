import click
import numpy as np
import os
import json

from scipy.io import mmread

from JacobiSolver import JacobiSolver
from GaussSeidelSolver import GaussSeidelSolver
from GradientSolver import GradientSolver
from ConjugateGradientSolver import ConjugateGradientSolver

from plot_generator import create_plot, create_confront_plot, create_sparsity_plot

DEFAULT_TOLERANCES = [10e-4, 10e-6, 10e-8, 10e-10]
SAVE_PATH = "results"

@click.command()
@click.option('--path', type=click.Path(exists=True), required=True, help="File contenente la matrice A in formato mtx.")
@click.option('--x', type=click.Path(exists=True), default = None, help="File contenente il vettore correct_x, che rappresenta la soluzione ideale del sistema.")
@click.option('--b', type=click.Path(exists=True), default = None, help="File contenente il vettore b, ovvero il vettore dei termini noti.")
@click.option('--tol', type=float, multiple=True, help="Array di scalari che rappresentano la tolleranza.")
def run_benchmark(path, b, x, tol):
    
    A = load_matrix(path)
    
    # Get the name of the matrix
    matrix_name = os.path.splitext(os.path.basename(path))[0]

    # We manage the matrix as a dense matrix
    A = A.toarray()

    

    # If the solution is not given it is initialized as an array of ones
    if x is None:
        x = np.ones(len(A))
    
    # If RHS is not given it gets calculated
    if b is None:
        b = np.dot(A,x)
    
    # if the tolerance is not given the default array is used
    if len(tol) == 0:
        tolerances = DEFAULT_TOLERANCES
    else:
        tolerances = list(tol)

    temp_solver = JacobiSolver()
    if temp_solver.check_matrix_properties(A):
        print("Matrix passed all checks (square, symmetric, positive definite)")



        total_data = []
        for tolerance in tolerances:
            jacobi = JacobiSolver().solve(A, b, tolerance, x)
            gauss_seidel = GaussSeidelSolver().solve(A, b, tolerance, x)
            gradient = GradientSolver().solve(A, b, tolerance, x)
            conjugate_gradient = ConjugateGradientSolver().solve(A, b, tolerance, x)
            single_result = {
                "tolerance": tolerance,
                "solutions": [ jacobi, gauss_seidel, gradient, conjugate_gradient ]
            }
            total_data.append(single_result)

        if not os.path.exists(f"{SAVE_PATH}/{matrix_name}"):       
            os.makedirs(f"{SAVE_PATH}/{matrix_name}")

        with open(f"{SAVE_PATH}/{matrix_name}/results.json", "w") as f:
            json.dump(total_data, f, indent=4)



        generate_plots(matrix_name)






def generate_plots(matrix_name):
    with open(f"{SAVE_PATH}/{matrix_name}/results.json", "r") as f:
        total_data = json.load(f)
        create_confront_plot(total_data, matrix_name)



def load_matrix(path):
    if path.endswith('.mtx'):
        return mmread(path)
    else:
        click.echo("La matrice deve essere in formato .mtx")

@click.command()
@click.option('--path', type=click.Path(exists=True), required=True, help="File contenente la matrice A in formato mtx.")
def print_sparsity_pattern(path):
    if path.endswith('.mtx'):
        mat = mmread(path)
        # Get the name of the matrix
        matrix_name = os.path.splitext(os.path.basename(path))[0]
        A = mat.toarray()
        create_sparsity_plot(A, matrix_name)


if __name__ == "__main__":
    run_benchmark()
    #print_sparsity_pattern()