import click
import numpy as np
from scipy.io import mmread

from JacobiSolver import JacobiSolver
from GaussSeidelSolver import GaussSeidelSolver
from GradientSolver import GradientSolver
from ConjugateGradientSolver import ConjugateGradientSolver

@click.command()
@click.option('--a', type=click.Path(exists=True), required=True, help="File contenente la matrice A in formato mtx.")
@click.option('--x', type=click.Path(exists=True), default = None, help="File contenente il vettore correct_x, che rappresenta la soluzione ideale del sistema.")
@click.option('--b', type=click.Path(exists=True), default = None, help="File contenente il vettore b, ovvero il vettore dei termini noti.")
@click.option('--tol', type=float, default = 10e-10, help = "Scalare che rappresenta la tolleranza.")
def run_benchmark(a, b, x, tol):
    A = a
    A = load_matrix(A)

    A = A.toarray()

    if x is None:
        x = np.ones(len(A))
    
    if b is None:
        b = np.dot(A,x)
    
    jacobi = JacobiSolver().solve(A, b, tol, x)
    gauss_seidel = GaussSeidelSolver().solve(A, b, tol, x)
    gradient = GradientSolver().solve(A, b, tol, x)
    conjugate_gradient = ConjugateGradientSolver().solve(A, b, tol, x)
        
    click.secho(f"\n{jacobi}", fg='cyan', bold=True)
    click.secho(f"\n{gauss_seidel}", fg='green', bold=True)
    click.secho(f"\n{gradient}", fg='yellow', bold=True)
    click.secho(f"\n{conjugate_gradient}", fg='red', bold=True)


def load_matrix(path):
    if path.endswith('.mtx'):
        return mmread(path)
    else:
        click.echo("La matrice deve essere in formato .mtx")



if __name__ == "__main__":
    run_benchmark()