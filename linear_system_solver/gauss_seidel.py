import numpy as np
import warnings
from util import is_square, is_diagonally_dominant, any_stopping_criteria_met
import timeit

def gauss_seidel_solver(A, b, tolerance, max_iterations, initial_x):

    start_time = timeit.default_timer()

    # Check whether the given matrix is square
    if not is_square(A):
        return False

    # Check whether the given matrix is diagonally dominant
    if not is_diagonally_dominant(A):
        warnings.warn("Gauss-Seidel method may diverge since A is not diagonally dominant.")

    # L is the lower triangular matrix of A
    L = np.tril(A)
    
    # Set up variables 
    iterations = 0
    x_old = np.copy(initial_x)
    x_new = np.copy(x_old)

    # needed to get at least one iteration, alternatively iterate once here or set x_new = x_old + 1
    abs_error = tolerance + 1
    rel_error = tolerance + 1


    # iterate until any stopping critieria is met
    while not any_stopping_criteria_met(iterations, rel_error, max_iterations, tolerance):

        # calculate the residue vector
        x_old = np.copy(x_new)
        residue_k = b - np.dot(A, x_old)

        # calculate the new x vector
        x_new = _gauss_seidel_step(L, residue_k, x_old)
        iterations = iterations + 1

        # calculate absolute and relative error values
        abs_error = np.linalg.norm(x_new - x_old, ord=np.inf)
        rel_error = np.linalg.norm(residue_k)/np.linalg.norm(b)

    end_time = timeit.default_timer()

    execution_time = end_time - start_time

    result = {
        "solution": x_new,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "iterations": iterations,
        "time_spent": execution_time
    }

    return result

# function that calculates the next step for gauss-seidel
def _gauss_seidel_step(L, residue_k, x_k):
    y = _tril_matrix_solver(L, residue_k)
    return x_k + y
    

# direct method for solving triangular linear systems
def _tril_matrix_solver(matrix, rhs):
    solutions = np.zeros(len(matrix))
    solutions[0] = rhs[0] / matrix[0][0]
    for i in range(1, len(matrix), 1) :
        b = rhs[i]
        u = matrix[i][i]
        solutions[i] = (1/u) * (b - np.dot(matrix[i], solutions))
    return solutions