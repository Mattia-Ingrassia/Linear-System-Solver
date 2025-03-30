import numpy as np
import timeit
import warnings
from util import is_diagonally_dominant, any_stopping_criteria_met

def jacobi_solver(A, b, tollerance, max_iterations, initial_x):
    
    start_time = timeit.default_timer()

    # Check whether the given matrix is diagonally dominant
    if not is_diagonally_dominant(A):
        warnings.warn("Jacobi method may diverge since A is not diagonally dominant.")

    # Get the diagonal matrix of A
    D = np.diag(np.diag(A))

    # Get the inverse diagonal matrix of A
    D_inv = np.linalg.inv(D) 
    
    # Set up variables 
    iterations = 0
    x_old = initial_x
    x_new = np.copy(x_old)

    # needed to get at least one iteration, alternatively iterate once here or set x_new = x_old + 1
    abs_error = tollerance + 1
    rel_error = tollerance + 1


    # iterate until any stopping critieria is met
    while not any_stopping_criteria_met(iterations, rel_error, max_iterations, tollerance):
        x_old = np.copy(x_new)

        # calculate the residue vector
        residue = np.dot(A, x_old) - b
        
        # calculate the new x vector
        x_new = x_old + np.matmul(D_inv, - residue)
        iterations = iterations + 1
        
        # calculate absolute and relative error values
        abs_error = np.linalg.norm(x_new - x_old, ord=np.inf)
        rel_error = np.linalg.norm(residue, ord=np.inf)/np.linalg.norm(b, ord=np.inf)

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