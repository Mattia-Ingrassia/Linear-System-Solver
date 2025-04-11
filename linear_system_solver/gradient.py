import numpy as np
from util import any_stopping_criteria_met
import timeit


def gradient_solver(A, b, tolerance, max_iterations, initial_x):
    
    start_time = timeit.default_timer()
    
    # Set up variables 
    iterations = 0
    x_old = np.copy(initial_x)
    x_new = np.copy(x_old)

    # needed to get at least one iteration, alternatively iterate once here or set x_new = x_old + 1
    abs_error = tolerance + 1
    rel_error = tolerance + 1

    # iterate until any stopping critieria is met
    while not any_stopping_criteria_met(iterations, rel_error, max_iterations, tolerance):
        
        # calculate the residue vector and its transposed value
        x_old = np.copy(x_new)
        residue = b - np.dot(A, x_old)
        residue_transposed = np.transpose(residue)

        # calculate the alpha step and the new x
        alpha = ( np.dot(residue_transposed, residue)) / np.dot(residue_transposed, np.dot(A, residue))
        x_new = x_old + np.dot(alpha, residue)
        iterations = iterations + 1

        # calculate absolute and relative error values
        abs_error = np.linalg.norm(x_new - x_old, ord=np.inf)
        rel_error = np.linalg.norm(residue)/np.linalg.norm(b)

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


