import numpy as np
from util import any_stopping_criteria_met
import timeit


def conjugate_gradient_solver(A, b, tolerance, max_iterations, initial_x):
    
    start_time = timeit.default_timer()
    
    # Set up variables 
    iterations = 0
    x_old = np.copy(initial_x)
    x_new = np.copy(x_old)

    # needed to get at least one iteration, alternatively iterate once here or set x_new = x_old + 1
    abs_error = tolerance + 1
    rel_error = tolerance + 1

    residue_new = b - np.dot(A, x_old)

    direction_new = np.copy(residue_new)

    # iterate until any stopping critieria is met
    while not any_stopping_criteria_met(iterations, rel_error, max_iterations, tolerance):
        
        x_old = np.copy(x_new)
        direction_old = np.copy(direction_new)
        residue = np.copy(residue_new)

        direction_transposed = np.transpose(direction_old)
        
        
        alpha_beta_denominator = np.dot(direction_transposed, np.dot(A, direction_old))
        
        # calculate the alpha step and the new x
        alpha = ( np.dot(direction_transposed, residue)) / alpha_beta_denominator
        
        x_new = x_old + np.dot(alpha, direction_old)

        #new residue
        residue_new = b - np.dot(A, x_new)
    
        
        beta = ( np.dot(direction_transposed, np.dot(A, residue_new)) ) / alpha_beta_denominator

        
        direction_new = residue_new - np.dot(beta, direction_old)        
        
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


