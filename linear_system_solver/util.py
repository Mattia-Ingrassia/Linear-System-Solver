import numpy as np

def any_stopping_criteria_met(iterations, rel_error, max_iterations, tollerance):
    """
    Checks whether the stopping criteria are met.

    Parameters:
    - iterations: number of current iteration
    - rel_error: current relative error
    -   

    Returns:
    - True if the matrix is diagonally dominant for the given axis,
      False otherwise.

    """

    if((iterations < max_iterations) and (rel_error > tollerance)):
        return False
    return True

def is_diagonally_dominant(matrix, axis=1):
    """
    Checks whether the given matrix is diagonally dominant.

    Parameters:
    - matrix: given matrix
    - axis: 1 for rows (default), 0 for columns 

    Returns:
    - True if the matrix is diagonally dominant for the given axis,
      False otherwise.

    """
    
    # Get only the absolute value of the matrix elements
    abs_matrix = np.abs(matrix)
    
    # Calculate the Diagonal values of the matrix
    D = abs_matrix.diagonal()

    # Sum the elements of the row, subtracting the diagonal element
    # If axis is 0, then do the same operation but on columns instead
    S = abs_matrix.sum(axis) - D

    # If any element of D is less or equal to the respective element of S,
    # it means that the matrix is not diagonal dominant
    if np.any(D <= S):
        return False
    else:
        return True