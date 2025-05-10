from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import timeit
import warnings

class Solver(ABC):

    class Method(Enum):
        JACOBI = "Jacobi"
        GAUSS_SEIDL = "Gauss Seidel"
        GRADIENT = "Gradient"
        CONJUGATE_GRADIENT = "Conjugate Gradient"
    
    def __init__(self):
        self.A = None
        self.b = None
        self.tolerance = None
        self.correct_x = None
        self.initial_x = None
        self.D_inv = None
        self.L = None
        self.direction = None
        self.iterations = 0
        self.max_iterations = 20000
        self._solver_name = "Generic solver"
    
    def solve(self, A, b, tolerance, correct_x):
        self.A = A
        self.b = b
        self.tolerance = tolerance
        self.correct_x = correct_x
        self.initial_x = np.zeros(len(b))

        if self._solver_name == Solver.Method.JACOBI.value:
            self.D_inv = np.diag(1 / np.diag(A))

        if self._solver_name == Solver.Method.GAUSS_SEIDL.value:
            self.L = np.tril(A)

        if self._solver_name == Solver.Method.CONJUGATE_GRADIENT.value:
            self.direction = self._get_residue(self.initial_x)
        
        x = np.copy(self.initial_x)

        start_time = timeit.default_timer()

        self.iterations = 0

        while self._check_stopping_criteria(x):
            x = self.iterate(A, b, np.copy(x))
            self.iterations += 1
            
        end_time = timeit.default_timer()
    
        execution_time = end_time - start_time
    
        result = {
            "solution": x.tolist(),
            "relative_error": self._get_relative_error(x),
            "iterations": self.iterations,
            "time_spent": execution_time,
            "solver_name": self._solver_name
            }

        return result
    
    
    @abstractmethod
    def iterate(self):
        pass
    
    def _get_residue(self, x):
        return self.b - np.dot(self.A, x)
    
    def _get_relative_error(self, x):
        return np.linalg.norm(x - self.correct_x)/np.linalg.norm(self.correct_x)
    
    def _check_stopping_criteria(self, x):
        if(self.iterations > self.max_iterations):
            warnings.warn(f"{self._solver_name} with tolerance {self.tolerance} stopped because the max iterations number has been reached. Method hasn't converged.")
            return False
        else:
            if np.linalg.norm(self._get_residue(x)) / np.linalg.norm(self.b) < self.tolerance :
                print(f"{self._solver_name} with tolerance {self.tolerance} stopped because the correct result with desired tolerance has been reached. Method has converged.")
                return False
            else:
                return True
            

    def check_matrix_properties(self, matrix):
        # 1. check if the matrix is square
        if self.is_square(matrix):
            # 2. check if the matrix is diagonally dominant
            if not self.is_diagonally_dominant(matrix):
                warnings.warn("Jacobi might not converge since the matrix is not diagonally dominant.")
            
            # 3. check if the matrix is symmetric
            if self.is_symmetric(matrix):
                # 4. check if the matrix is positive definite
                if self.is_positive_definite(matrix):    
                    return True            
                
        return False

    def is_symmetric(self, matrix):
        """
        Checks whether the given matrix is symmetric.

        Parameters:
        - matrix: Given matrix, passed as a square NumPy array.

        Returns:
        - True if the matrix is symmetric.
        - False otherwise.

        """
        relative_tolerance = 10e-14
        absolute_tolerance = 10e-14
        # allclose returns true if the first two parameters are equal given a tolerance interval
        if np.allclose(matrix, matrix.T, rtol=relative_tolerance, atol=absolute_tolerance):
            return True
        else:
            warnings.warn("Given matrix is not symmetric")
            return False

    def is_square(self, matrix):
        """
        Checks whether the given matrix is square.

        Parameters:
        - matrix: Given matrix, passed as a square NumPy array.

        Returns:
        - True if the matrix is square.
        - False otherwise.

        """
        if(len(matrix.shape) == 2):
            if (matrix.shape[0] == matrix.shape[1]):
                return True
            else:
                warnings.warn("Number of rows and columns of the given matrix don't match")
        else:
            warnings.warn("Given matrix is not bi-dimensional")
        return False

    def is_positive_definite(self, matrix):
        """
        Checks whether the given matrix is positive definite.

        Parameters:
        - matrix: Given matrix, passed as a square NumPy array.

        Returns:
        - True if the matrix is definite positive.
        - False otherwise.

        """
        # If all eigen values of the given matrix are positive, it means
        # that the matrix is positive definite
        if np.all(np.linalg.eigvals(matrix) > 0):
            return True
        else:
            warnings.warn("Matrix is not positive definite.")
            return False


    def is_diagonally_dominant(self, matrix, axis=1):
        """
        Checks whether the given matrix is diagonally dominant.

        Parameters:
        - matrix: Given matrix, passed as a square NumPy array.
        - axis: 1 for row dominance (default), 0 for column dominance.

        Returns:
        - True if the matrix is diagonally dominant along the specified axis,
        False otherwise.

        """
        
        # Get only the absolute value of the matrix elements
        abs_matrix = np.abs(matrix)
        
        # Calculate the Diagonal values of the matrix
        D = np.diag(abs_matrix)
        
        # Sum the elements of the row, subtracting the diagonal element
        # If axis is 0, then do the same operation but on columns instead
        S = abs_matrix.sum(axis) - D

        # If any element of D is less or equal to the respective element of S,
        # it means that the matrix is not diagonally dominant
        if np.any(D <= S):
            return False
        else:
            return True