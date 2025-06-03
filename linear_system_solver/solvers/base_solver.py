from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from typing import Optional
import timeit
import warnings

class SolverMethod(str, Enum):
    """Enum for available solver methods"""
    JACOBI = "Jacobi"
    GAUSS_SEIDEL = "Gauss Seidel"
    GRADIENT = "Gradient"
    CONJUGATE_GRADIENT = "Conjugate Gradient"


class Solver(ABC):
    """
    Abstract base class for linear system solvers.
    
    This class defines the interface that all custom solver will implement, providing several needed functionalities.
    """
    
    def __init__(self):
        self.method = None  
        self.max_iterations = 20000
        self.verbose = True
        self.tolerance = None
        self.iterations = 0

    def solve(self, 
              A: NDArray[np.float64], 
              b: NDArray[np.float64], 
              tolerance: float,
              correct_x: NDArray[np.float64],
              max_iterations: Optional[int] = 20000,
              verbose: bool = True):

        """
        Solve the linear system Ax = b.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
        
        b : ndarray
            Right-hand side vector
        
        tolerance : float
            Convergence tolerance
        
        correct_x : ndarray
            True solution for error calculation. Default is a vector containing only ones.

        max_iterations : int, optional
            Maximum number of iterations. Default is self.max_iterations = 20000.

        verbose : bool, optional
            Whether to print progress messages. Default is True.
            
        Returns
        -------
        A custom object containing:
            - solution: solution computed by the method.
            - relative_error: relative error of the computed solution.
            - iterations: number of iterations before convergence of the given method.
            - time_spent: time spent (in seconds) by the given method to compute the solution.
            - solver_name: name of the method used to compute the solution.
            - converged: bool, True if the method convergedm, False otherwise.
          
        """
        
        # Set up solver settings
        self.verbose = verbose
        self.tolerance = tolerance

        if max_iterations is not None:
            self.max_iterations = max_iterations

        initial_x = np.zeros(len(b))

        # Check if system is solvable
        if not self._validate_inputs(A, b, initial_x):
            raise ValueError("Invalid input matrices. Check dimensions and properties.")

        self._setup_solver(A, b, initial_x)


        # Start iteration logic

        x = np.copy(initial_x)
        self.iterations = 0
        has_converged = False
        start_time = timeit.default_timer()


        while self._check_stopping_criteria(A, b, x):
            x = self.iterate(A, b, x)
            self.iterations += 1

        if self.iterations <= self.max_iterations:
            has_converged = True
        else: 
            has_converged = False


        end_time = timeit.default_timer()
        execution_time = end_time - start_time

        result = {
            "solution": x.tolist(),
            "relative_error": self._get_relative_error(x, correct_x),
            "iterations": self.iterations,
            "time_spent": execution_time,
            "solver_name": self.method,
            "converged": has_converged
            }

        return result
    
    
    @abstractmethod
    def iterate(self, 
                A: NDArray[np.float64],
                b: NDArray[np.float64], 
                x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Execute one iteration of the solver.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
        b : ndarray
            Right-hand side vector
        x : ndarray
            Current solution vector
            
        Returns
        -------
        ndarray
            New x vector after one iteration.
        """
        pass
    
    def _get_residue(self, 
                     A: NDArray[np.float64], 
                     b: NDArray[np.float64], 
                     x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculates the residue of the given x.

        Parameters
        ----------
        A : ndarray
            Coefficient matrix
        b : ndarray
            Right-hand side vector
        x : ndarray
            Current solution vector
            
        Returns
        -------
        ndarray
            Residue vector.
        """
        return b - np.dot(A, x)
    
    
    def _get_relative_error(self,
                             x: NDArray[np.float64], 
                             correct_x: NDArray[np.float64]) -> np.floating:
        """
        Calculates the relative error of the given x.

        Parameters
        ----------
        x : ndarray
            Vector containing the computed solution x.
        correct_x : ndarray
            Vector containing the given correct solution x.
        
        Returns
        -------
        np.float64
            Relative error of the computed solution.
        """
        return np.linalg.norm(x - correct_x) / np.linalg.norm(correct_x)
    

    def _check_stopping_criteria(self, 
                                 A:  NDArray[np.float64],
                                 b:  NDArray[np.float64],
                                 x:  NDArray[np.float64]) -> bool:
        """
        Checks if any of the stopping criteria has been met.

        Parameters
        ----------
        A : ndarray
            Coefficient matrix
        b : ndarray
            Right-hand side vector
        x : ndarray
            Current solution vector
        
        Returns
        -------
        bool
            False if any of the stopping criteria is met, True otherwise.
        """
        if self.tolerance is None:
            warnings.warn(f"{self.method} error: tolerance is None.")
            return False
        elif(self.iterations > self.max_iterations):
            warnings.warn(f"{self.method} with tolerance {self.tolerance} stopped because the max iterations number has been reached. Method hasn't converged.")
            return False
        else:
            if np.linalg.norm(self._get_residue(A, b, x)) / np.linalg.norm(b) < self.tolerance :
                return False
            else:
                return True
            
    def _validate_inputs(self, 
                         A: NDArray[np.float64],
                         b: NDArray[np.float64],
                         initial_x: NDArray[np.float64]) -> bool:
        """
        Validate input matrices dimensions and properties.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
        b : ndarray
            Right-hand side vector
        initial_x : ndarray
            Initial guess
            
        Returns
        -------
        bool
            True if inputs are valid, False otherwise.
        """
        
        if A.shape[0] != A.shape[1]:
            warnings.warn("Matrix A is not square")
            return False
        
        if A.shape[0] != len(b):
            warnings.warn(f"Matrix dimensions mismatch: A is {A.shape}, b is {len(b)}")
            return False
            
        if len(initial_x) != len(b):
            warnings.warn(f"Vector dimensions mismatch: initial_x is {len(initial_x)}, b is {len(b)}")
            return False
        
        return True
    

    def _setup_solver(self, 
                      A: NDArray[np.float64], 
                      b: NDArray[np.float64],
                      initial_x: NDArray[np.float64]) -> None:
        """Setup solver-specific operations"""
        pass



    def _is_diagonally_dominant(self, 
                                  A: NDArray[np.float64]) -> bool:
        """
        Check if the matrix is diagonally dominant.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
            
        Returns
        -------
        bool
            True if the matrix is diagonally dominant, False otherwise.
        """
        # Get only the absolute value of the matrix elements
        abs_matrix = np.abs(A)
        # Calculate the Diagonal values of the matrix
        D = np.diag(abs_matrix)
        # Sum the elements of the row, subtracting the diagonal element
        S = abs_matrix.sum(axis=1) - D

        # If any element of D is less or equal to the respective element of S,
        # it means that the matrix is not diagonally dominant
        if np.any(D <= S):
            return False
        else:
            return True


    def _is_symmetric(self, 
                      A: NDArray[np.float64]) -> bool:
        """
        Checks whether the given matrix is symmetric.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
            
        Returns
        -------
        bool
            True if the matrix is symmetric, False otherwise.        

        """
        relative_tolerance = 10e-14
        absolute_tolerance = 10e-14
        # allclose returns true if the first two parameters are equal given a tolerance interval
        if np.allclose(A, A.T, rtol=relative_tolerance, atol=absolute_tolerance):
            return True
        else:
            warnings.warn("Given matrix is not symmetric")
            return False



    def _is_positive_definite(self, 
                             A: NDArray[np.float64]) -> bool:
        """
        Checks whether the given matrix is positive definite.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
            
        Returns
        -------
        bool
            True if the matrix is positive definite, False otherwise.        

        """
        # If all eigen values of the given matrix are positive, it means
        # that the matrix is positive definite
        if np.all(np.linalg.eigvals(A) > 0):
            return True
        else:
            warnings.warn("Matrix is not positive definite.")
            return False

        


    def _is_symmetric_positive_definite(self, 
                                        A: NDArray[np.float64]) -> bool:
        """
        Checks whether the given matrix is symmetric and positive definite.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
            
        Returns
        -------
        bool
            True if the matrix is symmetric and positive definite, False otherwise.        

        """
        if self._is_symmetric(A) and self._is_positive_definite(A):
            return True
        else:
            return False

