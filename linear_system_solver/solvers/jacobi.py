import numpy as np
from numpy.typing import NDArray
import warnings

from .base_solver import Solver, SolverMethod


class JacobiSolver(Solver):
    """
    Implementation of the Jacobi iterative method for solving linear systems.
    """
    def __init__(self):
        super().__init__()
        self.method = SolverMethod.JACOBI.value
        self.D_inv = None

    def _setup_solver(self, 
                      A: NDArray[np.float64], 
                      b: NDArray[np.float64],
                      initial_x: NDArray[np.float64]) -> None:
        """
        Setup solver-specific operations

        For Jacobi method, we precompute the inverse of the diagonal matrix D.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
        b : ndarray
            Right-hand side vector
        initial_x:
            Initial x vector
        """
        # Check if matrix is diagonally dominant for Jacobi method
        if not self._is_diagonally_dominant(A):
            warnings.warn("Jacobi might not converge since the matrix is not diagonally dominant.", stacklevel=2)

        
        # Get the inverse of diagonal elements        
        self.D_inv = np.diag(1 / np.diag(A)) 




    def iterate(self, 
                A: NDArray[np.float64], 
                b: NDArray[np.float64], 
                x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Execute one iteration of the Jacobi method.
        
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
            New solution vector after one iteration.
        """
        x_old = np.copy(x)
        x_new = x_old + np.matmul(self.D_inv, self._get_residue(A, b, x_old))
        return x_new
    


    