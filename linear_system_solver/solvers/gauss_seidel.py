import numpy as np
from numpy.typing import NDArray
import warnings

from .base_solver import Solver, SolverMethod

class GaussSeidelSolver(Solver):
    """
    Implementation of the Gauss Seidel iterative method for solving linear systems.
    """
    def __init__(self):
        super().__init__()
        self.method = SolverMethod.GAUSS_SEIDEL.value
        self.L = None


    def _setup_solver(self, 
                      A: NDArray[np.float64], 
                      b: NDArray[np.float64],
                      initial_x: NDArray[np.float64]) -> None:
        """
        Setup solver-specific operations

        For Gauss-Seidel method, we compute the triangular lower matrix.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
        b : ndarray
            Right-hand side vector
        initial_x:
            Initial x vector
        """
        # Check if matrix is diagonally dominant for Gauss Seidel method
        if not self._is_diagonally_dominant(A):
            warnings.warn("Gauss-Seidel might not converge since the matrix is not diagonally dominant.", stacklevel=2)

        self.L = np.tril(A)


    def iterate(self, 
                A: NDArray[np.float64], 
                b: NDArray[np.float64], 
                x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Execute one iteration of the Gauss-Seidel method.
        
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
        x_new = x_old + self._tril_matrix_solver(self._get_residue(A, b, x_old))
        return x_new
    


    # direct method for solving triangular linear systems
    def _tril_matrix_solver(self,
                            rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Solve the triangular lower matrix.
        
        Parameters
        ----------
        rhs : ndarray
            Right-hand side vector
            
        Returns
        -------
        ndarray
            New solution vector 
        """
        matrix = self.L
        solutions = np.zeros(len(matrix))
        solutions[0] = rhs[0] / matrix[0][0]
        for i in range(1, len(matrix)) :
            b = rhs[i]
            u = matrix[i][i]
            solutions[i] = (1/u) * (b - np.dot(matrix[i, :i], solutions[:i]))
        return solutions

        
