import numpy as np
from numpy.typing import NDArray

from .base_solver import Solver, SolverMethod

class GradientSolver(Solver):
    """
    Implementation of the Gradient iterative method for solving linear systems.
    """
    def __init__(self):
        super().__init__()
        self.method = SolverMethod.GRADIENT.value


    def iterate(self,
                A: NDArray[np.float64], 
                b: NDArray[np.float64], 
                x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Execute one iteration of the Gradient method.
        
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
        # calculate the residue vector and its transposed value
        x_old = np.copy(x)
        residue = self._get_residue(A, b, x_old)

        # calculate the alpha step and the new x
        alpha = np.dot(residue.T, residue) / np.dot(residue.T, np.dot(A, residue))
        x_new = x_old + np.dot(alpha, residue)
        
        return x_new
    

