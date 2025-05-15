import numpy as np
from numpy.typing import NDArray

from .base_solver import Solver, SolverMethod

class ConjugateGradientSolver(Solver):
    """
    Implementation of the Conjugate Gradient iterative method for solving linear systems.
    """
    def __init__(self):
        super().__init__()
        self.method = SolverMethod.CONJUGATE_GRADIENT.value
        self.direction = None

    def _setup_solver(self, 
                      A: NDArray[np.float64], 
                      b: NDArray[np.float64],
                      initial_x: NDArray[np.float64]) -> None:
        """
        Setup solver-specific operations

        For Conjugate Gradient method, we compute the initial direction.
        
        Parameters
        ----------
        A : ndarray
            Coefficient matrix
        b : ndarray
            Right-hand side vector
        initial_x:
            Initial x vector
        
        """

        self.direction = self._get_residue(A, b, initial_x)


    def iterate(self, A, b, x):
        """
        Execute one iteration of the Conjugate Gradient method.
        
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
        
        direction_old = np.copy(self.direction)
        
        residue = self._get_residue(A, b, x_old)
        
        alpha_beta_denominator = np.dot(direction_old.T, np.dot(A, direction_old))
        
        # calculate the alpha step and the new x
        alpha = np.dot(direction_old.T, residue) / alpha_beta_denominator
        
        x_new = x_old + np.dot(alpha, direction_old)

        #new residue
        residue_new = self._get_residue(A, b, x_new)
    
        
        beta =  np.dot(direction_old.T, np.dot(A, residue_new)) / alpha_beta_denominator

        
        direction_new = residue_new - np.dot(beta, direction_old)        
        

        self.direction = np.copy(direction_new)
        return x_new
    








