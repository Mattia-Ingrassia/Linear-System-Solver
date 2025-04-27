from Solver import Solver
import numpy as np

class GradientSolver(Solver):
    
    def __init__(self):
        super().__init__()
        self._solver_name = Solver.Method.GRADIENT.value

    def iterate(self, A, b, x):
        # calculate the residue vector and its transposed value
        x_old = np.copy(x)
        residue = self._get_residue(x_old)

        # calculate the alpha step and the new x
        alpha = ( np.dot(residue.T, residue)) / np.dot(residue.T, np.dot(A, residue))
        x_new = x_old + np.dot(alpha, residue)
        
        return x_new