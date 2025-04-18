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
        residue_transposed = np.transpose(residue)

        # calculate the alpha step and the new x
        alpha = ( np.dot(residue_transposed, residue)) / np.dot(residue_transposed, np.dot(A, residue))
        x_new = x_old + np.dot(alpha, residue)
        
        return x_new