from Solver import Solver
import numpy as np

class JacobiSolver(Solver):
    
    def __init__(self):
        super().__init__()
        self._solver_name = Solver.Method.JACOBI.value

    def iterate(self, A, b, x):
        x_old = np.copy(x)
        x_new = x_old + np.matmul(self.D_inv, + self._get_residue(x_old))
        return x_new