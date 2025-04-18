from Solver import Solver
import numpy as np

class GaussSeidelSolver(Solver):
    
    def __init__(self):
        super().__init__()
        self._solver_name = Solver.Method.GAUSS_SEIDL.value

    def iterate(self, A, b, x):
        x_old = np.copy(x)
        x_new = x_old + self._tril_matrix_solver(self._get_residue(x_old))
        return x_new
    


    # direct method for solving triangular linear systems
    def _tril_matrix_solver(self, rhs):
        matrix = self.L
        solutions = np.zeros(len(matrix))
        solutions[0] = rhs[0] / matrix[0][0]
        for i in range(1, len(matrix), 1) :
            b = rhs[i]
            u = matrix[i][i]
            solutions[i] = (1/u) * (b - np.dot(matrix[i], solutions))
        return solutions
    
