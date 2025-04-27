from Solver import Solver
import numpy as np

class ConjugateGradientSolver(Solver):
    
    def __init__(self):
        super().__init__()
        self._solver_name = Solver.Method.CONJUGATE_GRADIENT.value

    def iterate(self, A, b, x):
        x_old = np.copy(x)
        
        direction_old = np.copy(self.direction)
        
        residue = self._get_residue(x_old)
        
        alpha_beta_denominator = np.dot(direction_old.T, np.dot(A, direction_old))
        
        # calculate the alpha step and the new x
        alpha = np.dot(direction_old.T, residue) / alpha_beta_denominator
        
        x_new = x_old + np.dot(alpha, direction_old)

        #new residue
        residue_new = self._get_residue(x_new)
    
        
        beta =  np.dot(direction_old.T, np.dot(A, residue_new)) / alpha_beta_denominator

        
        direction_new = residue_new - np.dot(beta, direction_old)        
        

        self.direction = np.copy(direction_new)
        return x_new