import numpy as np
from scipy import linalg


class Esprit():
    def __init__(self, array):
        self.array = array
        self.d = self.array.element_spacing
        self.M = self.array.num_antenna
        self.J1 = np.hstack((np.eye(self.M-1), np.zeros((self.M-1,1))))
        self.J2 = np.hstack((np.zeros((self.M-1, 1)), np.eye(self.M-1)))

    def estimate(self, x: np.ndarray, num_targets: int , formulation :str ='ls') -> np.ndarray:
        """
        :param formulation: `ls` for Least Squares and 
            `tls` for Total Least Squares 
        """
        R = (x.conj().T @ x) / x.shape[0]

        eigenvalues, eigenvectors = linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        Es = eigenvectors[:, idx[:num_targets]]

        E1 = self.J1 @ Es
        E2 = self.J2 @ Es

        if formulation == 'ls':
            Phi = linalg.inv(E1.T @ E1) @ E1.T @ E2
        elif formulation == 'tls':
            stacked = np.hstack([E1, E2])
            
            U, _, Vh = np.linalg.svd(stacked, full_matrices=False)
            V = Vh.conj().T
            
            V11 = V[:num_targets, :num_targets]
            V21 = V[num_targets:, :num_targets]
            V12 = V[:num_targets, num_targets:]
            V22 = V[num_targets:, num_targets:]
            
            Phi = -V12 @ np.linalg.inv(V22)
        else:
            raise ValueError("formulation must be 'ls' or 'tls'")
        
        eigenvalues_phi = linalg.eigvals(Phi)
        angles = np.arcsin(np.angle(eigenvalues_phi) / (2 * np.pi * self.d))
        
        return np.sort(angles)

