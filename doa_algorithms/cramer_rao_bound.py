import numpy as np
from scipy import linalg


class CramerRaoBound:
    def __init__(self, array):
        self.array = array
        self.rng = self.array.rng

    def estimate(self, sig: np.ndarray, target_angle_rad: np.ndarray, snr_db: float):
        A = self.array.steering_matrix(target_angle_rad)
        D = self.array.steering_matrix_derivative(target_angle_rad)

        snr_linear = 10 ** (snr_db / 10)

        # S = (linalg.pinv(A.conj().T @ A) @ A.conj().T @ sig.T).T
        S = self.rng.standard_normal((sig.shape[0], len(target_angle_rad))) + 1j * self.rng.standard_normal((sig.shape[0], len(target_angle_rad)))
        S *= np.sqrt(snr_linear / 2)

        R = (S.conj().T @ S) / sig.shape[0]

        P_A = A @ linalg.pinv(A.conj().T @ A) @ A.conj().T
        P_A = np.eye(P_A.shape[0], dtype=P_A.dtype) - P_A

        H = D.conj().T @ P_A @ D

        crb = np.real(H * R.T)
        crb = linalg.inv(crb)

        crb *= 1 / (2*sig.shape[0])

        return np.trace(crb)
