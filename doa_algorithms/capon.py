import numpy as np
from scipy import linalg


class Capon:
    def __init__(self, array, doas_list):
        super().__init__()
        self.array = array
        self.doas_list = doas_list
        self.epsilon = 1e-10
        self.num_angles = len(self.doas_list)
        self._manifold_m = None

    @property
    def manifold_matrix_1d(self) -> np.ndarray:
        if self._manifold_m is None:
            steering_vectors = [
                self.array.receive_steering_vector(angle) for angle in self.doas_list
            ]
            self._manifold_m = np.stack(steering_vectors, axis=0)
        return self._manifold_m

    def estimate(self, input_signal: np.ndarray) -> np.ndarray:
        R = input_signal.conj().T @ input_signal / input_signal.shape[0]
        R += self.epsilon * np.eye(self.array.num_antenna)
        try:
            R_inv = linalg.inv(R)
        except linalg.LinAlgError:
            raise ValueError("Failed to invert R.")

        A = self.manifold_matrix_1d
        spec_denominator = np.einsum('in,nm,im->i', A, R_inv, A.conj())
        spectrum = 1.0 / (np.real(spec_denominator) + 1e-6)
        return spectrum
