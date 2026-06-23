import numpy as np
from scipy import linalg


class Music:
    def __init__(
        self,
        array,
        doas_list: list | np.ndarray,
    ):
        super().__init__()
        self.array = array
        self.doas_list = doas_list
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

    def _calculate_noise_projector(self, input_signal: np.ndarray, num_sources: int) -> np.ndarray:
        if input_signal.ndim != 2:
            raise ValueError("Input signal must be a 2D array.")
        R = (input_signal.conj().T @ input_signal) / input_signal.shape[0]
        eigenvectors = linalg.svd(R)[0]
        En = eigenvectors[:, num_sources:]
        return En @ En.conj().T

    def estimate(self, input_signal: np.ndarray, num_sources: int) -> np.ndarray:
        Pn = self._calculate_noise_projector(input_signal, num_sources)
        A = self.manifold_matrix_1d
        spec_denominator = np.einsum('in,nm,im->i', A, Pn, A.conj())
        music_sp = 1.0 / (np.real(spec_denominator) + 1e-6)
        return music_sp
