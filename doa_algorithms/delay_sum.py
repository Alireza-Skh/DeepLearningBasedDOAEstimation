import numpy as np


class DelaySum:
    def __init__(
        self,
        array,
        all_doas
    ):
        self.all_doas = all_doas
        self.array = array
        self.steering_matrix_cache = None

    def _manifold_matrix(self) -> np.ndarray:
        if self.steering_matrix_cache is None or self.steering_matrix_cache_key != self.array.num_antenna:
            self.steering_matrix_cache = self.array.steering_matrix(
                self.all_doas, self.array.num_antenna)
            self.steering_matrix_cache_key = self.array.num_antenna
        return self.steering_matrix_cache

    def estimate(self, input_signal: np.ndarray) -> np.ndarray:
        R = np.cov(input_signal, rowvar=False)

        A = self._manifold_matrix()
        p_theta = A.conj().T @ R @ A
        p_theta = np.diag(p_theta)
        return np.abs(p_theta)
