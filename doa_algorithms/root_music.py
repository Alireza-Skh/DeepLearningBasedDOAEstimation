import numpy as np
from scipy import linalg


class RootMUSIC:
    def __init__(self, array, num_targets):
        self.array = array
        self.num_targets = num_targets
        self.d = array.element_spacing * array._lambda

    def estimate(self, input_signal: np.ndarray):
        R = (input_signal.conj().T @ input_signal) / input_signal.shape[0]

        _, _, vh = linalg.svd(R)
        eigenvectors = vh.conj().T

        if self.num_targets >= self.array.num_antenna:
            raise ValueError(f"Number of sources ({self.num_targets}) must be less than number of antennas ({self.array.num_antenna})")

        noise_eigenvectors = eigenvectors[:, self.num_targets:]

        C = noise_eigenvectors @ noise_eigenvectors.conj().T

        M = self.array.num_antenna
        a = np.zeros(2*M - 1, dtype=complex)

        for k in range(-(M-1), M):
            if k >= 0:
                a[M - 1 + k] = np.sum(np.diag(C, k))
            else:
                a[M - 1 + k] = np.sum(np.diag(C, k))

        roots = np.roots(a[::-1])
        inside_roots = roots[np.abs(roots) < 1.0]

        if len(inside_roots) < self.num_targets:
            roots_sorted = roots[np.argsort(np.abs(np.abs(roots) - 1.0))]
            selected_roots = roots_sorted[:self.num_targets]
        else:
            inside_roots_sorted = inside_roots[np.argsort(np.abs(inside_roots))[::-1]]
            selected_roots = inside_roots_sorted[:self.num_targets]

        angles = np.angle(selected_roots)
        sin_thetas = angles * self.array._lambda / (2 * np.pi * self.d)

        valid_indices = np.abs(sin_thetas) <= 1.0
        sin_thetas = sin_thetas[valid_indices]

        if len(sin_thetas) == 0:
            return np.array([])

        return np.arcsin(sin_thetas)
