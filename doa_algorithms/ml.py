import numpy as np
from scipy import linalg 


class MaximumLikelihoodEstimator:
    def __init__(self, array, num_targets: int, angles_bound: tuple):
        self.array = array
        self.num_targets = num_targets
        self.angles_grid = np.linspace(*angles_bound)

    def _get_steering_vector(self, theta) -> np.ndarray:
        return self.array.receive_steering_vector(theta)

    def _projection_matrix(self, A) -> np.ndarray:
        try:
            A_H_A_inv = linalg.inv(A.T @ A.conj())
        except linalg.LinAlgError:
            A_H_A_inv = linalg.pinv(A.T @ A.conj())
        P_A = A_H_A_inv @ A.T
        return P_A

    def _build_manifold_matrix(self, params_list: list) -> np.ndarray:
        stv_list = [self._get_steering_vector(theta) for theta in params_list]
        return np.column_stack(stv_list)

    def _get_null_projection(self, A) -> np.ndarray:
        P_A = self._projection_matrix(A)
        I = np.eye(P_A.shape[0], dtype=complex)
        P_null = I - P_A
        return P_null

    def _inint_estimation(self, R_X) -> list:
        estimated_params = []
        stv = np.zeros((R_X.shape[0], self.num_targets), dtype=complex)

        for k in range(self.num_targets):
            max_spectrum_val = -np.inf
            best_param = None

            for theta in self.angles_grid:
                stv[:, k] = self._get_steering_vector(theta)

                A_k = stv[:, :k+1]
                P_A = self._projection_matrix(A_k)

                spectrum_val = np.trace(P_A @ R_X).real

                if spectrum_val > max_spectrum_val:
                    max_spectrum_val = spectrum_val
                    best_param = theta

            best_param = best_param
            estimated_params.append(best_param)
            stv[:, k] = self._get_steering_vector(best_param)

        return estimated_params

    def _refine_estimates(self, estimated_params: list, R_X, max_iterations: int, tolerance: float):
        for p in range(max_iterations):
            params_before_iter = np.array(estimated_params).copy()

            for k in range(self.num_targets):
                other_params = estimated_params[:k] + estimated_params[k+1:]

                if not other_params:
                    continue

                A_other = self._build_manifold_matrix(other_params)
                P_other_perp = self._get_null_projection(A_other)
                R_proj = P_other_perp.conj().T @ R_X @ P_other_perp

                max_spectrum_val_theta = -np.inf
                best_theta = estimated_params[k]

                for theta in self.angles_grid:
                    a = self._get_steering_vector(theta)[:, np.newaxis]
                    spectrum_val = (a.conj().T @ R_proj @ a).real.item()
                    if spectrum_val > max_spectrum_val_theta:
                        max_spectrum_val_theta = spectrum_val
                        best_theta = theta
                
                best_theta = best_theta
                estimated_params[k] = best_theta

            current_params = np.array(estimated_params)
            param_change = np.linalg.norm(current_params - params_before_iter)

            print(f"  Iteration {p+1}/{max_iterations}, Parameter change: {param_change:.4f}")
            if param_change < tolerance:
                print("  Convergence reached.")
                break

        return estimated_params

    def estimate(self, received_signal, refine_estimates: bool = True, max_iterations: int = 10, tolerance: float = 1e-3):
        num_snapshots = received_signal.shape[1]
        R_X = (received_signal.conj().T @ received_signal) / num_snapshots

        estimated_params = self._inint_estimation(R_X)
        if refine_estimates:
            print("Starting refinement...")
            estimated_params = self._refine_estimates(estimated_params, R_X, max_iterations, tolerance)

        return estimated_params