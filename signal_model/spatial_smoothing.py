import numpy as np


def fbss(input_signal, num_antenna, num_subarray):
    """
    Compute the forward backward spatially smoothed (FBSS) covariance matrix of a signal.

    Parameters:
    - `input_signal`: 2D array where each row represents a time series data from all antennas.
    - `num_antenna`: Total number of antennas.
    - `num_subarray`: Number of subarrays to consider for smoothing.

    Returns:
    - 2D array representing the spatially smoothed covariance matrix.
    """
    if num_subarray < 1 or num_subarray > num_antenna:
        raise ValueError(f'The number of subarrays must be within [1, {num_antenna}].')

    R = np.cov(input_signal, rowvar=False)
    Rf = R[:num_antenna-num_subarray+1, :num_antenna-num_subarray+1].copy()
    for i in range(1, num_subarray):
        Rf += R[i:i+num_antenna-num_subarray+1, i:i+num_antenna-num_subarray+1]
    Rf /= num_subarray

    return 0.5 * (Rf + np.flip(Rf).conj())


def improved_spatial_smoothed_covariance(input_signal, num_antenna, num_subarray):
    """
    Compute the improved spatially smoothed covariance matrix for a given input signal.

    Parameters:
    - `input_signal` : numpy.ndarray
    The input signal with dimensions (num_samples, num_antennas), where `num_samples` is the number of time samples
    and `num_antennas` is the total number of antennas in the array.

    - `num_antenna` : int
        The total number of antennas in the array.

    - `num_subarray` : int
        The number of subarrays used for smoothing. It should be less than or equal to `num_antennas`.

    Returns:
    - `R1` : numpy.ndarray
        The improved spatially smoothed covariance matrix with dimensions (p, p), where `p = num_antenna - num_subarray + 1`.
    """

    p = num_antenna-num_subarray+1
    R = np.cov(input_signal, rowvar=False)
    Rf_ii = Rf_jj = Rf_ij = Rf_ji = 0
    for i in range(num_subarray):
        for j in range(num_subarray):
            Rf_ii += R[i:i+p, i:i+p]
            Rf_jj += R[j:j+p, j:j+p]
            Rf_ij += R[i:i+p, j:j+p]
            Rf_ji += R[j:j+p, i:i+p]

    R1 = (Rf_ii @ np.flip(Rf_ii)) + (Rf_jj @ np.flip(Rf_jj))
    R1 += (Rf_ij @ np.flip(Rf_ij)) + (Rf_ji @ np.flip(Rf_ji))
    return R1 / 2*num_subarray
