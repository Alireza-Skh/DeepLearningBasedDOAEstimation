import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d

class SpectrumPeakFinder:
    def __init__(self, expected_peaks, min_res, sampling_freq, filter_type='butterworth', filter_params=None):
        """
        Initialize the peak finder with iterative peak detection approach.

        Parameters:
        - expected_peaks: int, exact number of peaks to find
        - filter_type: str, type of filter ('butterworth', 'gaussian', 'savgol', 'none')
        - filter_params: dict, parameters for the chosen filter
        """
        self.expected_peaks = expected_peaks
        self.filter_type = filter_type
        self.min_res = min_res
        self.min_idx = int(np.round(min_res * sampling_freq))

        default_params = {
            'butterworth': {'cutoff_freq': 0.1, 'order': 4},
            'gaussian': {'sigma': 2.0},
            'savgol': {'window_length': 11, 'polyorder': 3}
        }

        if filter_params is None:
            self.filter_params = default_params.get(filter_type, {})
        else:
            self.filter_params = filter_params

    def apply_lowpass_filter(self, spectrum):
        if self.filter_type == 'none':
            return spectrum.copy()

        elif self.filter_type == 'butterworth':
            cutoff = self.filter_params.get('cutoff_freq', 0.1)
            order = self.filter_params.get('order', 4)
            b, a = butter(order, cutoff, btype='low', analog=False)
            filtered_spectrum = filtfilt(b, a, spectrum)

        elif self.filter_type == 'gaussian':
            sigma = self.filter_params.get('sigma', 2.0)
            filtered_spectrum = gaussian_filter1d(spectrum, sigma=sigma)

        elif self.filter_type == 'savgol':
            window_length = self.filter_params.get('window_length', 11)
            polyorder = self.filter_params.get('polyorder', 3)

            if window_length % 2 == 0:
                window_length += 1
            window_length = max(window_length, polyorder + 1)
            window_length = min(window_length, len(spectrum))

            filtered_spectrum = savgol_filter(
                spectrum, window_length, polyorder)

        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        return filtered_spectrum

    def find_single_peak_with_width(self, spectrum):
        peaks, _ = find_peaks(spectrum)
        spec_len = len(spectrum)

        if len(peaks) == 0:
            return -1, -1
        elif len(peaks) > 1:
            peak_heights = spectrum[peaks]
            highest_peak_idx = peaks[np.argmax(peak_heights)]
            peak_amp = spectrum[highest_peak_idx]
        else:
            highest_peak_idx = peaks
            peak_amp = spectrum[highest_peak_idx]

        half_max = max(peak_amp / 2, self.min_res)
        left_idx = highest_peak_idx.item()
        right_idx = highest_peak_idx.item()
        mean_spec = np.mean(spectrum[left_idx-self.min_idx:right_idx+self.min_idx])

        while left_idx > 0 and (spectrum[left_idx] > half_max or spectrum[left_idx] > mean_spec):
            left_idx -= 1
        while right_idx < spec_len - 1 and (spectrum[right_idx] > half_max or spectrum[right_idx] > mean_spec):
            right_idx += 1

        width = right_idx - left_idx

        left_bound = int(max(0, highest_peak_idx - width/2))
        right_bound = int(min(len(spectrum), highest_peak_idx + width/2 + 1))

        peak_region = spectrum[left_bound:right_bound]
        if len(peak_region) > 0 and np.sum(peak_region) > 0:
            indices_in_region = np.arange(left_bound, right_bound)
            weights = peak_region / np.sum(peak_region)
            mean_index = np.sum(indices_in_region * weights)
        else:
            mean_index = highest_peak_idx

        return int(mean_index), width

    def zero_out_peak_region(self, spectrum, peak_index, peak_width):
        spectrum_copy = spectrum.copy()

        left_bound = int(max(0, peak_index - peak_width))
        right_bound = int(min(len(spectrum), peak_index + peak_width + 1))
        spectrum_copy[left_bound:right_bound] = 0

        return spectrum_copy

    def find_peak_indices(self, spectrum, min_prominence_ratio=0.05, return_filtered_spectrum=False):
        filtered_spectrum = self.apply_lowpass_filter(spectrum)
        filtered_spectrum /= np.max(spectrum)
        working_spectrum = filtered_spectrum.copy()

        peak_indices = []

        for _ in range(self.expected_peaks):
            peak_idx, peak_width = self.find_single_peak_with_width(working_spectrum)
            peak_prom = np.max(working_spectrum) * min_prominence_ratio 
            peak_indices.append(peak_idx)

            working_spectrum = self.zero_out_peak_region(
                working_spectrum, peak_idx, peak_width
            )

            if np.max(working_spectrum) < peak_prom:
                break
        
        for _ in range(np.abs(len(peak_indices) - self.expected_peaks)):
            peak_indices.append(-1) 

        if peak_indices:
            sorted_order = np.argsort(peak_indices)
            peak_indices = [peak_indices[i] for i in sorted_order]

        if return_filtered_spectrum:
            return peak_indices, filtered_spectrum
        return peak_indices
