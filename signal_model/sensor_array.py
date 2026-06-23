from time import sleep
import numpy as np
from numpy.random import default_rng
import torch


class ULA:
    C = 3e8

    def __init__(
        self,
        num_antenna: int,
        freq: int,
        num_samples: int,
        angles_bound: list | tuple,
        element_spacing: float = 0.5,  # lambda / 2
        min_resolution: float = 1,
        baseband_mode: bool = True,
        coherent: bool = False,
        angle_type: str = "rad",
        array_imperfections: str = "none",
        rng=default_rng(seed=42),
    ) -> None:
        super().__init__()
        if angle_type.lower() not in ["rad", "deg"]:
            raise ValueError(
                f"Invalid angle type '{angle_type}'. Supported types are 'deg' and 'rad'.")
        if freq <= 0:
            raise ValueError("Frequency must be positive.")
        if num_antenna <= 0:
            raise ValueError("Number of antennas must be positive.")
        if element_spacing <= 0:
            raise ValueError("Element spacing must be positive.")
        if array_imperfections.lower() not in ("none", "gain", "phase", "pos", "all"):
            raise ValueError(
                f"Invalid array_imperfections type '{array_imperfections}'.")

        self._is_degrees = angle_type.lower() == "deg"
        self.freq = freq
        self.num_antenna = num_antenna
        self.num_samples = num_samples
        self.baseband_mode = baseband_mode
        self.coherent = coherent
        self.element_spacing = element_spacing
        self.min_resolution = min_resolution
        self.angles_bound = angles_bound
        self.rng = rng
        self.angles_list = self._angles_list
        self.array_imperfections = array_imperfections.lower()

        self.d = self._tx_indices * self.element_spacing * self._lambda
        self._generate_imperfections()

        if not self.baseband_mode:
            self.sampling_freq = 4 * self.freq
            self.sampling_interval = 1.0 / self.sampling_freq
            self.max_time = np.round(
                self.num_samples * self.sampling_interval, decimals=int(np.log10(self.freq))
            )

    @property
    def _lambda(self) -> float:
        return self.C / self.freq

    @property
    def _time(self) -> np.ndarray:
        return np.arange(self.num_samples) * self.sampling_interval

    @property
    def _tx_indices(self) -> np.ndarray:
        return np.arange(self.num_antenna)

    @property
    def _angles_list(self) -> np.ndarray:
        return np.linspace(*self.angles_bound, endpoint=False, dtype=np.float32)

    def gen_rand_angles(self, num_targets) -> np.ndarray:
        rand_angles = self.rng.choice(self.angles_list, self.angles_bound[-1]//4)

        keep = []
        for angle in rand_angles:
            if not keep or abs(angle - keep[-1]) >= self.min_resolution:
                keep.append(angle)
                keep.sort()
            if len(keep) == num_targets:
                break

        if len(keep) < num_targets:
            raise ValueError(
                f"Could not find {num_targets} angles with minimum resolution {self.min_resolution}")

        return np.array(sorted(keep))

    def check_angle_type(self, angle) -> float:
        if self._is_degrees:
            angle = np.deg2rad(angle)
        return angle

    def _generate_imperfections(self) -> None:
        self.gain_errors = np.ones(self.num_antenna)
        self.phase_errors = np.zeros(self.num_antenna)
        self.pos_errors = np.zeros(self.num_antenna)

        if self.array_imperfections == "none":
            return

        if "gain" in self.array_imperfections or "all" in self.array_imperfections:
            gain_db = (self.rng.random(self.num_antenna) - 0.5) * 5  # ±2.5 dB
            gain_db[0] = 0.0
            self.gain_errors = 10**(gain_db / 20)

        if "phase" in self.array_imperfections or "all" in self.array_imperfections:
            phase_deg = (self.rng.random(self.num_antenna) - 0.5) * 60  # ±30°
            phase_deg[0] = 0.0
            self.phase_errors = np.deg2rad(phase_deg)

        if "pos" in self.array_imperfections or "all" in self.array_imperfections:
            pos_fraction = (self.rng.random(self.num_antenna) - 0.5) * 0.2  # ±10%
            pos_fraction[0] = 0.0
            self.pos_errors = pos_fraction * self.element_spacing * self._lambda
            self.d += self.pos_errors 
        
        self.complex_weights = self.gain_errors * np.exp(1j * self.phase_errors)

    def receive_steering_vector(self, angle: float) -> np.ndarray:
        angle = self.check_angle_type(angle)
        stv = np.exp(-2j * np.pi * self.d * np.sin(angle) / self._lambda)

        if self.array_imperfections != "none":
            stv *= self.complex_weights

        return stv

    def receive_steering_vector_cuda(self, angle: float) -> torch.Tensor:
        device = angle.device

        if self._is_degrees:
            angle = torch.deg2rad(angle)

        d = torch.as_tensor(self.d).to(device)
        stv = torch.exp(-2j * torch.pi * d * torch.sin(angle) / self._lambda).to(device)

        if self.array_imperfections != "none":
            stv *= torch.as_tensor(self.complex_weights).to(device)

        return stv

    def steering_matrix(self, angles: np.ndarray) -> np.ndarray:
        angles = self.check_angle_type(angles)
        stm = np.exp(-2j * np.pi * self.d[:, None] * np.sin(angles)[None, :] / self._lambda)

        if self.array_imperfections != "none":
            stm *= self.complex_weights[:, None]

        return stm

    def steering_matrix_derivative(self, angles: np.ndarray) -> np.ndarray:
        angles = self.check_angle_type(angles)
        base_stm = np.exp(-2j * np.pi * self.d[:, None] * np.sin(angles)[None, :] / self._lambda)
        derivative_factor = -2j * np.pi * self.d[:, None] * np.cos(angles)[None, :] / self._lambda
        derivative = derivative_factor * base_stm

        if self.array_imperfections != "none":
            derivative *= self.complex_weights[:, None]
            
        return derivative

    def random_noise(self, shape=None) -> np.ndarray:
        if shape is None:
            shape = (self.num_antenna, self.num_samples)
        
        noise = (self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape))
        return noise / np.sqrt(2)

    def p_signal(self, snr_db:float, num_targets: int) -> np.ndarray:
        shape = (num_targets, self.num_samples)

        if self.baseband_mode:
            real_part = self.rng.standard_normal(shape)
            imaginary_part = self.rng.standard_normal(shape)
            sig = real_part + 1j * imaginary_part
        else:
            doppler_freqs = self.rng.uniform(-self.freq * 0.1, self.freq*0.1, num_targets)
            random_phases = self.rng.uniform(0, 2*np.pi, num_targets)

            sig = np.zeros(shape, dtype=complex)
            for i in range(num_targets):
                sig[i, :] = (
                    np.cos(2 * np.pi * (self.freq + doppler_freqs[i]) * self._time + random_phases[i])
                )

        if self.coherent:
            reference_signal = sig[0:1, :]
            sig = np.tile(reference_signal, (num_targets, 1))

        random_amplitudes = self.rng.uniform(0.1, 1.0, (num_targets, 1))
        sig *= random_amplitudes

        sig_norm = np.sqrt(np.mean(np.abs(sig)**2))
        sig *= np.sqrt(10 ** (snr_db * 0.1)) / sig_norm

        return sig 

    def receive_waveform(self, target_angles: np.ndarray, snr_db: float) -> np.ndarray:
        stm = self.steering_matrix(target_angles)
        p_sig = self.p_signal(snr_db, len(target_angles))
        noise = self.random_noise()
        r_sig = np.dot(stm, p_sig)
        return (r_sig + noise).T
