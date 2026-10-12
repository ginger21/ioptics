import warnings
from typing import List, Optional

import numpy as np
import torch


class OpticalComponent(torch.nn.Module):
    """Base class for optical component of optical neural network"""

    def __init__(self, ports: List[int], dof: int):
        super().__init__()
        self.ports = ports
        self.dof = dof
        self.register_buffer("_1j", torch.tensor(1j, dtype=torch.complex64))

    def __repr__(self):
        """String representation, can be overridden in child classes"""
        return "<OpticalComponent ports={}>".format(self.ports)

    def get_transfer_matrix(self) -> torch.Tensor:
        """Logic for computing the transfer operator of the component"""
        raise NotImplementedError(
            "get_transfer_matrix() must be overriden in child class!"
        )

    def reset_parameters(self) -> None:
        pass



class PhaseShifter(OpticalComponent):
    """Class for single-mode phase shifter"""

    def __init__(self, m: int, phi: Optional[float] = None, phase_noise_std: float = 0.0, noise_aware_training: bool = False):
        super().__init__([m], dof=1)
        if not isinstance(m, int) or m < 0:
            raise ValueError("m must be a non-negative integer")
        if phase_noise_std < 0:
            raise ValueError("phase_noise_std must be non-negative")
        self.m = m
        if phi is None:
            self.phi = torch.nn.Parameter(
                2 * torch.pi * torch.rand(1, dtype=torch.float32)
            )
        else:
            self.phi = (
                phi
                if isinstance(phi, torch.nn.Parameter)
                else torch.nn.Parameter(torch.tensor(phi, dtype=torch.float32))
            )
        self.phase_noise_std = phase_noise_std
        self.noise_aware_training = noise_aware_training
        if self.phase_noise_std > 0:
            self.register_buffer("phase_noise_phi", torch.normal(
                mean=0.0, std=torch.tensor(self.phase_noise_std, dtype=torch.float32, device=self.phi.device)
            ))
        else:
            self.register_buffer("phase_noise_phi", torch.tensor(0.0, dtype=torch.float32, device=self.phi.device))

    def __repr__(self):
        return "<PhaseShifter, port = {}, phi = {}>".format(self.m, self.phi)

    def extra_repr(self):
        return f"m={self.m}, phase_noise_std={self.phase_noise_std}, noise_aware_training={self.noise_aware_training}"

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.phi.data = 2 * torch.pi * torch.rand_like(self.phi)
            if self.phase_noise_std > 0:
                self.phase_noise_phi.copy_(
                    torch.normal(
                        mean=0.0,
                        std=torch.tensor(
                            self.phase_noise_std,
                            dtype=torch.float32,
                            device=self.phase_noise_phi.device,
                        ),
                    )
                )
            else:
                self.phase_noise_phi.zero_()

    def get_transfer_matrix(self) -> torch.Tensor:
        # add phase noise
        apply_noise = not self.training or self.noise_aware_training
        if apply_noise:
            phi = self.phi + self.phase_noise_phi.to(self.phi.device, self.phi.dtype)
        else:
            phi = self.phi

        return torch.exp(self._1j * phi.to(torch.complex64)).squeeze()


class Beamsplitter(OpticalComponent):
    """Class for a perfect 50:50 beamsplitter"""

    def __init__(self, m: int, n: int):
        """
        :param m: first waveguide index
        :param n: second waveguide index
        """
        super().__init__([m, n], dof=2)
        if not isinstance(m, int) or not isinstance(n, int) or m < 0 or n < 0:
            raise ValueError("m and n must be non-negative integers")
        self.m = m
        self.n = n

    def __repr__(self):
        return "<Beamsplitter, ports = {}>".format(self.ports)

    def get_transfer_matrix(self) -> torch.Tensor:
        scale = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=self._1j.device))
        return scale * torch.tensor([[1+0j, 0+1j], [0+1j, 1+0j]], dtype=torch.complex64, device=self._1j.device)


class MZI(OpticalComponent):
    """'Class for a programmable phase-shifting Mach-Zehnder interferometer"""

    def __init__(
        self,
        m: int,
        n: int,
        theta: Optional[float] = None,
        phi: Optional[float] = None,
        phase_noise_std: float = 0.0,
        noise_aware_training: bool = False,
    ):
        super().__init__([m, n], dof=2)
        if not isinstance(m, int) or not isinstance(n, int) or m < 0 or n < 0:
            raise ValueError("m and n must be non-negative integers")
        if phase_noise_std < 0:
            raise ValueError("phase_noise_std must be non-negative")
        self.m = m  # input waveguide A index (0-indexed)
        self.n = n  # input waveguide B index
        if theta is None:
            self.theta = torch.nn.Parameter(
                2 * torch.pi * torch.rand(1, dtype=torch.float32)
            )
        else:
            self.theta = (
                theta
                if isinstance(theta, torch.nn.Parameter)
                else torch.nn.Parameter(torch.tensor(theta, dtype=torch.float32))
            )

        if phi is None:
            self.phi = torch.nn.Parameter(
                2 * torch.pi * torch.rand(1, dtype=torch.float32)
            )
        else:
            self.phi = (
                phi
                if isinstance(phi, torch.nn.Parameter)
                else torch.nn.Parameter(torch.tensor(phi, dtype=torch.float32))
            )

        self.phase_noise_std = phase_noise_std
        self.noise_aware_training = noise_aware_training

        if self.phase_noise_std > 0:
            self.register_buffer(
                "phase_noise_theta",
                torch.normal(
                    mean=0.0,
                    std=torch.tensor(
                        self.phase_noise_std, dtype=torch.float32, device=self.theta.device
                    ),
                ),
            )
            self.register_buffer(
                "phase_noise_phi",
                torch.normal(
                    mean=0.0,
                    std=torch.tensor(
                        self.phase_noise_std, dtype=torch.float32, device=self.theta.device
                    ),
                ),
            )
        else:
            self.register_buffer(
                "phase_noise_theta",
                torch.tensor(0.0, dtype=torch.float32, device=self.theta.device),
            )
            self.register_buffer(
                "phase_noise_phi",
                torch.tensor(0.0, dtype=torch.float32, device=self.theta.device),
            )

    def __repr__(self):
        return "<MZI theta={}, phi={}>".format(self.theta, self.phi)

    def extra_repr(self):
        return f"m={self.m}, n={self.n}, phase_noise_std={self.phase_noise_std}, noise_aware_training={self.noise_aware_training}"

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.theta.data = 2 * torch.pi * torch.rand_like(self.theta)
            self.phi.data = 2 * torch.pi * torch.rand_like(self.phi)
            if self.phase_noise_std > 0:
                noise_std = torch.tensor(
                    self.phase_noise_std,
                    dtype=torch.float32,
                    device=self.phase_noise_theta.device,
                )
                self.phase_noise_theta.copy_(torch.normal(mean=0.0, std=noise_std))
                self.phase_noise_phi.copy_(torch.normal(mean=0.0, std=noise_std))
            else:
                self.phase_noise_theta.zero_()
                self.phase_noise_phi.zero_()

    def get_transfer_matrix(self) -> torch.Tensor:
        # Add phase noise
        apply_noise = not self.training or self.noise_aware_training
        if apply_noise:
            phi = self.phi + self.phase_noise_phi.to(self.phi.device, self.phi.dtype)
            theta = self.theta + self.phase_noise_theta.to(
                self.theta.device, self.theta.dtype
            )
        else:
            phi = self.phi
            theta = self.theta

        transfer_matrix = torch.zeros(2, 2, dtype=torch.complex64, device=self._1j.device)

        transfer_matrix[0, 0] = torch.exp(self._1j * phi) * (
            torch.exp(self._1j * theta) - 1
        )
        transfer_matrix[0, 1] = (
            self._1j * torch.exp(self._1j * phi) * (1 + torch.exp(self._1j * theta))
        )
        transfer_matrix[1, 0] = self._1j * (torch.exp(self._1j * theta) + 1)
        transfer_matrix[1, 1] = 1 - torch.exp(self._1j * theta)

        transfer_matrix = 0.5 * transfer_matrix

        return transfer_matrix


class WaveGuide(OpticalComponent):
    """'Class for waveguide without BS, PS or MZI in ONN."""

    def __init__(self, m: int):
        super().__init__([m], dof=1)
        if not isinstance(m, int) or m < 0:
            raise ValueError("m must be a non-negative integer")
        self.m = m  # input waveguide A index (0-indexed)

    def __repr__(self):
        return "<Waveguide ports = {}>".format(self.ports)

    def get_transfer_matrix(self) -> torch.Tensor:
        return torch.tensor(1 + 0j, dtype=torch.complex64, device=self._1j.device)


class MZM(OpticalComponent):
    """Class for a Mach-Zehnder Modulator"""

    def __init__(self, m: int, theta: Optional[float] = None, phase_noise_std: float = 0.0, noise_aware_training: bool = False):
        super().__init__([m], dof=1)
        if not isinstance(m, int) or m < 0:
            raise ValueError("m must be a non-negative integer")
        if phase_noise_std < 0:
            raise ValueError("phase_noise_std must be non-negative")
        self.m = m
        self.phase_noise_std = phase_noise_std
        self.noise_aware_training = noise_aware_training
        if theta is None:
            self.theta = torch.nn.Parameter(
                2 * torch.pi * torch.rand(1, dtype=torch.float32)
            )
        else:
            self.theta = (
                theta
                if isinstance(theta, torch.nn.Parameter)
                else torch.nn.Parameter(torch.tensor(theta, dtype=torch.float32))
            )

        if self.phase_noise_std > 0:
            self.register_buffer(
                "phase_noise_theta",
                torch.normal(
                    mean=0.0,
                    std=torch.tensor(
                        self.phase_noise_std, dtype=torch.float32, device=self.theta.device
                    ),
                ),
            )
        else:
            self.register_buffer(
                "phase_noise_theta",
                torch.tensor(0.0, dtype=torch.float32, device=self.theta.device),
            )

    def __repr__(self):
        return "<MZM, port = {}, theta = {}>".format(self.m, self.theta)

    def extra_repr(self):
        return f"m={self.m}, phase_noise_std={self.phase_noise_std}, noise_aware_training={self.noise_aware_training}"

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.theta.data = 2 * torch.pi * torch.rand_like(self.theta)
            if self.phase_noise_std > 0:
                self.phase_noise_theta.copy_(
                    torch.normal(
                        mean=0.0,
                        std=torch.tensor(
                            self.phase_noise_std,
                            dtype=torch.float32,
                            device=self.phase_noise_theta.device,
                        ),
                    )
                )
            else:
                self.phase_noise_theta.zero_()

    def get_transfer_matrix(self) -> torch.Tensor:
        apply_noise = not self.training or self.noise_aware_training
        if apply_noise:
            theta = self.theta + self.phase_noise_theta.to(
                self.theta.device, self.theta.dtype
            )
        else:
            theta = self.theta

        transfer_matrix_PS = torch.eye(2, dtype=torch.complex64, device=self._1j.device)
        transfer_matrix_PS[0][0] = torch.exp(self._1j * theta)

        _sqrt2_half = (torch.sqrt(torch.tensor(2.0, dtype=torch.float32)) / 2).item()
        transfer_matrix_MMI12 = torch.tensor(
            [[_sqrt2_half], [_sqrt2_half]],
            dtype=torch.complex64,
            device=self._1j.device,
        )
        transfer_matrix_MMI21 = torch.tensor(
            [_sqrt2_half, _sqrt2_half],
            dtype=torch.complex64,
            device=self._1j.device,
        )

        transfer_matrix = (
            transfer_matrix_MMI21 @ transfer_matrix_PS @ transfer_matrix_MMI12
        )

        return transfer_matrix.squeeze()


class Laser(OpticalComponent):
    """Class for a Laser"""

    def __init__(self, RIN: float = -155, linewidth: float = 1e5):
        super().__init__([], dof=1)
        if linewidth < 0:
            raise ValueError("linewidth must be non-negative")
        self.RIN = RIN
        self.linewidth = linewidth

    def __repr__(self):
        return "<Laser, RIN = {}, linewidth = {}>".format(self.RIN, self.linewidth)

    def RIN_noise(
        self, power, frequency: float = 1e9, generator: Optional[torch.Generator] = None
    ):
        if self.RIN <= -200.0 or self.RIN == -float("inf"):
            return torch.zeros_like(power.real)
        RIN_linear = 10 ** (self.RIN / 10)
        noise_variance = RIN_linear * power**2 * frequency
        noise_std = torch.sqrt(noise_variance)
        noise = torch.randn(power.shape, dtype=power.dtype, device=power.device, generator=generator)
        noise_power = noise * noise_std
        return noise_power

    def phase_noise(
        self, power, dt: float = 1e-12, generator: Optional[torch.Generator] = None
    ):
        if self.linewidth <= 0.0:
            return torch.zeros_like(power.real)
        phase_variance = 2 * torch.pi * self.linewidth * dt
        phase_std = torch.sqrt(
            torch.tensor(phase_variance, dtype=power.dtype, device=power.device)
        )
        noise = torch.randn(power.real.shape, dtype=power.real.dtype, device=power.device, generator=generator)
        phase_noise = noise * phase_std

        return phase_noise

    def simulate(self, E, generator: Optional[torch.Generator] = None):
        if not torch.is_tensor(E):
            raise TypeError("E must be a torch.Tensor")
        input_power_raw = torch.abs(E) ** 2
        noisy_power = input_power_raw + self.RIN_noise(input_power_raw, generator=generator)
        min_pwr = torch.min(noisy_power).item()
        if min_pwr < -1e-6:
            neg_mask = noisy_power < 0
            neg_ratio = neg_mask.float().mean().item()
            warnings.warn(
                f"Laser RIN noise produced negative optical power (min: {min_pwr:.3e}, "
                f"affected: {neg_ratio*100:.1f}%). Negative values are clamped to 0 to prevent NaN, "
                "which indicates the small-signal Gaussian RIN approximation has broken down "
                "at low optical power or excessive noise levels.",
                category=UserWarning,
                stacklevel=2,
            )
        input_power = torch.clamp(noisy_power, min=0.0)

        amplitude = torch.sqrt(input_power)
        if torch.is_complex(E):
            original_phase = torch.angle(E)
        else:
            original_phase = torch.where(
                E < 0,
                torch.tensor(torch.pi, device=E.device, dtype=E.dtype),
                torch.zeros_like(E),
            )
        phase_jitter = self.phase_noise(input_power, generator=generator)
        total_phase = original_phase + phase_jitter
        optical_field = amplitude * torch.exp(self._1j * total_phase)

        return optical_field


class Detector(OpticalComponent):
    """Class for a Detector"""

    def __init__(
        self,
        bandwidth: float = 1e9,
        dark_current: float = 1e-9,
        responsivity: float = 1.0e-3,
        impedance: float = 50.0,
    ):
        super().__init__([], dof=1)
        if bandwidth < 0:
            raise ValueError("bandwidth must be non-negative")
        if dark_current < 0:
            raise ValueError("dark_current must be non-negative")
        if responsivity < 0:
            raise ValueError("responsivity must be non-negative")
        if impedance <= 0:
            raise ValueError("impedance must be positive")
        self.bandwidth = bandwidth
        self.dark_current = dark_current  # 1e-9: 1 nA @ Si Room temperature
        self.responsivity = responsivity  # 1.0 A/W (Si @ 1550nm), 1e-3 A/mW
        self.impedance = impedance  # 50 ohm

    def __repr__(self):
        return "<Detector, bandwidth = {}, dark_current = {}, responsivity = {}, impedance = {}>".format(
            self.bandwidth, self.dark_current, self.responsivity, self.impedance
        )

    def total_noise(
        self, q, k_b, temperature, photocurrent, generator: Optional[torch.Generator] = None
    ):
        if q < 0 or k_b < 0 or temperature < 0:
            raise ValueError("q, k_b, and temperature must be non-negative")
        pos_current = torch.clamp(photocurrent, min=0.0)
        shot_noise_var = 2 * q * pos_current * self.bandwidth
        dark_noise_var = 2 * q * self.dark_current * self.bandwidth
        thermal_noise_var = 4 * k_b * temperature * self.bandwidth / self.impedance

        total_noise_var = shot_noise_var + (dark_noise_var + thermal_noise_var)
        total_noise_std = torch.sqrt(total_noise_var)

        noise = torch.randn(
            photocurrent.shape,
            dtype=photocurrent.dtype,
            device=photocurrent.device,
            generator=generator,
        )
        total_noise = noise * total_noise_std
        return total_noise

    def simulate(self, power, q, k_b, temperature, generator: Optional[torch.Generator] = None):
        if not torch.is_complex(power):
            min_power = torch.min(power).item()
            if min_power < -1e-6:
                raise ValueError(
                    f"Optical power must be non-negative, but got minimum value {min_power:.3e}. "
                    "Optical power cannot be negative in physical detection."
                )
            safe_power = torch.clamp(power, min=0.0)
        else:
            safe_power = torch.abs(power)

        photocurrent = self.responsivity * safe_power
        photocurrent = photocurrent + self.dark_current

        photocurrent = photocurrent + self.total_noise(
            q, k_b, temperature, photocurrent, generator=generator
        )

        return photocurrent


class TIA(OpticalComponent):
    """Class for a Transimpedance Amplifier"""

    def __init__(self, transimpedance: float = 1e2):
        super().__init__([], dof=1)
        if transimpedance <= 0:
            raise ValueError("transimpedance must be positive")
        self.transimpedance = transimpedance  # 100 ohm

    def __repr__(self):
        return "<TIA, transimpedance = {}>".format(self.transimpedance)

    def simulate(self, photocurrent):
        return self.transimpedance * photocurrent


class ADC(OpticalComponent):
    """Class for an Analog-to-Digital Converter"""

    def __init__(self, bits: int = 12, v_range: float = 2.0):
        super().__init__([], dof=1)
        if not isinstance(bits, int) or bits <= 0:
            raise ValueError("bits must be a positive integer")
        if v_range <= 0:
            raise ValueError("v_range must be positive")
        self.bits = bits
        self.v_range = v_range
        self.lsb = self.v_range / (2**self.bits)  # least significant bit

    def __repr__(self):
        return "<ADC, bits = {}>".format(self.bits)

    def simulate(self, voltage):
        # keep the voltage in the v_range
        clipped = torch.clamp(voltage, 0, self.v_range)

        # Quantize by rounding to the nearest LSB, then clamp to available codes.
        digital_output = torch.clamp(
            torch.round(clipped / self.lsb), 0, 2**self.bits - 1
        )

        quantized = digital_output * self.lsb

        return quantized