import math
import warnings
from typing import List, Optional

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook


_GLOBAL_PARAM_EPOCH: int = 0


def get_global_param_epoch() -> int:
    return _GLOBAL_PARAM_EPOCH


def bump_global_param_epoch(*_args, **_kwargs) -> None:
    global _GLOBAL_PARAM_EPOCH
    _GLOBAL_PARAM_EPOCH += 1


register_optimizer_step_post_hook(bump_global_param_epoch)


class OpticalParameter(torch.nn.Parameter):
    """Parameter subclass that tracks per-parameter and global mutation epochs."""

    def _touch(self) -> None:
        global _GLOBAL_PARAM_EPOCH
        _GLOBAL_PARAM_EPOCH += 1
        self._param_epoch = _GLOBAL_PARAM_EPOCH

    @property
    def data(self):
        self._touch()
        return super().data

    @data.setter
    def data(self, val):
        self._touch()
        torch.nn.Parameter.data.__set__(self, val)

    def copy_(self, *args, **kwargs):
        self._touch()
        return super().copy_(*args, **kwargs)

    def add_(self, *args, **kwargs):
        self._touch()
        return super().add_(*args, **kwargs)

    def sub_(self, *args, **kwargs):
        self._touch()
        return super().sub_(*args, **kwargs)

    def mul_(self, *args, **kwargs):
        self._touch()
        return super().mul_(*args, **kwargs)

    def div_(self, *args, **kwargs):
        self._touch()
        return super().div_(*args, **kwargs)

    def zero_(self):
        self._touch()
        return super().zero_()

    def fill_(self, *args, **kwargs):
        self._touch()
        return super().fill_(*args, **kwargs)

    def __iadd__(self, other):
        self._touch()
        return super().__iadd__(other)

    def __isub__(self, other):
        self._touch()
        return super().__isub__(other)

    def __imul__(self, other):
        self._touch()
        return super().__imul__(other)

    def __itruediv__(self, other):
        self._touch()
        return super().__itruediv__(other)

    def __setitem__(self, idx, val):
        self._touch()
        return super().__setitem__(idx, val)

    def view(self, *shape):
        self._touch()
        return super().view(*shape)

    def detach(self):
        self._touch()
        return super().detach()



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
            self.phi = OpticalParameter(
                2 * torch.pi * torch.rand(1, dtype=torch.float32)
            )
        else:
            self.phi = (
                phi
                if isinstance(phi, OpticalParameter)
                else OpticalParameter(phi.detach().clone() if isinstance(phi, torch.Tensor) else torch.tensor(phi, dtype=torch.float32))
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
            self.phi.copy_(2 * torch.pi * torch.rand_like(self.phi))
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
        apply_noise = not self.training or self.noise_aware_training
        phi = self.phi + self.phase_noise_phi if apply_noise else self.phi
        return torch.exp(1j * phi).squeeze()


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
        scale = 2.0 ** -0.5
        self.register_buffer(
            "_transfer_matrix",
            torch.tensor([[scale + 0j, 1j * scale], [1j * scale, scale + 0j]], dtype=torch.complex64),
            persistent=False,
        )

    def __repr__(self):
        return "<Beamsplitter, ports = {}>".format(self.ports)

    def get_transfer_matrix(self) -> torch.Tensor:
        return self._transfer_matrix


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
            self.theta = OpticalParameter(
                2 * torch.pi * torch.rand(1, dtype=torch.float32)
            )
        else:
            self.theta = (
                theta
                if isinstance(theta, OpticalParameter)
                else OpticalParameter(theta.detach().clone() if isinstance(theta, torch.Tensor) else torch.tensor(theta, dtype=torch.float32))
            )

        if phi is None:
            self.phi = OpticalParameter(
                2 * torch.pi * torch.rand(1, dtype=torch.float32)
            )
        else:
            self.phi = (
                phi
                if isinstance(phi, OpticalParameter)
                else OpticalParameter(phi.detach().clone() if isinstance(phi, torch.Tensor) else torch.tensor(phi, dtype=torch.float32))
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
            self.theta.copy_(2 * torch.pi * torch.rand_like(self.theta))
            self.phi.copy_(2 * torch.pi * torch.rand_like(self.phi))
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
        apply_noise = not self.training or self.noise_aware_training
        if apply_noise:
            phi = self.phi + self.phase_noise_phi
            theta = self.theta + self.phase_noise_theta
        else:
            phi = self.phi
            theta = self.theta

        exp_jphi = torch.exp(1j * phi)
        exp_jtheta = torch.exp(1j * theta)

        t00 = 0.5 * exp_jphi * (exp_jtheta - 1.0)
        t01 = 0.5j * exp_jphi * (1.0 + exp_jtheta)
        t10 = 0.5j * (exp_jtheta + 1.0)
        t11 = 0.5 * (1.0 - exp_jtheta)

        row0 = torch.stack([t00.squeeze(), t01.squeeze()])
        row1 = torch.stack([t10.squeeze(), t11.squeeze()])
        return torch.stack([row0, row1])


class WaveGuide(OpticalComponent):
    """'Class for waveguide without BS, PS or MZI in ONN."""

    def __init__(self, m: int):
        super().__init__([m], dof=1)
        if not isinstance(m, int) or m < 0:
            raise ValueError("m must be a non-negative integer")
        self.m = m  # input waveguide A index (0-indexed)
        self.register_buffer(
            "_transfer_matrix",
            torch.tensor(1 + 0j, dtype=torch.complex64),
            persistent=False,
        )

    def __repr__(self):
        return "<Waveguide ports = {}>".format(self.ports)

    def get_transfer_matrix(self) -> torch.Tensor:
        return self._transfer_matrix


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
            self.theta = OpticalParameter(
                2 * torch.pi * torch.rand(1, dtype=torch.float32)
            )
        else:
            self.theta = (
                theta
                if isinstance(theta, OpticalParameter)
                else OpticalParameter(theta.detach().clone() if isinstance(theta, torch.Tensor) else torch.tensor(theta, dtype=torch.float32))
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
            self.theta.copy_(2 * torch.pi * torch.rand_like(self.theta))
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
        theta = self.theta + self.phase_noise_theta if apply_noise else self.theta
        return (0.5 * (torch.exp(1j * theta) + 1.0)).squeeze()


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
        phase_std = math.sqrt(2 * math.pi * self.linewidth * dt)
        noise = torch.randn(power.real.shape, dtype=power.real.dtype, device=power.device, generator=generator)
        return noise * phase_std

    def simulate(self, E, generator: Optional[torch.Generator] = None):
        if not torch.is_tensor(E):
            raise TypeError("E must be a torch.Tensor")
        input_power_raw = torch.abs(E) ** 2
        noisy_power = input_power_raw + self.RIN_noise(input_power_raw, generator=generator)
        if not torch.compiler.is_compiling():
            min_pwr = torch.amin(noisy_power).item()
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
                torch.full_like(E, torch.pi),
                torch.zeros_like(E),
            )
        phase_jitter = self.phase_noise(input_power, generator=generator)
        total_phase = original_phase + phase_jitter
        optical_field = amplitude * torch.exp(1j * total_phase)

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
            if not torch.compiler.is_compiling():
                min_power = torch.amin(power).item()
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

    def __init__(
        self,
        transimpedance: float = 1e2,
        *,
        gain_error: float = 1.0,
        offset: float = 0.0,
        noise_std: float = 0.0,
        saturation_min: float = 0.0,
        saturation_max: float = float("inf"),
        auto_gain: bool = False,
        adc_v_range: float = 2.0,
    ):
        super().__init__([], dof=1)
        if transimpedance <= 0:
            raise ValueError("transimpedance must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if saturation_min > saturation_max:
            raise ValueError("saturation_min must be <= saturation_max")
        if adc_v_range <= 0:
            raise ValueError("adc_v_range must be positive")
        self.transimpedance = transimpedance  # 100 ohm
        self.gain_error = gain_error
        self.offset = offset
        self.noise_std = noise_std
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max
        self.auto_gain = auto_gain
        self.adc_v_range = adc_v_range

    def __repr__(self):
        return "<TIA, transimpedance = {}>".format(self.transimpedance)

    def simulate(
        self, photocurrent: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if self.auto_gain:
            max_current = torch.amax(photocurrent)
            target_voltage = 0.85 * self.adc_v_range
            dynamic_gain = torch.clamp(
                target_voltage / torch.clamp(max_current, min=1e-9), max=1e7
            )
            eff_transimpedance = torch.where(
                max_current > 1e-9,
                dynamic_gain,
                torch.tensor(
                    self.transimpedance,
                    dtype=photocurrent.dtype,
                    device=photocurrent.device,
                ),
            )
            voltage = photocurrent * eff_transimpedance
        else:
            voltage = self.transimpedance * photocurrent

        v = voltage * self.gain_error + self.offset
        if self.noise_std > 0.0:
            v = v + torch.randn(
                v.shape, dtype=v.dtype, device=v.device, generator=generator
            ) * self.noise_std
        if self.saturation_min != 0.0 or self.saturation_max != float("inf"):
            v = torch.clamp(v, min=self.saturation_min, max=self.saturation_max)
        return v


class ADC(OpticalComponent):
    """Class for an Analog-to-Digital Converter"""

    def __init__(
        self,
        bits: int = 12,
        v_range: float = 2.0,
        *,
        gain_error: float = 1.0,
        offset: float = 0.0,
        transition_noise_lsb: float = 0.0,
        inl_coefficient: float = 0.0,
    ):
        super().__init__([], dof=1)
        if not isinstance(bits, int) or bits <= 0:
            raise ValueError("bits must be a positive integer")
        if v_range <= 0:
            raise ValueError("v_range must be positive")
        if transition_noise_lsb < 0:
            raise ValueError("transition_noise_lsb must be non-negative")
        self.bits = bits
        self.v_range = v_range
        self.lsb = self.v_range / (2**self.bits)  # least significant bit
        self.gain_error = gain_error
        self.offset = offset
        self.transition_noise_lsb = transition_noise_lsb
        self.inl_coefficient = inl_coefficient

    def __repr__(self):
        return "<ADC, bits = {}>".format(self.bits)

    def simulate(
        self, voltage: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if torch.is_complex(voltage):
            voltage = voltage.real

        v = voltage * self.gain_error + self.offset

        if self.inl_coefficient != 0.0:
            v_norm = v / self.v_range
            v = v + self.inl_coefficient * (v_norm**3) * self.v_range

        if self.transition_noise_lsb > 0.0:
            v = v + torch.randn(
                v.shape, dtype=v.dtype, device=v.device, generator=generator
            ) * (self.transition_noise_lsb * self.lsb)

        clipped = torch.clamp(v, 0.0, self.v_range)
        digital_output = torch.clamp(
            torch.round(clipped / self.lsb), 0, 2**self.bits - 1
        )
        return torch.clamp(digital_output * self.lsb, 0.0, self.v_range)