import torch
import math
from typing import Optional


class Abs(torch.nn.Module):
    """Maps z -> |z|, corresponding to amplitude measurement by a photodetector."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class AbsSquared(torch.nn.Module):
    """Maps z -> |z|^2, corresponding to power measurement by a photodetector."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)**2


class SquareActivation(torch.nn.Module):
    """Maps z -> z^2, corresponding to square activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(x)


class SPMActivation(torch.nn.Module):
    """
    Lossless Self-Phase Modulation (SPM) activation function.
    Formula: f(z) = z * exp(-i * gain * |z|^2)
    """

    def __init__(self, gain: float):
        super().__init__()
        if not math.isfinite(gain):
            raise ValueError(f"gain must be a finite real number, got {gain}")
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = self.gain if (self.gain.device == x.device and self.gain.dtype == x.real.dtype) else self.gain.to(device=x.device, dtype=x.real.dtype)
        return x * torch.exp(-1j * gain * torch.abs(x)**2)


class ElectroOpticActivation(torch.nn.Module):
    """
    Electro-optic activation function with intensity modulation.
    Formula: f(z) = i * sqrt(1 - alpha) * exp(-i/2 * (g*|z|^2 + phi_b)) * cos(1/2 * (g*|z|^2 + phi_b)) * z
    """

    def __init__(
        self,
        alpha: float = 0.1,
        responsivity: float = 0.8,
        area: float = 1.0,
        V_pi: float = 10.0,
        V_bias: float = 10.0,
        R: float = 1e3,
        impedance: float = 120 * math.pi,
        g: Optional[float] = None,
        phi_b: Optional[float] = None,
    ):
        super().__init__()
        if not (0 <= alpha < 1):
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")

        if (g is None) != (phi_b is None):
            raise ValueError("g and phi_b must be provided together or both omitted")

        if g is not None and phi_b is not None:
            g_val = g
            phi_b_val = phi_b
        else:
            if V_pi / impedance <= 0:
                raise ValueError(f"V_pi/impedance must be positive, got {V_pi/impedance}")
            g_val = math.pi * alpha * R * responsivity * area * 1e-12 / 2 / V_pi / impedance
            phi_b_val = math.pi * V_bias / V_pi

        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("g", torch.tensor(g_val, dtype=torch.float32))
        self.register_buffer("phi_b", torch.tensor(phi_b_val, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.real.dtype
        device = x.device
        need_cast = self.alpha.device != device or self.alpha.dtype != dtype
        alpha = self.alpha.to(device=device, dtype=dtype) if need_cast else self.alpha
        g = self.g.to(device=device, dtype=dtype) if need_cast else self.g
        phi_b = self.phi_b.to(device=device, dtype=dtype) if need_cast else self.phi_b

        abs_sq = torch.abs(x)**2
        arg = 0.5 * (g * abs_sq + phi_b)

        return 1j * torch.sqrt(1 - alpha) * torch.exp(-1j * arg) * torch.cos(arg) * x


class SigmoidLikeActivation(torch.nn.Module):
    """
    Digital backend sigmoid-like output response with additive Gaussian noise.

    Power-domain behavior:
    1) Compute input power P = |x|^2
    2) Apply sigmoid to power
    3) Add Gaussian noise in power domain
    4) Reconstruct field by amplitude scaling to preserve input phase
    """

    def __init__(
        self,
        midpoint: float = 0.5,
        steepness: float = 10.0,
        noise_std: float = 0.1,
        noise_mean: float = 0.0,
    ):
        super().__init__()
        if not math.isfinite(steepness):
            raise ValueError(f"steepness must be a finite real number, got {steepness}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        self.register_buffer("midpoint", torch.tensor(midpoint, dtype=torch.float32))
        self.register_buffer("steepness", torch.tensor(steepness, dtype=torch.float32))
        self.register_buffer("noise_std", torch.tensor(noise_std, dtype=torch.float32))
        self.register_buffer("noise_mean", torch.tensor(noise_mean, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.abs(x)**2

        dtype = p.dtype
        device = p.device
        need_cast = self.midpoint.device != device or self.midpoint.dtype != dtype

        midpoint = self.midpoint.to(device=device, dtype=dtype) if need_cast else self.midpoint
        steepness = self.steepness.to(device=device, dtype=dtype) if need_cast else self.steepness

        p_act = torch.sigmoid(steepness * (p - midpoint))

        noise_std = self.noise_std.to(device=device, dtype=dtype) if need_cast else self.noise_std
        noise_mean = self.noise_mean.to(device=device, dtype=dtype) if need_cast else self.noise_mean

        if self.noise_std > 0:
            p_act = p_act + torch.randn_like(p_act) * noise_std + noise_mean

        p_noisy = torch.clamp(p_act, min=0.0)
        scale = torch.sqrt(p_noisy / (p + 1e-12))

        return x * scale


class TanhLikeActivation(torch.nn.Module):
    """
    Saturating tanh-like power activation for optical fields.

    Power-domain behavior:
    1) Compute input power P = |x|^2
    2) Apply P_out = tanh(gain * P)
    3) Reconstruct field by amplitude scaling to preserve input phase
    """

    def __init__(self, gain: float = float(math.atanh(0.8))):
        super().__init__()
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError(f"gain must be a finite positive real number, got {gain}")
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.abs(x) ** 2
        dtype = p.dtype
        device = p.device

        gain = self.gain if (self.gain.device == device and self.gain.dtype == dtype) else self.gain.to(device=device, dtype=dtype)
        p_out = torch.tanh(gain * p)

        scale = torch.sqrt(p_out / (p + 1e-12))
        return x * scale


