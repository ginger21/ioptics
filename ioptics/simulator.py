from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .components import Laser, Detector, TIA, ADC, MZI, MZM, PhaseShifter
from .layers import MeshCore, ArrayCore


# ---------------------------------------------------------------------------
# Enums / typed literals
# ---------------------------------------------------------------------------

class HardwareProfile(str, Enum):
    """Pre-defined simulator operating hardware specifications/profiles."""
    IDEAL = "ideal"
    LAB = "lab"
    HARDWARE_LIKE = "hardware_like"


class InputDomain(str, Enum):
    """Domain interpretation for the ``run`` input tensor."""
    FIELD = "field"
    POWER = "power"


class LossModel(str, Enum):
    """Strategy for computing insertion loss."""
    LAYER = "layer"
    COMPONENT = "component"


class ExecutionMode(str, Enum):
    """Operating execution mode for a simulation run."""
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    MC = "mc"


class OutputDomain(str, Enum):
    """Interpretation of the optical model's output tensor."""
    AUTO = "auto"
    FIELD = "field"
    POWER = "power"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulatorConfig:
    """Typed, immutable container for simulator constants and component params."""

    # Physical constants
    temperature: float = 300.0          # Kelvin
    electron_charge: float = 1.602e-19  # Coulombs
    boltzmann: float = 1.381e-23        # J/K

    # Optical loss parameters (layer-count model)
    layer_loss_db: float = 1.2          # dB per MZI mesh layer/column
    coupling_loss: float = 0.5          # dB

    # Loss model strategy
    loss_model: str = LossModel.LAYER.value  # "layer" | "component"

    # Component-level loss coefficients (dB per component instance)
    component_loss_db: float = 0.6      # dB per MZI instance
    mzm_loss: float = 0.4               # dB per MZM
    phase_shifter_loss: float = 0.15    # dB per PhaseShifter

    # Laser parameters
    laser_rin: float = -155.0           # dB/Hz
    laser_linewidth: float = 1e5        # Hz

    # Detector parameters
    detector_bandwidth: float = 1e9     # Hz
    detector_dark_current: float = 1e-9 # A
    detector_responsivity: float = 1.0e-3  # A/mW
    detector_impedance: float = 50.0    # Ohms

    # TIA parameters
    tia_transimpedance: float = 100.0   # Ohms

    # ADC parameters
    adc_bits: int = 12
    adc_v_range: float = 2.0            # Volts

    # Component noise toggles per profile
    laser_noise: bool = False
    detector_noise: bool = False

    # TIA non-ideality knobs (P2)
    tia_saturation_min: float = 0.0          # Volts; clamp floor (0 = no lower clipping)
    tia_saturation_max: float = float("inf")  # Volts; clamp ceiling (inf = no clipping)
    tia_gain_error: float = 1.0               # multiplicative gain (1 = ideal)
    tia_offset: float = 0.0                   # Volts; additive DC offset
    tia_noise_std: float = 0.0                # Volts; additive Gaussian noise std

    # ADC non-ideality knobs (P2)
    adc_gain_error: float = 1.0                     # multiplicative gain (1 = ideal)
    adc_offset: float = 0.0                         # Volts; additive DC offset
    adc_transition_noise_lsb: float = 0.0           # LSB units; DNL-like transition noise
    adc_inl_coefficient: float = 0.0                # cubic INL-like distortion coefficient

    # Domain interpretation
    input_domain: str = InputDomain.FIELD.value
    output_domain: str = OutputDomain.AUTO.value

    # Auto-gain & digital normalization options
    auto_gain: bool = False
    normalize: bool = False

    @staticmethod
    def from_profile(profile: HardwareProfile | str) -> SimulatorConfig:
        """Return a pre-configured ``SimulatorConfig`` for the given profile."""
        prof = HardwareProfile(profile)
        if prof == HardwareProfile.IDEAL:
            return SimulatorConfig(
                layer_loss_db=0.0,
                coupling_loss=0.0,
                laser_noise=False,
                detector_noise=False,
            )
        if prof == HardwareProfile.LAB:
            return SimulatorConfig(
                layer_loss_db=1.0,
                coupling_loss=0.3,
                laser_noise=True,
                detector_noise=True,
            )
        if prof == HardwareProfile.HARDWARE_LIKE:
            return SimulatorConfig(
                layer_loss_db=1.5,
                coupling_loss=0.8,
                laser_rin=-150.0,
                laser_linewidth=5e5,
                detector_bandwidth=5e8,
                detector_dark_current=5e-9,
                laser_noise=True,
                detector_noise=True,
            )
        raise ValueError(f"Unknown profile: {profile}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _count_optical_layers(optical_model: torch.nn.Module) -> int:
    """Count MeshCore / ArrayCore sub-modules in *optical_model*."""
    total = 0
    for module in optical_model.modules():
        if isinstance(module, (MeshCore, ArrayCore)):
            total += 1
    return total


def _count_optical_components(
    optical_model: torch.nn.Module,
) -> Dict[str, int]:
    """Count MZI, MZM, and PhaseShifter component instances in *optical_model*.

    Returns a dict with keys ``"mzi"``, ``"mzm"``, ``"phase_shifter"``.
    Only leaf-level components are counted (sub-modules of MeshCore/ArrayCore
    are not double-counted).
    """
    counts: Dict[str, int] = {"mzi": 0, "mzm": 0, "phase_shifter": 0}
    # Track which modules have already been accounted for by a parent Core
    core_modules: set = set()
    for module in optical_model.modules():
        if isinstance(module, (MeshCore, ArrayCore)):
            core_modules.add(id(module))

    for name, module in optical_model.named_modules():
        # Skip Core-level containers -- we count their leaf components instead
        if id(module) in core_modules:
            continue
        if isinstance(module, MZI):
            counts["mzi"] += 1
        elif isinstance(module, MZM):
            counts["mzm"] += 1
        elif isinstance(module, PhaseShifter):
            counts["phase_shifter"] += 1
    return counts


# ---------------------------------------------------------------------------
# Trace structure
# ---------------------------------------------------------------------------

@dataclass
class SimulationTrace:
    """Intermediate values captured during ``simulator.run``."""
    input_optical_field: torch.Tensor
    model_output_before_loss: torch.Tensor
    model_output_after_loss: torch.Tensor
    photocurrent: torch.Tensor
    voltage: torch.Tensor
    final_output: torch.Tensor


# ---------------------------------------------------------------------------
# TIA / ADC non-ideality helpers (P2)
# ---------------------------------------------------------------------------

def _apply_tia_non_idealities(
    voltage: torch.Tensor,
    *,
    gain_error: float,
    offset: float,
    noise_std: float,
    saturation_min: float,
    saturation_max: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Apply TIA non-idealities to a voltage tensor."""
    v = voltage * gain_error + offset
    if noise_std > 0.0:
        v = v + torch.randn(
            v.shape, dtype=v.dtype, device=v.device, generator=generator
        ) * noise_std
    if saturation_min != 0.0 or saturation_max != float("inf"):
        v = torch.clamp(v, min=saturation_min, max=saturation_max)
    return v


def _apply_adc_non_idealities(
    voltage: torch.Tensor,
    *,
    bits: int,
    v_range: float,
    gain_error: float,
    offset: float,
    transition_noise_lsb: float,
    inl_coefficient: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Apply ADC non-idealities and quantize a voltage tensor."""
    if torch.is_complex(voltage):
        voltage = voltage.real
    lsb = v_range / (2 ** bits)

    # Gain / offset on analog input
    v = voltage * gain_error + offset

    # INL (cubic distortion on normalized input)
    if inl_coefficient != 0.0:
        v_norm = v / v_range
        v = v + inl_coefficient * v_norm ** 3 * v_range

    # Transition noise (DNL-like) -- added before quantization
    if transition_noise_lsb > 0.0:
        v = v + torch.randn(
            v.shape, dtype=v.dtype, device=v.device, generator=generator
        ) * (transition_noise_lsb * lsb)

    # Single quantization step
    clipped = torch.clamp(v, 0.0, v_range)
    code = torch.clamp(torch.round(clipped / lsb), 0, 2 ** bits - 1)
    return torch.clamp(code * lsb, 0.0, v_range)


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------

class Simulator:
    """Optical-neural-network simulator with configurable profiles and tracing."""

    def __init__(
        self,
        config: Optional[SimulatorConfig] = None,
        *,
        profile: str = HardwareProfile.IDEAL.value,
        auto_gain: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        # Resolve profile
        self._profile = HardwareProfile(profile)

        if config is not None:
            self._config = config
        else:
            self._config = SimulatorConfig.from_profile(self._profile)

        overrides = {}
        if auto_gain is not None:
            overrides["auto_gain"] = auto_gain
        if normalize is not None:
            overrides["normalize"] = normalize
        if overrides:
            self._config = replace(self._config, **overrides)

        # Convenience aliases
        self.temperature: float = self._config.temperature
        self.electron_charge: float = self._config.electron_charge
        self.boltzmann: float = self._config.boltzmann
        self.layer_loss_db: float = self._config.layer_loss_db
        self.coupling_loss: float = self._config.coupling_loss
        self.loss_model: str = self._config.loss_model

    # ------------------------------------------------------------------
    # Loss computation helpers
    # ------------------------------------------------------------------

    def _compute_loss_db(self, optical_model: torch.nn.Module) -> float:
        """Compute total insertion + coupling loss in dB for *optical_model*."""
        if self._config.loss_model == LossModel.COMPONENT.value:
            return self._compute_component_loss(optical_model)
        # Default: layer-count model
        total_layers = _count_optical_layers(optical_model)
        return total_layers * self._config.layer_loss_db + self._config.coupling_loss

    def _compute_component_loss(self, optical_model: torch.nn.Module) -> float:
        """Compute loss from individual component counts plus coupling loss."""
        counts = _count_optical_components(optical_model)
        loss = (
            counts["mzi"] * self._config.component_loss_db
            + counts["mzm"] * self._config.mzm_loss
            + counts["phase_shifter"] * self._config.phase_shifter_loss
            + self._config.coupling_loss
        )
        return loss

    # ------------------------------------------------------------------
    # Primary simulation step
    # ------------------------------------------------------------------

    def run(
        self,
        input_feature: torch.Tensor,
        optical_model: torch.nn.Module,
        *,
        mode: str = ExecutionMode.STOCHASTIC.value,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        return_trace: bool = False,
        input_domain: str = InputDomain.FIELD.value,
        mc_samples: int = 100,
        output_domain: Optional[str] = None,
        auto_gain: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, SimulationTrace] | Dict[str, Any]:
        """Run a forward simulation.

        Parameters
        ----------
        input_feature :
            Input tensor interpreted according to *input_domain*.
        optical_model :
            A ``torch.nn.Module`` containing MeshCore / ArrayCore layers.
        mode :
            Execution mode. One of ``"deterministic"``, ``"stochastic"`` (default),
            or ``"mc"``. In ``"mc"`` mode the result is a Monte Carlo
            aggregate dictionary (same structure as :meth:`run_mc`) with
            *mc_samples* iterations.
        seed :
            Optional integer seed for reproducibility. Mutually exclusive with
            *generator*.
        generator :
            Optional ``torch.Generator``. Mutually exclusive with *seed*.
        return_trace :
            If ``True``, also return a ``SimulationTrace`` with intermediate
            pipeline values. Not supported in ``"mc"`` mode.
        input_domain :
            ``"field"`` (default) -- *input_feature* is optical field amplitude.
            ``"power"`` -- *input_feature* is optical power; converted to field.
        mc_samples :
            Number of Monte Carlo samples when ``mode="mc"``. Ignored for
            other modes.
        output_domain :
            Interpretation of the model's output. ``"auto"`` (default) infers domain
            from tensor properties or model attribute. ``"field"`` treats output as
            field amplitude (loss factor $10^{-dB/20}$, detector sees $|E|^2$).
            ``"power"`` treats output as power (loss factor $10^{-dB/10}$,
            detector sees power directly).

        Returns
        -------
        output : ``torch.Tensor``
        (output, trace) : ``Tuple[torch.Tensor, SimulationTrace]`` if *return_trace*.
        dict : Monte Carlo aggregate when ``mode="mc"``.
        """
        # Validate mode
        try:
            exec_mode = ExecutionMode(mode)
        except ValueError:
            valid = [m.value for m in ExecutionMode]
            raise ValueError(
                f"Invalid execution mode: {mode!r}. Valid modes are: {valid}"
            ) from None

        # Delegate for MC mode
        if exec_mode == ExecutionMode.MC:
            return self.run_mc(
                input_feature,
                optical_model,
                n_samples=mc_samples,
                seed=seed,
                input_domain=input_domain,
            )

        # For deterministic mode, effectively disable noise by using a noise-free config
        effective_config = self._config
        if exec_mode == ExecutionMode.DETERMINISTIC:
            effective_config = SimulatorConfig(
                layer_loss_db=0.0,
                coupling_loss=0.0,
                laser_rin=-155.0,
                laser_linewidth=0.0,
                laser_noise=False,
                detector_noise=False,
                loss_model=LossModel.LAYER.value,
                # Carry over remaining defaults
                component_loss_db=self._config.component_loss_db,
                mzm_loss=self._config.mzm_loss,
                phase_shifter_loss=self._config.phase_shifter_loss,
                # Explicitly disable TIA/ADC non-idealities
                tia_gain_error=1.0,
                tia_offset=0.0,
                tia_noise_std=0.0,
                tia_saturation_min=0.0,
                tia_saturation_max=float("inf"),
                adc_gain_error=1.0,
                adc_offset=0.0,
                adc_transition_noise_lsb=0.0,
                adc_inl_coefficient=0.0,
            )

        return self._run_internal(
            input_feature,
            optical_model,
            config_override=effective_config,
            seed=seed,
            generator=generator,
            return_trace=return_trace,
            input_domain=input_domain,
            output_domain=output_domain,
            auto_gain=auto_gain,
            normalize=normalize,
        )

    def _run_internal(
        self,
        input_feature: torch.Tensor,
        optical_model: torch.nn.Module,
        *,
        config_override: Optional[SimulatorConfig] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        return_trace: bool = False,
        input_domain: str = InputDomain.FIELD.value,
        output_domain: Optional[str] = None,
        auto_gain: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, SimulationTrace]:
        """Internal single-run simulation with an optional config override."""
        # Validate mutual exclusivity
        if seed is not None and generator is not None:
            raise ValueError("Provide either 'seed' or 'generator', not both.")

        cfg = config_override if config_override is not None else self._config

        # Resolve effective generator
        effective_gen: Optional[torch.Generator] = None
        if seed is not None:
            effective_gen = torch.Generator().manual_seed(seed)
        elif generator is not None:
            effective_gen = generator

        # --- Input domain handling ---
        domain = InputDomain(input_domain)
        if domain == InputDomain.FIELD:
            E_field = input_feature
        elif domain == InputDomain.POWER:
            # Power -> field: E = sqrt(P)  (phase assumed zero)
            pwr = input_feature.real if torch.is_complex(input_feature) else input_feature
            min_pwr = torch.min(pwr).item()
            if min_pwr < -1e-6:
                raise ValueError(
                    f"Input optical power must be non-negative, but got minimum value {min_pwr:.3e}."
                )
            pwr = torch.clamp(pwr, min=0.0)
            E_field = torch.sqrt(pwr.to(torch.float32))
        else:
            raise ValueError(f"Invalid input_domain: {input_domain!r}")

        # --- Laser ---
        laser = Laser(
            RIN=cfg.laser_rin if cfg.laser_noise else -float("inf"),
            linewidth=cfg.laser_linewidth if cfg.laser_noise else 0.0,
        )
        input_optical_field = laser.simulate(E_field, generator=effective_gen)

        # --- Optical model ---
        model_output_before_loss = optical_model(input_optical_field)

        # --- Resolve effective output domain ---
        domain_str = output_domain or cfg.output_domain
        eff_out_domain = OutputDomain(domain_str)

        if eff_out_domain == OutputDomain.AUTO:
            # a) Check model hint attribute
            model_hint = getattr(optical_model, "output_domain", None)
            if model_hint in (OutputDomain.FIELD.value, OutputDomain.POWER.value):
                eff_out_domain = OutputDomain(model_hint)
            # b) Complex tensors represent optical fields, never optical power
            elif torch.is_complex(model_output_before_loss):
                eff_out_domain = OutputDomain.FIELD
            # c) Check if last module in sequential/composite model is a detector/power conversion
            elif (
                isinstance(optical_model, torch.nn.Sequential)
                and len(optical_model) > 0
                and optical_model[-1].__class__.__name__ in ("Abs", "AbsSquared", "SquareActivation")
            ):
                eff_out_domain = OutputDomain.POWER
            # d) Infer from tensor properties
            elif torch.all(model_output_before_loss >= 0) and torch.all(torch.isfinite(model_output_before_loss)):
                eff_out_domain = OutputDomain.POWER
            else:
                eff_out_domain = OutputDomain.FIELD

        # --- Insertion / coupling loss ---
        if cfg.loss_model == LossModel.COMPONENT.value:
            loss_db = self._compute_component_loss_for_config(optical_model, cfg)
        else:
            total_layers = _count_optical_layers(optical_model)
            loss_db = total_layers * cfg.layer_loss_db + cfg.coupling_loss

        if eff_out_domain == OutputDomain.FIELD:
            # Field amplitude attenuation: 10^(-dB/20)
            loss_factor = 10 ** (-loss_db / 20)
            model_output_after_loss = model_output_before_loss * loss_factor
            detected_optical_power = torch.abs(model_output_after_loss) ** 2
        else:
            # Power attenuation: 10^(-dB/10)
            loss_factor = 10 ** (-loss_db / 10)
            model_output_after_loss = model_output_before_loss * loss_factor
            if torch.is_complex(model_output_after_loss):
                # Safeguard: if user forced POWER mode on complex output, convert to intensity
                detected_optical_power = torch.abs(model_output_after_loss) ** 2
            else:
                min_pwr = torch.min(model_output_after_loss).item()
                if min_pwr < -1e-6:
                    raise ValueError(
                        f"Model output optical power must be non-negative, but got minimum value {min_pwr:.3e}. "
                        "Check model architecture or output domain configuration."
                    )
                detected_optical_power = torch.clamp(model_output_after_loss, min=0.0)

        # --- Detector ---
        detector = Detector(
            bandwidth=cfg.detector_bandwidth,
            dark_current=cfg.detector_dark_current if cfg.detector_noise else 0.0,
            responsivity=cfg.detector_responsivity,
            impedance=cfg.detector_impedance,
        )
        photocurrent = detector.simulate(
            detected_optical_power,
            q=cfg.electron_charge if cfg.detector_noise else 0.0,
            k_b=cfg.boltzmann if cfg.detector_noise else 0.0,
            temperature=cfg.temperature if cfg.detector_noise else 0.0,
            generator=effective_gen,
        )

        # --- TIA ---
        eff_auto_gain = auto_gain if auto_gain is not None else cfg.auto_gain
        if eff_auto_gain:
            # Auto-gain: dynamically scale transimpedance so maximum signal reaches ~85% of ADC range
            max_current = torch.max(photocurrent).item()
            if max_current > 1e-9:
                target_voltage = 0.85 * cfg.adc_v_range
                eff_transimpedance = min(target_voltage / max_current, 1e7)
            else:
                eff_transimpedance = cfg.tia_transimpedance
        else:
            eff_transimpedance = cfg.tia_transimpedance

        tia = TIA(transimpedance=eff_transimpedance)
        voltage = tia.simulate(photocurrent)

        # Apply TIA non-idealities (P2)
        voltage = _apply_tia_non_idealities(
            voltage,
            gain_error=cfg.tia_gain_error,
            offset=cfg.tia_offset,
            noise_std=cfg.tia_noise_std,
            saturation_min=cfg.tia_saturation_min,
            saturation_max=cfg.tia_saturation_max,
            generator=effective_gen,
        )

        # --- ADC (non-idealities + quantization) ---
        output = _apply_adc_non_idealities(
            voltage,
            bits=cfg.adc_bits,
            v_range=cfg.adc_v_range,
            gain_error=cfg.adc_gain_error,
            offset=cfg.adc_offset,
            transition_noise_lsb=cfg.adc_transition_noise_lsb,
            inl_coefficient=cfg.adc_inl_coefficient,
            generator=effective_gen,
        )

        # --- Digital Normalization ---
        eff_normalize = normalize if normalize is not None else cfg.normalize
        if eff_normalize:
            if eff_auto_gain:
                norm_scale = 0.85 * cfg.adc_v_range
            else:
                norm_scale = cfg.detector_responsivity * eff_transimpedance
                if norm_scale <= 0.0:
                    norm_scale = cfg.adc_v_range
            output = torch.clamp(output / norm_scale, 0.0, 1.0)

        if return_trace:
            trace = SimulationTrace(
                input_optical_field=input_optical_field,
                model_output_before_loss=model_output_before_loss,
                model_output_after_loss=model_output_after_loss,
                photocurrent=photocurrent,
                voltage=voltage,
                final_output=output,
            )
            return output, trace  # type: ignore[return-value]

        return output

    def _compute_component_loss_for_config(
        self, optical_model: torch.nn.Module, cfg: SimulatorConfig
    ) -> float:
        """Compute component-level loss using *cfg* coefficients."""
        counts = _count_optical_components(optical_model)
        return (
            counts["mzi"] * cfg.component_loss_db
            + counts["mzm"] * cfg.mzm_loss
            + counts["phase_shifter"] * cfg.phase_shifter_loss
            + cfg.coupling_loss
        )

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def run_mc(
        self,
        input_feature: torch.Tensor,
        optical_model: torch.nn.Module,
        *,
        n_samples: int = 100,
        seed: Optional[int] = None,
        input_domain: str = InputDomain.FIELD.value,
    ) -> Dict[str, Any]:
        """Run *n_samples* Monte Carlo iterations and return aggregate statistics.

        Parameters
        ----------
        input_feature :
            Same as :meth:`run`.
        optical_model :
            Same as :meth:`run`.
        n_samples :
            Number of Monte Carlo iterations.
        seed :
            Optional seed for reproducibility.  Each sample advances the
            generator state so results are deterministic for a given seed.
        input_domain :
            Same as :meth:`run`.

        Returns
        -------
        dict with keys: ``mean``, ``std``, ``p05``, ``p95``, ``samples``.
        """
        # Use a local generator for reproducibility
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        samples: List[torch.Tensor] = []
        for _ in range(n_samples):
            # Fork a child generator so each run sees independent noise
            child = torch.Generator()
            child.manual_seed(int(torch.randint(0, 2**31, (1,), generator=rng).item()))
            sample = self.run(
                input_feature,
                optical_model,
                generator=child,
                input_domain=input_domain,
            )
            samples.append(sample)

        stacked = torch.stack(samples)
        flat = stacked.view(stacked.shape[0], -1)

        return {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "p05": torch.quantile(flat.float(), 0.05, dim=0),
            "p95": torch.quantile(flat.float(), 0.95, dim=0),
            "samples": stacked,
        }

    # ------------------------------------------------------------------
    # Sweep API
    # ------------------------------------------------------------------

    def run_sweep(
        self,
        input_feature: torch.Tensor,
        optical_model: torch.nn.Module,
        *,
        sweep_params: Mapping[str, Sequence[Any]],
        mode: str = ExecutionMode.STOCHASTIC.value,
        mc_samples: int = 100,
        seed: Optional[int] = None,
        input_domain: str = InputDomain.FIELD.value,
    ) -> List[Dict[str, Any]]:
        """Run a grid sweep over *sweep_params* and return structured results.

        Parameters
        ----------
        input_feature :
            Input tensor passed to every simulation call.
        optical_model :
            Optical model passed to every simulation call.
        sweep_params :
            Mapping from parameter name to a sequence of values. The Cartesian
            product of all value sequences is enumerated. Supported parameter
            names include ``"mode"``, ``"mc_samples"``, ``"seed"``,
            ``"input_domain"``, as well as any ``SimulatorConfig`` field
            (e.g. ``"layer_loss_db"``, ``"laser_rin"``).
        mode :
            Default simulation mode for each sweep point.  Can be overridden
            per point via ``sweep_params["mode"]``.
        mc_samples :
            Default Monte Carlo sample count for ``mode="mc"``.  Can be
            overridden via ``sweep_params["mc_samples"]``.
        seed :
            If provided, each combination receives a deterministic sub-seed
            derived from *seed* + combination index, ensuring reproducibility.
        input_domain :
            Default input domain.  Can be overridden per point.

        Returns
        -------
        List of dicts, one per combination.  Each dict contains:
            - ``"params"``: dict of the parameter values for this combination
            - ``"output"``: the simulation output (tensor or MC aggregate dict)
        """
        if not sweep_params:
            raise ValueError("sweep_params must contain at least one parameter to sweep")

        # Build Cartesian product
        keys = list(sweep_params.keys())
        values = [list(sweep_params[k]) for k in keys]
        combinations = list(itertools.product(*values))

        # Determine config-field keys vs run-time keys
        config_fields = {
            f.name for f in SimulatorConfig.__dataclass_fields__.values()
        }
        run_keys = {"mode", "mc_samples", "seed", "input_domain"}

        results: List[Dict[str, Any]] = []
        for idx, combo in enumerate(combinations):
            combo_dict = dict(zip(keys, combo))

            # Derive per-combination seed for determinism
            combo_seed: Optional[int] = None
            if seed is not None:
                combo_seed = seed + idx

            # Separate config overrides from run-time arguments
            config_overrides: Dict[str, Any] = {}
            run_kwargs: Dict[str, Any] = {}
            for k, v in combo_dict.items():
                if k in config_fields:
                    config_overrides[k] = v
                elif k in run_keys:
                    run_kwargs[k] = v
                else:
                    raise ValueError(f"Unknown sweep parameter: {k!r}")

            # Build a simulator instance with config overrides if needed
            if config_overrides:
                base = self._config
                merged_fields = {
                    f.name: getattr(base, f.name) for f in SimulatorConfig.__dataclass_fields__.values()
                }
                merged_fields.update(config_overrides)
                sweep_sim = Simulator(config=SimulatorConfig(**merged_fields))
            else:
                sweep_sim = self

            # Resolve effective run arguments
            eff_mode = run_kwargs.get("mode", mode)
            eff_mc = run_kwargs.get("mc_samples", mc_samples)
            eff_domain = run_kwargs.get("input_domain", input_domain)

            output = sweep_sim.run(
                input_feature,
                optical_model,
                mode=eff_mode,
                seed=combo_seed,
                mc_samples=eff_mc,
                input_domain=eff_domain,
            )

            results.append({
                "params": combo_dict,
                "output": output,
            })

        return results
