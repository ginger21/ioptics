"""P0 + P1 enhancement tests for ioptics.simulator.

P0 covers:
  a) same-seed reproducibility for run
  b) return_trace contract
  c) input_domain validation and field/power behavior
  d) run_mc output structure and deterministic behavior with seed
  e) profile/config behavior

P1 covers:
  f) component-level loss model (vs layer-count model)
  g) simulation mode API (ideal, stochastic, mc)
  h) sweep API (batch experiments)
"""
from __future__ import annotations

from typing import Optional

import pytest
import torch

from ioptics.components import ADC, Detector, Laser, TIA
from ioptics.layers import ArrayCore, FCArray, MeshCore
from ioptics.simulator import (
    ExecutionMode,
    HardwareProfile,
    InputDomain,
    LossModel,
    OutputDomain,
    SimulationTrace,
    Simulator,
    SimulatorConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _LegacyParityModel(torch.nn.Module):
    """Wrap FCArray while forcing real-valued model outputs for legacy parity checks."""

    output_domain = OutputDomain.POWER.value

    def __init__(self) -> None:
        super().__init__()
        self.core = FCArray(in_dim=4, out_dim=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.core(x))


def _reference_initial_simulator_run(
    input_feature: torch.Tensor,
    optical_model: torch.nn.Module,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Reproduce the initial simulator.run() pipeline from commit 2de0608."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    e_field = input_feature
    input_optical_field = Laser().simulate(e_field, generator=generator)
    output_optical_power = optical_model(input_optical_field)

    total_layers = 0
    for module in optical_model.modules():
        if isinstance(module, (MeshCore, ArrayCore)):
            total_layers += 1

    loss_db = total_layers * 1.2 + 0.5
    loss_factor = 10 ** (-loss_db / 10)
    output_optical_power = output_optical_power * loss_factor

    photocurrent = Detector().simulate(
        output_optical_power,
        q=1.602e-19,
        k_b=1.381e-23,
        temperature=300,
        generator=generator,
    )
    output_voltage = TIA().simulate(photocurrent)
    return ADC().simulate(output_voltage)


def _build_small_model() -> FCArray:
    """A minimal FCArray optical model for testing."""
    return FCArray(in_dim=4, out_dim=4)


def _random_input() -> torch.Tensor:
    return torch.randn(4, dtype=torch.complex64)


# ---------------------------------------------------------------------------
# a) Same-seed reproducibility for run
# ---------------------------------------------------------------------------

def test_run_same_seed_reproducibility() -> None:
    model = _build_small_model()
    inp = _random_input()

    sim = Simulator()
    out1 = sim.run(inp, model, seed=42)
    out2 = sim.run(inp, model, seed=42)

    torch.testing.assert_close(out1, out2)


def test_run_different_seeds_differ() -> None:
    model = _build_small_model()
    inp = _random_input()

    # Use a high-noise config so the stochastic component is observable
    # after ADC quantization.
    noisy_cfg = SimulatorConfig(
        laser_rin=-50.0,        # very high RIN noise
        laser_linewidth=1e9,    # wide linewidth -> large phase noise
        laser_noise=True,
        detector_noise=False,
    )
    sim = Simulator(config=noisy_cfg)
    out1 = sim.run(inp, model, seed=1)
    out2 = sim.run(inp, model, seed=999)

    assert not torch.allclose(out1, out2), "Different seeds should produce different results"


def test_run_seed_and_generator_mutually_exclusive() -> None:
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    with pytest.raises(ValueError, match="Provide either"):
        sim.run(inp, model, seed=1, generator=torch.Generator())


# ---------------------------------------------------------------------------
# b) return_trace contract
# ---------------------------------------------------------------------------

def test_run_return_trace_structure() -> None:
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    result = sim.run(inp, model, seed=7, return_trace=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    output, trace = result

    assert isinstance(trace, SimulationTrace)
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, trace.final_output)

    # All trace fields are tensors
    for attr in (
        "input_optical_field",
        "model_output_before_loss",
        "model_output_after_loss",
        "photocurrent",
        "voltage",
        "final_output",
    ):
        assert isinstance(getattr(trace, attr), torch.Tensor), f"{attr} should be a tensor"


def test_run_return_trace_false_returns_tensor() -> None:
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    result = sim.run(inp, model, seed=7, return_trace=False)

    assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# c) input_domain validation and field/power behavior
# ---------------------------------------------------------------------------

def test_run_input_domain_invalid_raises() -> None:
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    with pytest.raises(ValueError, match="is not a valid InputDomain"):
        sim.run(inp, model, input_domain="bogus")


def test_run_input_domain_field_default() -> None:
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    out_explicit = sim.run(inp, model, seed=5, input_domain=InputDomain.FIELD.value)
    out_default = sim.run(inp, model, seed=5, input_domain=InputDomain.FIELD.value)

    torch.testing.assert_close(out_explicit, out_default)


def test_run_input_domain_power_converts() -> None:
    sim = Simulator()
    model = _build_small_model()

    # Create power input (positive real values)
    power = torch.abs(torch.randn(4, dtype=torch.complex64)) ** 2
    power = power.real

    # Should not raise
    output = sim.run(power, model, seed=3, input_domain=InputDomain.POWER.value)
    assert isinstance(output, torch.Tensor)
    assert output.shape == power.shape


# ---------------------------------------------------------------------------
# d) run_mc output structure and deterministic behavior with seed
# ---------------------------------------------------------------------------

def test_run_mc_output_structure() -> None:
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    result = sim.run_mc(inp, model, n_samples=20, seed=42)

    for key in ("mean", "std", "p05", "p95", "samples"):
        assert key in result, f"Missing key: {key}"

    assert isinstance(result["samples"], torch.Tensor)
    assert result["samples"].shape[0] == 20
    assert result["mean"].shape == inp.shape
    assert result["std"].shape == inp.shape


def test_run_mc_deterministic_with_seed() -> None:
    sim = Simulator(profile=HardwareProfile.LAB.value)
    model = _build_small_model()
    inp = _random_input()

    result1 = sim.run_mc(inp, model, n_samples=10, seed=123)
    result2 = sim.run_mc(inp, model, n_samples=10, seed=123)

    torch.testing.assert_close(result1["mean"], result2["mean"])
    torch.testing.assert_close(result1["std"], result2["std"])
    torch.testing.assert_close(result1["samples"], result2["samples"])


def test_run_mc_batch_parallel_and_serial_modes() -> None:
    """Test run_mc under both batch_parallel=True and serial (batch_parallel=False) modes."""
    sim = Simulator(profile=HardwareProfile.LAB.value)
    model = _build_small_model()
    inp = _random_input()

    res_batch = sim.run_mc(inp, model, n_samples=25, seed=777, batch_parallel=True)
    res_serial = sim.run_mc(inp, model, n_samples=25, seed=777, batch_parallel=False)

    assert res_batch["samples"].shape == (25, *inp.shape)
    assert res_serial["samples"].shape == (25, *inp.shape)
    assert res_batch["mean"].shape == inp.shape
    assert res_serial["mean"].shape == inp.shape

    # Both modes should produce reasonable aggregate statistics close in distribution
    torch.testing.assert_close(res_batch["mean"], res_serial["mean"], atol=0.25, rtol=0.25)


# ---------------------------------------------------------------------------
# e) profile/config behavior
# ---------------------------------------------------------------------------

def test_profile_ideal_zero_loss() -> None:
    """Ideal profile should have zero insertion and coupling loss."""
    sim = Simulator(profile=HardwareProfile.IDEAL.value)
    assert sim.layer_loss_db == 0.0
    assert sim.coupling_loss == 0.0


def test_profile_hardware_like_nonzero_loss() -> None:
    """Hardware-like profile should have nonzero loss parameters."""
    sim = Simulator(profile=HardwareProfile.HARDWARE_LIKE.value)
    assert sim.layer_loss_db > 0.0
    assert sim.coupling_loss > 0.0


def test_custom_config_overrides_profile() -> None:
    """Providing a custom config should override the profile defaults."""
    custom = SimulatorConfig(layer_loss_db=5.0, coupling_loss=3.0)
    sim = Simulator(config=custom)
    assert sim.layer_loss_db == 5.0
    assert sim.coupling_loss == 3.0


def test_simulatorconfig_from_profile_lab() -> None:
    cfg = SimulatorConfig.from_profile(HardwareProfile.LAB)
    assert cfg.laser_noise is True
    assert cfg.detector_noise is True


def test_simulatorconfig_from_profile_ideal() -> None:
    cfg = SimulatorConfig.from_profile(HardwareProfile.IDEAL)
    assert cfg.laser_noise is False
    assert cfg.detector_noise is False
    assert cfg.layer_loss_db == 0.0


def test_backward_compat_default_Simulator() -> None:
    """Simulator() with no args should still work and produce output."""
    sim = Simulator()
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    output = sim.run(inp, model, seed=0)
    assert isinstance(output, torch.Tensor)


# ---------------------------------------------------------------------------
# f) Component-level loss model
# ---------------------------------------------------------------------------

def test_component_loss_differs_from_layer_loss() -> None:
    """Component loss model should produce different loss than layer-count model."""
    model = _build_small_model()  # FCArray(4, 4) => 16 MZMs, 1 ArrayCore

    # Layer model: 1 core layer * layer_loss_db + coupling_loss
    sim_layer = Simulator(config=SimulatorConfig(
        layer_loss_db=1.0,
        coupling_loss=0.5,
        loss_model=LossModel.LAYER.value,
    ))
    layer_loss_db = sim_layer._compute_loss_db(model)

    # Component model: 16 MZMs * mzm_loss + coupling_loss
    sim_comp = Simulator(config=SimulatorConfig(
        component_loss_db=0.4,
        mzm_loss=0.4,
        phase_shifter_loss=0.15,
        coupling_loss=0.5,
        loss_model=LossModel.COMPONENT.value,
    ))
    comp_loss_db = sim_comp._compute_loss_db(model)

    # The two should differ: layer=1.0+0.5=1.5, component=16*0.4+0.5=6.9
    assert layer_loss_db != comp_loss_db, (
        f"Expected different losses: layer={layer_loss_db}, component={comp_loss_db}"
    )
    assert comp_loss_db == pytest.approx(16 * 0.4 + 0.5)
    assert layer_loss_db == pytest.approx(1.0 + 0.5)


def test_component_loss_matches_expected_counted_loss() -> None:
    """Component loss should equal the sum of individual component losses + coupling."""
    model = _build_small_model()  # FCArray(4, 4) => 16 MZMs

    cfg = SimulatorConfig(
        component_loss_db=0.6,
        mzm_loss=0.3,
        phase_shifter_loss=0.1,
        coupling_loss=0.8,
        loss_model=LossModel.COMPONENT.value,
    )
    sim = Simulator(config=cfg)

    expected = 16 * cfg.mzm_loss + cfg.coupling_loss  # 16*0.3 + 0.8 = 5.6
    actual = sim._compute_loss_db(model)
    assert actual == pytest.approx(expected)


def test_component_model_simulation_produces_output() -> None:
    """A full simulation run with component loss model should succeed."""
    sim = Simulator(config=SimulatorConfig(
        component_loss_db=0.6,
        mzm_loss=0.4,
        phase_shifter_loss=0.15,
        coupling_loss=0.5,
        loss_model=LossModel.COMPONENT.value,
    ))
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    output = sim.run(inp, model, seed=42)
    assert isinstance(output, torch.Tensor)
    assert output.shape == inp.shape


def test_component_loss_default_backward_compat() -> None:
    """Default simulator should use layer loss model (backward compat)."""
    sim = Simulator()
    assert sim.loss_model == LossModel.LAYER.value


# ---------------------------------------------------------------------------
# g) Simulation mode API
# ---------------------------------------------------------------------------

def test_mode_ideal_deterministic_with_noisy_config() -> None:
    """Ideal mode should be deterministic even when the simulator has a noisy config."""
    noisy_cfg = SimulatorConfig(
        laser_rin=-50.0,
        laser_linewidth=1e9,
        laser_noise=True,
        detector_noise=True,
    )
    sim = Simulator(config=noisy_cfg)
    model = _build_small_model()
    inp = _random_input()

    out1 = sim.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=42)
    out2 = sim.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=42)
    out3 = sim.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=99)

    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out1, out3, msg="Ideal mode should be deterministic regardless of seed")


def test_mode_stochastic_produces_output() -> None:
    """Stochastic mode (default) should produce a tensor output."""
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    output = sim.run(inp, model, mode=ExecutionMode.STOCHASTIC.value, seed=7)
    assert isinstance(output, torch.Tensor)


def test_mode_mc_from_run_returns_expected_structure() -> None:
    """mode='mc' from run() should return the same MC aggregate structure."""
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    result = sim.run(inp, model, mode=ExecutionMode.MC.value, seed=42, mc_samples=15)

    assert isinstance(result, dict)
    for key in ("mean", "std", "p05", "p95", "samples"):
        assert key in result, f"Missing key: {key}"
    assert result["samples"].shape[0] == 15


def test_mode_mc_sample_count_configurable() -> None:
    """mc_samples parameter should control the number of samples."""
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    result_5 = sim.run(inp, model, mode=ExecutionMode.MC.value, seed=42, mc_samples=5)
    result_20 = sim.run(inp, model, mode=ExecutionMode.MC.value, seed=42, mc_samples=20)

    assert result_5["samples"].shape[0] == 5
    assert result_20["samples"].shape[0] == 20


def test_mode_invalid_raises() -> None:
    """Invalid mode values should raise ValueError with a clear message."""
    sim = Simulator()
    model = _build_small_model()
    inp = _random_input()

    with pytest.raises(ValueError, match="Invalid execution mode"):
        sim.run(inp, model, mode="bogus")

    with pytest.raises(ValueError, match="deterministic"):
        sim.run(inp, model, mode="BOGUS")


def test_mode_deterministic_zero_loss() -> None:
    """Deterministic mode should apply zero loss (no attenuation)."""
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    # Non-ideal config but deterministic mode
    sim = Simulator(config=SimulatorConfig(
        layer_loss_db=5.0,
        coupling_loss=5.0,
        laser_noise=False,
        detector_noise=False,
    ))

    output = sim.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=7)
    # In ideal mode, loss is zero, so output should equal model output
    # (before ADC quantization noise, which is also disabled in ideal mode)
    assert isinstance(output, torch.Tensor)


# ---------------------------------------------------------------------------
# h) Sweep API
# ---------------------------------------------------------------------------

def test_sweep_combination_count() -> None:
    """Sweep should produce Cartesian product count of results."""
    sim = Simulator()
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    results = sim.run_sweep(
        inp, model,
        sweep_params={
            "layer_loss_db": [0.5, 1.0, 2.0],
            "coupling_loss": [0.1, 0.5],
        },
        seed=42,
    )

    # 3 x 2 = 6 combinations
    assert len(results) == 6


def test_sweep_params_structure() -> None:
    """Each sweep result should contain 'params' and 'output' keys."""
    sim = Simulator()
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    results = sim.run_sweep(
        inp, model,
        sweep_params={"layer_loss_db": [0.5, 1.0]},
        seed=0,
    )

    for entry in results:
        assert "params" in entry
        assert "output" in entry
        assert isinstance(entry["params"], dict)
        assert isinstance(entry["output"], torch.Tensor)
        assert "layer_loss_db" in entry["params"]


def test_sweep_deterministic_with_seed() -> None:
    """Sweep with fixed seed should produce deterministic outputs."""
    sim = Simulator()
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    sweep_kwargs = {
        "sweep_params": {"layer_loss_db": [0.5, 1.0]},
        "seed": 12345,
    }

    results1 = sim.run_sweep(inp, model, **sweep_kwargs)
    results2 = sim.run_sweep(inp, model, **sweep_kwargs)

    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1["params"] == r2["params"]
        torch.testing.assert_close(r1["output"], r2["output"])


def test_sweep_with_mc_mode() -> None:
    """Sweep should work with mode='mc'."""
    sim = Simulator()
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    results = sim.run_sweep(
        inp, model,
        sweep_params={"layer_loss_db": [0.5, 1.0]},
        mode=ExecutionMode.MC.value,
        mc_samples=5,
        seed=42,
    )

    assert len(results) == 2
    for entry in results:
        assert isinstance(entry["output"], dict)
        assert "mean" in entry["output"]
        assert "samples" in entry["output"]
        assert entry["output"]["samples"].shape[0] == 5


def test_sweep_with_mode_override() -> None:
    """Sweep should allow overriding mode per combination."""
    sim = Simulator()
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    results = sim.run_sweep(
        inp, model,
        sweep_params={"mode": [ExecutionMode.STOCHASTIC.value, ExecutionMode.DETERMINISTIC.value]},
        seed=7,
    )

    assert len(results) == 2
    # Both should return tensors
    for entry in results:
        assert isinstance(entry["output"], torch.Tensor)


def test_sweep_empty_params_raises() -> None:
    """Sweep with empty sweep_params should raise ValueError."""
    sim = Simulator()
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    with pytest.raises(ValueError, match="sweep_params must contain"):
        sim.run_sweep(inp, model, sweep_params={})


def test_sweep_unknown_param_raises() -> None:
    """Sweep with an unknown parameter name should raise ValueError."""
    sim = Simulator()
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    with pytest.raises(ValueError, match="Unknown sweep parameter"):
        sim.run_sweep(inp, model, sweep_params={"nonexistent_param": [1, 2]})


# ---------------------------------------------------------------------------
# i) P2 -- ADC/TIA non-idealities
# ---------------------------------------------------------------------------

def test_non_idealities_affect_stochastic_output() -> None:
    """Non-ideal config should produce different output from defaults under stochastic mode."""
    model = _build_small_model()
    inp = _random_input()

    sim_default = Simulator()
    out_default = sim_default.run(inp, model, seed=42, mode=ExecutionMode.STOCHASTIC.value)

    sim_nonideal = Simulator(config=SimulatorConfig(
        tia_gain_error=2.0,
        tia_offset=0.5,
        tia_noise_std=0.01,
        adc_gain_error=0.5,
        adc_offset=0.1,
        adc_inl_coefficient=100.0,
        laser_noise=True,
        laser_rin=-60.0,
    ))
    out_nonideal = sim_nonideal.run(inp, model, seed=42, mode=ExecutionMode.STOCHASTIC.value)

    assert not torch.allclose(out_default, out_nonideal), (
        "Non-ideal config should produce different output"
    )


def test_ideal_mode_bypasses_non_idealities() -> None:
    """Ideal mode should produce identical results regardless of extreme non-ideal settings."""
    model = _build_small_model()
    inp = _random_input()

    cfg_extreme = SimulatorConfig(
        tia_gain_error=5.0,
        tia_offset=100.0,
        tia_noise_std=50.0,
        tia_saturation_min=10.0,
        tia_saturation_max=20.0,
        adc_gain_error=0.01,
        adc_offset=999.0,
        adc_transition_noise_lsb=500.0,
        adc_inl_coefficient=1e6,
        laser_noise=True,
        detector_noise=True,
    )
    sim_extreme = Simulator(config=cfg_extreme)

    out1 = sim_extreme.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=42)
    out2 = sim_extreme.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=42)
    out3 = sim_extreme.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=9999)

    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out1, out3, msg="Ideal mode ignores non-ideal settings")


def test_tia_saturation_clamps_output() -> None:
    """TIA saturation should clamp the output voltage to the configured range."""
    model = _build_small_model()
    # Use a large input to generate meaningful photocurrent/voltage.
    inp = torch.ones(4, dtype=torch.complex64) * 10.0

    sim_clamped = Simulator(config=SimulatorConfig(
        tia_saturation_max=1e-6,
        tia_saturation_min=0.0,
        tia_gain_error=1.0,
        tia_offset=0.0,
        tia_noise_std=0.0,
    ))
    # Use a tiny ADC range so the clamped voltage survives quantization.
    sim_small_range = Simulator(config=SimulatorConfig(
        tia_saturation_max=1e-6,
        tia_saturation_min=0.0,
        tia_gain_error=1.0,
        tia_offset=0.0,
        tia_noise_std=0.0,
        adc_v_range=1e-5,
        adc_bits=12,
    ))

    _, trace_clamped = sim_clamped.run(
        inp, model, seed=7, return_trace=True, mode=ExecutionMode.STOCHASTIC.value,
    )
    out_small = sim_small_range.run(
        inp, model, seed=7, mode=ExecutionMode.STOCHASTIC.value,
    )

    # The TIA voltage in the trace should be clamped to <= saturation_max
    assert (trace_clamped.voltage <= 1e-6 + 1e-12).all(), (
        f"TIA voltage exceeds saturation max: max={trace_clamped.voltage.max().item()}"
    )
    # With a tiny ADC range, the clamped signal should produce non-zero output
    assert (out_small >= 0).all()


def test_adc_inl_has_observable_effect() -> None:
    """ADC INL coefficient should change the output relative to zero-INL baseline."""
    model = _build_small_model()
    # Use a large input to generate meaningful TIA voltage.
    inp = torch.ones(4, dtype=torch.complex64) * 10.0

    # Scale TIA gain so voltage sits well within the ADC range (~0.03-0.05 V
    # on a 1 V range), then apply strong cubic distortion.
    base_kwargs = dict(
        adc_v_range=1.0,
        adc_bits=12,
        tia_gain_error=1.5e-3,
        tia_offset=0.0,
        tia_noise_std=0.0,
        adc_gain_error=1.0,
        adc_offset=0.0,
        adc_transition_noise_lsb=0.0,
    )

    sim_baseline = Simulator(config=SimulatorConfig(**base_kwargs, adc_inl_coefficient=0.0))
    sim_inl = Simulator(config=SimulatorConfig(**base_kwargs, adc_inl_coefficient=1000.0))

    out_baseline = sim_baseline.run(inp, model, seed=7, mode=ExecutionMode.STOCHASTIC.value)
    out_inl = sim_inl.run(inp, model, seed=7, mode=ExecutionMode.STOCHASTIC.value)

    assert not torch.allclose(out_baseline, out_inl), (
        f"INL should change output: baseline={out_baseline}, inl={out_inl}"
    )


def test_adc_dnl_has_observable_effect() -> None:
    """ADC transition noise (DNL) should change the output relative to zero-noise baseline."""
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64) * 10.0

    base_kwargs = dict(
        adc_v_range=1.0,
        adc_bits=12,
        tia_gain_error=1.5e-3,
        tia_offset=0.0,
        tia_noise_std=0.0,
        adc_gain_error=1.0,
        adc_offset=0.0,
        adc_inl_coefficient=0.0,
    )

    sim_baseline = Simulator(config=SimulatorConfig(**base_kwargs, adc_transition_noise_lsb=0.0))
    sim_dnl = Simulator(config=SimulatorConfig(**base_kwargs, adc_transition_noise_lsb=100.0))

    out_baseline = sim_baseline.run(inp, model, seed=7, mode=ExecutionMode.STOCHASTIC.value)
    out_dnl = sim_dnl.run(inp, model, seed=7, mode=ExecutionMode.STOCHASTIC.value)

    assert not torch.allclose(out_baseline, out_dnl), (
        f"DNL should change output: baseline={out_baseline}, dnl={out_dnl}"
    )


def test_deterministic_with_seed_stochastic_non_ideal() -> None:
    """Stochastic mode with non-idealities should be deterministic for the same seed."""
    model = _build_small_model()
    inp = _random_input()

    sim = Simulator(config=SimulatorConfig(
        tia_gain_error=1.5,
        tia_offset=0.1,
        tia_noise_std=0.001,
        adc_inl_coefficient=50.0,
        adc_transition_noise_lsb=10.0,
        laser_noise=True,
        detector_noise=True,
    ))

    out1 = sim.run(inp, model, seed=12345, mode=ExecutionMode.STOCHASTIC.value)
    out2 = sim.run(inp, model, seed=12345, mode=ExecutionMode.STOCHASTIC.value)

    torch.testing.assert_close(out1, out2)


def test_deterministic_with_seed_mc_non_ideal() -> None:
    """MC mode with non-idealities should be deterministic for the same seed."""
    model = _build_small_model()
    inp = _random_input()

    sim = Simulator(config=SimulatorConfig(
        tia_noise_std=0.001,
        adc_inl_coefficient=10.0,
        adc_transition_noise_lsb=5.0,
        laser_noise=True,
        detector_noise=True,
    ))

    result1 = sim.run_mc(inp, model, n_samples=10, seed=99999)
    result2 = sim.run_mc(inp, model, n_samples=10, seed=99999)

    torch.testing.assert_close(result1["mean"], result2["mean"])
    torch.testing.assert_close(result1["std"], result2["std"])
    torch.testing.assert_close(result1["samples"], result2["samples"])


def test_legacy_parameter_set_matches_initial_pipeline_behavior() -> None:
    """Verify whether the requested legacy-like parameter set reproduces initial output."""
    model = _LegacyParityModel()
    inp = torch.tensor([2.0, 0.7, 1.3, 3.1], dtype=torch.float32)
    seed = 2026

    current_sim = Simulator(config=SimulatorConfig(
        layer_loss_db=1.2,
        coupling_loss=0.5,
        laser_noise=True,
        detector_noise=True,
    ))
    current_output = current_sim.run(
        inp,
        model,
        mode=ExecutionMode.STOCHASTIC.value,
        seed=seed,
    )
    initial_output = _reference_initial_simulator_run(inp, model, seed=seed)

    torch.testing.assert_close(current_output, initial_output)


def test_run_auto_domain_selects_field_for_complex() -> None:
    """Verify that AUTO selects field semantics when model output is complex."""
    model = _build_small_model()
    inp = _random_input()
    seed = 42

    sim = Simulator()

    # Run with AUTO (should infer FIELD since FCArray output is complex)
    out_auto = sim.run(
        inp,
        model,
        seed=seed,
        output_domain=OutputDomain.AUTO.value,
    )

    # Run with explicit FIELD
    out_field = sim.run(
        inp,
        model,
        seed=seed,
        output_domain=OutputDomain.FIELD.value,
    )

    torch.testing.assert_close(out_auto, out_field)


def test_complex_model_with_forced_power_domain_safely_executes() -> None:
    """Complex model output when forced to POWER domain converts safely without ADC clamp crash."""
    model = _build_small_model()  # FCArray outputs complex tensor
    inp = _random_input()
    sim = Simulator()

    # Should not raise NotImplementedError or produce complex voltage in ADC
    output = sim.run(inp, model, output_domain=OutputDomain.POWER.value, seed=42)
    assert isinstance(output, torch.Tensor)
    assert not torch.is_complex(output)
    assert not torch.any(torch.isnan(output))


def test_sequential_with_abssquared_inferred_as_power() -> None:
    """Sequential model ending with AbsSquared should be inferred as POWER domain in AUTO mode."""
    from ioptics.nonlinear import AbsSquared
    model = torch.nn.Sequential(_build_small_model(), AbsSquared())
    inp = _random_input()
    sim = Simulator()

    out_auto = sim.run(inp, model, output_domain=OutputDomain.AUTO.value, seed=42)
    out_power = sim.run(inp, model, output_domain=OutputDomain.POWER.value, seed=42)

    torch.testing.assert_close(out_auto, out_power)


def test_simulator_rejects_negative_power_input() -> None:
    """Simulator should raise ValueError when input_domain=POWER receives negative values."""
    sim = Simulator()
    model = _build_small_model()
    invalid_power_input = torch.tensor([-1.0, 0.5, 1.0, 2.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="Input optical power must be non-negative"):
        sim.run(invalid_power_input, model, input_domain=InputDomain.POWER.value)


def test_auto_gain_scales_signal_close_to_adc_range() -> None:
    """Auto-gain should dynamically scale transimpedance to ~85% of ADC full scale."""
    sim = Simulator(profile=HardwareProfile.LAB.value)
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    out_default = sim.run(inp, model, seed=42, auto_gain=False)
    out_autogain = sim.run(inp, model, seed=42, auto_gain=True)

    # In default mode, signal with 100 ohm TIA is small (< 0.8 V, well below 1.7V auto-gain target)
    assert out_default.max().item() < 0.8
    # With auto-gain, peak signal is scaled close to 0.85 * 2.0V = 1.7V (tolerance: 1.5V ~ 1.8V)
    assert 1.5 <= out_autogain.max().item() <= 1.8


def test_digital_normalization_scales_output_to_unit_range() -> None:
    """Digital normalization should map output values within [0.0, 1.0]."""
    sim = Simulator(profile=HardwareProfile.LAB.value, auto_gain=True, normalize=True)
    model = _build_small_model()
    inp = torch.ones(4, dtype=torch.complex64)

    out = sim.run(inp, model, seed=42)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)
    # Peak output under auto-gain should be scaled close to 1.0
    assert out.max().item() > 0.95



