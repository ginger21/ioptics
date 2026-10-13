import pytest
import torch

from ioptics.components import (
    ADC,
    Beamsplitter,
    Detector,
    Laser,
    MZI,
    MZM,
    PhaseShifter,
    TIA,
    WaveGuide,
)


def test_phase_shifter_transfer_matrix_matches_phi() -> None:
    phi = 0.3
    component = PhaseShifter(m=0, phi=phi)

    matrix = component.get_transfer_matrix()

    expected = torch.exp(1j * torch.tensor(phi, dtype=torch.complex64))
    torch.testing.assert_close(matrix, expected, atol=1e-6, rtol=0.0)


def test_beamsplitter_is_unitary() -> None:
    component = Beamsplitter(m=0, n=1)

    matrix = component.get_transfer_matrix()
    identity = matrix.conj().T @ matrix

    torch.testing.assert_close(
        identity,
        torch.eye(2, dtype=torch.complex64),
        atol=1e-6,
        rtol=0.0,
    )


def test_mzi_matches_expected_formula_zero_phases() -> None:
    component = MZI(m=0, n=1, theta=0.0, phi=0.0)

    matrix = component.get_transfer_matrix()

    expected = 0.5 * torch.tensor(
        [[0 + 0j, 0 + 2j], [0 + 2j, 0 + 0j]], dtype=torch.complex64
    )
    torch.testing.assert_close(matrix, expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "theta,phi",
    [
        (0.0, 0.0),
        (0.3, 0.7),
        (torch.pi / 4, torch.pi / 3),
        (torch.pi, 2 * torch.pi),
        (-0.5, 1.2),
    ],
)
def test_mzi_is_unitary_arbitrary_phases(theta: float, phi: float) -> None:
    """MZI transfer matrix must be unitary: U^H @ U == I."""
    component = MZI(m=0, n=1, theta=theta, phi=phi)
    u = component.get_transfer_matrix()
    identity = u.conj().T @ u

    torch.testing.assert_close(
        identity,
        torch.eye(2, dtype=torch.complex64),
        atol=1e-6,
        rtol=0.0,
    )


def test_mzi_energy_conservation() -> None:
    """Lossless MZI must conserve optical power for arbitrary complex inputs."""
    torch.manual_seed(42)
    component = MZI(m=0, n=1, theta=0.42, phi=1.23)
    u = component.get_transfer_matrix()

    input_fields = torch.randn(5, 2, dtype=torch.complex64)
    output_fields = input_fields @ u.T

    input_power = (input_fields.abs() ** 2).sum(dim=-1)
    output_power = (output_fields.abs() ** 2).sum(dim=-1)

    torch.testing.assert_close(input_power, output_power, atol=1e-6, rtol=1e-6)


def test_waveguide_identity() -> None:
    component = WaveGuide(m=0)

    matrix = component.get_transfer_matrix()

    torch.testing.assert_close(
        matrix,
        torch.tensor(1 + 0j, dtype=torch.complex64),
        atol=1e-6,
        rtol=0.0,
    )


def test_mzm_half_scale_at_zero_phase() -> None:
    component = MZM(m=0, theta=0.0)

    matrix = component.get_transfer_matrix()

    expected = torch.tensor(1 + 0j, dtype=torch.complex64)
    torch.testing.assert_close(matrix, expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("theta", [0.0, 0.3, torch.pi / 2, torch.pi, -1.1])
def test_mzm_closed_form_matches_mmi_ps_mmi_chain(theta: float) -> None:
    """MZM closed-form transfer coefficient must match MMI21 @ PS @ MMI12 exactly."""
    component = MZM(m=0, theta=theta)
    actual = component.get_transfer_matrix()

    ps = torch.eye(2, dtype=torch.complex64)
    ps[0, 0] = torch.exp(1j * torch.tensor(theta, dtype=torch.complex64))
    s = 1.0 / (2.0 ** 0.5)
    mmi12 = torch.tensor([[s], [s]], dtype=torch.complex64)
    mmi21 = torch.tensor([s, s], dtype=torch.complex64)
    expected = (mmi21 @ ps @ mmi12).squeeze()

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_laser_simulate_zero_input_stays_zero() -> None:
    torch.manual_seed(0)
    component = Laser(RIN=-155, linewidth=1e5)
    input_field = torch.zeros(4, dtype=torch.float32)

    output = component.simulate(input_field)

    torch.testing.assert_close(output, torch.zeros_like(output))


def test_detector_zero_noise_matches_responsivity() -> None:
    component = Detector(bandwidth=1e9, dark_current=0.0, responsivity=2.0)
    power = torch.tensor([1.0, 2.0], dtype=torch.float32)

    current = component.simulate(power, q=0.0, k_b=0.0, temperature=0.0)

    torch.testing.assert_close(current, torch.tensor([2.0, 4.0]))


def test_noise_generator_is_reproducible() -> None:
    gen1 = torch.Generator().manual_seed(1234)
    gen2 = torch.Generator().manual_seed(1234)

    laser = Laser(RIN=-155, linewidth=1e5)
    input_field = torch.ones(4, dtype=torch.complex64)

    out1 = laser.simulate(input_field, generator=gen1)
    out2 = laser.simulate(input_field, generator=gen2)

    torch.testing.assert_close(out1, out2)

    detector = Detector(bandwidth=1e9, dark_current=0.0, responsivity=2.0)
    power = torch.tensor([1.0, 2.0], dtype=torch.float32)

    current1 = detector.simulate(power, q=1.0, k_b=1.0, temperature=1.0, generator=gen1)
    current2 = detector.simulate(power, q=1.0, k_b=1.0, temperature=1.0, generator=gen2)

    torch.testing.assert_close(current1, current2)


def test_tia_scales_photocurrent() -> None:
    component = TIA(transimpedance=10.0)
    photocurrent = torch.tensor([0.1, 0.2])

    voltage = component.simulate(photocurrent)

    torch.testing.assert_close(voltage, torch.tensor([1.0, 2.0]))


def test_adc_quantizes_and_clips() -> None:
    component = ADC(bits=2, v_range=1.0)
    voltage = torch.tensor([-0.1, 0.1, 0.6, 1.4])

    quantized = component.simulate(voltage)

    expected = torch.tensor([0.0, 0.0, 0.5, 0.75])
    torch.testing.assert_close(quantized, expected)


def test_detector_impedance() -> None:
    component = Detector(impedance=75.0)
    assert component.impedance == 75.0


def test_laser_preserves_original_field_phase() -> None:
    """Laser should preserve input complex field phase rather than discarding it."""
    # Linewidth 0 means zero phase jitter
    laser = Laser(RIN=-155, linewidth=0.0)
    # Phase = pi / 4
    input_field = torch.tensor([1.0 + 1.0j], dtype=torch.complex64)
    output = laser.simulate(input_field)

    assert not torch.any(torch.isnan(output))
    # Original phase should be preserved
    torch.testing.assert_close(torch.angle(output), torch.angle(input_field), atol=1e-5, rtol=1e-5)


def test_laser_rin_negative_power_clamped_no_nan() -> None:
    """Large RIN noise on low input power should clamp cleanly and never produce NaN."""
    # High RIN to trigger negative values in input_power + RIN_noise
    laser = Laser(RIN=-50, linewidth=1e5)
    input_field = torch.tensor([1e-6 + 1e-6j], dtype=torch.complex64)
    # Run multiple times to exercise random noise distribution
    for _ in range(50):
        output = laser.simulate(input_field)
        assert not torch.any(torch.isnan(output)), "Laser output produced NaN!"
        assert not torch.any(torch.isinf(output)), "Laser output produced Inf!"


def test_detector_shot_noise_is_channel_independent() -> None:
    """Shot noise variance must be proportional to each channel's photocurrent, not global mean."""
    detector = Detector(dark_current=0.0, bandwidth=1e9)
    # Channel 0 has zero power (dark), Channel 1 has high power
    power = torch.tensor([0.0, 100.0], dtype=torch.float32)

    # Sample multiple times to calculate empirical noise variance for each channel
    samples = []
    for _ in range(200):
        # Temperature = 0 to isolate shot noise from thermal noise
        current = detector.simulate(power, q=1.602e-19, k_b=0.0, temperature=0.0)
        samples.append(current)

    samples_t = torch.stack(samples)  # (200, 2)
    var = samples_t.var(dim=0)

    # Channel 0 (dark) shot noise variance should be 0 since dark_current is 0
    assert var[0].item() == 0.0, f"Dark channel had non-zero shot noise: {var[0].item()}"
    # Channel 1 (illuminated) shot noise variance must be positive
    assert var[1].item() > 0.0, f"Illuminated channel had zero variance: {var[1].item()}"


def test_detector_rejects_negative_optical_power() -> None:
    """Detector should raise ValueError on significantly negative optical power."""
    detector = Detector()
    invalid_power = torch.tensor([-0.5, 1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="Optical power must be non-negative"):
        detector.simulate(invalid_power, q=1.602e-19, k_b=1.381e-23, temperature=300)


def test_laser_warns_when_rin_noise_produces_negative_power() -> None:
    """Laser should emit a UserWarning when high RIN on weak power causes negative values."""
    laser = Laser(RIN=-50, linewidth=1e5)
    field = torch.ones(4, dtype=torch.complex64)

    with pytest.warns(UserWarning, match="Laser RIN noise produced negative optical power"):
        output = laser.simulate(field)
    assert not torch.any(torch.isnan(output))


def test_tia_non_idealities_standalone() -> None:
    """TIA component should apply gain error, offset, noise, saturation, and auto-gain independently."""
    tia = TIA(
        transimpedance=100.0,
        gain_error=1.5,
        offset=0.2,
        noise_std=0.05,
        saturation_min=0.1,
        saturation_max=1.8,
    )
    photocurrent = torch.tensor([0.0, 0.005, 0.02], dtype=torch.float32)

    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    v1 = tia.simulate(photocurrent, generator=gen1)
    v2 = tia.simulate(photocurrent, generator=gen2)

    torch.testing.assert_close(v1, v2)
    assert torch.all(v1 >= 0.1)
    assert torch.all(v1 <= 1.8)

    # Auto-gain mode
    tia_auto = TIA(transimpedance=100.0, auto_gain=True, adc_v_range=2.0)
    small_current = torch.tensor([1e-5, 2e-5], dtype=torch.float32)
    v_auto = tia_auto.simulate(small_current)
    torch.testing.assert_close(v_auto.max(), torch.tensor(0.85 * 2.0), atol=1e-5, rtol=1e-5)


def test_adc_non_idealities_standalone() -> None:
    """ADC component should apply gain error, offset, INL, DNL, and complex protection independently."""
    adc_ideal = ADC(bits=12, v_range=2.0)
    adc_nonideal = ADC(
        bits=12,
        v_range=2.0,
        gain_error=1.1,
        offset=0.05,
        inl_coefficient=10.0,
        transition_noise_lsb=5.0,
    )
    voltage = torch.tensor([0.2, 0.5, 1.0, 1.5], dtype=torch.float32)

    gen1 = torch.Generator().manual_seed(99)
    gen2 = torch.Generator().manual_seed(99)
    out1 = adc_nonideal.simulate(voltage, generator=gen1)
    out2 = adc_nonideal.simulate(voltage, generator=gen2)
    out_ideal = adc_ideal.simulate(voltage)

    torch.testing.assert_close(out1, out2)
    assert not torch.allclose(out1, out_ideal)
    assert torch.all(out1 >= 0.0) and torch.all(out1 <= 2.0)

    # Complex input safeguard
    complex_v = torch.tensor([0.5 + 0.3j, 1.0 - 0.2j], dtype=torch.complex64)
    out_complex = adc_ideal.simulate(complex_v)
    assert not torch.is_complex(out_complex)



