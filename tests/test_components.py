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
