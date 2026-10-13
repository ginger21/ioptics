"""Comprehensive unit tests for 1D intra-column thermal crosstalk modeling in ioptics."""
from __future__ import annotations

import pytest
import torch

from ioptics.backward_functions import BackwardType, CustomForwardFunction
from ioptics.components import MZI, MZM, PhaseShifter
from ioptics.differentiators import ParameterShiftDifferentiator
from ioptics.layers import ArrayColumn, ClementsMesh, FCArray, MeshColumn, MeshCore
from ioptics.simulator import ExecutionMode, HardwareProfile, Simulator, SimulatorConfig


# ---------------------------------------------------------------------------
# 1. Column-level matrix generation and setter validation
# ---------------------------------------------------------------------------

def test_column_thermal_crosstalk_matrix_construction() -> None:
    # 2 MZIs in column (ports (0, 1) and (2, 3), pitch = 2 waveguides -> dist = 1)
    col = MeshColumn(
        N=4,
        components=[MZI(0, 1), MZI(2, 3)],
        thermal_crosstalk_coeff=0.05,
    )
    mat = col.get_thermal_crosstalk_matrix()
    assert mat is not None
    assert mat.shape == (2, 2)
    expected = torch.tensor([[1.0, 0.05], [0.05, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(mat, expected)


def test_column_thermal_crosstalk_distance_decay() -> None:
    # 3 MZMs in ArrayColumn at ports 0, 1, 2 (pitch = 1 waveguide)
    col = ArrayColumn(
        N=3,
        components=[MZM(0), MZM(1), MZM(2)],
        thermal_crosstalk_coeff=0.1,
    )
    mat = col.get_thermal_crosstalk_matrix()
    assert mat is not None
    assert mat.shape == (3, 3)
    expected = torch.tensor(
        [
            [1.0, 0.1, 0.01],
            [0.1, 1.0, 0.1],
            [0.01, 0.1, 1.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(mat, expected)


def test_column_thermal_crosstalk_zero_returns_none() -> None:
    col = MeshColumn(N=4, components=[MZI(0, 1), MZI(2, 3)], thermal_crosstalk_coeff=0.0)
    assert col.get_thermal_crosstalk_matrix() is None


def test_column_thermal_crosstalk_invalid_coeff_raises() -> None:
    with pytest.raises(ValueError, match="thermal_crosstalk_coeff must be in"):
        MeshColumn(N=4, components=[MZI(0, 1), MZI(2, 3)], thermal_crosstalk_coeff=-0.1)

    with pytest.raises(ValueError, match="thermal_crosstalk_coeff must be in"):
        MeshColumn(N=4, components=[MZI(0, 1), MZI(2, 3)], thermal_crosstalk_coeff=1.0)


def test_column_set_custom_crosstalk_matrix() -> None:
    col = MeshColumn(N=4, components=[MZI(0, 1), MZI(2, 3)])
    custom_mat = torch.tensor([[1.0, 0.08], [0.04, 1.0]], dtype=torch.float32)
    col.set_thermal_crosstalk_matrix(custom_mat)
    torch.testing.assert_close(col.get_thermal_crosstalk_matrix(), custom_mat)

    # Reset
    col.set_thermal_crosstalk_matrix(None)
    assert col.get_thermal_crosstalk_matrix() is None


def test_column_set_custom_crosstalk_matrix_shape_and_complex_validation() -> None:
    col = MeshColumn(N=4, components=[MZI(0, 1), MZI(2, 3)])
    with pytest.raises(ValueError, match="shape"):
        col.set_thermal_crosstalk_matrix(torch.eye(3))

    with pytest.raises(ValueError, match="real-valued"):
        col.set_thermal_crosstalk_matrix(torch.eye(2, dtype=torch.complex64))


# ---------------------------------------------------------------------------
# 2. Conservation of Unitarity under Thermal Crosstalk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [2, 4, 6])
def test_clements_mesh_unitarity_preserved_under_thermal_crosstalk(L: int) -> None:
    torch.manual_seed(101)
    mesh = ClementsMesh(L=L, thermal_crosstalk_coeff=0.05)
    U = mesh.get_transfer_matrix()
    I_expected = torch.eye(L, dtype=U.dtype, device=U.device)
    U_dagger_U = U.conj().T @ U
    torch.testing.assert_close(U_dagger_U, I_expected, atol=1e-5, rtol=1e-5)

    # Energy conservation test
    x = torch.randn(5, L, dtype=torch.complex64)
    y = mesh(x)
    torch.testing.assert_close(torch.norm(x, dim=-1), torch.norm(y, dim=-1), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. Numeric Gradient Equivalence with PyTorch Autograd
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backward_type", [BackwardType.PARAMETER_SHIFT.value, BackwardType.ADJOINT_VARIANT.value])
@pytest.mark.parametrize("L", [2, 4])
def test_clements_mesh_gradient_matches_autograd_with_thermal_crosstalk(
    backward_type: str, L: int
) -> None:
    torch.manual_seed(202)
    crosstalk = 0.04
    layer_custom = ClementsMesh(L=L, backward_type=backward_type, thermal_crosstalk_coeff=crosstalk)
    layer_default = ClementsMesh(L=L, backward_type=BackwardType.DEFAULT.value, thermal_crosstalk_coeff=crosstalk)

    with torch.no_grad():
        for p_def, p_cust in zip(layer_default.parameters(), layer_custom.parameters()):
            p_def.copy_(p_cust)

    # 1. Real input
    x_real = torch.randn(3, L, dtype=torch.float32)
    x_cust_r = x_real.clone().requires_grad_(True)
    x_def_r = x_real.clone().requires_grad_(True)

    out_cust_r = layer_custom(x_cust_r)
    loss_cust_r = (out_cust_r.abs() ** 2).sum()
    loss_cust_r.backward()

    out_def_r = layer_default(x_def_r)
    loss_def_r = (out_def_r.abs() ** 2).sum()
    loss_def_r.backward()

    torch.testing.assert_close(x_cust_r.grad, x_def_r.grad, atol=1e-4, rtol=1e-3)
    for p_cust, p_def in zip(layer_custom.parameters(), layer_default.parameters()):
        torch.testing.assert_close(p_cust.grad, p_def.grad, atol=1e-4, rtol=1e-3)

    # 2. Complex input
    for p in layer_custom.parameters():
        p.grad = None
    for p in layer_default.parameters():
        p.grad = None

    x_complex = torch.randn(2, L, dtype=torch.complex64)
    x_cust_c = x_complex.clone().requires_grad_(True)
    x_def_c = x_complex.clone().requires_grad_(True)

    out_cust_c = layer_custom(x_cust_c)
    loss_cust_c = (out_cust_c.abs() ** 2).sum()
    loss_cust_c.backward()

    out_def_c = layer_default(x_def_c)
    loss_def_c = (out_def_c.abs() ** 2).sum()
    loss_def_c.backward()

    torch.testing.assert_close(x_cust_c.grad, x_def_c.grad, atol=1e-4, rtol=1e-3)
    for p_cust, p_def in zip(layer_custom.parameters(), layer_default.parameters()):
        torch.testing.assert_close(p_cust.grad, p_def.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("backward_type", [BackwardType.PARAMETER_SHIFT.value, BackwardType.ADJOINT_VARIANT.value])
def test_fcarray_gradient_matches_autograd_with_thermal_crosstalk(backward_type: str) -> None:
    torch.manual_seed(303)
    in_dim, out_dim = 4, 3
    crosstalk = 0.03
    layer_custom = FCArray(in_dim, out_dim, backward_type=backward_type, thermal_crosstalk_coeff=crosstalk)
    layer_default = FCArray(in_dim, out_dim, backward_type=BackwardType.DEFAULT.value, thermal_crosstalk_coeff=crosstalk)

    with torch.no_grad():
        for p_def, p_cust in zip(layer_default.parameters(), layer_custom.parameters()):
            p_def.copy_(p_cust)

    x = torch.randn(2, in_dim, dtype=torch.complex64, requires_grad=True)
    x_def = x.detach().clone().requires_grad_(True)

    out_custom = layer_custom(x)
    loss_custom = (out_custom.abs() ** 2).sum()
    loss_custom.backward()

    out_def = layer_default(x_def)
    loss_def = (out_def.abs() ** 2).sum()
    loss_def.backward()

    torch.testing.assert_close(x.grad, x_def.grad, atol=1e-4, rtol=1e-3)
    for p_cust, p_def in zip(layer_custom.parameters(), layer_default.parameters()):
        torch.testing.assert_close(p_cust.grad, p_def.grad, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. Double-Precision gradcheck under Thermal Crosstalk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backward_type", [BackwardType.PARAMETER_SHIFT.value, BackwardType.ADJOINT_VARIANT.value])
def test_gradcheck_under_thermal_crosstalk(backward_type: str) -> None:
    torch.manual_seed(404)
    L = 4
    layer = ClementsMesh(L=L, backward_type=backward_type, thermal_crosstalk_coeff=0.04)
    input_tensor = torch.randn(2, L, dtype=torch.float64, requires_grad=True)
    params = tuple(layer.parameters())

    def loss_func(inp: torch.Tensor) -> torch.Tensor:
        out = CustomForwardFunction.apply(layer, inp, backward_type, *params)
        return (out.abs() ** 2).sum()

    assert torch.autograd.gradcheck(loss_func, (input_tensor,), eps=1e-6, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# 5. ParameterShiftDifferentiator Batch Accelerated Path with Thermal Crosstalk
# ---------------------------------------------------------------------------

def test_parameter_shift_differentiator_batch_parity_under_crosstalk() -> None:
    torch.manual_seed(505)
    model = torch.nn.Sequential(ClementsMesh(L=4, thermal_crosstalk_coeff=0.05))
    inputs = torch.randn(2, 4, dtype=torch.complex64)
    target = torch.randn(2, 4, dtype=torch.complex64)

    def loss_fn(pred, tgt):
        return ((pred - tgt).abs() ** 2).sum()

    diff = ParameterShiftDifferentiator()
    grads = diff.differentiate(model, inputs, loss_fn, target)
    assert grads is not None
    assert grads.numel() == sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Compare with serial evaluation by temporarily removing forward_parameter_shifts
    saved_fn = MeshCore.forward_parameter_shifts
    try:
        delattr(MeshCore, "forward_parameter_shifts")
        grads_serial = diff.differentiate(model, inputs, loss_fn, target)
        torch.testing.assert_close(grads, grads_serial, atol=1e-4, rtol=1e-3)
    finally:
        MeshCore.forward_parameter_shifts = saved_fn


# ---------------------------------------------------------------------------
# 6. Simulator Integration Tests
# ---------------------------------------------------------------------------

def test_simulator_hardware_profile_thermal_crosstalk_defaults() -> None:
    sim_ideal = Simulator(profile=HardwareProfile.IDEAL.value)
    assert sim_ideal.thermal_crosstalk_coeff == 0.0

    sim_lab = Simulator(profile=HardwareProfile.LAB.value)
    assert sim_lab.thermal_crosstalk_coeff == 0.01

    sim_hw = Simulator(profile=HardwareProfile.HARDWARE_LIKE.value)
    assert sim_hw.thermal_crosstalk_coeff == 0.03


def test_simulator_deterministic_mode_bypasses_thermal_crosstalk() -> None:
    model = FCArray(in_dim=4, out_dim=4)
    inp = torch.randn(4, dtype=torch.complex64)

    # Lab profile with 1% thermal crosstalk
    sim_lab = Simulator(profile=HardwareProfile.LAB.value)
    out_stoch = sim_lab.run(inp, model, mode=ExecutionMode.STOCHASTIC.value, seed=42)

    # In deterministic mode, crosstalk and noises are zeroed
    out_det1 = sim_lab.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=42)
    out_det2 = sim_lab.run(inp, model, mode=ExecutionMode.DETERMINISTIC.value, seed=99)
    torch.testing.assert_close(out_det1, out_det2)
    assert not torch.allclose(out_stoch, out_det1)


def test_simulator_run_dynamic_crosstalk_override() -> None:
    model = ClementsMesh(L=4)
    inp = torch.randn(4, dtype=torch.complex64)
    sim = Simulator()

    out_base = sim.run(inp, model, seed=123)
    out_crosstalk = sim.run(inp, model, seed=123, thermal_crosstalk_coeff=0.08)

    assert not torch.allclose(out_base, out_crosstalk)
    # Ensure simulator restored model's override cleanly
    assert model._sim_crosstalk_coeff_override is None
