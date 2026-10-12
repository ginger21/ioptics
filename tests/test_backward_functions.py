import pytest
import torch

from ioptics.backward_functions import BackwardType, CustomForwardFunction
from ioptics.layers import ClementsMesh


def test_custom_forward_dtype_behavior_float64_to_complex128() -> None:
    """Test that float64 input produces complex128 output using PARAMETER_SHIFT."""
    layer = ClementsMesh(L=4, backward_type=BackwardType.PARAMETER_SHIFT.value)
    input_tensor = torch.ones(4, dtype=torch.float64)

    output = layer(input_tensor)

    assert output.dtype == torch.complex128, f"Expected output dtype complex128, got {output.dtype}"
    assert output.shape == (4,), f"Expected output shape (4,), got {output.shape}"


def test_custom_forward_dimension_mismatch_error_adjoint_variant() -> None:
    """Test that dimension mismatch raises ValueError for ADJOINT_VARIANT custom backward."""
    layer = ClementsMesh(L=2, backward_type=BackwardType.ADJOINT_VARIANT.value)
    bad_input = torch.zeros(1, 3)  # Wrong last dimension size

    with pytest.raises(ValueError):
        _ = layer(bad_input)


def test_backward_types_consistency() -> None:
    """Test that all backward types produce same forward output."""
    input_tensor = (
        torch.randn(2, 4, dtype=torch.float32)
        + 1j * torch.randn(2, 4, dtype=torch.float32)
    ).to(torch.complex64)

    layer_default = ClementsMesh(L=4, backward_type=BackwardType.DEFAULT.value)
    layer_param_shift = ClementsMesh(L=4, backward_type=BackwardType.PARAMETER_SHIFT.value)
    layer_adjoint = ClementsMesh(L=4, backward_type=BackwardType.ADJOINT_VARIANT.value)

    # Set the same parameters to ensure consistent forward behavior
    with torch.no_grad():
        for p_default, p_param_shift, p_adjoint in zip(
            layer_default.parameters(),
            layer_param_shift.parameters(),
            layer_adjoint.parameters()
        ):
            p_param_shift.copy_(p_default)
            p_adjoint.copy_(p_default)

    output_default = layer_default(input_tensor)
    output_param_shift = layer_param_shift(input_tensor)
    output_adjoint = layer_adjoint(input_tensor)

    torch.testing.assert_close(output_default, output_param_shift, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output_default, output_adjoint, atol=1e-5, rtol=1e-5)


def test_backward_types_dimension_mismatch_errors() -> None:
    """Test that dimension mismatch errors are properly raised for all custom backward types."""
    input_3d = torch.zeros(1, 3)

    # Test PARAMETER_SHIFT
    layer_param_shift = ClementsMesh(L=2, backward_type=BackwardType.PARAMETER_SHIFT.value)
    with pytest.raises(ValueError):
        _ = layer_param_shift(input_3d)

    # Test ADJOINT_VARIANT
    layer_adjoint = ClementsMesh(L=2, backward_type=BackwardType.ADJOINT_VARIANT.value)
    with pytest.raises(ValueError):
        _ = layer_adjoint(input_3d)


def test_backward_function_different_input_shapes() -> None:
    """Test custom backward with various input shapes."""
    layer = ClementsMesh(L=3, backward_type=BackwardType.ADJOINT_VARIANT.value)

    # Test 1D input
    input_1d = torch.ones(3, dtype=torch.complex64)
    output_1d = layer(input_1d)
    assert output_1d.shape == (3,)

    # Test 2D batch input
    input_2d = torch.ones(5, 3, dtype=torch.complex64)
    output_2d = layer(input_2d)
    assert output_2d.shape == (5, 3)

    # Test 3D batch input
    input_3d = torch.ones(2, 4, 3, dtype=torch.complex64)
    output_3d = layer(input_3d)
    assert output_3d.shape == (2, 4, 3)

    # All should be close to baseline calculation
    for input_tensor, output_tensor in [(input_1d, output_1d), (input_2d, output_2d), (input_3d, output_3d)]:
        transfer_matrix = layer.get_transfer_matrix().to(
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        baseline = torch.matmul(transfer_matrix, input_tensor.unsqueeze(-1)).squeeze(-1)
        baseline = baseline[..., layer.output_ports]
        torch.testing.assert_close(output_tensor, baseline, atol=1e-5, rtol=1e-5)


def test_backward_real_float32_input_parameter_shift_grad_dtype() -> None:
    layer = ClementsMesh(L=2, backward_type=BackwardType.PARAMETER_SHIFT.value)
    layer_default = ClementsMesh(L=2, backward_type=BackwardType.DEFAULT.value)
    with torch.no_grad():
        for p_def, p_ps in zip(layer_default.parameters(), layer.parameters()):
            p_def.copy_(p_ps)

    input_tensor = torch.randn(4, 2, dtype=torch.float32, requires_grad=True)
    input_default = input_tensor.detach().clone().requires_grad_(True)

    output = layer(input_tensor)
    loss = output.abs().sum()
    loss.backward()

    output_def = layer_default(input_default)
    loss_def = output_def.abs().sum()
    loss_def.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.dtype == torch.float32
    assert input_tensor.grad.shape == input_tensor.shape
    torch.testing.assert_close(input_tensor.grad, input_default.grad, atol=1e-4, rtol=1e-3)
    for p_ps, p_def in zip(layer.parameters(), layer_default.parameters()):
        assert p_ps.grad is not None
        torch.testing.assert_close(p_ps.grad, p_def.grad, atol=1e-4, rtol=1e-3)


def test_backward_real_float32_input_adjoint_variant_grad_dtype() -> None:
    layer = ClementsMesh(L=2, backward_type=BackwardType.ADJOINT_VARIANT.value)
    layer_default = ClementsMesh(L=2, backward_type=BackwardType.DEFAULT.value)
    with torch.no_grad():
        for p_def, p_adj in zip(layer_default.parameters(), layer.parameters()):
            p_def.copy_(p_adj)

    input_tensor = torch.randn(3, 2, dtype=torch.float32, requires_grad=True)
    input_default = input_tensor.detach().clone().requires_grad_(True)

    output = layer(input_tensor)
    loss = output.abs().sum()
    loss.backward()

    output_def = layer_default(input_default)
    loss_def = output_def.abs().sum()
    loss_def.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.dtype == torch.float32
    assert input_tensor.grad.shape == input_tensor.shape
    torch.testing.assert_close(input_tensor.grad, input_default.grad, atol=1e-4, rtol=1e-3)
    for p_adj, p_def in zip(layer.parameters(), layer_default.parameters()):
        assert p_adj.grad is not None
        torch.testing.assert_close(p_adj.grad, p_def.grad, atol=1e-4, rtol=1e-3)


def test_backward_real_float64_input_adjoint_variant_grad_dtype() -> None:
    layer = ClementsMesh(L=2, backward_type=BackwardType.ADJOINT_VARIANT.value)
    layer_default = ClementsMesh(L=2, backward_type=BackwardType.DEFAULT.value)
    with torch.no_grad():
        for p_def, p_adj in zip(layer_default.parameters(), layer.parameters()):
            p_def.copy_(p_adj)

    input_tensor = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    input_default = input_tensor.detach().clone().requires_grad_(True)

    output = layer(input_tensor)
    loss = output.abs().sum()
    loss.backward()

    output_def = layer_default(input_default)
    loss_def = output_def.abs().sum()
    loss_def.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.dtype == torch.float64
    assert input_tensor.grad.shape == input_tensor.shape
    torch.testing.assert_close(input_tensor.grad, input_default.grad, atol=1e-5, rtol=1e-4)
    for p_adj, p_def in zip(layer.parameters(), layer_default.parameters()):
        assert p_adj.grad is not None
        torch.testing.assert_close(p_adj.grad, p_def.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("L", [2, 4])
@pytest.mark.parametrize(
    "backward_type",
    [BackwardType.PARAMETER_SHIFT.value, BackwardType.ADJOINT_VARIANT.value],
)
def test_hard_numerical_gradient_assertion_compared_with_autograd(
    L: int,
    backward_type: str,
) -> None:
    """Rigorous numeric assertion comparing custom backward strategies with PyTorch Autograd."""
    torch.manual_seed(42)
    layer_custom = ClementsMesh(L=L, backward_type=backward_type)
    layer_default = ClementsMesh(L=L, backward_type=BackwardType.DEFAULT.value)

    with torch.no_grad():
        for p_def, p_cust in zip(layer_default.parameters(), layer_custom.parameters()):
            p_def.copy_(p_cust)

    # 1. Real input test
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

    # Zero grads for next test
    for p in layer_custom.parameters():
        p.grad = None
    for p in layer_default.parameters():
        p.grad = None

    # 2. Complex input test
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


@pytest.mark.parametrize(
    "backward_type",
    [BackwardType.PARAMETER_SHIFT.value, BackwardType.ADJOINT_VARIANT.value],
)
def test_custom_forward_gradcheck(backward_type: str) -> None:
    """Verify analytical gradients against finite differences using torch.autograd.gradcheck."""
    torch.manual_seed(42)
    L = 2
    layer = ClementsMesh(L=L, backward_type=backward_type)
    input_tensor = torch.randn(2, L, dtype=torch.float64, requires_grad=True)
    params = tuple(layer.parameters())

    def loss_func(inp: torch.Tensor) -> torch.Tensor:
        out = CustomForwardFunction.apply(layer, inp, backward_type, *params)
        return (out.abs() ** 2).sum()

    assert torch.autograd.gradcheck(loss_func, (input_tensor,), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize(
    "backward_type",
    [BackwardType.PARAMETER_SHIFT.value, BackwardType.ADJOINT_VARIANT.value],
)
def test_custom_forward_parameter_grad_order_matches_forward_arg_order(
    backward_type: str,
) -> None:
    layer = ClementsMesh(L=2, backward_type=backward_type)
    params = tuple(layer.parameters())
    assert len(params) >= 2

    input_tensor = torch.randn(4, 2, dtype=torch.float32)

    for param in params:
        param.grad = None
    output = layer(input_tensor)
    output.abs().sum().backward()
    baseline_grads = [param.grad.detach().clone() for param in params]

    for param in params:
        param.grad = None
    reversed_output = CustomForwardFunction.apply(
        layer,
        input_tensor,
        backward_type,
        *tuple(reversed(params)),
    )
    reversed_output.abs().sum().backward()

    for baseline_grad, param in zip(baseline_grads, params):
        assert param.grad is not None
        torch.testing.assert_close(param.grad, baseline_grad, atol=1e-5, rtol=1e-5)
