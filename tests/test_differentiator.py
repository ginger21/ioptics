import torch
import pytest
from torch.nn.utils import parameters_to_vector

from ioptics.differentiators import BernoulliDifferentiator, PhysicalBernoulliDifferentiator, ParameterShiftDifferentiator
from ioptics.layers import ClementsMesh
from ioptics.models import OpticalNN


def _abs_mse_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean((outputs.abs() - targets) ** 2)


def test_bernoulli_differentiator_returns_flat_gradient_with_expected_size() -> None:
    torch.manual_seed(42)

    model = OpticalNN(ClementsMesh(L=10))

    expected_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 10)

    differentiator = BernoulliDifferentiator(step_size=0.01, eta=0.1)
    gradient = differentiator.differentiate(model, inputs, _abs_mse_loss, y_label=targets)

    assert isinstance(gradient, torch.Tensor)
    assert gradient.dim() == 1
    assert gradient.numel() == expected_params_count


def test_bernoulli_differentiator_scales_linearly_with_eta_under_fixed_seed() -> None:
    torch.manual_seed(123)

    model = OpticalNN(ClementsMesh(L=5))

    inputs = torch.randn(3, 5)
    targets = torch.randn(3, 5)

    torch.manual_seed(123)
    differentiator1 = BernoulliDifferentiator(step_size=0.01, eta=0.02)
    grad1 = differentiator1.differentiate(model, inputs, _abs_mse_loss, y_label=targets)

    torch.manual_seed(123)
    differentiator2 = BernoulliDifferentiator(step_size=0.01, eta=0.04)
    grad2 = differentiator2.differentiate(model, inputs, _abs_mse_loss, y_label=targets)

    torch.testing.assert_close(grad2, grad1 * 2.0, atol=1e-6, rtol=1e-5)


class FakeSimulator:
    def __init__(self) -> None:
        self.call_count = 0

    def run(self, inputs: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        self.call_count += 1
        return model(inputs)


def test_physical_bernoulli_differentiator_uses_simulator_and_respects_eta_scaling() -> None:
    torch.manual_seed(456)

    simulator = FakeSimulator()
    model = OpticalNN(ClementsMesh(L=3))

    inputs = torch.randn(2, 3)
    targets = torch.randn(2, 3)

    torch.manual_seed(456)
    differentiator1 = PhysicalBernoulliDifferentiator(simulator, step_size=0.01, eta=0.01)
    simulator.call_count = 0
    grad1 = differentiator1.differentiate(model, inputs, _abs_mse_loss, y_label=targets)
    calls_eta1 = simulator.call_count

    torch.manual_seed(456)
    differentiator2 = PhysicalBernoulliDifferentiator(simulator, step_size=0.01, eta=0.02)
    simulator.call_count = 0
    grad2 = differentiator2.differentiate(model, inputs, _abs_mse_loss, y_label=targets)
    calls_eta2 = simulator.call_count

    torch.testing.assert_close(grad2, grad1 * 2.0, atol=1e-6, rtol=1e-5)
    assert calls_eta1 == 2
    assert calls_eta2 == 2


def test_bernoulli_differentiator_restores_original_parameters_after_differentiate() -> None:
    torch.manual_seed(789)

    model = OpticalNN(ClementsMesh(L=4))
    original_params = parameters_to_vector(model.parameters()).clone()

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    differentiator = BernoulliDifferentiator(step_size=0.01, eta=0.1)
    differentiator.differentiate(model, inputs, _abs_mse_loss, y_label=targets)

    params_after = parameters_to_vector(model.parameters())
    torch.testing.assert_close(params_after, original_params)


def test_parameter_shift_differentiator_returns_flat_gradient_with_expected_size() -> None:
    torch.manual_seed(101)

    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    differentiator = ParameterShiftDifferentiator()
    gradient = differentiator.differentiate(model, inputs, _abs_mse_loss, y_label=targets)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert isinstance(gradient, torch.Tensor)
    assert gradient.dim() == 1
    assert gradient.numel() == trainable_count


def test_bernoulli_differentiator_rejects_non_positive_step_size():
    with pytest.raises(ValueError, match="step_size must be a finite positive number"):
        BernoulliDifferentiator(step_size=-0.01)
    with pytest.raises(ValueError, match="step_size must be a finite positive number"):
        BernoulliDifferentiator(step_size=0)
    with pytest.raises(ValueError, match="step_size must be a finite positive number"):
        BernoulliDifferentiator(step_size=float('inf'))
    with pytest.raises(ValueError, match="step_size must be a finite positive number"):
        BernoulliDifferentiator(step_size=float('nan'))


def test_bernoulli_differentiator_rejects_non_finite_eta():
    with pytest.raises(ValueError, match="eta must be a finite positive number"):
        BernoulliDifferentiator(eta=-0.01)
    with pytest.raises(ValueError, match="eta must be a finite positive number"):
        BernoulliDifferentiator(eta=0)
    with pytest.raises(ValueError, match="eta must be a finite positive number"):
        BernoulliDifferentiator(eta=float('inf'))
    with pytest.raises(ValueError, match="eta must be a finite positive number"):
        BernoulliDifferentiator(eta=float('nan'))


def test_bernoulli_differentiator_raises_when_y_label_is_none():
    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4)

    differentiator = BernoulliDifferentiator()
    with pytest.raises(ValueError, match="y_label cannot be None"):
        differentiator.differentiate(model, inputs, lambda x, y: torch.sum(x))


def test_bernoulli_differentiator_raises_when_no_trainable_parameters():
    model = OpticalNN(ClementsMesh(L=4))
    # Make all parameters non-trainable
    for param in model.parameters():
        param.requires_grad = False

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    differentiator = BernoulliDifferentiator()
    with pytest.raises(ValueError, match="model has no trainable parameters"):
        differentiator.differentiate(model, inputs, _abs_mse_loss, y_label=targets)


def test_bernoulli_differentiator_raises_when_loss_is_not_scalar():
    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    def non_scalar_loss_fn(outputs, targets):
        return outputs  # Return non-scalar tensor

    differentiator = BernoulliDifferentiator()
    with pytest.raises(ValueError, match="loss_fn must return a scalar tensor"):
        differentiator.differentiate(model, inputs, non_scalar_loss_fn, y_label=targets)


def test_physical_bernoulli_differentiator_raises_when_inputs_device_mismatch():
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not (has_cuda or has_mps):
        pytest.skip("CUDA or MPS device not available for device mismatch test")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    model = OpticalNN(ClementsMesh(L=4))
    # Move model to CPU by default, then try with inputs on GPU
    inputs = torch.randn(2, 4, device=device)  # Input on CUDA/MPS
    targets = torch.randn(2, 4)

    differentiator = PhysicalBernoulliDifferentiator()
    with pytest.raises(ValueError, match="inputs and model parameters must be on the same device"):
        differentiator.differentiate(model, inputs, _abs_mse_loss, y_label=targets)


def test_parameter_shift_differentiator_raises_when_y_label_is_none():
    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4)

    differentiator = ParameterShiftDifferentiator()
    with pytest.raises(ValueError, match="y_label cannot be None"):
        differentiator.differentiate(model, inputs, lambda x, y: torch.sum(x))


def test_parameter_shift_differentiator_raises_when_no_trainable_parameters():
    model = OpticalNN(ClementsMesh(L=4))
    # Make all parameters non-trainable
    for param in model.parameters():
        param.requires_grad = False

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    differentiator = ParameterShiftDifferentiator()
    with pytest.raises(ValueError, match="model has no trainable parameters"):
        differentiator.differentiate(model, inputs, _abs_mse_loss, y_label=targets)


def test_parameter_shift_differentiator_raises_when_loss_is_not_scalar():
    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    def non_scalar_loss_fn(outputs, targets):
        return outputs  # Return non-scalar tensor

    differentiator = ParameterShiftDifferentiator()
    with pytest.raises(ValueError, match="loss_fn must return a scalar tensor"):
        differentiator.differentiate(model, inputs, non_scalar_loss_fn, y_label=targets)


def test_parameter_shift_differentiator_raises_when_inputs_device_mismatch():
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if not (has_cuda or has_mps):
        pytest.skip("CUDA or MPS device not available for device mismatch test")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4, device=device)  # Input on CUDA/MPS
    targets = torch.randn(2, 4)

    differentiator = ParameterShiftDifferentiator()
    with pytest.raises(ValueError, match="inputs and model parameters must be on the same device"):
        differentiator.differentiate(model, inputs, _abs_mse_loss, y_label=targets)


def test_parameter_shift_differentiator_restores_parameters_on_exception():
    """Verify parameters are restored even if an exception occurs during shifting."""
    torch.manual_seed(42)

    model = OpticalNN(ClementsMesh(L=4))
    original_params = parameters_to_vector(model.parameters()).clone()

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    # Create a loss function that raises after first call
    call_count = [0]

    def failing_loss_fn(outputs, targets):
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("Intentional failure")
        return torch.mean((outputs - targets) ** 2)

    differentiator = ParameterShiftDifferentiator()

    with pytest.raises(RuntimeError):
        differentiator.differentiate(model, inputs, failing_loss_fn, y_label=targets)

    # Parameters should be restored despite the exception
    params_after = parameters_to_vector(model.parameters())
    torch.testing.assert_close(params_after, original_params)


def test_physical_bernoulli_differentiator_raises_when_y_label_is_none():
    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4)

    differentiator = PhysicalBernoulliDifferentiator()
    with pytest.raises(ValueError, match="y_label cannot be None"):
        differentiator.differentiate(model, inputs, lambda x, y: torch.sum(x))


def test_physical_bernoulli_differentiator_raises_when_no_trainable_parameters():
    model = OpticalNN(ClementsMesh(L=4))
    # Make all parameters non-trainable
    for param in model.parameters():
        param.requires_grad = False

    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    differentiator = PhysicalBernoulliDifferentiator()
    with pytest.raises(ValueError, match="model has no trainable parameters"):
        differentiator.differentiate(model, inputs, _abs_mse_loss, y_label=targets)


def test_physical_bernoulli_differentiator_raises_when_loss_is_not_scalar():
    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    def non_scalar_loss_fn(outputs, targets):
        return outputs  # Return non-scalar tensor

    differentiator = PhysicalBernoulliDifferentiator()
    with pytest.raises(ValueError, match="loss_fn must return a scalar tensor"):
        differentiator.differentiate(model, inputs, non_scalar_loss_fn, y_label=targets)


def test_differentiators_support_complex_inputs_with_simulator():
    """Verify that PhysicalBernoulli and ParameterShift fully support complex optical inputs."""
    from ioptics.simulator import Simulator, HardwareProfile

    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4, dtype=torch.complex64)
    targets = torch.randn(2, 4, dtype=torch.float32)

    def complex_mse_loss(outputs, targets):
        mag = outputs.abs() if torch.is_complex(outputs) else outputs
        return torch.nn.functional.mse_loss(mag, targets)

    sim = Simulator(profile=HardwareProfile.LAB)
    differentiator_pbd = PhysicalBernoulliDifferentiator(simulator=sim)
    grads_pbd = differentiator_pbd.differentiate(model, inputs, complex_mse_loss, y_label=targets)
    assert len(grads_pbd) > 0
    for grad in grads_pbd:
        assert grad is not None
        assert torch.isfinite(grad).all()

    differentiator_ps = ParameterShiftDifferentiator()
    grads_ps = differentiator_ps.differentiate(model, inputs, complex_mse_loss, y_label=targets)
    assert len(grads_ps) > 0
    for grad in grads_ps:
        assert grad is not None
        assert torch.isfinite(grad).all()


def test_differentiators_reject_nonfinite_loss():
    model = OpticalNN(ClementsMesh(L=4))
    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 4)

    def nan_loss_fn(outputs, targets):
        return torch.tensor(float('nan'))

    def inf_loss_fn(outputs, targets):
        return torch.tensor(float('inf'))

    differentiator = BernoulliDifferentiator()
    with pytest.raises(ValueError, match="loss must be finite"):
        differentiator.differentiate(model, inputs, nan_loss_fn, y_label=targets)
    with pytest.raises(ValueError, match="loss must be finite"):
        differentiator.differentiate(model, inputs, inf_loss_fn, y_label=targets)


def test_parameter_shift_differentiator_batched_matches_serial_and_fcarray() -> None:
    """Verify batched ParameterShiftDifferentiator matches serial two-point shift on ClementsMesh and FCArray."""
    from ioptics.layers import FCArray
    from ioptics.nonlinear import Abs

    torch.manual_seed(2026)
    mesh = ClementsMesh(L=6)
    fc = FCArray(in_dim=6, out_dim=4)
    model = OpticalNN(mesh, fc, Abs())
    inputs = torch.randn(4, 6, dtype=torch.float32)
    targets = torch.randint(0, 4, (4,))
    loss_fn = torch.nn.CrossEntropyLoss()

    diff = ParameterShiftDifferentiator()
    grads_batched = diff.differentiate(model, inputs, loss_fn, y_label=targets)

    # Compute reference serial two-point shift by wrapping model in a non-Sequential container
    class NonSequentialWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

    wrapped_model = NonSequentialWrapper(model)
    grads_serial = diff.differentiate(wrapped_model, inputs, loss_fn, y_label=targets)

    torch.testing.assert_close(grads_batched, grads_serial, atol=1e-5, rtol=1e-4)
