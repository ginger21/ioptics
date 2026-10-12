import torch
import pytest

from ioptics.layers import ClementsMesh
from ioptics.models import OpticalNN, TrainingHistory


class _ZeroGradientDifferentiator:
    def differentiate(self, model, inputs, loss_fn, y_label=None, **kwargs):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return torch.zeros(num_params, device=inputs.device)


class _NoSimulatorDifferentiator:
    def differentiate(self, model, inputs, loss_fn, y_label=None, **kwargs):
        raise NotImplementedError


class _NoneSimulatorDifferentiator:
    simulator = None

    def differentiate(self, model, inputs, loss_fn, y_label=None, **kwargs):
        raise NotImplementedError


def test_calculate_accuracy_batch_size_mismatch():
    """Test calculate_accuracy raises ValueError when X_input batch size doesn't match y_label length."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    # Input with 8 samples but labels with only 5
    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)  # 8 samples
    y = torch.randint(0, 4, (5,))  # Only 5 labels

    with pytest.raises(ValueError, match=r"X_input batch size \(8\) must match y_label size \(5\)"):
        model.calculate_accuracy(X, y)


def test_calculate_accuracy_preserves_training_state_train():
    """Test calculate_accuracy preserves the original training state when initially in train mode."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    # Initially in train mode
    model.train()
    original_training_state = model.training

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    # Call calculate_accuracy
    accuracy = model.calculate_accuracy(X, y)

    # Verify the training state was preserved
    assert model.training == original_training_state, f"Training state changed from {original_training_state} to {model.training}"


def test_calculate_accuracy_preserves_training_state_eval():
    """Test calculate_accuracy preserves the original training state when initially in eval mode."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    # Initially in eval mode
    model.eval()
    original_training_state = model.training

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    # Call calculate_accuracy
    accuracy = model.calculate_accuracy(X, y)

    # Verify the training state was preserved
    assert model.training == original_training_state, f"Training state changed from {original_training_state} to {model.training}"


def test_calculate_accuracy_empty_y_label():
    """Test calculate_accuracy raises ValueError when y_label is empty."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y_empty = torch.tensor([])  # Empty tensor

    with pytest.raises(ValueError, match="y_label cannot be empty"):
        model.calculate_accuracy(X, y_empty)


def test_evaluate_consistent_with_calculate_accuracy():
    """Test evaluate(..., metric_fn=None) returns the same value as calculate_accuracy."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    accuracy_result = model.calculate_accuracy(X, y)
    evaluate_result = model.evaluate(X, y, metric_fn=None)

    assert accuracy_result == evaluate_result, f"Results differ: calculate_accuracy={accuracy_result}, evaluate={evaluate_result}"


def test_evaluate_custom_metric():
    """Test evaluate returns custom metric value and preserves training state."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    # Define a custom metric function (mean squared error for demonstration)
    def mse_metric(outputs, targets):
        predictions = outputs.argmax(dim=-1)
        correct_mask = (predictions == targets)
        return (1 - correct_mask.float().mean()).item()  # Error rate as MSE-like metric

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    # Initially in train mode
    model.train()
    original_training_state = model.training

    # Call evaluate with custom metric
    evaluate_result = model.evaluate(X, y, metric_fn=mse_metric)

    # Check that result is reasonable (between 0 and 1 for error rate)
    assert 0.0 <= evaluate_result <= 1.0, f"Evaluation result out of expected range: {evaluate_result}"

    # Verify the training state was preserved
    assert model.training == original_training_state, f"Training state changed from {original_training_state} to {model.training}"


def test_evaluate_with_tensor_metric():
    """Test evaluate handles 0-dim tensor metric values correctly."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    # Define a custom metric function that returns a 0-dim tensor
    def tensor_metric(outputs, targets):
        predictions = outputs.argmax(dim=-1)
        correct_mask = (predictions == targets)
        return torch.sum(correct_mask.float()).mean()  # Return 0-dim tensor

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    evaluate_result = model.evaluate(X, y, metric_fn=tensor_metric)

    # Result should be a scalar number
    assert isinstance(evaluate_result, float), f"Expected float result, got {type(evaluate_result)}"


def test_evaluate_non_scalar_metric_raises_value_error():
    """Test evaluate raises ValueError when metric_fn returns non-scalar tensor."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    def non_scalar_metric(outputs, targets):
        return outputs.real  # Non-scalar tensor

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    with pytest.raises(ValueError, match="metric_fn must return a scalar"):
        model.evaluate(X, y, metric_fn=non_scalar_metric)


def test_evaluate_batch_size_mismatch():
    """Test evaluate raises ValueError when X_input batch size doesn't match y_label length."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    # Input with 8 samples but labels with only 5
    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)  # 8 samples
    y = torch.randint(0, 4, (5,))  # Only 5 labels

    with pytest.raises(ValueError, match=r"X_input batch size \(8\) must match y_label size \(5\)"):
        model.evaluate(X, y)


def test_evaluate_empty_y_label():
    """Test evaluate raises ValueError when y_label is empty."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y_empty = torch.tensor([])  # Empty tensor

    with pytest.raises(ValueError, match="y_label cannot be empty"):
        model.evaluate(X, y_empty)


def test_evaluate_metric_fn_returns_string_raises_type_error():
    """Test evaluate raises TypeError when metric_fn returns a string."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    def string_metric(outputs, targets):
        return "accuracy"  # String return

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    with pytest.raises(TypeError, match="metric_fn must return a numeric scalar"):
        model.evaluate(X, y, metric_fn=string_metric)


def test_evaluate_metric_fn_returns_bool_raises_type_error():
    """Test evaluate raises TypeError when metric_fn returns a boolean."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    def bool_metric(outputs, targets):
        return True  # Boolean return

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    with pytest.raises(TypeError, match="metric_fn must return a numeric scalar"):
        model.evaluate(X, y, metric_fn=bool_metric)


def test_evaluate_metric_fn_returns_complex_tensor_raises_type_error():
    """Test evaluate raises TypeError when metric_fn returns 0-dim complex tensor."""
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)

    def complex_tensor_metric(outputs, targets):
        return torch.tensor(1 + 1j, dtype=torch.complex64)  # 0-dim complex tensor

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    with pytest.raises(TypeError, match="metric_fn must return a numeric scalar"):
        model.evaluate(X, y, metric_fn=complex_tensor_metric)


def test_fit_returns_stable_history_and_resets_by_default():
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)
    differentiator = _ZeroGradientDifferentiator()
    loss_function = torch.nn.CrossEntropyLoss()

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    history_1 = model.fit(
        loss_function=loss_function,
        X_input=X,
        y_label=y,
        epochs=2,
        differentiator=differentiator,
        verbose=False,
        batch_size=4,
    )

    assert isinstance(history_1, TrainingHistory)
    assert len(history_1.loss_history) == 2
    assert len(history_1.accuracy_history) == 2

    history_2 = model.fit(
        loss_function=loss_function,
        X_input=X,
        y_label=y,
        epochs=1,
        differentiator=differentiator,
        verbose=False,
        batch_size=4,
    )

    # reset_history=True by default, so second run only contains one epoch
    assert len(history_2.loss_history) == 1
    assert len(history_2.accuracy_history) == 1
    assert len(model.loss_history) == 1
    assert len(model.accuracy_history) == 1



def test_fit_appends_when_reset_history_is_false():
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)
    differentiator = _ZeroGradientDifferentiator()
    loss_function = torch.nn.CrossEntropyLoss()

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    history_1 = model.fit(
        loss_function=loss_function,
        X_input=X,
        y_label=y,
        epochs=2,
        differentiator=differentiator,
        verbose=False,
        batch_size=4,
    )
    first_loss_snapshot = tuple(history_1.loss_history)
    first_accuracy_snapshot = tuple(history_1.accuracy_history)

    history_2 = model.fit(
        loss_function=loss_function,
        X_input=X,
        y_label=y,
        epochs=1,
        differentiator=differentiator,
        verbose=False,
        batch_size=4,
        reset_history=False,
    )

    assert len(history_2.loss_history) == 3
    assert len(history_2.accuracy_history) == 3
    assert history_2.loss_history[:2] == first_loss_snapshot
    assert history_2.accuracy_history[:2] == first_accuracy_snapshot


def test_fit_restores_training_state_when_model_starts_in_eval_mode():
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)
    differentiator = _ZeroGradientDifferentiator()
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    assert model.training is False

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    model.fit(
        loss_function=loss_function,
        X_input=X,
        y_label=y,
        epochs=1,
        differentiator=differentiator,
        verbose=False,
        batch_size=4,
    )

    assert model.training is False


def test_fit_metric_fn_returns_string_raises_type_error():
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)
    differentiator = _ZeroGradientDifferentiator()
    loss_function = torch.nn.CrossEntropyLoss()

    def string_metric(outputs, targets):
        return "invalid"

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    with pytest.raises(TypeError, match="metric_fn must return a numeric scalar"):
        model.fit(
            loss_function=loss_function,
            X_input=X,
            y_label=y,
            epochs=1,
            differentiator=differentiator,
            metric_fn=string_metric,
            verbose=False,
            batch_size=4,
        )


@pytest.mark.parametrize(
    "differentiator",
    [_NoSimulatorDifferentiator(), _NoneSimulatorDifferentiator()],
)
def test_fit_simulation_raises_when_differentiator_has_no_usable_simulator(differentiator):
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)
    loss_function = torch.nn.CrossEntropyLoss()

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    with pytest.raises(TypeError, match="differentiator 必须有 simulator 属性，请使用 PhysicalBernoulliDifferentiator"):
        model.fit_simulation(
            loss_function=loss_function,
            X_input=X,
            y_label=y,
            epochs=1,
            differentiator=differentiator,
            verbose=False,
            batch_size=4,
        )


def test_fit_with_simulator_parameter():
    from ioptics.simulator import Simulator
    layer = ClementsMesh(L=4)
    model = OpticalNN(layer)
    loss_function = torch.nn.CrossEntropyLoss()
    differentiator = _ZeroGradientDifferentiator()
    sim = Simulator()

    X = torch.randn(8, 4) + 1j * torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))

    history = model.fit(
        loss_function=loss_function,
        X_input=X,
        y_label=y,
        epochs=2,
        differentiator=differentiator,
        simulator=sim,
        verbose=False,
        batch_size=4,
    )
    assert len(history.loss_history) == 2
    assert len(history.accuracy_history) == 2
