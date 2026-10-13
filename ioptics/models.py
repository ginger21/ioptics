import torch
from typing import Callable, List, Optional, Tuple, NamedTuple

from .differentiators import Differentiator
from .training import OpticalTrainer


class TrainingHistory(NamedTuple):
    """Immutable training history snapshot."""

    loss_history: Tuple[float, ...]
    accuracy_history: Tuple[float, ...]


class OpticalNN(torch.nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.loss_history = []
        self.accuracy_history = []
        self._trainer = OpticalTrainer()

    def _reset_history(self) -> None:
        self.loss_history.clear()
        self.accuracy_history.clear()

    def _snapshot_history(self) -> TrainingHistory:
        return TrainingHistory(
            loss_history=tuple(self.loss_history),
            accuracy_history=tuple(self.accuracy_history),
        )

    def _model_device(self) -> torch.device:
        first_param = next(self.parameters(), None)
        if first_param is None:
            return torch.device("cpu")
        return first_param.device

    @staticmethod
    def _ensure_batch_dimension(outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim == 1:
            return outputs.unsqueeze(0)
        return outputs

    def _compute_batch_stats(
        self,
        outputs: torch.Tensor,
        batch_y: torch.Tensor,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        outputs = self._ensure_batch_dimension(outputs)
        current_loss = loss_function(outputs, batch_y)
        correct = (outputs.argmax(dim=-1) == batch_y).sum().item()
        return current_loss, correct

    def _update_parameters(
        self,
        gradients: torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, ...],
        optimizer: Optional[torch.optim.Optimizer],
        lr: float,
    ) -> None:
        self._trainer.update_parameters(self, gradients, optimizer, lr)

    def _restore_training_mode(self, original_training_state: bool) -> None:
        """Restore the original training mode based on the saved state."""
        if original_training_state:
            self.train()
        else:
            self.eval()

    def _fit_impl(
        self,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        X_input: torch.Tensor,
        y_label: torch.Tensor,
        epochs: int,
        differentiator: Differentiator,
        output_getter: Callable[[torch.Tensor], torch.Tensor],
        log_formatter: Callable[[int, float, float], str],
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]] = None,
        reset_history: bool = False,
    ) -> Tuple[List[float], List[float]]:
        if reset_history:
            self._reset_history()

        return self._trainer.fit_impl(
            model=self,
            loss_function=loss_function,
            X_input=X_input,
            y_label=y_label,
            epochs=epochs,
            differentiator=differentiator,
            output_getter=output_getter,
            log_formatter=log_formatter,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            metric_fn=metric_fn,
        )

    @staticmethod
    def _coerce_metric_scalar(result: float | torch.Tensor) -> float:
        """
        Coerce the result from a metric function to a float scalar.

        Args:
            result: The result from the metric function

        Returns:
            float: The scalar value converted to float

        Raises:
            ValueError: If result is a non-scalar tensor
            TypeError: If result is not a numeric scalar
        """
        if isinstance(result, torch.Tensor):
            if result.dim() != 0:
                raise ValueError("metric_fn must return a scalar")
            # Check if the tensor contains numeric data (excluding complex and boolean)
            if result.dtype.is_complex or result.dtype == torch.bool:
                raise TypeError("metric_fn must return a numeric scalar")
            if not (result.dtype.is_floating_point or result.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]):
                raise TypeError("metric_fn must return a numeric scalar")
            result = result.item()
        elif isinstance(result, bool):  # Explicitly check for bool (since bool is subclass of int)
            raise TypeError("metric_fn must return a numeric scalar")
        elif not isinstance(result, (int, float)):
            # If not tensor and not numeric type, raise error
            raise TypeError("metric_fn must return a numeric scalar")
        return float(result)

    def _evaluate_impl(self, X_input: torch.Tensor, y_label: torch.Tensor,
                      metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]] = None) -> float:
        original_training_state = self.training
        try:
            self.eval()
            device = self._model_device()
            X_input = X_input.to(device)
            y_label = y_label.to(device)

            label_count = y_label.numel()

            if label_count == 0:
                raise ValueError("y_label cannot be empty")

            if X_input.shape[0] != label_count:
                raise ValueError(
                    f"X_input batch size ({X_input.shape[0]}) must match y_label size ({label_count})"
                )

            total = label_count
            with torch.no_grad():
                outputs = torch.abs(self(X_input))
                outputs = self._ensure_batch_dimension(outputs)

                if metric_fn is not None:
                    result = metric_fn(outputs, y_label)
                    return self._coerce_metric_scalar(result)
                else:
                    correct = (outputs.argmax(dim=-1) == y_label).sum().item()
                    return float(100 * correct / total)
        finally:
            self._restore_training_mode(original_training_state)

    def calculate_accuracy(self, X_input: torch.Tensor, y_label: torch.Tensor) -> float:
        """Return classification accuracy percentage for the provided batch."""
        return self._evaluate_impl(X_input, y_label, metric_fn=None)

    def evaluate(self, X_input: torch.Tensor, y_label: torch.Tensor,
                 metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]] = None) -> float:
        """
        Evaluate model outputs with default accuracy or a custom scalar metric.

        Behavior:
        - metric_fn is None: returns accuracy percentage in [0, 100].
        - metric_fn is provided: must return a numeric scalar (Python number or 0-d tensor).

        Raises:
        - ValueError: empty labels, batch-size mismatch, or non-scalar tensor metric.
        - TypeError: non-numeric metric output (e.g. bool, string, complex tensor).
        """
        return self._evaluate_impl(X_input, y_label, metric_fn)

    def fit(self,
            loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            X_input: torch.Tensor,
            y_label: torch.Tensor,
            epochs: int,
            differentiator: Differentiator,
            simulator: Optional[object] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            lr: float = 0.01,
            batch_size: int = 32,
            verbose: bool = True,
            metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]] = None,
            reset_history: bool = True,
            ) -> TrainingHistory:
        """
        Train the optical neural network with the provided data and loss function.

        Args:
            loss_function: Function that computes loss given predictions and targets
            X_input: Input tensor for training
            y_label: Target tensor for training
            epochs: Number of training epochs
            differentiator: Differentiator object for computing gradients
            simulator: Optional Simulator object for hardware-in-the-loop simulation
            optimizer: Optional PyTorch optimizer (uses internal update if None)
            lr: Learning rate for parameter updates
            batch_size: Size of training batches
            verbose: Whether to print training progress
            metric_fn: Optional custom metric function to track during training
            reset_history: Whether to clear previous training history before starting

        Returns:
            TrainingHistory: Immutable snapshot of loss and accuracy history during training
        """
        active_sim = simulator
        if active_sim is None and hasattr(differentiator, 'simulator') and differentiator.simulator is not None:
            active_sim = differentiator.simulator

        if active_sim is not None:
            output_getter = lambda batch_X: active_sim.run(batch_X, self)
            log_formatter = lambda epoch, avg_loss, avg_accuracy: (
                f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%"
            )
        else:
            output_getter = lambda batch_X: torch.abs(self(batch_X))
            log_formatter = lambda epoch, avg_loss, avg_accuracy: (
                f"Epoch {epoch}, Loss: {avg_loss}, Accuracy: {avg_accuracy}"
            )

        self._fit_impl(
            loss_function=loss_function,
            X_input=X_input,
            y_label=y_label,
            epochs=epochs,
            differentiator=differentiator,
            output_getter=output_getter,
            log_formatter=log_formatter,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            metric_fn=metric_fn,
            reset_history=reset_history,
        )
        return self._snapshot_history()
