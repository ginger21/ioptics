import torch
from typing import Callable, List, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset

from .differentiators import Differentiator


class OpticalTrainer:
    def update_parameters(
        self,
        model: torch.nn.Module,
        gradients: torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, ...],
        optimizer: Optional[torch.optim.Optimizer],
        lr: float,
    ) -> None:
        # Normalize gradients to a single flat tensor
        if isinstance(gradients, (list, tuple)):
            if len(gradients) == 0:
                raise ValueError("Gradients list/tuple cannot be empty")
            for i, grad in enumerate(gradients):
                if not isinstance(grad, torch.Tensor):
                    raise TypeError(f"Gradient element at index {i} must be a torch.Tensor, got {type(grad)}")
            gradients = torch.cat([g.flatten() for g in gradients])
        elif isinstance(gradients, torch.Tensor):
            gradients = gradients.flatten()
        else:
            raise TypeError(f"Gradients must be a tensor or sequence of tensors, got {type(gradients)}")

        # Validate gradient element count against trainable parameters
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if gradients.numel() != trainable_params_count:
            raise ValueError(
                f"Gradient tensor has {gradients.numel()} elements, "
                f"but model has {trainable_params_count} trainable parameters"
            )

        if optimizer is not None:
            optimizer.zero_grad()
            start_idx = 0
            for p in model.parameters():
                if p.requires_grad:
                    end_idx = start_idx + p.numel()
                    p.grad = gradients[start_idx:end_idx].view(p.shape)
                    start_idx = end_idx
            optimizer.step()
            return

        with torch.no_grad():
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            start_idx = 0
            for param in trainable_params:
                end_idx = start_idx + param.numel()
                param -= lr * gradients[start_idx:end_idx].view(param.shape)
                start_idx = end_idx

    def fit_impl(
        self,
        model: torch.nn.Module,
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
    ) -> Tuple[List[float], List[float]]:
        original_training_state = model.training

        try:
            model.train()

            device = model._model_device()
            dataset = TensorDataset(X_input, y_label)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            num_samples = X_input.shape[0]

            for e in range(epochs):
                epoch_loss_t = torch.zeros((), dtype=torch.float64, device=device)
                epoch_correct_t = torch.zeros((), dtype=torch.long, device=device)
                epoch_metric = 0.0

                for batch_X, batch_y in dataloader:
                    if batch_X.device != device:
                        batch_X = batch_X.to(device)
                    if batch_y.device != device:
                        batch_y = batch_y.to(device)

                    gradients = differentiator.differentiate(
                        model, batch_X, loss_function, y_label=batch_y
                    )
                    self.update_parameters(model, gradients, optimizer, lr)

                    with torch.no_grad():
                        outputs = output_getter(batch_X)
                        outputs = model._ensure_batch_dimension(outputs)
                        current_loss = loss_function(outputs, batch_y)
                        batch_len = batch_X.size(0)
                        epoch_loss_t = epoch_loss_t + current_loss.detach() * batch_len

                        if metric_fn is not None:
                            metric_value = model._coerce_metric_scalar(metric_fn(outputs, batch_y))
                            epoch_metric += metric_value * batch_len
                        else:
                            epoch_correct_t = epoch_correct_t + (outputs.argmax(dim=-1) == batch_y).sum()

                avg_loss = epoch_loss_t.item() / num_samples
                if metric_fn is not None:
                    avg_metric = epoch_metric / num_samples
                    model.accuracy_history.append(avg_metric)
                else:
                    avg_accuracy = 100.0 * epoch_correct_t.item() / num_samples
                    model.accuracy_history.append(avg_accuracy)

                model.loss_history.append(avg_loss)

                if verbose:
                    current_acc = model.accuracy_history[-1]
                    print(log_formatter(e + 1, avg_loss, current_acc))
        finally:
            model._restore_training_mode(original_training_state)

        return model.loss_history, model.accuracy_history
