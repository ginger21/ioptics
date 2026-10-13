"""
Differentiators for calculating gradients of model parameters.
"""

import abc

import torch
from torch.distributions import Bernoulli
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .components import bump_global_param_epoch
from .simulator import Simulator

class Differentiator(metaclass=abc.ABCMeta):
    """
    Base class for differentiators.

    A differentiator is responsible for calculating the gradients of a model's
    parameters with respect to some loss function or output.
    """

    @abc.abstractmethod
    def differentiate(self, model, inputs, loss_fn, **kwargs):
        """
        Calculates the gradients of the model's parameters.

        Args:
            model (torch.nn.Module): The model whose parameters' gradients are to be calculated.
            inputs (torch.Tensor): The input data for the model.
            loss_fn (callable): The loss function to use for differentiation.
            **kwargs: Additional keyword arguments for differentiation.

        Returns:
            torch.Tensor: The gradients for the model parameters.
        """
        raise NotImplementedError("Subclasses must implement this method.")

def _validate_common(model, inputs, loss_fn, y_label):
    """
    Common validation helper for differentiators.

    Args:
        model: The model to validate
        inputs: Input tensor
        loss_fn: Loss function to validate
        y_label: Target labels to validate

    Returns:
        List of trainable parameters

    Raises:
        TypeError: If loss_fn is not callable
        ValueError: If y_label is None, no trainable params, or device mismatch
    """
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    if y_label is None:
        raise ValueError("y_label cannot be None")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("model has no trainable parameters")

    device = inputs.device
    if next(iter(trainable_params)).device != device:
        raise ValueError("inputs and model parameters must be on the same device")

    return trainable_params


def _validate_scalar_loss(loss):
    """
    Validate that loss is a scalar tensor.

    Args:
        loss: The loss value to validate

    Raises:
        ValueError: If loss is not a scalar tensor or contains non-finite values
    """
    if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
        raise ValueError("loss_fn must return a scalar tensor")
    if not torch.isfinite(loss):
        raise ValueError("loss must be finite (non-NaN/Inf)")


class BernoulliDifferentiator(Differentiator):
    """
    Differentiator using Bernoulli random vector perturbation for gradient estimation.
    """

    def __init__(self, step_size=0.05, eta=0.02):
        if not (isinstance(step_size, (int, float)) and step_size > 0 and torch.isfinite(torch.tensor(step_size))):
            raise ValueError("step_size must be a finite positive number")
        if not (isinstance(eta, (int, float)) and eta > 0 and torch.isfinite(torch.tensor(eta))):
            raise ValueError("eta must be a finite positive number")
        self.step_size = step_size
        self.eta = eta

    def differentiate(self, model, inputs, loss_fn, y_label=None, **kwargs):
        """
        Args:
            model (torch.nn.Module): The model whose parameters' gradients are to be calculated.
            inputs (torch.Tensor): The input data for the model.
            loss_fn (callable): The loss function to use for differentiation.
            y_label (torch.Tensor): The target labels for the loss function.
            **kwargs: Additional keyword arguments for differentiation.

        Returns:
            gradients (torch.Tensor): The estimated gradients for each parameter.
        """
        trainable_params = _validate_common(model, inputs, loss_fn, y_label)

        params_vector_origin = parameters_to_vector(trainable_params)
        num_params = params_vector_origin.numel()

        # Generate Bernoulli random vector (+1 or -1)
        device = inputs.device
        probs = torch.ones(num_params, device=device) * 0.5
        delta_pi = (Bernoulli(probs).sample() * 2 - 1) * self.step_size

        try:
            # Forward perturbation
            vector_to_parameters(params_vector_origin + delta_pi, trainable_params)
            output_plus = model(inputs)
            loss_plus = loss_fn(output_plus, y_label)
            _validate_scalar_loss(loss_plus)

            # Backward perturbation
            vector_to_parameters(params_vector_origin - delta_pi, trainable_params)
            output_minus = model(inputs)
            loss_minus = loss_fn(output_minus, y_label)
            _validate_scalar_loss(loss_minus)
        finally:
            # Restore original parameters
            vector_to_parameters(params_vector_origin, trainable_params)

        # Gradient estimation
        s = (loss_plus - loss_minus) / (2 * torch.norm(delta_pi))
        gradient = self.eta * s * delta_pi

        return gradient

class PhysicalBernoulliDifferentiator(Differentiator):
    """
    使用物理仿真器 (simulator) 的 Bernoulli 梯度估计器。
    在参数扰动后，通过物理仿真器计算输出，更真实地模拟硬件行为。
    梯度计算公式为：s = (loss_plus - loss_minus) / (2 * ||delta_pi||)
                 gradient = eta * s * delta_pi
    """

    def __init__(self, simulator=None, step_size=0.05, eta=0.02):
        """
        Args:
            simulator: ioptics.simulator.Simulator 实例
            step_size: 参数扰动的步长
            eta: 梯度缩放系数
        """
        if not (isinstance(step_size, (int, float)) and step_size > 0 and torch.isfinite(torch.tensor(step_size))):
            raise ValueError("step_size must be a finite positive number")
        if not (isinstance(eta, (int, float)) and eta > 0 and torch.isfinite(torch.tensor(eta))):
            raise ValueError("eta must be a finite positive number")

        self.simulator = simulator if simulator is not None else Simulator()
        self.step_size = step_size
        self.eta = eta

    def differentiate(self, model, inputs, loss_fn, y_label=None, **kwargs):
        """
        使用物理仿真器进行梯度估计。

        Args:
            model: 光学神经网络模型
            inputs: 输入特征 (功率值，单位 mW)
            loss_fn: 损失函数
            y_label: 目标标签

        Returns:
            gradient: 估计的梯度
        """
        trainable_params = _validate_common(model, inputs, loss_fn, y_label)

        params_vector_origin = parameters_to_vector(trainable_params)
        num_params = params_vector_origin.numel()

        # 生成 Bernoulli 随机向量 (+1 or -1)
        device = inputs.device
        probs = torch.ones(num_params, device=device) * 0.5
        delta_pi = (Bernoulli(probs).sample() * 2 - 1) * self.step_size

        try:
            # 正向扰动
            vector_to_parameters(params_vector_origin + delta_pi, trainable_params)
            # 使用 simulator 进行物理仿真
            output_plus = self.simulator.run(inputs, model)
            loss_plus = loss_fn(output_plus, y_label)
            _validate_scalar_loss(loss_plus)

            # 反向扰动
            vector_to_parameters(params_vector_origin - delta_pi, trainable_params)
            # 使用 simulator 进行物理仿真
            output_minus = self.simulator.run(inputs, model)
            loss_minus = loss_fn(output_minus, y_label)
            _validate_scalar_loss(loss_minus)
        finally:
            # 恢复原始参数
            vector_to_parameters(params_vector_origin, trainable_params)

        # 梯度估计
        s = (loss_plus - loss_minus) / (2 * torch.norm(delta_pi))
        gradient = self.eta * s * delta_pi

        return gradient


class ParameterShiftDifferentiator(Differentiator):
    """
    Differentiator using parameter shift for gradient estimation.

    Uses a two-point parameter shift rule: d/dθ f(θ) ≈ [f(θ+s) - f(θ-s)] / (2*sin(s))
    with fixed shift s = π/2.
    """

    def differentiate(self, model, inputs, loss_fn, y_label=None, **kwargs):
        """
        Args:
            model (torch.nn.Module): The model whose parameters' gradients are to be calculated.
            inputs (torch.Tensor): The input data for the model.
            loss_fn (callable): The loss function to use for differentiation.
            y_label (torch.Tensor): The target labels for the loss function.
            **kwargs: Additional keyword arguments for differentiation.
        Returns:
            gradients (torch.Tensor): The estimated gradients for each trainable parameter scalar,
            with length equal to the total number of trainable parameters.
        """
        trainable_params = _validate_common(model, inputs, loss_fn, y_label)

        # Compute baseline output, loss, and dloss_df once before iterating parameters
        output = model(inputs)
        loss = loss_fn(output, y_label)
        _validate_scalar_loss(loss)
        dloss_df = torch.autograd.grad(loss, output, allow_unused=True)[0]

        grad_list = []

        for param in trainable_params:
            # Process each parameter element separately
            flat_param = param.view(-1)
            for idx in range(flat_param.numel()):
                with torch.no_grad():
                    origin_value = flat_param[idx].detach().clone()
                    try:
                        flat_param[idx].add_(torch.pi / 2)
                        bump_global_param_epoch()
                        output_plus = model(inputs)

                        flat_param[idx].sub_(torch.pi)  # This makes it (original + pi/2 - pi) = (original - pi/2)
                        bump_global_param_epoch()
                        output_minus = model(inputs)

                        df_dtheta = (output_plus - output_minus) / 2

                        dl_dtheta = (dloss_df * df_dtheta).sum()

                        grad_list.append(dl_dtheta)
                    finally:
                        flat_param[idx].copy_(origin_value)
                        bump_global_param_epoch()

        # Gradient estimation - stack all individual gradients
        gradient = torch.stack(grad_list)

        return gradient
