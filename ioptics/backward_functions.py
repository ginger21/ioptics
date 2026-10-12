import torch
import torch.nn.functional as F
from enum import Enum
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


def _sum_all_dims(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    return tensor.sum(dim=tuple(range(tensor.ndim)))


class BackwardType(Enum):
    PARAMETER_SHIFT = "parameter shift"
    ADJOINT_VARIANT = "adjoint variant"
    DEFAULT = "default"


class BackwardStrategy(ABC):
    """反向传播策略抽象基类"""

    @abstractmethod
    def compute_gradients(
        self,
        grad_output: torch.Tensor,
        optical_layer,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        saved_forward_transfer_matrix: Optional[torch.Tensor] = None,
        parameter_refs: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        pass


class ParameterShiftStrategy(BackwardStrategy):
    def compute_gradients(
        self,
        grad_output: torch.Tensor,
        optical_layer,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        saved_forward_transfer_matrix: Optional[torch.Tensor] = None,
        parameter_refs: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        transfer_matrix = (
            saved_forward_transfer_matrix
            if saved_forward_transfer_matrix is not None
            else optical_layer.get_transfer_matrix()
        )
        transfer_matrix = transfer_matrix.to(device=grad_output.device, dtype=grad_output.dtype)
        grad_transfer_matrix = transfer_matrix[optical_layer.output_ports, :]

        grad_input = F.linear(grad_output, grad_transfer_matrix.conj().T)

        params = parameter_refs if parameter_refs is not None else tuple(optical_layer.parameters())
        grad_list = []

        with torch.no_grad():
            for param in params:
                origin_value = param.data.clone()
                try:
                    param.data.copy_(origin_value + torch.pi / 2)
                    output_plus = optical_layer(input_tensor)

                    df_dtheta = (output_plus - output_tensor) * (1 - 1j) / 2
                    grad_theta = _sum_all_dims(grad_output.conj() * df_dtheta).real

                    grad_list.append(
                        grad_theta.to(device=param.device, dtype=param.dtype).reshape_as(param)
                    )
                finally:
                    param.data.copy_(origin_value)

        return grad_input, *grad_list


class AdjointStrategy(BackwardStrategy):
    """伴随方法（Adjoint Method）反向传播策略

    正向传播时记录每层输入端光场 A，
    反向传播时逐层回传伴随场 A*，
    利用 Re(adj^H · ∂T/∂param · a_fwd) 计算各相位参数梯度。
    """

    def compute_gradients(
        self,
        grad_output: torch.Tensor,
        optical_layer,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        saved_forward_transfer_matrix: Optional[torch.Tensor] = None,
        parameter_refs: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        from .components import MZI, PhaseShifter, MZM

        # 确定复数精度：如果输入或输出为float64或complex128，则使用complex128，否则使用complex64
        use_complex128 = (
            input_tensor.dtype == torch.float64
            or input_tensor.dtype == torch.complex128
            or grad_output.dtype == torch.complex128
        )
        internal_dtype = torch.complex128 if use_complex128 else torch.complex64

        _1j = torch.tensor(1j, dtype=internal_dtype)

        with torch.no_grad():
            device = input_tensor.device

            # 只在需要时进行类型转换，保持与输入一致的精度
            if input_tensor.dtype != internal_dtype and not torch.is_complex(input_tensor):
                input_tensor = input_tensor.to(internal_dtype)
            elif torch.is_complex(input_tensor) and input_tensor.dtype != internal_dtype:
                input_tensor = input_tensor.to(internal_dtype)

            if grad_output.dtype != internal_dtype:
                grad_output = grad_output.to(internal_dtype)

            L = optical_layer.L

            # === 正向传播：逐层记录每个 column 输入端的光场 ===
            forward_fields = []
            current_field = input_tensor

            for column in optical_layer.columnlayers:
                forward_fields.append(current_field)
                T_col = column.get_transfer_matrix().to(device, dtype=internal_dtype)
                current_field = F.linear(current_field, T_col)

            # === 初始化伴随场：将 grad_output 扩展到完整维度 ===
            full_grad = torch.zeros(
                *input_tensor.shape[:-1], L,
                dtype=internal_dtype, device=device
            )
            full_grad[..., optical_layer.output_ports] = grad_output
            adjoint_field = full_grad

            # === 反向传播伴随场并计算梯度 ===
            all_grads = []
            param_grad_map: Dict[int, torch.Tensor] = {}

            for i in reversed(range(len(optical_layer.columnlayers))):
                column = optical_layer.columnlayers[i]
                A_fwd = forward_fields[i]

                layer_grads = []

                for component in column.components:
                    if isinstance(component, MZI):
                        m, n = component.m, component.n
                        theta = (component.theta + component.phase_noise_theta).to(internal_dtype)
                        phi = (component.phi + component.phase_noise_phi).to(internal_dtype)

                        a = torch.stack([A_fwd[..., m], A_fwd[..., n]], dim=-1)
                        adj = torch.stack([adjoint_field[..., m], adjoint_field[..., n]], dim=-1)

                        e_itheta = torch.exp(_1j * theta)
                        e_iphi = torch.exp(_1j * phi)
                        e_i_pt = e_iphi * e_itheta

                        # ∂T/∂θ
                        dT_dtheta = torch.zeros(2, 2, dtype=internal_dtype, device=device)
                        dT_dtheta[0, 0] = 0.5 * _1j * e_i_pt
                        dT_dtheta[0, 1] = -0.5 * e_i_pt
                        dT_dtheta[1, 0] = -0.5 * e_itheta
                        dT_dtheta[1, 1] = -0.5 * _1j * e_itheta

                        # ∂T/∂φ
                        T_comp = component.get_transfer_matrix().to(device, dtype=internal_dtype)
                        dT_dphi = torch.zeros(2, 2, dtype=internal_dtype, device=device)
                        dT_dphi[0, 0] = _1j * T_comp[0, 0]
                        dT_dphi[0, 1] = _1j * T_comp[0, 1]

                        # grad = Re(adj^H · ∂T/∂param · a)
                        grad_theta = _sum_all_dims(adj.conj() * F.linear(a, dT_dtheta)).real
                        grad_phi = _sum_all_dims(adj.conj() * F.linear(a, dT_dphi)).real

                        grad_theta_param = grad_theta.reshape_as(component.theta)
                        grad_phi_param = grad_phi.reshape_as(component.phi)
                        param_grad_map[id(component.theta)] = grad_theta_param
                        param_grad_map[id(component.phi)] = grad_phi_param
                        layer_grads.extend([grad_theta_param, grad_phi_param])

                    elif isinstance(component, PhaseShifter):
                        m = component.m
                        phi = (component.phi + component.phase_noise_phi).to(internal_dtype)

                        a_m = A_fwd[..., m]
                        adj_m = adjoint_field[..., m]

                        grad_phi = _sum_all_dims(adj_m.conj() * _1j * torch.exp(_1j * phi) * a_m).real
                        grad_phi_param = grad_phi.reshape_as(component.phi)
                        param_grad_map[id(component.phi)] = grad_phi_param
                        layer_grads.append(grad_phi_param)

                    elif isinstance(component, MZM):
                        m = component.m
                        theta = (component.theta + component.phase_noise_theta).to(internal_dtype)

                        a_m = A_fwd[..., m]
                        adj_m = adjoint_field[..., m]

                        dT_dtheta = 0.5 * _1j * torch.exp(_1j * theta)
                        grad_theta = _sum_all_dims(adj_m.conj() * dT_dtheta * a_m).real
                        grad_theta_param = grad_theta.reshape_as(component.theta)
                        param_grad_map[id(component.theta)] = grad_theta_param
                        layer_grads.append(grad_theta_param)

                all_grads = layer_grads + all_grads

                # 反向传播伴随场: A* = T^H · A*
                T_col = column.get_transfer_matrix().to(device, dtype=internal_dtype)
                adjoint_field = F.linear(adjoint_field, T_col.conj().T)

            if saved_forward_transfer_matrix is not None:
                grad_transfer_matrix = saved_forward_transfer_matrix[
                    optical_layer.output_ports, :
                ].to(device=device, dtype=internal_dtype)
                grad_input = F.linear(grad_output, grad_transfer_matrix.conj().T)
            else:
                grad_input = adjoint_field

            if parameter_refs is not None:
                ordered_grads = []
                for param in parameter_refs:
                    grad_param = param_grad_map.get(id(param))
                    if grad_param is None:
                        grad_param = torch.zeros_like(param)
                    else:
                        grad_param = grad_param.to(device=param.device, dtype=param.dtype).reshape_as(param)
                    ordered_grads.append(grad_param)
                return grad_input, *ordered_grads

        return grad_input, *all_grads


class CustomForwardFunction(torch.autograd.Function):
    # 策略实例字典
    _strategies: Dict[str, BackwardStrategy] = {
        BackwardType.PARAMETER_SHIFT.value: ParameterShiftStrategy(),
        BackwardType.ADJOINT_VARIANT.value: AdjointStrategy(),
    }

    @staticmethod
    def forward(ctx, optical_layer, input_tensor, backward_type, *parameters):
        ctx.optical_layer = optical_layer
        ctx.backward_type = backward_type

        if not torch.is_complex(input_tensor):
            if input_tensor.dtype == torch.float64:
                input_tensor_converted = input_tensor.to(torch.complex128)
            else:
                input_tensor_converted = input_tensor.to(torch.complex64)
        else:
            input_tensor_converted = input_tensor

        transfer_matrix = optical_layer.get_transfer_matrix().to(
            device=input_tensor_converted.device, dtype=input_tensor_converted.dtype
        )

        if input_tensor_converted.shape[-1] != transfer_matrix.shape[1]:
            raise ValueError(
                f"Input tensor last dimension size ({input_tensor_converted.shape[-1]}) "
                f"does not match transfer matrix second dimension size ({transfer_matrix.shape[1]})"
            )

        output_tensor = F.linear(input_tensor_converted, transfer_matrix)
        selected_output = output_tensor[..., optical_layer.output_ports]

        ctx.save_for_backward(input_tensor, selected_output, transfer_matrix, *parameters)

        return selected_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, output_tensor, transfer_matrix, *parameter_refs = ctx.saved_tensors

        optical_layer = ctx.optical_layer
        backward_type = ctx.backward_type

        strategy = CustomForwardFunction._strategies[backward_type]

        grad_input, *grad_list = strategy.compute_gradients(
            grad_output,
            optical_layer,
            input_tensor,
            output_tensor,
            saved_forward_transfer_matrix=transfer_matrix,
            parameter_refs=tuple(parameter_refs),
        )

        # 确保 grad_input 的 dtype 和 complex 属性与原始输入保持一致
        if torch.is_complex(input_tensor):
            # 如果原始输入是复数，则保持复数
            if grad_input.dtype != input_tensor.dtype:
                grad_input = grad_input.to(dtype=input_tensor.dtype)
        else:
            # 如果原始输入是实数，则取实部并转换为相应实数类型
            if input_tensor.dtype == torch.float64:
                grad_input = grad_input.real.to(dtype=torch.float64)
            else:  # 默认转换为 float32
                grad_input = grad_input.real.to(dtype=torch.float32)

        return None, grad_input, None, *grad_list
