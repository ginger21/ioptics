import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum
from typing import List, Iterable, cast, Optional, Union
from .components import OpticalComponent, MZI, MZM
from functools import reduce
from .backward_functions import BackwardType, CustomForwardFunction


class ForwardMode(Enum):
    MATRIX = "matrix"
    FIELD_PROP = "field_prop"

class ColumnLayer(torch.nn.Module):
    def __init__(self, N:int, components:List[OpticalComponent]):
        super().__init__()
        self.N = N  # number of input waveguides
        for component in components:
            for port in component.ports:
                if port < 0 or port >= N:
                    raise ValueError(f"Port index {port} out of range [0, {N})")
        self.components = torch.nn.ModuleList(components)
        
    def __iter__(self) -> Iterable[torch.nn.Module]:
        yield from self.components

    def extra_repr(self):
        return f"N={self.N}, components={len(self.components)}"

    def reset_parameters(self) -> None:
        for component in self.components:
            component.reset_parameters()

    def get_transfer_matrix(self) -> torch.Tensor:
        raise NotImplementedError("get_transfer_matrix() must be overriden in child class!")

class MeshColumn(ColumnLayer):
    '''class for physical column of optical components.'''
    def __init__(self, N:int, components:List[OpticalComponent]):
        super().__init__(N, components)
        self._all_mzi = len(self.components) > 0 and all(isinstance(c, MZI) for c in self.components)
        if self._all_mzi:
            m_list = [c.m for c in self.components]
            n_list = [c.n for c in self.components]
            self.register_buffer("_mzi_m_idx", torch.tensor(m_list, dtype=torch.long), persistent=False)
            self.register_buffer("_mzi_n_idx", torch.tensor(n_list, dtype=torch.long), persistent=False)
        else:
            self._mzi_m_idx = None
            self._mzi_n_idx = None

    def get_transfer_matrix(self) -> torch.Tensor:
        if len(self.components) == 0:
            return torch.eye(self.N, dtype=torch.complex64)

        if self._all_mzi:
            thetas = []
            phis = []
            for c in self.components:
                c_apply_noise = not self.training or c.noise_aware_training
                if c_apply_noise:
                    thetas.append(c.theta + c.phase_noise_theta.to(c.theta.device, c.theta.dtype))
                    phis.append(c.phi + c.phase_noise_phi.to(c.phi.device, c.phi.dtype))
                else:
                    thetas.append(c.theta)
                    phis.append(c.phi)

            thetas = torch.cat([t.view(1) for t in thetas])
            phis = torch.cat([p.view(1) for p in phis])
            _1j = self.components[0]._1j

            exp_jphi = torch.exp(_1j * phis)
            exp_jtheta = torch.exp(_1j * thetas)

            t00 = 0.5 * exp_jphi * (exp_jtheta - 1.0)
            t01 = 0.5 * _1j * exp_jphi * (1.0 + exp_jtheta)
            t10 = 0.5 * _1j * (exp_jtheta + 1.0)
            t11 = 0.5 * (1.0 - exp_jtheta)

            T = torch.eye(self.N, dtype=t00.dtype, device=t00.device)
            m = self._mzi_m_idx.to(device=T.device)
            n = self._mzi_n_idx.to(device=T.device)
            T[m, m] = t00
            T[m, n] = t01
            T[n, m] = t10
            T[n, n] = t11
            return T

        # General fallback for mixed or custom components
        first_T = self.components[0].get_transfer_matrix()
        transfer_matrix = torch.eye(
            self.N,
            dtype=first_T.dtype,
            device=first_T.device,
        )

        for inx, component in enumerate(self.components):
            T = first_T if inx == 0 else component.get_transfer_matrix()
            T = T.to(device=transfer_matrix.device, dtype=transfer_matrix.dtype)
            if component.dof == 1:
                m = component.m
                transfer_matrix[m, m] = T
            elif component.dof == 2:
                m = component.m
                n = component.n
                transfer_matrix[m, m] = T[0, 0]
                transfer_matrix[m, n] = T[0, 1]
                transfer_matrix[n, m] = T[1, 0]
                transfer_matrix[n, n] = T[1, 1]
            else:
                raise ValueError(f"The dof must be 1 or 2, now it is {component.dof}!")
        return transfer_matrix

class ArrayColumn(ColumnLayer):
    """A column of optical components producing a single output row (length N).

    In the weight-matrix view (L x N), each ArrayColumn produces one row of the
    weight matrix. L (number of columns/outputs) may differ from N (inputs).
    """
    def __init__(self, N:int, components:List[OpticalComponent]):
        super().__init__(N, components)

    def get_transfer_matrix(self) -> torch.Tensor:
        # Returns a 1D tensor of shape (N,) representing one row of the weight matrix
        if len(self.components) == 0:
            return torch.zeros(self.N, dtype=torch.complex64)

        first_T = self.components[0].get_transfer_matrix()
        transfer_matrix = torch.zeros(
            self.N,
            dtype=first_T.dtype,
            device=first_T.device,
        )

        #transfer_matrix_list = []
        #componets_list = sorted(self.components, key=lambda component:component.m)
        for inx, component in enumerate(self.components):
            # component = cast(OpticalComponent, component)
            T = first_T if inx == 0 else component.get_transfer_matrix()
            T = T.to(device=transfer_matrix.device, dtype=transfer_matrix.dtype)
            if component.dof == 1:
                m = component.m
                # m = cast(int, m)
                transfer_matrix[m] = T
                # transfer_matrix_list.append(T)
            elif component.dof == 2:
                raise ValueError(f"The dof must be 1, now it is {component.dof}!")
            # transfer_matrix = torch.stack(transfer_matrix_list)
        return transfer_matrix


class CoreLayer(torch.nn.Module):
    def __init__(
        self,
        L:int,
        columnlayers:List[ColumnLayer],
        output_ports:Optional[slice] = None,
        backward_type: str = BackwardType.DEFAULT.value,
        forward_mode: str = ForwardMode.MATRIX.value,
    ):
        super().__init__()
        self.L = L
        self.output_ports = self._normalize_output_ports(output_ports, L)

        if backward_type not in [bt.value for bt in BackwardType]:
            raise ValueError(f"Unsupported backward_type: {backward_type}")
        self.backward_type = backward_type

        if forward_mode not in [fm.value for fm in ForwardMode]:
            raise ValueError(f"Unsupported forward_mode: {forward_mode}")
        self.forward_mode = forward_mode

        self.columnlayers = torch.nn.ModuleList(columnlayers)

    def _normalize_output_ports(self, output_ports: Optional[object], L: int) -> Union[slice, List[int]]:
        if output_ports is None:
            return slice(0, L)
        elif isinstance(output_ports, slice):
            # Validate slice bounds
            start = output_ports.start if output_ports.start is not None else 0
            stop = output_ports.stop if output_ports.stop is not None else L

            if output_ports.step == 0:
                raise ValueError("Slice step cannot be zero")

            if start < 0 or start > L or stop < 0 or stop > L:
                raise ValueError(f"Slice bounds [{start}:{stop}] out of range [0, {L}]")

            return output_ports
        elif isinstance(output_ports, (list, tuple)):
            # Convert to list to maintain order as provided (don't sort)
            output_ports_list = list(output_ports)
            for idx in output_ports_list:
                if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= L:
                    raise ValueError(f"Index {idx} out of range [0, {L})")
            return output_ports_list
        else:
            raise TypeError(f"output_ports must be slice, list, tuple, or None, got {type(output_ports)}")

    def __iter__(self) -> Iterable[torch.nn.Module]:
        yield from self.columnlayers

    def extra_repr(self):
        return f"L={self.L}, output_ports={self.output_ports}, backward_type={self.backward_type}, forward_mode={self.forward_mode}"

    def reset_parameters(self) -> None:
        for columnlayer in self.columnlayers:
            columnlayer.reset_parameters()

    def get_transfer_matrix(self) -> torch.Tensor:
        raise NotImplementedError("get_transfer_matrix() must be overriden in child class!")

    def _default_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:

        if input_tensor.ndim < 1:
            raise ValueError(
                f"Input tensor must have at least 1 dimension, got shape {tuple(input_tensor.shape)}"
            )

        if not torch.is_complex(input_tensor):
            if input_tensor.dtype == torch.float64:
                input_tensor = input_tensor.to(torch.complex128)
            else:
                input_tensor = input_tensor.to(torch.complex64)

        transfer_matrix = self.get_transfer_matrix().to(device=input_tensor.device, dtype=input_tensor.dtype)

        if transfer_matrix.ndim != 2:
            raise ValueError(
                f"Transfer matrix must be 2D, got shape {tuple(transfer_matrix.shape)}"
            )

        if transfer_matrix.shape[0] != self.L:
            raise ValueError(
                f"Transfer matrix first dimension size ({transfer_matrix.shape[0]}) "
                f"does not match layer output size L ({self.L})"
            )

        if input_tensor.shape[-1] != transfer_matrix.shape[1]:
            raise ValueError(
                f"Input tensor last dimension size ({input_tensor.shape[-1]}) "
                f"does not match transfer matrix second dimension size ({transfer_matrix.shape[1]})"
            )

        output_tensor = F.linear(input_tensor, transfer_matrix)

        return output_tensor[..., self.output_ports]
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.backward_type == BackwardType.DEFAULT.value:
            # 直接计算，使用PyTorch自动微分
            return self._default_forward(input_tensor)
        else:
            # 使用自定义反向传播
            return CustomForwardFunction.apply(self, input_tensor, self.backward_type, *self.parameters())

class MeshCore(CoreLayer):
    '''class for physical mesh of optical columnlayers.'''
    def __init__(
        self,
        L:int,
        columnlayers:List[ColumnLayer],
        output_ports:Optional[slice] = None,
        backward_type: str = BackwardType.DEFAULT.value,
        forward_mode: str = ForwardMode.MATRIX.value,
    ):
        super().__init__(L, columnlayers, output_ports, backward_type, forward_mode)

    def _field_prop_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        current_field = input_tensor
        for layer in self.columnlayers:
            T_col = layer.get_transfer_matrix().to(device=current_field.device, dtype=current_field.dtype)
            current_field = F.linear(current_field, T_col)
        return current_field[..., self.output_ports]

    def _default_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.forward_mode == ForwardMode.FIELD_PROP.value:
            if input_tensor.ndim < 1:
                raise ValueError(
                    f"Input tensor must have at least 1 dimension, got shape {tuple(input_tensor.shape)}"
                )
            if not torch.is_complex(input_tensor):
                if input_tensor.dtype == torch.float64:
                    input_tensor = input_tensor.to(torch.complex128)
                else:
                    input_tensor = input_tensor.to(torch.complex64)
            if input_tensor.shape[-1] != self.L:
                raise ValueError(
                    f"Input tensor last dimension size ({input_tensor.shape[-1]}) "
                    f"does not match layer input size ({self.L})"
                )
            return self._field_prop_forward(input_tensor)
        return super()._default_forward(input_tensor)

    def get_transfer_matrix(self) -> torch.Tensor:
        if len(self.columnlayers) == 0:
            raise ValueError("columnlayers cannot be empty")

        # reversed: signal propagates through columnlayers in order, so later layers multiply on the left
        transfer_matrix = reduce(torch.matmul,
                [layer.get_transfer_matrix() for layer in reversed(self.columnlayers)])
        return transfer_matrix

class ClementsMesh(MeshCore):
    def __init__(
        self,
        L:int,
        output_ports: Optional[slice] = None,
        phase_noise_std: float = 0.0,
        noise_aware_training: bool = False,
        backward_type: str = BackwardType.DEFAULT.value,
        forward_mode: str = ForwardMode.MATRIX.value,
    ):
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")

        self.phase_noise_std = phase_noise_std
        self.noise_aware_training = noise_aware_training
        columnlayer_list = []
        for layer_index in range(L):
            if L % 2 == 0:  # even number of waveguides
                if layer_index % 2 == 0:
                    columnlayer = MeshColumn(L, [MZI(m, m+1, phase_noise_std=phase_noise_std, noise_aware_training=noise_aware_training) for m in range(0, L, 2)])
                else:
                    columnlayer = MeshColumn(L, [MZI(m, m+1, phase_noise_std=phase_noise_std, noise_aware_training=noise_aware_training) for m in range(1, L-1, 2)])
            else:  # odd number of waveguides
                if layer_index % 2 == 0:
                    columnlayer = MeshColumn(L, [MZI(m, m+1, phase_noise_std=phase_noise_std, noise_aware_training=noise_aware_training) for m in range(0, L-1, 2)])
                else:
                    columnlayer = MeshColumn(L, [MZI(m, m+1, phase_noise_std=phase_noise_std, noise_aware_training=noise_aware_training) for m in range(1, L, 2)])
            columnlayer_list.append(columnlayer)
        super().__init__(L, columnlayer_list, output_ports = output_ports, backward_type = backward_type, forward_mode = forward_mode)

    def extra_repr(self):
        return f"L={self.L}, phase_noise_std={self.phase_noise_std}, noise_aware_training={self.noise_aware_training}, backward_type={self.backward_type}, forward_mode={self.forward_mode}"

class ArrayCore(CoreLayer):
    """Core layer for array-based optical computation using weight-matrix (L x N) semantics.

    Similar to a fully-connected layer: L output ports (rows) each receive weighted
    sums from N input ports (columns). L may differ from N. Each ArrayColumn
    produces one row of the weight matrix.
    """
    def __init__(self, L:int, columnlayers:List[ColumnLayer], output_ports:Optional[slice] = None, backward_type: str = BackwardType.DEFAULT.value):
        super().__init__(L, columnlayers, output_ports, backward_type)

    def extra_repr(self):
        n_value = self.columnlayers[0].N if self.columnlayers else "n/a"
        return f"L={self.L}, N={n_value}"

    def get_transfer_matrix(self) -> torch.Tensor:
        if len(self.columnlayers) == 0:
            raise ValueError("columnlayers cannot be empty")

        # self.L = cast(int, self.L)
        # n = self.columnlayers[0].N
        # n = cast(int, n)
        first_T = self.columnlayers[0].get_transfer_matrix()
        # Weight matrix shape: (L rows x N cols). Each columnlayer produces one row.
        transfer_matrix = torch.zeros(
            (self.L, self.columnlayers[0].N),
            dtype=first_T.dtype,
            device=first_T.device,
        )
        for inx, layer in enumerate(self.columnlayers):
            # layer = cast(ColumnLayer, layer)
            T = first_T if inx == 0 else layer.get_transfer_matrix()
            T = T.to(device=transfer_matrix.device, dtype=transfer_matrix.dtype)
            transfer_matrix[inx] = T
        return transfer_matrix

class FCArray(ArrayCore):
    """Fully-connected optical array: maps in_dim inputs to out_dim outputs.

    Each output channel (row) is implemented as an ArrayColumn containing in_dim
    MZM modulators. Each MZM modulates a single input channel with no cross-channel
    coupling (independent, element-wise modulation). The resulting rows collectively
    form an LxN weight matrix (L = out_dim rows, N = in_dim columns), producing
    the same linear transformation as a classical fully-connected layer.
    """
    def __init__(self, in_dim:int, out_dim:int, output_ports:Optional[slice] = None, backward_type: str = BackwardType.DEFAULT.value, noise_aware_training: bool = False):
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_aware_training = noise_aware_training
        columnlayer_list = []
        # Build out_dim columns, each receiving in_dim inputs (L=out_dim, N=in_dim)
        for layer_index in range(out_dim):
            columnlayer = ArrayColumn(in_dim, [MZM(m, noise_aware_training=noise_aware_training) for m in range(in_dim)])
            columnlayer_list.append(columnlayer)

        super().__init__(out_dim, columnlayer_list, output_ports, backward_type)

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, noise_aware_training={self.noise_aware_training}, backward_type={self.backward_type}"

