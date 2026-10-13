import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum
from typing import List, Iterable, cast, Optional, Union, Tuple
from .components import OpticalComponent, MZI, MZM, PhaseShifter, get_global_param_epoch
from functools import reduce
from .backward_functions import BackwardType, CustomForwardFunction


_get_tensor_version = torch.Tensor._version.__get__
_get_tensor_item = torch.Tensor.item


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
        self.cache_enabled: bool = True
        self._col_params: Optional[Tuple[torch.Tensor, ...]] = None
        self._col_any_requires_grad: bool = False
        self._cached_col_tm: Optional[torch.Tensor] = None
        self._cached_col_sig: Optional[tuple] = None
        self._cached_col_has_grad: bool = False
        self._cached_col_epoch: int = -1
        self._base_col_tm: Optional[torch.Tensor] = None
        self._base_col_sig: Optional[tuple] = None
        self._base_col_has_grad: bool = False

    def _init_col_cache_tracking(self) -> None:
        params = tuple(self.parameters())
        self._col_params = params
        self._col_any_requires_grad = any(p.requires_grad for p in params)

    def _get_col_signature(self) -> tuple:
        if self._col_params is None:
            self._init_col_cache_tracking()
        params = self._col_params
        if not params:
            return (self.training,)
        p0 = params[0]
        if p0.is_cpu:
            return (
                self.training,
                p0.data_ptr(),
                params[-1].data_ptr(),
                tuple(map(_get_tensor_item, params)),
            )
        return (
            self.training,
            p0.data_ptr(),
            params[-1].data_ptr(),
            tuple(map(_get_tensor_version, params)),
        )

    def pin_base_cache(self) -> None:
        self._base_col_tm = self._cached_col_tm
        self._base_col_sig = self._cached_col_sig
        self._base_col_has_grad = self._cached_col_has_grad

    def _on_col_tm_backward(self, _grad: torch.Tensor) -> None:
        self._cached_col_tm = None
        self._cached_col_sig = None
        self._cached_col_has_grad = False
        self._cached_col_epoch = -1
        self._base_col_tm = None
        self._base_col_sig = None
        self._base_col_has_grad = False

    def invalidate_cache(self, reset_tracking: bool = False) -> None:
        self._cached_col_tm = None
        self._cached_col_sig = None
        self._cached_col_has_grad = False
        self._cached_col_epoch = -1
        self._base_col_tm = None
        self._base_col_sig = None
        self._base_col_has_grad = False
        if reset_tracking:
            self._col_params = None

    def train(self, mode: bool = True):
        if self.training != mode:
            self.invalidate_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.invalidate_cache(reset_tracking=True)
        return super()._apply(fn)

    def __iter__(self) -> Iterable[torch.nn.Module]:
        yield from self.components

    def extra_repr(self):
        return f"N={self.N}, components={len(self.components)}"

    def reset_parameters(self) -> None:
        for component in self.components:
            component.reset_parameters()
        self.invalidate_cache()

    def _compute_transfer_matrix(self) -> torch.Tensor:
        raise NotImplementedError("get_transfer_matrix() must be overriden in child class!")

    def get_transfer_matrix(self, use_cache: bool = True) -> torch.Tensor:
        if not use_cache or not self.cache_enabled or torch.compiler.is_compiling():
            return self._compute_transfer_matrix()

        if self._col_params is None:
            self._init_col_cache_tracking()

        need_grad = torch.is_grad_enabled() and self._col_any_requires_grad
        sig = self._get_col_signature()

        if self._cached_col_tm is not None and (not need_grad or self._cached_col_has_grad):
            if sig == self._cached_col_sig:
                return self._cached_col_tm

        if self._base_col_tm is not None and (not need_grad or self._base_col_has_grad):
            if sig == self._base_col_sig:
                self._cached_col_tm = self._base_col_tm
                self._cached_col_sig = self._base_col_sig
                self._cached_col_has_grad = self._base_col_has_grad
                return self._cached_col_tm

        tm = self._compute_transfer_matrix()
        sig_after = self._get_col_signature()
        self._cached_col_tm = tm
        self._cached_col_sig = sig_after
        self._cached_col_has_grad = tm.requires_grad
        if tm.requires_grad:
            tm.register_hook(self._on_col_tm_backward)
        return tm

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
            m_t = self._mzi_m_idx
            n_t = self._mzi_n_idx
            scatter_idx = torch.cat([m_t * N + m_t, m_t * N + n_t, n_t * N + m_t, n_t * N + n_t])
            self.register_buffer("_mzi_scatter_idx", scatter_idx, persistent=False)
            base_eye = torch.eye(N, dtype=torch.complex64)
            for p in set(m_list + n_list):
                base_eye[p, p] = 0.0
            self.register_buffer("_base_eye_flat", base_eye.reshape(-1), persistent=False)
        else:
            self._mzi_m_idx = None
            self._mzi_n_idx = None
            self._mzi_scatter_idx = None
            self._base_eye_flat = None

    def _compute_transfer_matrix(self) -> torch.Tensor:
        if len(self.components) == 0:
            return torch.eye(self.N, dtype=torch.complex64)

        if self._all_mzi:
            thetas = [
                (c.theta + c.phase_noise_theta).view(1)
                if (not self.training or c.noise_aware_training)
                else c.theta.view(1)
                for c in self.components
            ]
            phis = [
                (c.phi + c.phase_noise_phi).view(1)
                if (not self.training or c.noise_aware_training)
                else c.phi.view(1)
                for c in self.components
            ]

            theta_vec = torch.cat(thetas)
            phi_vec = torch.cat(phis)

            exp_jphi = torch.exp(1j * phi_vec)
            exp_jtheta = torch.exp(1j * theta_vec)

            t00 = 0.5 * exp_jphi * (exp_jtheta - 1.0)
            t01 = 0.5j * exp_jphi * (1.0 + exp_jtheta)
            t10 = 0.5j * (exp_jtheta + 1.0)
            t11 = 0.5 * (1.0 - exp_jtheta)

            vals = torch.cat([t00, t01, t10, t11])
            base_flat = (
                self._base_eye_flat
                if (self._base_eye_flat.dtype == vals.dtype and self._base_eye_flat.device == vals.device)
                else self._base_eye_flat.to(device=vals.device, dtype=vals.dtype)
            )
            scatter_idx = (
                self._mzi_scatter_idx
                if self._mzi_scatter_idx.device == vals.device
                else self._mzi_scatter_idx.to(device=vals.device)
            )
            return base_flat.scatter(0, scatter_idx, vals).view(self.N, self.N)

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
        self._all_mzm = len(self.components) > 0 and all(isinstance(c, MZM) for c in self.components)
        self._all_ps = len(self.components) > 0 and all(isinstance(c, PhaseShifter) for c in self.components)
        if self._all_mzm or self._all_ps:
            m_list = [c.m for c in self.components]
            self._is_contiguous_full = (
                len(m_list) == N and m_list == list(range(N))
            )
            self.register_buffer("_comp_m_idx", torch.tensor(m_list, dtype=torch.long), persistent=False)
        else:
            self._is_contiguous_full = False
            self._comp_m_idx = None

    def _compute_transfer_matrix(self) -> torch.Tensor:
        # Returns a 1D tensor of shape (N,) representing one row of the weight matrix
        if len(self.components) == 0:
            return torch.zeros(self.N, dtype=torch.complex64)

        if self._all_mzm:
            thetas = [
                (c.theta + c.phase_noise_theta).view(1)
                if (not self.training or c.noise_aware_training)
                else c.theta.view(1)
                for c in self.components
            ]
            theta_vec = torch.cat(thetas)
            vals = 0.5 * (torch.exp(1j * theta_vec) + 1.0)
            if self._is_contiguous_full:
                return vals
            idx = self._comp_m_idx if self._comp_m_idx.device == vals.device else self._comp_m_idx.to(device=vals.device)
            return torch.zeros(self.N, dtype=vals.dtype, device=vals.device).scatter(0, idx, vals)

        if self._all_ps:
            phis = [
                (c.phi + c.phase_noise_phi).view(1)
                if (not self.training or c.noise_aware_training)
                else c.phi.view(1)
                for c in self.components
            ]
            phi_vec = torch.cat(phis)
            vals = torch.exp(1j * phi_vec)
            if self._is_contiguous_full:
                return vals
            idx = self._comp_m_idx if self._comp_m_idx.device == vals.device else self._comp_m_idx.to(device=vals.device)
            return torch.zeros(self.N, dtype=vals.dtype, device=vals.device).scatter(0, idx, vals)

        first_T = self.components[0].get_transfer_matrix()
        transfer_matrix = torch.zeros(
            self.N,
            dtype=first_T.dtype,
            device=first_T.device,
        )

        for inx, component in enumerate(self.components):
            T = first_T if inx == 0 else component.get_transfer_matrix()
            T = T.to(device=transfer_matrix.device, dtype=transfer_matrix.dtype)
            if component.dof == 1:
                m = component.m
                transfer_matrix[m] = T
            elif component.dof == 2:
                raise ValueError(f"The dof must be 1, now it is {component.dof}!")
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
        self.cache_enabled: bool = True
        self._cached_transfer_matrix: Optional[torch.Tensor] = None
        self._cache_signature: Optional[tuple] = None
        self._cache_has_grad: bool = False
        self._cache_all_params: Optional[Tuple[torch.Tensor, ...]] = None
        self._cache_any_requires_grad: bool = False
        self._base_col_tms: Optional[List[torch.Tensor]] = None
        self._base_suffix: Optional[List[Optional[torch.Tensor]]] = None
        self._base_prefix: Optional[List[Optional[torch.Tensor]]] = None
        self._base_tm: Optional[torch.Tensor] = None
        self._base_has_grad: bool = False

    def _init_cache_tracking(self) -> None:
        all_params = tuple(self.parameters())
        self._cache_all_params = all_params
        if len(all_params) <= 256:
            self._cache_eval_params = all_params
        else:
            step = max(1, len(all_params) // 32)
            self._cache_eval_params = all_params[::step] + (all_params[-1],)
        self._cache_any_requires_grad = any(p.requires_grad for p in all_params)

    def _get_cache_signature(self) -> tuple:
        if self._cache_all_params is None:
            self._init_cache_tracking()
        all_params = self._cache_all_params
        if not all_params:
            return (self.training, get_global_param_epoch())
        tracked = all_params if self.training else self._cache_eval_params
        p0 = all_params[0]
        p_last = all_params[-1]
        return (
            self.training,
            get_global_param_epoch(),
            p0.data_ptr(),
            p_last.data_ptr(),
            tuple(map(_get_tensor_version, tracked)),
        )

    def _clear_matrix_cache_only(self) -> None:
        self._cached_transfer_matrix = None
        self._cache_signature = None
        self._cache_has_grad = False
        self._base_col_tms = None
        self._base_suffix = None
        self._base_prefix = None
        self._base_tm = None
        self._base_has_grad = False

    def _on_cached_tm_backward(self, _grad: torch.Tensor) -> None:
        self._clear_matrix_cache_only()

    def invalidate_cache(self, reset_tracking: bool = False) -> None:
        self._clear_matrix_cache_only()
        if reset_tracking:
            self._cache_all_params = None
        for col in self.columnlayers:
            if hasattr(col, "invalidate_cache"):
                col.invalidate_cache(reset_tracking=reset_tracking)

    def train(self, mode: bool = True):
        if self.training != mode:
            self.invalidate_cache(reset_tracking=False)
        return super().train(mode)

    def _apply(self, fn):
        self.invalidate_cache(reset_tracking=True)
        return super()._apply(fn)

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
        self.invalidate_cache()

    def _compute_transfer_matrix(self, use_cache: bool = True) -> torch.Tensor:
        raise NotImplementedError("get_transfer_matrix() must be overriden in child class!")

    def get_transfer_matrix(self, use_cache: bool = True) -> torch.Tensor:
        if not use_cache or not self.cache_enabled or torch.compiler.is_compiling():
            return self._compute_transfer_matrix(use_cache=False)

        if self._cache_all_params is None:
            self._init_cache_tracking()

        need_grad = torch.is_grad_enabled() and self._cache_any_requires_grad
        if self._cached_transfer_matrix is not None:
            if (not need_grad or self._cache_has_grad) and self._get_cache_signature() == self._cache_signature:
                return self._cached_transfer_matrix

        tm = self._compute_transfer_matrix(use_cache=True)
        self._cached_transfer_matrix = tm
        self._cache_signature = self._get_cache_signature()
        self._cache_has_grad = tm.requires_grad
        if tm.requires_grad:
            tm.register_hook(self._on_cached_tm_backward)
        return tm

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

        transfer_matrix = self.get_transfer_matrix()
        if transfer_matrix.device != input_tensor.device or transfer_matrix.dtype != input_tensor.dtype:
            transfer_matrix = transfer_matrix.to(device=input_tensor.device, dtype=input_tensor.dtype)

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
        if self.backward_type == BackwardType.DEFAULT.value or not torch.is_grad_enabled():
            # 直接计算，使用PyTorch自动微分或无梯度前向
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
            T_col = layer.get_transfer_matrix()
            if T_col.device != current_field.device or T_col.dtype != current_field.dtype:
                T_col = T_col.to(device=current_field.device, dtype=current_field.dtype)
            current_field = F.linear(current_field, T_col)
        return current_field[..., self.output_ports]

    def _default_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.forward_mode == ForwardMode.FIELD_PROP.value:
            if (
                not self.training
                or self._cached_transfer_matrix is not None
                or self._base_col_tms is not None
            ) and self.cache_enabled and not torch.compiler.is_compiling():
                return super()._default_forward(input_tensor)
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

    def _compute_transfer_matrix(self, use_cache: bool = True) -> torch.Tensor:
        if len(self.columnlayers) == 0:
            raise ValueError("columnlayers cannot be empty")

        if not use_cache or not self.cache_enabled or torch.compiler.is_compiling():
            return reduce(
                torch.matmul,
                [
                    layer.get_transfer_matrix(use_cache=False)
                    if isinstance(layer, ColumnLayer)
                    else layer.get_transfer_matrix()
                    for layer in reversed(self.columnlayers)
                ],
            )

        col_tms = [
            layer.get_transfer_matrix(use_cache=True)
            if isinstance(layer, ColumnLayer)
            else layer.get_transfer_matrix()
            for layer in self.columnlayers
        ]
        K = len(col_tms)
        need_grad = torch.is_grad_enabled() and self._cache_any_requires_grad

        if (
            self._base_col_tms is not None
            and len(self._base_col_tms) == K
            and (not need_grad or self._base_has_grad)
        ):
            diff_idx = -1
            diff_count = 0
            for i, (cur_t, base_t) in enumerate(zip(col_tms, self._base_col_tms)):
                if cur_t is not base_t:
                    diff_idx = i
                    diff_count += 1
                    if diff_count > 1:
                        break

            if diff_count == 0 and self._base_tm is not None:
                return self._base_tm

            if diff_count == 1 and K >= 2 and self._base_suffix is not None:
                if self._base_prefix is None:
                    prefix: List[Optional[torch.Tensor]] = [None] * K
                    acc_p = self._base_col_tms[K - 1]
                    for i in range(K - 2, -1, -1):
                        prefix[i] = acc_p
                        if i > 0:
                            acc_p = torch.matmul(acc_p, self._base_col_tms[i])
                    self._base_prefix = prefix

                left = self._base_prefix[diff_idx]
                right = self._base_suffix[diff_idx]
                T_c = col_tms[diff_idx]
                if left is None:
                    return torch.matmul(T_c, right)
                if right is None:
                    return torch.matmul(left, T_c)
                return torch.matmul(torch.matmul(left, T_c), right)

        if K == 1:
            tm = col_tms[0]
            suffix: List[Optional[torch.Tensor]] = [None]
        else:
            suffix = [None] * K
            acc = col_tms[0]
            for i in range(1, K):
                suffix[i] = acc
                acc = torch.matmul(col_tms[i], acc)
            tm = acc

        self._base_col_tms = col_tms
        self._base_suffix = suffix
        self._base_prefix = None
        self._base_tm = tm
        self._base_has_grad = tm.requires_grad
        for layer in self.columnlayers:
            if isinstance(layer, ColumnLayer):
                layer.pin_base_cache()
        return tm

    def forward_parameter_shifts(
        self,
        input_tensor: torch.Tensor,
        shift: float = torch.pi * 0.5,
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]]:
        """Batch-compute outputs for +shift and -shift on all trainable scalar MZI/MZM/PS parameters in one pass."""
        if not all(
            isinstance(c, (MZI, MZM, PhaseShifter))
            and (c.theta.numel() == 1 if hasattr(c, "theta") else True)
            and (c.phi.numel() == 1 if hasattr(c, "phi") else True)
            for col in self.columnlayers
            for c in col.components
        ):
            return None

        if not torch.is_complex(input_tensor):
            internal_dtype = torch.complex128 if input_tensor.dtype == torch.float64 else torch.complex64
            inp_c = input_tensor.to(internal_dtype)
        else:
            internal_dtype = input_tensor.dtype
            inp_c = input_tensor

        device = inp_c.device
        L = self.L

        with torch.no_grad():
            fields = []
            col_tms = []
            cur = inp_c
            for col in self.columnlayers:
                fields.append(cur)
                T_c = col.get_transfer_matrix().to(device=device, dtype=internal_dtype)
                col_tms.append(T_c)
                cur = F.linear(cur, T_c)
            full_base = cur

            K_cols = len(col_tms)
            suffix_prod: List[torch.Tensor] = [None] * K_cols  # type: ignore[list-item]
            acc_s = torch.eye(L, dtype=internal_dtype, device=device)
            for i in reversed(range(K_cols)):
                suffix_prod[i] = acc_s
                if i > 0:
                    acc_s = torch.matmul(acc_s, col_tms[i])

            param_list: List[torch.Tensor] = []
            plus_chunks: List[torch.Tensor] = []
            minus_chunks: List[torch.Tensor] = []

            for i, col in enumerate(self.columnlayers):
                A_fwd = fields[i]
                S_i = suffix_prod[i]
                if getattr(col, "_all_mzi", False) and all(
                    c.theta.requires_grad and c.phi.requires_grad for c in col.components
                ):
                    m_idx = col._mzi_m_idx if col._mzi_m_idx.device == device else col._mzi_m_idx.to(device)
                    n_idx = col._mzi_n_idx if col._mzi_n_idx.device == device else col._mzi_n_idx.to(device)
                    thetas = torch.cat([
                        ((c.theta + c.phase_noise_theta) if (not col.training or c.noise_aware_training) else c.theta).view(1)
                        for c in col.components
                    ])
                    phis = torch.cat([
                        ((c.phi + c.phase_noise_phi) if (not col.training or c.noise_aware_training) else c.phi).view(1)
                        for c in col.components
                    ])
                    for c in col.components:
                        param_list.append(c.theta)
                        param_list.append(c.phi)

                    a_m = A_fwd[..., m_idx]
                    a_n = A_fwd[..., n_idx]
                    e_iphi = torch.exp(1j * phis).to(dtype=internal_dtype)
                    e_itheta = torch.exp(1j * thetas).to(dtype=internal_dtype)
                    y_m = 0.5 * e_iphi * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n)
                    y_n = 0.5j * (e_itheta + 1.0) * a_m + 0.5 * (1.0 - e_itheta) * a_n

                    e_it_p = torch.exp(1j * (thetas + shift)).to(dtype=internal_dtype)
                    e_it_m = torch.exp(1j * (thetas - shift)).to(dtype=internal_dtype)
                    dym_tp = 0.5 * e_iphi * ((e_it_p - 1.0) * a_m + 1j * (1.0 + e_it_p) * a_n) - y_m
                    dyn_tp = (0.5j * (e_it_p + 1.0) * a_m + 0.5 * (1.0 - e_it_p) * a_n) - y_n
                    dym_tm = 0.5 * e_iphi * ((e_it_m - 1.0) * a_m + 1j * (1.0 + e_it_m) * a_n) - y_m
                    dyn_tm = (0.5j * (e_it_m + 1.0) * a_m + 0.5 * (1.0 - e_it_m) * a_n) - y_n

                    e_ip_p = torch.exp(1j * (phis + shift)).to(dtype=internal_dtype)
                    e_ip_m = torch.exp(1j * (phis - shift)).to(dtype=internal_dtype)
                    dym_pp = 0.5 * e_ip_p * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n) - y_m
                    dym_pm = 0.5 * e_ip_m * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n) - y_m

                    S_m = S_i[:, m_idx].T  # (K, L)
                    S_n = S_i[:, n_idx].T  # (K, L)

                    dt_plus = dym_tp.unsqueeze(-1) * S_m + dyn_tp.unsqueeze(-1) * S_n
                    dt_minus = dym_tm.unsqueeze(-1) * S_m + dyn_tm.unsqueeze(-1) * S_n
                    dp_plus = dym_pp.unsqueeze(-1) * S_m
                    dp_minus = dym_pm.unsqueeze(-1) * S_m

                    lead_shape = inp_c.shape[:-1]
                    plus_chunks.append(torch.stack([dt_plus, dp_plus], dim=-2).reshape(*lead_shape, -1, L))
                    minus_chunks.append(torch.stack([dt_minus, dp_minus], dim=-2).reshape(*lead_shape, -1, L))
                else:
                    return None

            if not param_list:
                return ([], full_base.unsqueeze(-2)[..., self.output_ports], full_base.unsqueeze(-2)[..., self.output_ports])

            out_plus = (full_base.unsqueeze(-2) + torch.cat(plus_chunks, dim=-2))[..., self.output_ports]
            out_minus = (full_base.unsqueeze(-2) + torch.cat(minus_chunks, dim=-2))[..., self.output_ports]
            return param_list, out_plus, out_minus

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

    def _compute_transfer_matrix(self, use_cache: bool = True) -> torch.Tensor:
        if len(self.columnlayers) == 0:
            raise ValueError("columnlayers cannot be empty")

        if use_cache and self.cache_enabled and not torch.compiler.is_compiling():
            rows = [
                layer.get_transfer_matrix(use_cache=True)
                if isinstance(layer, ColumnLayer)
                else layer.get_transfer_matrix()
                for layer in self.columnlayers
            ]
            need_grad = torch.is_grad_enabled() and self._cache_any_requires_grad
            if (
                self._base_col_tms is not None
                and len(self._base_col_tms) == len(rows)
                and (not need_grad or self._base_has_grad)
                and self._base_tm is not None
                and all(r is b for r, b in zip(rows, self._base_col_tms))
            ):
                return self._base_tm
            first_T = rows[0]
            rows = [
                r if (r.device == first_T.device and r.dtype == first_T.dtype) else r.to(device=first_T.device, dtype=first_T.dtype)
                for r in rows
            ]
            tm = torch.stack(rows, dim=0)
            self._base_col_tms = rows
            self._base_tm = tm
            self._base_has_grad = tm.requires_grad
            return tm

        if all(
            getattr(col, "_all_mzm", False) and getattr(col, "_is_contiguous_full", False)
            for col in self.columnlayers
        ):
            thetas = [
                (c.theta + c.phase_noise_theta).view(1)
                if (not col.training or c.noise_aware_training)
                else c.theta.view(1)
                for col in self.columnlayers
                for c in col.components
            ]
            theta_mat = torch.cat(thetas).view(self.L, self.columnlayers[0].N)
            return 0.5 * (torch.exp(1j * theta_mat) + 1.0)

        first_T = (
            self.columnlayers[0].get_transfer_matrix(use_cache=False)
            if isinstance(self.columnlayers[0], ColumnLayer)
            else self.columnlayers[0].get_transfer_matrix()
        )
        rows = [first_T]
        for layer in self.columnlayers[1:]:
            row_t = (
                layer.get_transfer_matrix(use_cache=False)
                if isinstance(layer, ColumnLayer)
                else layer.get_transfer_matrix()
            )
            rows.append(row_t.to(device=first_T.device, dtype=first_T.dtype))
        return torch.stack(rows, dim=0)

    def forward_parameter_shifts(
        self,
        input_tensor: torch.Tensor,
        shift: float = torch.pi * 0.5,
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]]:
        """Batch-compute outputs for +shift and -shift on all trainable MZM parameters in one pass."""
        if not all(
            getattr(col, "_all_mzm", False)
            and getattr(col, "_is_contiguous_full", False)
            and all(c.theta.requires_grad and c.theta.numel() == 1 for c in col.components)
            for col in self.columnlayers
        ):
            return None

        if not torch.is_complex(input_tensor):
            internal_dtype = torch.complex128 if input_tensor.dtype == torch.float64 else torch.complex64
            inp_c = input_tensor.to(internal_dtype)
        else:
            internal_dtype = input_tensor.dtype
            inp_c = input_tensor

        L = self.L
        N = self.columnlayers[0].N
        with torch.no_grad():
            param_list = [c.theta for col in self.columnlayers for c in col.components]
            thetas = torch.cat([
                ((c.theta + c.phase_noise_theta) if (not col.training or c.noise_aware_training) else c.theta).view(1)
                for col in self.columnlayers
                for c in col.components
            ]).view(L, N)
            t_base = (0.5 * (torch.exp(1j * thetas) + 1.0)).to(device=inp_c.device, dtype=internal_dtype)
            t_plus = (0.5 * (torch.exp(1j * (thetas + shift)) + 1.0)).to(device=inp_c.device, dtype=internal_dtype)
            t_minus = (0.5 * (torch.exp(1j * (thetas - shift)) + 1.0)).to(device=inp_c.device, dtype=internal_dtype)

            full_base = F.linear(inp_c, t_base)  # (..., L)
            delta_p = (t_plus - t_base) * inp_c.unsqueeze(-2)  # (..., L, N)
            delta_m = (t_minus - t_base) * inp_c.unsqueeze(-2)  # (..., L, N)

            eye_L = torch.eye(L, dtype=internal_dtype, device=inp_c.device)  # (L, L)
            lead_shape = inp_c.shape[:-1]
            plus_all = full_base.unsqueeze(-2) + (delta_p.unsqueeze(-1) * eye_L.unsqueeze(-2)).reshape(*lead_shape, L * N, L)
            minus_all = full_base.unsqueeze(-2) + (delta_m.unsqueeze(-1) * eye_L.unsqueeze(-2)).reshape(*lead_shape, L * N, L)
            return param_list, plus_all[..., self.output_ports], minus_all[..., self.output_ports]

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

