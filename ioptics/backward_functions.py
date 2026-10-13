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
        from .components import MZI, PhaseShifter, MZM

        transfer_matrix = (
            saved_forward_transfer_matrix
            if saved_forward_transfer_matrix is not None
            else optical_layer.get_transfer_matrix()
        )
        transfer_matrix = transfer_matrix.to(device=grad_output.device, dtype=grad_output.dtype)
        grad_transfer_matrix = transfer_matrix[optical_layer.output_ports, :]

        grad_input = F.linear(grad_output, grad_transfer_matrix.conj().T)

        params = parameter_refs if parameter_refs is not None else tuple(optical_layer.parameters())
        if len(params) == 0:
            return (grad_input,)

        use_complex128 = (
            input_tensor.dtype == torch.float64
            or input_tensor.dtype == torch.complex128
            or grad_output.dtype == torch.complex128
        )
        internal_dtype = torch.complex128 if use_complex128 else torch.complex64

        has_columns = hasattr(optical_layer, "columnlayers") and len(optical_layer.columnlayers) > 0
        can_batch_columns = has_columns and all(
            isinstance(c, (MZI, MZM, PhaseShifter))
            and c.theta.numel() == 1 if hasattr(c, "theta") else True
            and (c.phi.numel() == 1 if hasattr(c, "phi") else True)
            for col in optical_layer.columnlayers
            for c in col.components
        )

        if can_batch_columns:
            with torch.no_grad():
                device = input_tensor.device
                inp_c = input_tensor if input_tensor.dtype == internal_dtype else input_tensor.to(internal_dtype)
                grad_out_c = grad_output if grad_output.dtype == internal_dtype else grad_output.to(internal_dtype)
                L = optical_layer.L
                is_array_core = optical_layer.columnlayers[0].__class__.__name__ == "ArrayColumn"
                shift_factor = (1.0 - 1.0j) * 0.5
                half_pi = torch.pi * 0.5

                param_grad_map: Dict[int, torch.Tensor] = {}

                if is_array_core:
                    full_grad = torch.zeros(*inp_c.shape[:-1], L, dtype=internal_dtype, device=device)
                    full_grad[..., optical_layer.output_ports] = grad_out_c

                    is_full_mzm_array = all(
                        getattr(col, "_all_mzm", False) and getattr(col, "_is_contiguous_full", False)
                        for col in optical_layer.columnlayers
                    )
                    if is_full_mzm_array:
                        N = optical_layer.columnlayers[0].N
                        thetas = torch.stack([
                            col._get_effective_phases().to(device=device)
                            for col in optical_layer.columnlayers
                        ], dim=0)
                        t_base = (0.5 * (torch.exp(1j * thetas) + 1.0)).to(dtype=internal_dtype)
                        t_plus = (0.5 * (torch.exp(1j * (thetas + half_pi)) + 1.0)).to(dtype=internal_dtype)
                        dt_ps = (t_plus - t_base) * shift_factor

                        x_flat = inp_c.reshape(-1, N)
                        g_flat = full_grad.reshape(-1, L)
                        grad_mat = ((g_flat.conj().T @ x_flat) * dt_ps).real

                        for i, col in enumerate(optical_layer.columnlayers):
                            row_g = grad_mat[i]
                            C_i = col._get_effective_crosstalk_matrix(device=row_g.device, dtype=row_g.dtype)
                            if C_i is not None:
                                row_g = torch.mv(C_i.T, row_g)
                            for m, c in enumerate(col.components):
                                param_grad_map[id(c.theta)] = row_g[m].to(device=c.theta.device, dtype=c.theta.dtype).reshape_as(c.theta)
                    else:
                        for i, col in enumerate(optical_layer.columnlayers):
                            g_row = full_grad[..., i]
                            if getattr(col, "_all_mzm", False):
                                m_idx = col._comp_m_idx if col._comp_m_idx.device == device else col._comp_m_idx.to(device)
                                thetas = col._get_effective_phases().to(device=device)
                                t_base = (0.5 * (torch.exp(1j * thetas) + 1.0)).to(dtype=internal_dtype)
                                t_plus = (0.5 * (torch.exp(1j * (thetas + half_pi)) + 1.0)).to(dtype=internal_dtype)
                                df = (t_plus - t_base) * inp_c[..., m_idx] * shift_factor
                                term = g_row.unsqueeze(-1).conj() * df
                                grad_vec = term.sum(dim=tuple(range(term.ndim - 1))).real if term.ndim > 1 else term.real
                                C_i = col._get_effective_crosstalk_matrix(device=grad_vec.device, dtype=grad_vec.dtype)
                                if C_i is not None:
                                    grad_vec = torch.mv(C_i.T, grad_vec)
                                for k, c in enumerate(col.components):
                                    param_grad_map[id(c.theta)] = grad_vec[k].to(device=c.theta.device, dtype=c.theta.dtype).reshape_as(c.theta)
                            elif getattr(col, "_all_ps", False):
                                m_idx = col._comp_m_idx if col._comp_m_idx.device == device else col._comp_m_idx.to(device)
                                phis = col._get_effective_phases().to(device=device)
                                t_base = torch.exp(1j * phis).to(dtype=internal_dtype)
                                t_plus = torch.exp(1j * (phis + half_pi)).to(dtype=internal_dtype)
                                df = (t_plus - t_base) * inp_c[..., m_idx] * shift_factor
                                term = g_row.unsqueeze(-1).conj() * df
                                grad_vec = term.sum(dim=tuple(range(term.ndim - 1))).real if term.ndim > 1 else term.real
                                C_i = col._get_effective_crosstalk_matrix(device=grad_vec.device, dtype=grad_vec.dtype)
                                if C_i is not None:
                                    grad_vec = torch.mv(C_i.T, grad_vec)
                                for k, c in enumerate(col.components):
                                    param_grad_map[id(c.phi)] = grad_vec[k].to(device=c.phi.device, dtype=c.phi.dtype).reshape_as(c.phi)
                            else:
                                for c in col.components:
                                    apply_noise = not col.training or c.noise_aware_training
                                    if isinstance(c, MZM):
                                        theta = (c.theta + c.phase_noise_theta) if apply_noise else c.theta
                                        t_base = (0.5 * (torch.exp(1j * theta) + 1.0)).to(dtype=internal_dtype)
                                        t_plus = (0.5 * (torch.exp(1j * (theta + half_pi)) + 1.0)).to(dtype=internal_dtype)
                                        df = (t_plus - t_base) * inp_c[..., c.m] * shift_factor
                                        gt = _sum_all_dims(g_row.conj() * df).real
                                        param_grad_map[id(c.theta)] = gt.to(device=c.theta.device, dtype=c.theta.dtype).reshape_as(c.theta)
                                    elif isinstance(c, PhaseShifter):
                                        phi = (c.phi + c.phase_noise_phi) if apply_noise else c.phi
                                        t_base = torch.exp(1j * phi).to(dtype=internal_dtype)
                                        t_plus = torch.exp(1j * (phi + half_pi)).to(dtype=internal_dtype)
                                        df = (t_plus - t_base) * inp_c[..., c.m] * shift_factor
                                        gp = _sum_all_dims(g_row.conj() * df).real
                                        param_grad_map[id(c.phi)] = gp.to(device=c.phi.device, dtype=c.phi.dtype).reshape_as(c.phi)
                else:
                    forward_fields = []
                    col_tms = []
                    cur = inp_c
                    for col in optical_layer.columnlayers:
                        forward_fields.append(cur)
                        T_c = col.get_transfer_matrix().to(device=device, dtype=internal_dtype)
                        col_tms.append(T_c)
                        cur = F.linear(cur, T_c)

                    full_grad = torch.zeros(*inp_c.shape[:-1], L, dtype=internal_dtype, device=device)
                    full_grad[..., optical_layer.output_ports] = grad_out_c
                    adj = full_grad

                    for i in reversed(range(len(optical_layer.columnlayers))):
                        col = optical_layer.columnlayers[i]
                        A_fwd = forward_fields[i]
                        is_pure_mzi = getattr(col, "_all_mzi", False)

                        if is_pure_mzi:
                            m_idx = col._mzi_m_idx if col._mzi_m_idx.device == device else col._mzi_m_idx.to(device)
                            n_idx = col._mzi_n_idx if col._mzi_n_idx.device == device else col._mzi_n_idx.to(device)
                            thetas, phis = col._get_effective_phases()
                            thetas = thetas.to(device=device)
                            phis = phis.to(device=device)

                            a_m = A_fwd[..., m_idx]
                            a_n = A_fwd[..., n_idx]
                            adj_m = adj[..., m_idx]
                            adj_n = adj[..., n_idx]

                            e_iphi = torch.exp(1j * phis).to(dtype=internal_dtype)
                            e_itheta = torch.exp(1j * thetas).to(dtype=internal_dtype)
                            y_m = 0.5 * e_iphi * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n)
                            y_n = 0.5j * (e_itheta + 1.0) * a_m + 0.5 * (1.0 - e_itheta) * a_n

                            # Batch +pi/2 shift on all thetas in this column simultaneously
                            e_itheta_p = torch.exp(1j * (thetas + half_pi)).to(dtype=internal_dtype)
                            y_m_tp = 0.5 * e_iphi * ((e_itheta_p - 1.0) * a_m + 1j * (1.0 + e_itheta_p) * a_n)
                            y_n_tp = 0.5j * (e_itheta_p + 1.0) * a_m + 0.5 * (1.0 - e_itheta_p) * a_n
                            df_m_t = (y_m_tp - y_m) * shift_factor
                            df_n_t = (y_n_tp - y_n) * shift_factor
                            term_t = adj_m.conj() * df_m_t + adj_n.conj() * df_n_t
                            grad_t_vec = term_t.sum(dim=tuple(range(term_t.ndim - 1))).real if term_t.ndim > 1 else term_t.real

                            # Batch +pi/2 shift on all phis in this column simultaneously
                            e_iphi_p = torch.exp(1j * (phis + half_pi)).to(dtype=internal_dtype)
                            y_m_pp = 0.5 * e_iphi_p * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n)
                            df_m_p = (y_m_pp - y_m) * shift_factor
                            term_p = adj_m.conj() * df_m_p
                            grad_p_vec = term_p.sum(dim=tuple(range(term_p.ndim - 1))).real if term_p.ndim > 1 else term_p.real

                            C_col = col._get_effective_crosstalk_matrix(device=grad_t_vec.device, dtype=grad_t_vec.dtype)
                            if C_col is not None:
                                grad_t_vec = torch.mv(C_col.T, grad_t_vec)
                                grad_p_vec = torch.mv(C_col.T, grad_p_vec)

                            for k, c in enumerate(col.components):
                                param_grad_map[id(c.theta)] = grad_t_vec[k].to(device=c.theta.device, dtype=c.theta.dtype).reshape_as(c.theta)
                                param_grad_map[id(c.phi)] = grad_p_vec[k].to(device=c.phi.device, dtype=c.phi.dtype).reshape_as(c.phi)
                        else:
                            for c in col.components:
                                apply_noise = not col.training or c.noise_aware_training
                                if isinstance(c, MZI):
                                    m, n = c.m, c.n
                                    theta = (c.theta + c.phase_noise_theta) if apply_noise else c.theta
                                    phi = (c.phi + c.phase_noise_phi) if apply_noise else c.phi
                                    a_m, a_n = A_fwd[..., m], A_fwd[..., n]
                                    adj_m, adj_n = adj[..., m], adj[..., n]
                                    e_iphi = torch.exp(1j * phi).to(dtype=internal_dtype)
                                    e_itheta = torch.exp(1j * theta).to(dtype=internal_dtype)
                                    y_m = 0.5 * e_iphi * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n)
                                    y_n = 0.5j * (e_itheta + 1.0) * a_m + 0.5 * (1.0 - e_itheta) * a_n

                                    e_itheta_p = torch.exp(1j * (theta + half_pi)).to(dtype=internal_dtype)
                                    y_m_tp = 0.5 * e_iphi * ((e_itheta_p - 1.0) * a_m + 1j * (1.0 + e_itheta_p) * a_n)
                                    y_n_tp = 0.5j * (e_itheta_p + 1.0) * a_m + 0.5 * (1.0 - e_itheta_p) * a_n
                                    gt = _sum_all_dims((adj_m.conj() * (y_m_tp - y_m) + adj_n.conj() * (y_n_tp - y_n)) * shift_factor).real

                                    e_iphi_p = torch.exp(1j * (phi + half_pi)).to(dtype=internal_dtype)
                                    y_m_pp = 0.5 * e_iphi_p * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n)
                                    gp = _sum_all_dims(adj_m.conj() * (y_m_pp - y_m) * shift_factor).real

                                    param_grad_map[id(c.theta)] = gt.to(device=c.theta.device, dtype=c.theta.dtype).reshape_as(c.theta)
                                    param_grad_map[id(c.phi)] = gp.to(device=c.phi.device, dtype=c.phi.dtype).reshape_as(c.phi)
                                elif isinstance(c, PhaseShifter):
                                    m = c.m
                                    phi = (c.phi + c.phase_noise_phi) if apply_noise else c.phi
                                    t_base = torch.exp(1j * phi).to(dtype=internal_dtype)
                                    t_plus = torch.exp(1j * (phi + half_pi)).to(dtype=internal_dtype)
                                    gp = _sum_all_dims(adj[..., m].conj() * (t_plus - t_base) * A_fwd[..., m] * shift_factor).real
                                    param_grad_map[id(c.phi)] = gp.to(device=c.phi.device, dtype=c.phi.dtype).reshape_as(c.phi)
                                elif isinstance(c, MZM):
                                    m = c.m
                                    theta = (c.theta + c.phase_noise_theta) if apply_noise else c.theta
                                    t_base = (0.5 * (torch.exp(1j * theta) + 1.0)).to(dtype=internal_dtype)
                                    t_plus = (0.5 * (torch.exp(1j * (theta + half_pi)) + 1.0)).to(dtype=internal_dtype)
                                    gt = _sum_all_dims(adj[..., m].conj() * (t_plus - t_base) * A_fwd[..., m] * shift_factor).real
                                    param_grad_map[id(c.theta)] = gt.to(device=c.theta.device, dtype=c.theta.dtype).reshape_as(c.theta)

                        adj = F.linear(adj, col_tms[i].conj().T)

                if all(id(p) in param_grad_map for p in params):
                    return grad_input, *[param_grad_map[id(p)] for p in params]

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

        with torch.no_grad():
            device = input_tensor.device

            if input_tensor.dtype != internal_dtype:
                input_tensor = input_tensor.to(internal_dtype)

            if grad_output.dtype != internal_dtype:
                grad_output = grad_output.to(internal_dtype)

            L = optical_layer.L
            is_array_core = hasattr(optical_layer, "columnlayers") and len(optical_layer.columnlayers) > 0 and optical_layer.columnlayers[0].__class__.__name__ == "ArrayColumn"

            # === 正向传播：逐层记录每个 column 输入端的光场 ===
            forward_fields = []
            current_field = input_tensor

            if is_array_core:
                for _ in optical_layer.columnlayers:
                    forward_fields.append(input_tensor)
            else:
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

            is_full_mzm_array = is_array_core and all(
                getattr(col, "_all_mzm", False) and getattr(col, "_is_contiguous_full", False)
                for col in optical_layer.columnlayers
            )

            if is_full_mzm_array:
                N = optical_layer.columnlayers[0].N
                theta_mat = torch.stack([
                    col._get_effective_phases().to(device=device)
                    for col in optical_layer.columnlayers
                ], dim=0)
                dT_dtheta = (0.5j * torch.exp(1j * theta_mat)).to(dtype=internal_dtype)

                x_flat = input_tensor.reshape(-1, N)
                adj_flat = adjoint_field.reshape(-1, L)
                grad_theta_mat = ((adj_flat.conj().T @ x_flat) * dT_dtheta).real

                for i, col in enumerate(optical_layer.columnlayers):
                    row_grads = grad_theta_mat[i]
                    C_i = col._get_effective_crosstalk_matrix(device=row_grads.device, dtype=row_grads.dtype)
                    if C_i is not None:
                        row_grads = torch.mv(C_i.T, row_grads)
                    for m, c in enumerate(col.components):
                        gt = row_grads[m]
                        if gt.device != c.theta.device or gt.dtype != c.theta.dtype:
                            gt = gt.to(device=c.theta.device, dtype=c.theta.dtype)
                        gt = gt.reshape_as(c.theta)
                        param_grad_map[id(c.theta)] = gt
                        all_grads.append(gt)
            else:
                for i in reversed(range(len(optical_layer.columnlayers))):
                    column = optical_layer.columnlayers[i]
                    A_fwd = forward_fields[i]

                    layer_grads = []

                    is_pure_mzi = len(column.components) > 0 and all(isinstance(c, MZI) for c in column.components)
                    is_pure_mzm = len(column.components) > 0 and all(isinstance(c, MZM) for c in column.components)
                    is_pure_ps = len(column.components) > 0 and all(isinstance(c, PhaseShifter) for c in column.components)

                    if is_pure_mzi:
                        m_idx = getattr(column, "_mzi_m_idx", None)
                        n_idx = getattr(column, "_mzi_n_idx", None)
                        if m_idx is None or n_idx is None:
                            m_idx = [c.m for c in column.components]
                            n_idx = [c.n for c in column.components]
                        else:
                            if m_idx.device != device:
                                m_idx = m_idx.to(device)
                            if n_idx.device != device:
                                n_idx = n_idx.to(device)

                        thetas, phis = column._get_effective_phases()
                        thetas = thetas.to(device=device)
                        phis = phis.to(device=device)

                        a_m = A_fwd[..., m_idx]
                        a_n = A_fwd[..., n_idx]
                        adj_m = adjoint_field[..., m_idx]
                        adj_n = adjoint_field[..., n_idx]

                        e_itheta = torch.exp(1j * thetas).to(dtype=internal_dtype)
                        e_iphi = torch.exp(1j * phis).to(dtype=internal_dtype)
                        e_i_pt = e_iphi * e_itheta

                        out0_theta = 0.5 * e_i_pt * (1j * a_m - a_n)
                        out1_theta = -0.5 * e_itheta * (a_m + 1j * a_n)
                        term_theta = adj_m.conj() * out0_theta + adj_n.conj() * out1_theta

                        if term_theta.ndim > 1:
                            grad_theta_vec = term_theta.sum(dim=tuple(range(term_theta.ndim - 1))).real
                        else:
                            grad_theta_vec = term_theta.real

                        out0_phi = 0.5j * e_iphi * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n)
                        term_phi = adj_m.conj() * out0_phi

                        if term_phi.ndim > 1:
                            grad_phi_vec = term_phi.sum(dim=tuple(range(term_phi.ndim - 1))).real
                        else:
                            grad_phi_vec = term_phi.real

                        C_col = column._get_effective_crosstalk_matrix(device=grad_theta_vec.device, dtype=grad_theta_vec.dtype)
                        if C_col is not None:
                            grad_theta_vec = torch.mv(C_col.T, grad_theta_vec)
                            grad_phi_vec = torch.mv(C_col.T, grad_phi_vec)

                        for k, component in enumerate(column.components):
                            gt = grad_theta_vec[k]
                            if gt.device != component.theta.device or gt.dtype != component.theta.dtype:
                                gt = gt.to(device=component.theta.device, dtype=component.theta.dtype)
                            gt = gt.reshape_as(component.theta)

                            gp = grad_phi_vec[k]
                            if gp.device != component.phi.device or gp.dtype != component.phi.dtype:
                                gp = gp.to(device=component.phi.device, dtype=component.phi.dtype)
                            gp = gp.reshape_as(component.phi)

                            param_grad_map[id(component.theta)] = gt
                            param_grad_map[id(component.phi)] = gp
                            layer_grads.extend([gt, gp])

                    elif is_pure_mzm:
                        m_idx = getattr(column, "_comp_m_idx", None)
                        if m_idx is None:
                            m_idx = [c.m for c in column.components]
                        elif m_idx.device != device:
                            m_idx = m_idx.to(device)

                        thetas = column._get_effective_phases().to(device=device)
                        a_m = A_fwd[..., m_idx]
                        adj_m = adjoint_field[..., i].unsqueeze(-1) if is_array_core else adjoint_field[..., m_idx]

                        dT_dtheta = (0.5j * torch.exp(1j * thetas)).to(dtype=internal_dtype)
                        term_theta = adj_m.conj() * dT_dtheta * a_m

                        if term_theta.ndim > 1:
                            grad_theta_vec = term_theta.sum(dim=tuple(range(term_theta.ndim - 1))).real
                        else:
                            grad_theta_vec = term_theta.real

                        C_col = column._get_effective_crosstalk_matrix(device=grad_theta_vec.device, dtype=grad_theta_vec.dtype)
                        if C_col is not None:
                            grad_theta_vec = torch.mv(C_col.T, grad_theta_vec)

                        for k, component in enumerate(column.components):
                            gt = grad_theta_vec[k]
                            if gt.device != component.theta.device or gt.dtype != component.theta.dtype:
                                gt = gt.to(device=component.theta.device, dtype=component.theta.dtype)
                            gt = gt.reshape_as(component.theta)
                            param_grad_map[id(component.theta)] = gt
                            layer_grads.append(gt)

                    elif is_pure_ps:
                        m_idx = getattr(column, "_comp_m_idx", None)
                        if m_idx is None:
                            m_idx = [c.m for c in column.components]
                        elif m_idx.device != device:
                            m_idx = m_idx.to(device)

                        phis = column._get_effective_phases().to(device=device)
                        a_m = A_fwd[..., m_idx]
                        adj_m = adjoint_field[..., i].unsqueeze(-1) if is_array_core else adjoint_field[..., m_idx]

                        dT_dphi = (1j * torch.exp(1j * phis)).to(dtype=internal_dtype)
                        term_phi = adj_m.conj() * dT_dphi * a_m

                        if term_phi.ndim > 1:
                            grad_phi_vec = term_phi.sum(dim=tuple(range(term_phi.ndim - 1))).real
                        else:
                            grad_phi_vec = term_phi.real

                        C_col = column._get_effective_crosstalk_matrix(device=grad_phi_vec.device, dtype=grad_phi_vec.dtype)
                        if C_col is not None:
                            grad_phi_vec = torch.mv(C_col.T, grad_phi_vec)

                        for k, component in enumerate(column.components):
                            gp = grad_phi_vec[k]
                            if gp.device != component.phi.device or gp.dtype != component.phi.dtype:
                                gp = gp.to(device=component.phi.device, dtype=component.phi.dtype)
                            gp = gp.reshape_as(component.phi)
                            param_grad_map[id(component.phi)] = gp
                            layer_grads.append(gp)

                    else:
                        for component in column.components:
                            c_apply_noise = not column.training or component.noise_aware_training
                            if isinstance(component, MZI):
                                m, n = component.m, component.n
                                theta = (component.theta + component.phase_noise_theta) if c_apply_noise else component.theta
                                phi = (component.phi + component.phase_noise_phi) if c_apply_noise else component.phi

                                a_m = A_fwd[..., m]
                                a_n = A_fwd[..., n]
                                adj_m = adjoint_field[..., m]
                                adj_n = adjoint_field[..., n]

                                e_itheta = torch.exp(1j * theta).to(dtype=internal_dtype)
                                e_iphi = torch.exp(1j * phi).to(dtype=internal_dtype)
                                e_i_pt = e_iphi * e_itheta

                                out0_theta = 0.5 * e_i_pt * (1j * a_m - a_n)
                                out1_theta = -0.5 * e_itheta * (a_m + 1j * a_n)
                                grad_theta = _sum_all_dims(adj_m.conj() * out0_theta + adj_n.conj() * out1_theta).real

                                out0_phi = 0.5j * e_iphi * ((e_itheta - 1.0) * a_m + 1j * (1.0 + e_itheta) * a_n)
                                grad_phi = _sum_all_dims(adj_m.conj() * out0_phi).real

                                grad_theta_param = grad_theta.to(device=component.theta.device, dtype=component.theta.dtype).reshape_as(component.theta)
                                grad_phi_param = grad_phi.to(device=component.phi.device, dtype=component.phi.dtype).reshape_as(component.phi)
                                param_grad_map[id(component.theta)] = grad_theta_param
                                param_grad_map[id(component.phi)] = grad_phi_param
                                layer_grads.extend([grad_theta_param, grad_phi_param])

                            elif isinstance(component, PhaseShifter):
                                m = component.m
                                phi = (component.phi + component.phase_noise_phi) if c_apply_noise else component.phi

                                a_m = A_fwd[..., m]
                                adj_m = adjoint_field[..., i] if is_array_core else adjoint_field[..., m]

                                grad_phi = _sum_all_dims(adj_m.conj() * (1j * torch.exp(1j * phi)).to(dtype=internal_dtype) * a_m).real
                                grad_phi_param = grad_phi.to(device=component.phi.device, dtype=component.phi.dtype).reshape_as(component.phi)
                                param_grad_map[id(component.phi)] = grad_phi_param
                                layer_grads.append(grad_phi_param)

                            elif isinstance(component, MZM):
                                m = component.m
                                theta = (component.theta + component.phase_noise_theta) if c_apply_noise else component.theta

                                a_m = A_fwd[..., m]
                                adj_m = adjoint_field[..., i] if is_array_core else adjoint_field[..., m]

                                dT_dtheta = (0.5j * torch.exp(1j * theta)).to(dtype=internal_dtype)
                                grad_theta = _sum_all_dims(adj_m.conj() * dT_dtheta * a_m).real
                                grad_theta_param = grad_theta.to(device=component.theta.device, dtype=component.theta.dtype).reshape_as(component.theta)
                                param_grad_map[id(component.theta)] = grad_theta_param
                                layer_grads.append(grad_theta_param)

                    all_grads = layer_grads + all_grads

                    if not is_array_core:
                        # 反向传播伴随场: A* = T^H · A*
                        T_col = column.get_transfer_matrix().to(device, dtype=internal_dtype)
                        adjoint_field = F.linear(adjoint_field, T_col.conj().T)

            if saved_forward_transfer_matrix is not None:
                grad_transfer_matrix = saved_forward_transfer_matrix[
                    optical_layer.output_ports, :
                ].to(device=device, dtype=internal_dtype)
                grad_input = F.linear(grad_output, grad_transfer_matrix.conj().T)
            elif is_array_core:
                T_total = optical_layer.get_transfer_matrix().to(device=device, dtype=internal_dtype)
                grad_input = F.linear(adjoint_field, T_total.conj().T)
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
