import pytest
import torch

from ioptics.backward_functions import BackwardType
from ioptics.components import MZI, PhaseShifter, MZM
from ioptics.layers import ClementsMesh, MeshColumn, MeshCore, ArrayColumn, ArrayCore, FCArray, CoreLayer


def test_corelayer_dimension_check_default_forward() -> None:
    layer = ClementsMesh(L=2)
    bad_input = torch.zeros(1, 3)

    with pytest.raises(ValueError):
        _ = layer(bad_input)


def test_custom_forward_dimension_check() -> None:
    layer = ClementsMesh(L=2, backward_type=BackwardType.PARAMETER_SHIFT.value)
    bad_input = torch.zeros(1, 3)

    with pytest.raises(ValueError):
        _ = layer(bad_input)


def test_columnlayer_port_range_validation() -> None:
    with pytest.raises(ValueError):
        _ = MeshColumn(2, [MZI(0, 2)])


def test_corelayer_preserves_complex128() -> None:
    layer = ClementsMesh(L=2)
    input_tensor = torch.zeros(2, dtype=torch.float64)

    output = layer(input_tensor)

    assert output.dtype == torch.complex128


def test_meshcolumn_identity_no_components():
    """Test that MeshColumn(N=3, []) returns eye(3)"""
    column = MeshColumn(N=3, components=[])
    transfer_matrix = column.get_transfer_matrix()

    expected = torch.eye(3, dtype=torch.complex64)
    torch.testing.assert_close(transfer_matrix, expected, atol=1e-5, rtol=0.0)


def test_meshcolumn_dof1_sets_diagonal():
    """Test that MeshColumn(N=2, [PhaseShifter(0, phi=0.0)]) diagonal [0,0] == exp(0j) == 1"""
    phase_shifter = PhaseShifter(0, phi=0.0)
    column = MeshColumn(N=2, components=[phase_shifter])
    transfer_matrix = column.get_transfer_matrix()

    # Check that diagonal element [0,0] equals exp(0j) = 1
    expected_element = torch.exp(torch.tensor(0j, dtype=torch.complex64))
    torch.testing.assert_close(transfer_matrix[0, 0], expected_element, atol=1e-5, rtol=0.0)

    # Also check that other elements are unchanged (identity matrix except [0,0])
    expected_matrix = torch.eye(2, dtype=torch.complex64)
    expected_matrix[0, 0] = 1.0 + 0j
    torch.testing.assert_close(transfer_matrix, expected_matrix, atol=1e-5, rtol=0.0)


def test_meshcolumn_dof2_sets_subblock():
    """Test that MeshColumn(N=2, [MZI(0,1,theta=0.0,phi=0.0)]) -> [0,0] block matches MZI(0,1).get_transfer_matrix()"""
    mzi = MZI(0, 1, theta=0.0, phi=0.0)
    column = MeshColumn(N=2, components=[mzi])
    column_matrix = column.get_transfer_matrix()

    mzi_matrix = mzi.get_transfer_matrix()

    # Check that the sub-block matches the MZI matrix
    torch.testing.assert_close(column_matrix[:2, :2], mzi_matrix, atol=1e-5, rtol=0.0)


def test_arraycolumn_returns_1d_row():
    """Test that ArrayColumn(N=2, [MZM(0,theta=0.0), MZM(1,theta=0.0)]).get_transfer_matrix() shape == (2,)"""
    array_col = ArrayColumn(N=2, components=[MZM(0, theta=0.0), MZM(1, theta=0.0)])
    transfer_matrix = array_col.get_transfer_matrix()

    assert transfer_matrix.shape == (2,), f"Expected shape (2,), got {transfer_matrix.shape}"
    assert transfer_matrix.dtype == torch.complex64


def test_arraycolumn_sparse_and_phaseshifter_vectorization() -> None:
    """Verify vectorized ArrayColumn on sparse MZM ports and PhaseShifter components."""
    mzm0 = MZM(0, theta=0.4)
    mzm2 = MZM(2, theta=1.2)
    sparse_mzm_col = ArrayColumn(N=4, components=[mzm0, mzm2])
    tm_mzm = sparse_mzm_col.get_transfer_matrix()
    expected_mzm = torch.zeros(4, dtype=torch.complex64)
    expected_mzm[0] = mzm0.get_transfer_matrix()
    expected_mzm[2] = mzm2.get_transfer_matrix()
    torch.testing.assert_close(tm_mzm, expected_mzm, atol=1e-6, rtol=1e-6)

    ps0 = PhaseShifter(0, phi=0.5)
    ps1 = PhaseShifter(1, phi=-0.7)
    ps_col = ArrayColumn(N=2, components=[ps0, ps1])
    tm_ps = ps_col.get_transfer_matrix()
    expected_ps = torch.stack([ps0.get_transfer_matrix(), ps1.get_transfer_matrix()])
    torch.testing.assert_close(tm_ps, expected_ps, atol=1e-6, rtol=1e-6)


def test_meshcore_single_column_matches_column():
    """Test that MeshCore(L=2, [MeshColumn(2,[MZI(0,1,theta=0.0,phi=0.0)])]) transfer matrix == MeshColumn's transfer matrix"""
    mzi = MZI(0, 1, theta=0.0, phi=0.0)
    mesh_column = MeshColumn(2, [mzi])
    mesh_core = MeshCore(L=2, columnlayers=[mesh_column])

    core_matrix = mesh_core.get_transfer_matrix()
    column_matrix = mesh_column.get_transfer_matrix()

    torch.testing.assert_close(core_matrix, column_matrix, atol=1e-5, rtol=0.0)


def test_clementsmesh_output_shape_even():
    """Test that ClementsMesh(L=4) forward on shape (4,) returns shape (4,)"""
    mesh = ClementsMesh(L=4)
    input_tensor = torch.ones(4)

    output = mesh(input_tensor)

    assert output.shape == (4,), f"Expected output shape (4,), got {output.shape}"


def test_clementsmesh_output_shape_odd():
    """Test that ClementsMesh(L=3) forward on shape (3,) returns shape (3,)"""
    mesh = ClementsMesh(L=3)
    input_tensor = torch.ones(3)

    output = mesh(input_tensor)

    assert output.shape == (3,), f"Expected output shape (3,), got {output.shape}"


def test_clementsmesh_transfer_matrix_is_unitary():
    """Test that ClementsMesh(L=4) transfer matrix U; U†U ≈ I (atol=1e-5)"""
    mesh = ClementsMesh(L=4)
    U = mesh.get_transfer_matrix()

    # Calculate U†U which should be identity
    U_dagger_U = torch.matmul(U.conj().t(), U)

    # Create identity matrix
    identity = torch.eye(4, dtype=torch.complex64)

    torch.testing.assert_close(U_dagger_U, identity, atol=1e-5, rtol=0.0)


def test_arraycore_output_shape_LxN():
    """Test that FCArray(in_dim=3, out_dim=2) forward on shape (3,) returns shape (2,)"""
    fc_array = FCArray(in_dim=3, out_dim=2)
    input_tensor = torch.ones(3)

    output = fc_array(input_tensor)

    assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"


def test_fcarray_transfer_matrix_shape():
    """Test that FCArray(in_dim=3, out_dim=2) get_transfer_matrix() shape == (2,3)"""
    fc_array = FCArray(in_dim=3, out_dim=2)
    transfer_matrix = fc_array.get_transfer_matrix()

    expected_shape = (2, 3)
    assert transfer_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {transfer_matrix.shape}"


def test_noise_not_applied_in_train_mode():
    """Test that ClementsMesh with phase_noise_std doesn't apply noise in train mode"""
    mesh = ClementsMesh(L=2, phase_noise_std=0.5)
    mesh.train()  # Set to training mode

    # Get transfer matrix multiple times - should be deterministic in train mode
    tm1 = mesh.get_transfer_matrix()
    tm2 = mesh.get_transfer_matrix()

    # Transfer matrices should be identical in train mode (no noise)
    torch.testing.assert_close(tm1, tm2, atol=1e-6, rtol=0.0)


def test_noise_applied_in_eval_mode():
    """Test that ClementsMesh applies phase noise in eval mode"""
    mesh = ClementsMesh(L=2, phase_noise_std=0.5)
    mesh.train()  # Start in train mode
    train_result = mesh.get_transfer_matrix()

    mesh.eval()  # Switch to eval mode
    eval_result = mesh.get_transfer_matrix()

    # In eval mode, noise should be applied, so results should differ
    # Use torch.allclose with tight tolerances to check they're different
    assert not torch.allclose(train_result, eval_result, atol=1e-3, rtol=1e-3), \
        "Train and eval mode results should differ due to noise application"


def test_noise_aware_training_applies_noise_in_train():
    """Test that ClementsMesh with noise_aware_training=True applies noise even in train mode"""
    # Create two meshes - one with noise aware training and one without
    mesh_with_noise = ClementsMesh(L=2, phase_noise_std=0.5, noise_aware_training=True)
    mesh_without_noise = ClementsMesh(L=2, phase_noise_std=0.0)  # No noise

    mesh_with_noise.train()  # Both in train mode
    mesh_without_noise.train()

    # Get transfer matrices
    tm_with_noise = mesh_with_noise.get_transfer_matrix()
    tm_without_noise = mesh_without_noise.get_transfer_matrix()

    # They should differ because one has noise applied
    assert not torch.allclose(tm_with_noise, tm_without_noise, atol=1e-3, rtol=1e-3), \
        "Results with and without noise should differ"


def test_output_ports_slice_validation():
    """Test that output_ports=slice(0, 10) for L=4 raises ValueError"""
    with pytest.raises(ValueError):
        _ = CoreLayer(L=4, columnlayers=[], output_ports=slice(0, 10))


def test_output_ports_list_validation():
    """Test that output_ports=[0, 4] for L=4 raises ValueError"""
    with pytest.raises(ValueError):
        _ = CoreLayer(L=4, columnlayers=[], output_ports=[0, 4])


def test_output_ports_slice_step_zero_validation() -> None:
    with pytest.raises(ValueError):
        _ = CoreLayer(L=4, columnlayers=[], output_ports=slice(0, 4, 0))


def test_output_ports_list_bool_validation() -> None:
    with pytest.raises(ValueError):
        _ = CoreLayer(L=4, columnlayers=[], output_ports=[True, 1])


def test_output_ports_list_selection():
    """Test that output_ports=[1, 3], L=4 returns output shape (2,) when input shape (4,)"""
    # Create minimal mock column layer that just returns identity matrix
    class DummyColumnLayer(torch.nn.Module):
        def __init__(self, N):
            super().__init__()
            self.N = N

        def get_transfer_matrix(self):
            return torch.eye(self.N, dtype=torch.complex64)

        def reset_parameters(self):
            pass

        def __iter__(self):
            yield from []

    # Test with a CoreLayer subclass that implements get_transfer_matrix
    class DummyCore(CoreLayer):
        def get_transfer_matrix(self):
            return torch.eye(4, dtype=torch.complex64)

    core_layer = DummyCore(
        L=4,
        columnlayers=[DummyColumnLayer(4), DummyColumnLayer(4), DummyColumnLayer(4), DummyColumnLayer(4)],
        output_ports=[1, 3],
    )

    # Perform forward pass
    input_tensor = torch.ones(4, dtype=torch.complex64)
    output = core_layer._default_forward(input_tensor)

    assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"


def test_clementsmesh_invalid_l_raises() -> None:
    with pytest.raises(ValueError):
        _ = ClementsMesh(L=0)


def test_fcarray_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        _ = FCArray(in_dim=0, out_dim=2)

    with pytest.raises(ValueError):
        _ = FCArray(in_dim=2, out_dim=0)


def test_corelayer_transfer_matrix_first_dim_validation() -> None:
    class DummyColumnLayer(torch.nn.Module):
        def __init__(self, N):
            super().__init__()
            self.N = N

        def get_transfer_matrix(self):
            return torch.eye(self.N, dtype=torch.complex64)

        def reset_parameters(self):
            pass

        def __iter__(self):
            yield from []

    class DummyCore(CoreLayer):
        def get_transfer_matrix(self):
            return torch.eye(3, 4, dtype=torch.complex64)

    core_layer = DummyCore(
        L=4,
        columnlayers=[DummyColumnLayer(4), DummyColumnLayer(4), DummyColumnLayer(4), DummyColumnLayer(4)],
    )

    with pytest.raises(ValueError):
        _ = core_layer(torch.ones(4, dtype=torch.float32))


def test_corelayer_input_rank_validation() -> None:
    class DummyColumnLayer(torch.nn.Module):
        def __init__(self, N):
            super().__init__()
            self.N = N

        def get_transfer_matrix(self):
            return torch.eye(self.N, dtype=torch.complex64)

        def reset_parameters(self):
            pass

        def __iter__(self):
            yield from []

    class DummyCore(CoreLayer):
        def get_transfer_matrix(self):
            return torch.eye(4, dtype=torch.complex64)

    core_layer = DummyCore(
        L=4,
        columnlayers=[DummyColumnLayer(4), DummyColumnLayer(4), DummyColumnLayer(4), DummyColumnLayer(4)],
    )

    with pytest.raises(ValueError):
        _ = core_layer(torch.tensor(1.0, dtype=torch.float32))


@pytest.mark.parametrize("L", [2, 3, 4, 6, 8])
def test_clements_mesh_transfer_matrix_is_unitary(L: int) -> None:
    """ClementsMesh transfer matrix must be unitary: U^H @ U == I and U @ U^H == I."""
    torch.manual_seed(42 + L)
    mesh = ClementsMesh(L=L)
    u = mesh.get_transfer_matrix()

    assert u.shape == (L, L)
    identity_right = u.conj().T @ u
    identity_left = u @ u.conj().T

    torch.testing.assert_close(
        identity_right,
        torch.eye(L, dtype=torch.complex64),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        identity_left,
        torch.eye(L, dtype=torch.complex64),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("L", [2, 4, 8])
def test_clements_mesh_energy_conservation(L: int) -> None:
    """Lossless ClementsMesh forward pass must conserve optical energy (Parseval identity)."""
    torch.manual_seed(100 + L)
    mesh = ClementsMesh(L=L)

    # Batch of random complex optical fields
    input_fields = torch.randn(10, L, dtype=torch.complex64)
    output_fields = mesh(input_fields)

    input_energy = (input_fields.abs() ** 2).sum(dim=-1)
    output_energy = (output_fields.abs() ** 2).sum(dim=-1)

    torch.testing.assert_close(output_energy, input_energy, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("L", [4, 6, 8])
def test_clements_mesh_field_prop_matches_matrix_mode(L: int) -> None:
    """Field-propagation forward mode must match matrix-based forward mode exactly."""
    from ioptics.layers import ForwardMode
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(2026 + L)
        mesh_matrix = ClementsMesh(L=L, forward_mode=ForwardMode.MATRIX.value)
        mesh_field = ClementsMesh(L=L, forward_mode=ForwardMode.FIELD_PROP.value)

        # Share weights
        with torch.no_grad():
            for p1, p2 in zip(mesh_matrix.parameters(), mesh_field.parameters()):
                p2.copy_(p1)

        inp = torch.randn(8, L, dtype=torch.complex64, requires_grad=True)
        inp_copy = inp.clone().detach().requires_grad_(True)

        out_matrix = mesh_matrix(inp)
        out_field = mesh_field(inp_copy)

        torch.testing.assert_close(out_field, out_matrix, atol=1e-5, rtol=1e-5)

        # Test backward pass equivalence
        loss_matrix = out_matrix.abs().sum()
        loss_field = out_field.abs().sum()

        loss_matrix.backward()
        loss_field.backward()

        torch.testing.assert_close(inp_copy.grad, inp.grad, atol=1e-5, rtol=1e-5)
        for p1, p2 in zip(mesh_matrix.parameters(), mesh_field.parameters()):
            assert p1.grad is not None and p2.grad is not None
            torch.testing.assert_close(p2.grad, p1.grad, atol=1e-5, rtol=1e-5)
    finally:
        torch.set_rng_state(rng_state)


def test_clements_mesh_and_fcarray_torch_compile_compatibility() -> None:
    """Verify functional scatter implementation compiles cleanly under torch.compile."""
    torch.manual_seed(42)
    mesh = ClementsMesh(L=4)
    fc = FCArray(in_dim=4, out_dim=3)
    x = torch.randn(4, 4, dtype=torch.complex64, requires_grad=True)
    x_comp = x.detach().clone().requires_grad_(True)

    out_eager = fc(mesh(x))
    (out_eager.abs() ** 2).sum().backward()

    compiled_fn = torch.compile(lambda inp: fc(mesh(inp)), backend="aot_eager")
    for p in list(mesh.parameters()) + list(fc.parameters()):
        p.grad = None

    out_comp = compiled_fn(x_comp)
    (out_comp.abs() ** 2).sum().backward()

    torch.testing.assert_close(out_comp, out_eager, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_comp.grad, x.grad, atol=1e-5, rtol=1e-5)


def test_transfer_matrix_cache_hit_and_invalidation() -> None:
    """Verify CoreLayer static transfer matrix caching and automatic invalidation."""
    mesh = ClementsMesh(L=4, phase_noise_std=0.2)
    fc = FCArray(in_dim=4, out_dim=3)

    # 1. Cache hit on repeated calls without parameter updates
    tm1 = mesh.get_transfer_matrix()
    tm2 = mesh.get_transfer_matrix()
    assert tm1 is tm2
    assert mesh._cached_transfer_matrix is tm1

    fc_tm1 = fc.get_transfer_matrix()
    fc_tm2 = fc.get_transfer_matrix()
    assert fc_tm1 is fc_tm2

    # 2. Switching train <-> eval invalidates cache
    mesh.eval()
    assert mesh._cached_transfer_matrix is None
    tm_eval1 = mesh.get_transfer_matrix()
    tm_eval2 = mesh.get_transfer_matrix()
    assert tm_eval1 is tm_eval2
    assert not torch.allclose(tm1, tm_eval1, atol=1e-3, rtol=1e-3)

    # 3. In-place parameter update invalidates cache automatically
    mesh.train()
    tm_train = mesh.get_transfer_matrix()
    with torch.no_grad():
        next(mesh.parameters()).add_(0.5)
    tm_after_update = mesh.get_transfer_matrix()
    assert tm_after_update is not tm_train
    assert not torch.allclose(tm_train, tm_after_update, atol=1e-4, rtol=1e-4)

    # 4. Backward pass invalidates autograd-attached cache so consecutive backwards succeed
    x = torch.randn(2, 4, dtype=torch.complex64)
    out1 = mesh(x)
    (out1.abs() ** 2).sum().backward()
    assert mesh._cached_transfer_matrix is None

    out2 = mesh(x)
    (out2.abs() ** 2).sum().backward()

    # 5. reset_parameters() invalidates cache
    _ = mesh.get_transfer_matrix()
    assert mesh._cached_transfer_matrix is not None
    mesh.reset_parameters()
    assert mesh._cached_transfer_matrix is None

    # 6. use_cache=False bypasses cache
    tm_cached = mesh.get_transfer_matrix()
    tm_uncached = mesh.get_transfer_matrix(use_cache=False)
    assert tm_uncached is not tm_cached
    torch.testing.assert_close(tm_uncached, tm_cached, atol=1e-6, rtol=1e-6)


def test_cache_single_param_perturbation_and_multilayer_non_backprop() -> None:
    """Verify column-level cache and prefix/suffix fast-path on finite-difference & parameter-shift."""
    from ioptics.differentiators import ParameterShiftDifferentiator
    from ioptics.models import OpticalNN
    from ioptics.nonlinear import Abs

    torch.manual_seed(2026)

    # 1. Single interior parameter .data mutation in eval() mode on large mesh (L=16)
    mesh = ClementsMesh(L=16)
    mesh.eval()
    x = torch.randn(4, 16, dtype=torch.complex64)
    _ = mesh(x)

    # Pick an interior parameter in the middle column (col 7, component 3)
    interior_param = mesh.columnlayers[7].components[3].theta
    eps = 1e-2
    orig_val = interior_param.data.clone()

    # Compute analytical gradient for that parameter first
    mesh.train()
    out_ag = mesh(x)
    loss_ag = (out_ag.abs() ** 2).sum()
    loss_ag.backward()
    expected_grad = interior_param.grad.detach().clone()

    # Now switch to eval() and compute finite-difference via .data.copy_()
    mesh.eval()
    with torch.no_grad():
        _ = mesh(x)  # populate cache in eval mode
        interior_param.data.copy_(orig_val + eps)
        loss_plus = (mesh(x).abs() ** 2).sum()
        interior_param.data.copy_(orig_val - eps)
        loss_minus = (mesh(x).abs() ** 2).sum()
        interior_param.data.copy_(orig_val)
        fd_grad = (loss_plus - loss_minus) / (2 * eps)

    torch.testing.assert_close(fd_grad.view_as(expected_grad), expected_grad, atol=1e-3, rtol=1e-2)

    # 2. Multi-layer OpticalNN with ParameterShiftDifferentiator: verify cached vs uncached match
    mesh1 = ClementsMesh(L=4)
    mesh2 = ClementsMesh(L=4)
    model = OpticalNN(mesh1, mesh2, Abs())
    inputs = torch.randn(6, 4, dtype=torch.float32)
    labels = torch.randint(0, 4, (6,))
    loss_fn = torch.nn.CrossEntropyLoss()

    diff = ParameterShiftDifferentiator()
    grads_cached = diff.differentiate(model, inputs, loss_fn, y_label=labels)

    mesh1.cache_enabled = False
    mesh2.cache_enabled = False
    for col in list(mesh1.columnlayers) + list(mesh2.columnlayers):
        col.cache_enabled = False
    grads_uncached = diff.differentiate(model, inputs, loss_fn, y_label=labels)

    torch.testing.assert_close(grads_cached, grads_uncached, atol=1e-5, rtol=1e-4)


