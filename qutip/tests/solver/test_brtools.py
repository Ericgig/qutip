import pytest
import numpy as np
from qutip.solver._brtools import matmul_var_data, _EigenBasisTransform
from qutip.solver._brtensor import (_br_term_dense, _br_term_sparse,
                                    brtensor, _BlochRedfieldElement)
import qutip

def _make_rand_data(shape):
    np.random.seed(11)
    array = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    return qutip.data.Dense(array)


transform = {
    0: lambda x: x,
    1: qutip.data.transpose,
    2: qutip.data.conj,
    3: qutip.data.adjoint
}

@pytest.mark.parametrize('datatype', qutip.data.to.dtypes)
@pytest.mark.parametrize('transleft', [0, 1, 2, 3],
                         ids=['', 'transpose', 'conj', 'dag'])
@pytest.mark.parametrize('transright', [0, 1, 2, 3],
                         ids=['', 'transpose', 'conj', 'dag'])
def test_matmul_var(datatype, transleft, transright):
    shape = (5, 5)
    np.random.seed(11)
    left = qutip.data.to(datatype, _make_rand_data(shape))
    right = qutip.data.to(datatype, _make_rand_data(shape))

    expected = qutip.data.matmul(
        transform[transleft](left),
        transform[transright](right),
        ).to_array()

    computed = matmul_var_data(left, right, transleft, transright).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14)


@pytest.mark.parametrize('sparse', [False, True], ids=['Dense', 'Sparse'])
def test_eigen_transform(sparse):
    a = qutip.destroy(5)
    f = lambda t, _: t
    op = qutip.QobjEvo([a*a.dag(), [a+a.dag(), f]])
    eigenT = _EigenBasisTransform(op, sparse=sparse)
    evecs_qevo = eigenT.as_Qobj()

    for t in [0, 1, 1.5]:
        eigenvals, ekets = op(t).eigenstates()
        np.testing.assert_allclose(eigenvals, eigenT.eigenvalues(t),
                                   rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(
            np.abs(np.hstack([eket.full() for eket in ekets])),
            np.abs(eigenT.evecs(t).to_array()),
            rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(np.abs(evecs_qevo(t).full()),
                                   np.abs(eigenT.evecs(t).to_array()),
                                   rtol=1e-14, atol=1e-14)


def test_eigen_transform_ket():
    N = 5
    a = qutip.destroy(N)
    op = a*a.dag() + a + a.dag()
    eigenT = _EigenBasisTransform(qutip.QobjEvo(op))
    op_diag = qutip.qdiags(eigenT.eigenvalues(0), [0])

    state = qutip.coherent(N, 1.1)

    expected = (op @ state).full()
    computed = eigenT.from_eigbasis(
        0,
        qutip.data.matmul(op_diag.data, eigenT.to_eigbasis(0, state.data))
    ).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14, atol=1e-14)


def test_eigen_transform_dm():
    N = 5
    a = qutip.destroy(N)
    op = a*a.dag() + a + a.dag()
    eigenT = _EigenBasisTransform(qutip.QobjEvo(op))
    op_diag = qutip.qdiags(eigenT.eigenvalues(0), [0])

    state = qutip.coherent_dm(N, 1.1)

    expected = (op @ state).full()
    computed = eigenT.from_eigbasis(
        0,
        qutip.data.matmul(op_diag.data, eigenT.to_eigbasis(0, state.data))
    ).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14, atol=1e-14)


def test_eigen_transform_oper_ket():
    N = 5
    a = qutip.destroy(N)
    op = a*a.dag() + a + a.dag()
    eigenT = _EigenBasisTransform(qutip.QobjEvo(op))
    op_diag = qutip.qdiags(eigenT.eigenvalues(0), [0])

    state = qutip.operator_to_vector(qutip.coherent_dm(N, 1.1))

    expected = (qutip.spre(op) @ state).full()
    computed = eigenT.from_eigbasis(
        0,
        qutip.data.matmul(qutip.spre(op_diag).data,
                          eigenT.to_eigbasis(0, state.data))
    ).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14, atol=1e-14)


def test_eigen_transform_super_ops():
    N = 5
    a = qutip.destroy(N)
    op = a*a.dag() + a + a.dag()
    eigenT = _EigenBasisTransform(qutip.QobjEvo(op))
    op_diag = qutip.qdiags(eigenT.eigenvalues(0), [0])

    state = qutip.sprepost(
        qutip.coherent_dm(N, 1.1),
        qutip.thermal_dm(N, 1.1)
    )

    expected = (qutip.spre(op) @ state).full()
    computed = eigenT.from_eigbasis(
        0,
        qutip.data.matmul(qutip.spre(op_diag).data,
                          eigenT.to_eigbasis(0, state.data))
    ).to_array()
    np.testing.assert_allclose(computed, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('func', [_br_term_dense, _br_term_sparse],
                         ids=['dense', 'sparse'])
def test_brterm_linbblad_comp(func):
    N = 5
    a = qutip.destroy(N) + qutip.destroy(N)**2/2
    A_op = a + a.dag()
    H = qutip.num(N)
    diag = H.eigenenergies()
    skew =  np.einsum('i,j->ji', np.ones(N), diag) - diag * np.ones((N, N))
    spectrum = (skew > 0) * 1.
    computation = func(A_op.data, spectrum, skew, 2).to_array()
    lindblad = qutip.lindblad_dissipator(a).full()
    np.testing.assert_allclose(computation, lindblad, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('cutoff', [0, 0.1, 1, 3, np.inf])
@pytest.mark.parametrize('spectra', [
    pytest.param(lambda skew: (skew > 0) * 1., id='pos_filter'),
    pytest.param(lambda skew: np.ones_like(skew), id='no_filter'),
    pytest.param(lambda skew: np.exp(skew)/10, id='smooth_filter'),
])
def test_brterm(cutoff, spectra):
    N = 5
    a = qutip.destroy(N) @ qutip.coherent_dm(N, 0.5) * 0.5
    A_op = a + a.dag()
    H = qutip.num(N)
    diag = H.eigenenergies()
    skew =  np.einsum('i,j->ji', np.ones(N), diag) - diag * np.ones((N, N))
    spectrum = spectra(skew)
    R_dense = _br_term_dense(A_op.data, spectrum, skew, cutoff).to_array()
    R_sparse = _br_term_sparse(A_op.data, spectrum, skew, cutoff).to_array()
    np.testing.assert_allclose(R_dense, R_sparse, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('cutoff', [0, 0.1, 1, 3, np.inf])
def test_brtensor(cutoff):
    N = 5
    H = qutip.num(N)
    a = qutip.destroy(N)
    A_op = a + a.dag()
    spectra = qutip.coefficient("(w>0)*0.5", args={'w':0})
    R = brtensor(H, A_op, spectra, cutoff<1e15, cutoff, fock_basis=True)
    R_eigs, evecs = brtensor(H, A_op, spectra, cutoff<1e15, cutoff,
                             fock_basis=False)
    assert isinstance(R, qutip.Qobj)
    assert isinstance(R_eigs, qutip.Qobj)
    assert isinstance(evecs, qutip.Qobj)
    state = qutip.operator_to_vector(qutip.rand_dm(N))
    fock_computed = R @ state
    eig_computed = R_eigs @ qutip.sprepost(evecs.dag(), evecs) @ state
    eig_computed = qutip.sprepost(evecs, evecs.dag()) @ eig_computed
    np.testing.assert_allclose(fock_computed.full(), eig_computed.full(),
                               rtol=1e-14, atol=1e-14)

@pytest.mark.parametrize('cutoff', [0, 0.1, 1, 3, np.inf])
def test_td_brtensor(cutoff):
    N = 5
    H = qutip.QobjEvo([qutip.num(N), "0.5+t**2"])
    a = qutip.destroy(N)
    A_op = qutip.QobjEvo([a + a.dag(), "t"])
    spectra = qutip.coefficient("(w>0)*0.5", args={'w':0})
    R = brtensor(H, A_op, spectra, cutoff<1e15, cutoff, fock_basis=True)
    R_eigs, evecs = brtensor(H, A_op, spectra, cutoff<1e15, cutoff,
                             fock_basis=False)
    assert isinstance(R, qutip.QobjEvo)
    assert isinstance(R_eigs, qutip.QobjEvo)
    assert isinstance(evecs, qutip.QobjEvo)
    state = qutip.operator_to_vector(qutip.rand_dm(N))
    fock_computed = R @ state
    eig_computed = R_eigs @ qutip.sprepost(evecs.dag(), evecs) @ state
    eig_computed = qutip.sprepost(evecs, evecs.dag()) @ eig_computed
    for t in [0, 0.5, 1.0]:
        np.testing.assert_allclose(
            (R(t) @ state).full(),
            R.matmul(t, state).full(),
            rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(fock_computed(t).full(),
                                   eig_computed(t).full(),
                                   rtol=1e-14, atol=1e-14)






















#
