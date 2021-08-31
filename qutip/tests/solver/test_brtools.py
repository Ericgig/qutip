import pytest
import numpy as np
from qutip.solver._brtools import matmul_var_data, _EigenBasisTransform
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


def test_eigen_tool():
    a = qutip.destroy(5)
    f = lambda t, _: t
    op = qutip.QobjEvo([a*a.dag(), [a+a.dag(), f]])
    eigenT = _EigenBasisTransform(op)
    evecs_qevo = eigenT.as_Qobj()

    for t in [0, 1, 1.5]:
        eigenvals, ekets = op(t).eigenstates()
        np.testing.assert_allclose(eigenvals, eigenT.eigenvalues(t))
        np.testing.assert_allclose(
            np.abs(np.hstack([eket.full() for eket in ekets])),
            np.abs(eigenT.evecs(t).to_array())
        )
        np.testing.assert_allclose(np.abs(evecs_qevo(t).full()),
                                   np.abs(eigenT.evecs(t).to_array()))


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
