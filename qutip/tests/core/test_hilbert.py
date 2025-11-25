import qutip
from qutip import spre, spost, sprepost, Qobj
import qutip.core.data as _data
from qutip.core._qobj.oper import Operator
from qutip.tests.core.data.conftest import (
    random_diag, random_dense, random_csr
)
import pytest
import numpy as np


def random_data(shape):
    if np.random.randint(3) == 2:
        return random_dense(shape, np.random.randint(2))
    elif np.random.randint(2):
        return random_csr(shape, 0.3, True)
    else:
        return random_diag(shape, 0.3, True)


def _make_data_oper_pair_single(dims, modes):
    shape = [1, 1]
    for mode in modes:
        shape[0] *= dims[0][mode]
        shape[1] *= dims[1][mode]
    a = random_data(shape)
    oper_a = Operator(a, modes=modes, dimension=dims)
    return a, oper_a


def _make_data_oper_pair_multi(dims, modes):
    N = 3
    shape = [1, 1]
    dims_a = [dims[0].copy(), dims[1].copy()]
    dims_b = [dims[0].copy(), dims[1].copy()]
    for mode in modes:
        shape[0] *= dims[0][mode]
        shape[1] *= dims[1][mode]
        dims_a[1][mode] = 1
        dims_b[0][mode] = 1

    dims_a[1][modes[0]] = N
    dims_b[0][modes[0]] = N

    a = random_data((shape[0], N))
    b = random_data((N, shape[1]))
    c = random_data(shape)
    oa = Operator(a, modes=modes, dimension=dims)
    ob = Operator(b, modes=modes, dimension=dims)
    oc = Operator(c, modes=modes, dimension=dims)
    return a @ b + c, a @ b + c


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (1, 10), (4, 2)])
@pytest.mark.parametrize("mode", [(0,), (0, 1)])
@pytest.mark.parametrize("factory", [
    _make_data_oper_pair_single,
    _make_data_oper_pair_multi,
])
def test_unary_operation(shape, mode, factory):
    dims = [[shape[0]], [shape[1]]]
    for _ in range(len(mode) - 1):
        dims[0].append(2)
        dims[1].append(2)
    a, oper_a = factory(dims, mode)

    assert a.conj() == (oper_a.conj()).to_data()
    assert a.transpose() == (oper_a.transpose()).to_data()
    assert a.adjoint() == (oper_a.adjoint()).to_data()

    assert (-a) == (-oper_a).to_data()
    assert (a * 2j) == (oper_a * 2j).to_data()
    assert (a / 3 ) == (oper_a / 3).to_data()


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (1, 10), (4, 2)])
@pytest.mark.parametrize("mode", [(0,), (0, 1)])
@pytest.mark.parametrize("factory", [
    _make_data_oper_pair_single,
    _make_data_oper_pair_multi,
])
def test_binary_operation(shape, mode, factory):
    dims = [[shape[0]], [shape[1]]]
    for _ in range(len(mode) - 1):
        dims[0].append(2)
        dims[1].append(2)
    a, oper_a = factory(dims, mode)
    b, oper_b = factory(dims[::-1], mode)

    assert (a + a) == (oper_a + oper_a).to_data()
    assert (a @ b) == (oper_a @ oper_b).to_data()
    assert (b - b) == (oper_b - oper_b).to_data()
    assert qutip.data.core.kron(a, b) == (oper_a & oper_b).to_data()


@pytest.mark.parametrize("mode", [(0,), (0, 1)])
@pytest.mark.parametrize("factory", [
    _make_data_oper_pair_single,
    _make_data_oper_pair_multi,
])
def test_super_operation(mode, factory):
    dims = [[2] * 4, [2] * 4]
    a, oper_a = factory(dims, mode)
    b, oper_b = factory(dims, mode)
    assert spre(Qobj(a)).data == oper_a.spre().to_data()
    assert spost(Qobj(a)).data == oper_a.spost().to_data()
    assert sprepost(Qobj(a), Qobj(b)).data == oper_a.sprepost(oper_b).to_data()


def test_super_tensor(mode, factory):
    dims = [[2], [2]]
    a, oper_a = _make_data_oper_pair_multi(dims, (0,))
    b, oper_b = _make_data_oper_pair_multi(dims, (0,))
    a, b = Qobj(a), Qobj(b)

    c = sprepost(a & b, a & b).data
    oper_c = oper_a.sprepost(oper_a) & oper_b.sprepost(oper_b)
    assert c == oper_c.to_data()
