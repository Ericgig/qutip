# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import pytest
import numbers
import numpy as np
import scipy.sparse as sp
from functools import partial
import qutip
from qutip import Qobj
from qutip.core.operators import *
from qutip.core.data import to

dtype_names = list(to._str2type.keys()) + list(to.dtypes)
dtype_types = list(to._str2type.values()) + list(to.dtypes)
dtypes = zip(dtype_names, dtype_types)
dtype_ids = [str(dtype) for dtype in dtype_names]

@pytest.fixture(params=dtypes, ids=dtype_ids)
def dtype(request):
    return request.param

N = 5


def test_jmat_12():
    "Spin 1/2 operators"
    spinhalf = jmat(1 / 2.)

    paulix = np.array([[0. + 0.j, 0.5 + 0.j],[0.5 + 0.j, 0. + 0.j]])
    pauliy = np.array([[0. + 0.j, 0. - 0.5j], [0. + 0.5j, 0. + 0.j]])
    pauliz = np.array([[0.5 + 0.j, 0. + 0.j], [0. + 0.j, -0.5 + 0.j]])
    sigma_p = np.array([[0. + 0.j, 1. + 0.j], [0. + 0.j, 0. + 0.j]])
    sigma_m = np.array([[0. + 0.j, 0. + 0.j], [1. + 0.j, 0. + 0.j]])

    assert np.allclose(spinhalf[0].full(), paulix)
    assert np.allclose(spinhalf[1].full(), pauliy)
    assert np.allclose(spinhalf[2].full(), pauliz)
    assert np.allclose(jmat(1 / 2., '+').full(), sigma_p)
    assert np.allclose(jmat(1 / 2., '-').full(), sigma_m)

    assert np.allclose(spin_Jx(1 / 2.).full(), paulix)
    assert np.allclose(spin_Jy(1 / 2.).full(), pauliy)
    assert np.allclose(spin_Jz(1 / 2.).full(), pauliz)
    assert np.allclose(spin_Jp(1 / 2.).full(), sigma_p)
    assert np.allclose(spin_Jm(1 / 2.).full(), sigma_m)

    assert np.allclose(sigmax().full(), paulix*2)
    assert np.allclose(sigmay().full(), pauliy*2)
    assert np.allclose(sigmaz().full(), pauliz*2)
    assert np.allclose(sigmap().full(), sigma_p)
    assert np.allclose(sigmam().full(), sigma_m)


def test_jmat_32():
    "Spin 3/2 operators"
    spin32 = jmat(3 / 2.)

    paulix32 = np.array(
        [[0.0000000, 0.8660254, 0.0000000, 0.0000000],
         [0.8660254, 0.0000000, 1.0000000, 0.0000000],
         [0.0000000, 1.0000000, 0.0000000, 0.8660254],
         [0.0000000, 0.0000000, 0.8660254, 0.0000000]],
        dtype=np.complex128)

    pauliy32 = np.array(
        [[0. + 0.j, 0. - 0.8660254j, 0. + 0.j, 0. + 0.j],
         [0. + 0.8660254j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
         [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. - 0.8660254j],
         [0. + 0.j, 0. + 0.j, 0. + 0.8660254j, 0. + 0.j]])

    pauliz32 = np.array([[1.5, 0.0, 0.0, 0.0],
                         [0.0, 0.5, 0.0, 0.0],
                         [0.0, 0.0,-0.5, 0.0],
                         [0.0, 0.0, 0.0,-1.5]],
                        dtype=np.complex128)

    assert np.allclose(spin32[0].full(), paulix32)
    assert np.allclose(spin32[1].full(), pauliy32)
    assert np.allclose(spin32[2].full(), pauliz32)


@pytest.mark.parametrize(['j_func1', 'j_func2'], [
    pytest.param(spin_Jx, partial(jmat, which="x"), id="x"),
    pytest.param(spin_Jy, partial(jmat, which="y"), id="y"),
    pytest.param(spin_Jz, partial(jmat, which="z"), id="z"),
    pytest.param(spin_Jp, partial(jmat, which="+"), id="+"),
    pytest.param(spin_Jm, partial(jmat, which="-"), id="-"),
])
@pytest.mark.parametrize(['spin', 'N'], [
    pytest.param(3/2., 4, id="1.5"),
    pytest.param(5/2., 6, id="2.5"),
    pytest.param(3.0, 7, id="3.0"),
])
def test_jmat(j_func1, j_func2, spin, N, dtype):
    "Spin 2 operators"
    spin_mat1 = j_func1(spin, dtype=dtype[0])
    spin_mat2 = j_func2(spin, dtype=dtype[0])
    assert spin_mat1 == spin_mat2
    assert spin_mat1.dims == [[N], [N]]
    assert spin_mat1.shape == (N, N)
    assert isinstance(spin_mat1.data, dtype[1])
    assert isinstance(spin_mat2.data, dtype[1])



@pytest.mark.parametrize(['oper_func', 'diag', 'offset', 'args'], [
    pytest.param(destroy, np.arange(1, N)**0.5, 1, (), id="destroy"),
    pytest.param(destroy, np.arange(6, N+5)**0.5, 1, (5,),
                 id="destroy_offset"),
    pytest.param(create, np.arange(1, N)**0.5, -1, (), id="create"),
    pytest.param(create, np.arange(6, N+5)**0.5, -1, (5,),
                 id="create_offset"),
    pytest.param(num, np.arange(N), 0, (), id="num"),
    pytest.param(num, np.arange(5, N+5), 0, (5,), id="num_offset"),
    pytest.param(charge, np.arange(-N, N+1), 0, (), id="charge"),
    pytest.param(charge, np.arange(2, N+1)/3, 0, (2, 1/3), id="charge_args"),
])
def test_diagonal_oper(oper_func, diag, offset, args, dtype):
    oper = oper_func(N, *args, dtype=dtype[0])
    assert isinstance(oper.data, dtype[1])
    assert oper == Qobj(sp.diags(diag, offset))


@pytest.mark.parametrize("to_test, expected", [
        (qzero, lambda x: np.zeros((x, x), dtype=complex)),
        (qeye, lambda x: np.eye(x, dtype=complex)),
    ])
@pytest.mark.parametrize("dimension", [1, 5, 100])
def test_simple_operator_creation(to_test, expected, dimension, dtype):
    qobj = to_test(dimension, dtype=dtype[0])
    assert isinstance(qobj.data, dtype[1])
    assert np.allclose(qobj.full(), expected(np.prod(dimension)))


@pytest.mark.parametrize("to_test", [qzero, qeye, identity])
@pytest.mark.parametrize("dimensions", [
        2,
        [2],
        [2, 3, 4],
        1,
        [1],
        [1, 1],
    ])
def test_implicit_tensor_creation(to_test, dimensions):
    implicit = to_test(dimensions)
    if isinstance(dimensions, numbers.Integral):
        dimensions = [dimensions]
    assert implicit.dims == [dimensions, dimensions]


@pytest.mark.parametrize("to_test", [qzero, qeye, identity])
def test_super_operator_creation(to_test):
    size = 2
    implicit = to_test([[size], [size]])
    explicit = qutip.to_super(to_test(size))
    assert implicit == explicit


def test_position(dtype):
    "position operator"
    N = 5
    pos = position(N, dtype=dtype[0])
    pos_matrix = (np.diag((np.arange(1, N)/2)**0.5, k=-1) +
                  np.diag((np.arange(1, N)/2)**0.5, k=1))
    assert np.allclose(pos.full(), pos_matrix)
    assert isinstance(pos.data, dtype[1])


def test_momentum(dtype):
    "momentum operator"
    N = 5
    mom = momentum(N, dtype=dtype[0])
    mom_matrix = (np.diag((np.arange(1, N)/2)**0.5, k=-1) +-
                  np.diag((np.arange(1, N)/2)**0.5, k=1)) * 1j
    assert np.allclose(mom.full(), mom_matrix)
    assert isinstance(mom.data, dtype[1])


def test_squeeze(dtype):
    "Squeezing operator"
    sq = squeeze(4, 0.1 + 0.1j, dtype=dtype[0])
    sqmatrix = np.array([[0.99500417 + 0.j, 0.00000000 + 0.j,
                          0.07059289 - 0.07059289j, 0.00000000 + 0.j],
                         [0.00000000 + 0.j, 0.98503746 + 0.j,
                          0.00000000 + 0.j, 0.12186303 - 0.12186303j],
                         [-0.07059289 - 0.07059289j, 0.00000000 + 0.j,
                          0.99500417 + 0.j, 0.00000000 + 0.j],
                         [0.00000000 + 0.j, -0.12186303 - 0.12186303j,
                          0.00000000 + 0.j, 0.98503746 + 0.j]])
    assert isinstance(sq.data, dtype[1])
    assert np.allclose(sq.full(), sqmatrix)


def test_displace(dtype):
    "Displacement operator"
    dp = displace(4, 0.25, dtype=dtype[0])
    dpmatrix = np.array(
        [[0.96923323 + 0.j, -0.24230859 + 0.j, 0.04282883 + 0.j, -
          0.00626025 + 0.j],
         [0.24230859 + 0.j, 0.90866411 + 0.j, -0.33183303 +
          0.j, 0.07418172 + 0.j],
         [0.04282883 + 0.j, 0.33183303 + 0.j, 0.84809499 +
          0.j, -0.41083747 + 0.j],
         [0.00626025 + 0.j, 0.07418172 + 0.j, 0.41083747 + 0.j,
          0.90866411 + 0.j]])

    assert np.allclose(dp.full(), dpmatrix)
    assert isinstance(dp.data, dtype[1])


def test_tunneling(dtype):
    "Tunneling operator"
    N = 5
    tn = tunneling(2*N+1, dtype=dtype[0])
    tn_matrix = np.diag(np.ones(2*N),k=-1) + np.diag(np.ones(2*N),k=1)
    assert np.allclose(tn.full(), tn_matrix)
    assert isinstance(tn.data, dtype[1])

    tn = tunneling(2*N+1, 2, dtype=dtype[0])
    tn_matrix = np.diag(np.ones(2*N-1),k=-2) + np.diag(np.ones(2*N-1),k=2)
    assert np.allclose(tn.full(), tn_matrix)
    assert isinstance(tn.data, dtype[1])


def test_commutator():
    A = qeye(N)
    B = destroy(N)
    assert commutator(A, B) == qzero(N)

    sx = sigmax()
    sy = sigmay()
    assert commutator(sx, sy)/2 == (sigmaz()*1j)
