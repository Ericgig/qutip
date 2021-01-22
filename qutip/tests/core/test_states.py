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
import numpy as np
import scipy.sparse as sp
import qutip
from qutip import Qobj
from qutip.core.states import *
from qutip.core.data import to
from functools import partial

dtype_names = list(to._str2type.keys()) + list(to.dtypes)
dtype_types = list(to._str2type.values()) + list(to.dtypes)
dtypes = zip(dtype_names, dtype_types)
dtype_ids = [str(dtype) for dtype in dtype_names]

@pytest.fixture(params=dtypes, ids=dtype_ids)
def dtype(request):
    return request.param



@pytest.mark.parametrize("size, n", [(2, 0), (2, 1), (100, 99)])
def test_basis_simple(size, n):
    qobj = basis(size, n)
    numpy = np.zeros((size, 1), dtype=complex)
    numpy[n, 0] = 1
    assert np.array_equal(qobj.full(), numpy)


@pytest.mark.parametrize("to_test", [basis, fock, fock_dm])
@pytest.mark.parametrize("size, n", [([2, 2], [0, 1]), ([2, 3, 4], [1, 2, 0])])
def test_implicit_tensor_basis_like(to_test, size, n):
    implicit = to_test(size, n)
    explicit = qutip.tensor(*[to_test(ss, nn) for ss, nn in zip(size, n)])
    assert implicit == explicit


@pytest.mark.parametrize("size, n, m", [
        ([2, 2], [0, 0], [1, 1]),
        ([2, 3, 4], [1, 2, 0], [0, 1, 3]),
    ])
def test_implicit_tensor_projection(size, n, m):
    implicit = projection(size, n, m)
    explicit = qutip.tensor(*[projection(ss, nn, mm)
                              for ss, nn, mm in zip(size, n, m)])
    assert implicit == explicit

@pytest.mark.parametrize("base, operator, args, opargs, eigenval", [
    pytest.param(basis, qutip.num, (10, 3), (10,), 3,
     id="basis"),
    pytest.param(basis, qutip.num, (10, 3, 1), (10, 1), 3,
     id="basis_offset"),
    pytest.param(fock, qutip.num, (10, 3), (10,), 3,
     id="fock"),
    pytest.param(fock_dm, qutip.num, (10, 3), (10,), 3,
     id="fock_dm"),
    pytest.param(fock_dm, qutip.num, (10, 3, 1), (10, 1), 3,
     id="fock_dm_offset"),
    pytest.param(coherent, qutip.destroy, (20, 0.75), (20,), 0.75,
     id="coherent"),
    pytest.param(coherent, qutip.destroy, (25, 1.25, 1), (25, 1), 1.25,
     id="coherent_offset"),
    pytest.param(coherent_dm, qutip.destroy, (25, 1.25), (25,), 1.25,
     id="coherent_dm"),
    pytest.param(phase_basis, qutip.phase, (10, 3), (10,), 3*2*np.pi/10,
     id="phase_basis"),
    pytest.param(phase_basis, qutip.phase, (10, 3, 1), (10, 1), 3*2*np.pi/10+1,
     id="phase_basis_phi0"),
    pytest.param(spin_state, qutip.spin_Jz, (3, 2), (3,), 2,
     id="spin_state"),
    pytest.param(zero_ket, qutip.qeye, (10,), (10,), 0,
     id="zero_ket"),
])
def test_div_basis(base, operator, args, opargs, eigenval, dtype):
    state = base(*args, dtype=dtype[0])
    oper = operator(*opargs)
    assert np.allclose((oper * state).full(), state.full() * eigenval)
    assert isinstance(state.data, dtype[1])


@pytest.mark.parametrize('dm', [
    partial(thermal_dm, n=1.),
    maximally_mixed_dm,
    partial(coherent_dm, alpha=0.5),
    partial(fock_dm, n=1),
    partial(spin_state, m=2, type='dm'),
    partial(spin_coherent, theta=1, phi=2, type='dm'),
], ids=[
    'thermal_dm', 'maximally_mixed_dm', 'coherent_dm',
    'fock_dm', 'spin_state', 'spin_coherent'
])
def test_dm(dm, dtype):
    N = 5
    rho = dm(N, dtype=dtype[0])
    # make sure rho has trace close to 1.0
    assert abs(rho.tr() - 1.0) < 1e-12
    assert isinstance(rho.data, dtype[1])

def test_CoherentState():
    """
    states: coherent state
    """
    N = 10
    alpha = 0.5
    c1 = coherent(N, alpha) # displacement method
    c2 = coherent(7, alpha, offset=3) # analytic method
    assert abs(qutip.expect(qutip.destroy(N), c1) - alpha) < 1e-10
    assert (qutip.Qobj(c1[3:]) - c2).norm() < 1e-7


def test_TripletStateNorm():
    """
    Test the states returned by function triplet_states are normalized.
    """
    for triplet in triplet_states():
        assert abs(triplet.norm() - 1.) < 1e-12


def test_ket2dm():
    N = 5
    state = qutip.coherent(N, 2)
    oper = ket2dm(state)
    assert np.abs(qutip.expect(oper, state) - 1) < 1e-12


def test_w_states(dtype):
    state = (qstate("uddd", dtype=dtype[0]) +
             qstate("dudd", dtype=dtype[0]) +
             qstate("ddud", dtype=dtype[0]) +
             qstate("dddu", dtype=dtype[0]))
    assert isinstance(state.data, dtype[1])
    ket_state = (ket("1000", dtype=dtype[0]) +
                 ket("gegg", dtype=dtype[0]) +
                 ket("uudu", dtype=dtype[0]) +
                 ket("HHHV", dtype=dtype[0]))
    assert isinstance(ket_state.data, dtype[1])
    assert state == ket_state
    w_ket = w_state(4, dtype=dtype[0])
    assert state * 0.5 == w_ket
    assert isinstance(w_ket.data, dtype[1])


def test_ghz_states(dtype):
    state = (qstate("uuu", dtype=dtype[0]) +
             qstate("ddd", dtype=dtype[0]))
    ghz_ket = ghz_state(3, dtype=dtype[0])
    assert state * 0.5**0.5 == ghz_ket
    assert isinstance(ghz_ket.data, dtype[1])
