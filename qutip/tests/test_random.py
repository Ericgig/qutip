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

import numpy as np
import pytest
from functools import partial
from qutip import qeye
from qutip.random_objects import *
from qutip.core.data import to

N = 6
rank = 3

def _valid_rand_unitary(rand_qobj):
    "Is it Unitary?"
    I = qeye(N)
    I.dims = rand_qobj.dims  # Do not test dims here
    assert rand_qobj * rand_qobj.dag() == I


def _valid_rand_herm(rand_qobj):
    "Is it hermitian?"
    assert rand_qobj.isherm


def _valid_rand_herm_PosDef(rand_qobj):
    "Is it hermitian + Positive semi-def"
    assert all(rand_qobj.eigenenergies() > 0)
    assert rand_qobj.isherm


def _valid_rand_herm_eigs(rand_qobj):
    "Is it hermitian with given eigenvalues"
    eigen_vals = np.linspace(0.0, 1.0, N)
    assert np.allclose(rand_qobj.eigenenergies(), eigen_vals)
    assert rand_qobj.isherm


def _make_rand_herm_eigs(N, dims=None, *, seed=None, dtype=None):
    eigen_vals = np.linspace(0.0, 1.0, N)
    return rand_herm(N, dims=dims, seed=seed, dtype=dtype)


def _valid_rand_dm(rand_qobj):
    "Is it a density matrix"
    assert abs(rand_qobj.tr() - 1.0) < 1e-14
    # verify all eigvals are >=0
    assert all(rand_qobj.eigenenergies() >= -1e-14)
    # verify hermitian
    assert rand_qobj.isherm


def _valid_rand_ginibre(rand_qobj):
    "Is it a density matrix of fixed rank"
    _valid_rand_dm(rand_qobj)
    assert sum([abs(E) >= 1e-14 for E in rand_qobj.eigenenergies()]) == rank


def _valid_rand_dm_eigs(rand_qobj):
    "Is it a density matrix with given eigenvalues"
    eigen_vals = np.linspace(0.0, 1.0, N)
    eigen_vals /= np.sum(eigen_vals)
    _valid_rand_dm(rand_qobj)
    assert np.allclose(rand_qobj.eigenenergies(), eigen_vals)


def _make_rand_dm_eigs(N, dims=None, *, seed=None, dtype=None):
    eigen_vals = np.linspace(0.0, 1.0, N)
    return rand_dm(N, dims=dims, seed=seed, dtype=dtype)


def _valid_rand_stochastic(rand_qobj, kind=0):
    'Is it a random stochastic'
    # kind: 0 -> left, 1-> right
    assert np.allclose(np.sum(rand_qobj.full(), axis=kind), 1, atol=1e-15)


def _valid_rand_ket(rand_qobj):
    "Is it a random ket"
    assert rand_qobj.type == 'ket'
    assert abs(rand_qobj.norm() - 1) < 1e-14


def _valid_rand_super(rand_qobj):
    "Is it a random super"
    assert rand_qobj.type == 'super'
    assert rand_qobj.issuper


def _valid_rand_super_bcsz_cptp(rand_qobj):
    "Is it a random super, cptp"
    assert rand_qobj.iscptp
    assert rand_qobj.type == 'super'
    assert rand_qobj.issuper


@pytest.fixture(params=[
    pytest.param((rand_unitary, _valid_rand_unitary),
                 id="rand_unitary"),
    pytest.param((rand_unitary_haar, _valid_rand_unitary),
                 id="rand_unitary_haar"),

    pytest.param((partial(rand_herm, density=0.2), _valid_rand_herm),
                 id="rand_herm_sparse"),
    pytest.param((partial(rand_herm, density=0.8), _valid_rand_herm),
                 id="rand_herm_dense"),
    pytest.param((partial(rand_herm, density=0.2, pos_def=True),
                  _valid_rand_herm_PosDef),
                 id="rand_herm_positive_sparse"),
    pytest.param((partial(rand_herm, density=0.8, pos_def=True),
                  _valid_rand_herm_PosDef),
                 id="rand_herm_positive_dense"),
    pytest.param((_make_rand_herm_eigs, _valid_rand_herm_eigs),
                 id="rand_herm_fixed_eigs"),

    pytest.param((rand_dm, _valid_rand_dm), id="rand_dm"),
    pytest.param((_make_rand_dm_eigs, _valid_rand_dm_eigs),
                 id="rand_dm_fixed_eigs"),
    pytest.param((rand_dm_hs, _valid_rand_dm), id="rand_dm_hs"),
    pytest.param((partial(rand_dm_ginibre, rank=rank), _valid_rand_ginibre),
                 id="rand_dm_ginibre"),

    pytest.param((partial(rand_stochastic, kind="left"),
                  _valid_rand_stochastic), id="rand_stochastic_left"),
    pytest.param((partial(rand_stochastic, kind="right"),
                  partial(_valid_rand_stochastic, kind=1)),
                 id="rand_stochastic_right"),

    pytest.param((rand_ket, _valid_rand_ket), id="rand_ket"),
    pytest.param((rand_ket_haar, _valid_rand_ket), id="rand_ket_haar"),

    pytest.param((rand_super, _valid_rand_super), id="rand_super"),
    pytest.param((rand_super_bcsz, _valid_rand_super_bcsz_cptp),
                 id="rand_super_bcsz"),
])
def rand_func_check(request):
    return request.param


def test_random_seeds(rand_func_check):
    func, check = rand_func_check
    seed = 12345
    U0 = func(N, seed=seed)
    U1 = func(N, seed=None)
    U2 = func(N, seed=seed)
    check(U0)
    check(U1)
    check(U2)
    assert U0 != U1
    assert U0 == U2

# random object accept `str` and base.Data
# Obtain all valid dtype from `to`
dtype_names = list(to._str2type.keys()) + list(to.dtypes)
dtype_types = list(to._str2type.values()) + list(to.dtypes)
@pytest.mark.parametrize('dtype', zip(dtype_names, dtype_types),
                         ids=[str(dtype) for dtype in dtype_names])
def test_random_type(rand_func_check, dtype):
    func, check = rand_func_check
    dtype_key, dtype_type = dtype
    rand_qobj = func(N, dtype=dtype_key)
    assert isinstance(rand_qobj.data, dtype_type)
    check(rand_qobj)


def check_rand_dims(rand_func_check, args, kwargs, dims):
    # TODO: promote this out of test_random, as it's generically useful
    #       in writing tests.
    func, check = rand_func_check
    rand_qobj = func(*args, **kwargs)
    check(rand_qobj)
    assert rand_qobj.dims == dims


@pytest.mark.parametrize('rand_kets', [
    pytest.param((rand_ket, _valid_rand_ket), id="rand_ket"),
    pytest.param((rand_ket_haar, _valid_rand_ket), id="rand_ket_haar"),
])
def test_rand_vector_dims(rand_kets):
    check_rand_dims(rand_kets, (N, ), {}, [[N], [1]])
    check_rand_dims(rand_kets, (6, ),
                    {'dims': [[2,3], [1,1]]}, [[2,3], [1,1]])


@pytest.mark.parametrize('rand_oper', [
    pytest.param((rand_unitary, _valid_rand_unitary),
                 id="rand_unitary"),
    pytest.param((rand_unitary_haar, _valid_rand_unitary),
                 id="rand_unitary_haar"),

    pytest.param((partial(rand_herm, density=0.2), _valid_rand_herm),
                 id="rand_herm_sparse"),
    pytest.param((partial(rand_herm, density=0.8), _valid_rand_herm),
                 id="rand_herm_dense"),
    pytest.param((partial(rand_herm, density=0.2, pos_def=True),
                  _valid_rand_herm_PosDef),
                 id="rand_herm_positive_sparse"),
    pytest.param((partial(rand_herm, density=0.8, pos_def=True),
                  _valid_rand_herm_PosDef),
                 id="rand_herm_positive_dense"),
    pytest.param((_make_rand_herm_eigs, _valid_rand_herm_eigs),
                 id="rand_herm_fixed_eigs"),

    pytest.param((rand_dm, _valid_rand_dm), id="rand_dm"),
    pytest.param((_make_rand_dm_eigs, _valid_rand_dm_eigs),
                 id="rand_dm_fixed_eigs"),
    pytest.param((rand_dm_hs, _valid_rand_dm), id="rand_dm_hs"),
    pytest.param((partial(rand_dm_ginibre, rank=rank), _valid_rand_ginibre),
                 id="rand_dm_ginibre"),

    pytest.param((partial(rand_stochastic, kind="left"),
                  _valid_rand_stochastic), id="rand_stochastic_left"),
    pytest.param((partial(rand_stochastic, kind="right"),
                  partial(_valid_rand_stochastic, kind=1)),
                 id="rand_stochastic_right"),
])
def test_rand_oper_dims(rand_oper):
    check_rand_dims(rand_oper, (N, ), {}, [[N], [N]])
    check_rand_dims(rand_oper, (6, ),
                    {'dims': [[2, 3], [2, 3]]}, [[2, 3], [2, 3]])


@pytest.mark.parametrize('rand_superop', [
    pytest.param((rand_super, _valid_rand_super), id="rand_super"),
    pytest.param((rand_super_bcsz, _valid_rand_super_bcsz_cptp),
                 id="rand_super_bcsz"),
])
def test_rand_super_dims(rand_superop):
    check_rand_dims(rand_superop, (N, ), {}, [[[N], [N]]] * 2)
    check_rand_dims(rand_superop, (6, ),
                    {'dims': [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]},
                    [[[2, 3], [2, 3]], [[2, 3], [2, 3]]])
