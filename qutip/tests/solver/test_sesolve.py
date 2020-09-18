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
from numpy.testing import assert_, run_module_suite

# disable the progress bar
import os

from qutip import (
    sigmax, sigmay, sigmaz, qeye, basis, expect, num, destroy, create,
    Cubic_Spline, QobjEvo
)
from qutip.solver import SolverOptions, sesolve
from qutip.solver.evolver import all_ode_method

os.environ['QUTIP_GRAPHICS'] = "NO"


delta = 1.0 * 2*np.pi
H1 = 0.5*delta*sigmax()
tlist = np.linspace(0, 20, 200)
S = Cubic_Spline(0, 20, np.exp(-alpha * tlist))
args = {'alpha': 0.5}


@pytest.mark.parametrize(['unitary_op'],
    pytest.param(None, id="state"),
    pytest.param(qeye(2), id="unitary"),
])
@pytest.mark.parametrize(['H', 'analytical'],
    [pytest.param(H1,
                  lambda t, args: t,
                  id='const_H'),
     pytest.param(lambda t, args: H1 * np.exp(-args['alpha'] * t),
                  lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                   / args['alpha']),
                  id='func_H'),
     pytest.param([[H1, lambda t, args: np.exp(-args['alpha'] * t)]],
                  lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                   / args['alpha']),
                  id='list_func_H'),
     pytest.param([[H1, 'exp(-alpha*t)']],
                  lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                   / args['alpha']),
                  id='list_str_H'),
     pytest.param([[H1, S]],
                  lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                   / args['alpha']),
                  id='list_cubic_spline_H'),
     pytest.param([[H1, np.exp(-alpha * tlist)]],
                  lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                   / args['alpha']),
                  id='list_array_H'),
     pytest.param(QobjEvo([[H1, 'exp(-alpha*t)']], args={'alpha': 0.5}),
                  lambda t, args: ((1 - np.exp(-args['alpha'] * t))
                                   / args['alpha']),
                  id='QobjEvo_H'),
])
def test_sesolve(H, delta, psi0, tlist, analytical,
                 unitary_op, tol=5e-3):
    """
    Compare integrated evolution with analytical result
    If U0 is not None then operator evo is checked
    Otherwise state evo
    """
    psi0 = basis(2, 0)

    if unitary_op is None:
        output = sesolve(H, psi0, tlist, [sigmax(), sigmay(), sigmaz()],
                         args=args)
        sx, sy, sz = output.expect[0], output.expect[1], output.expect[2]
    else:
        output = sesolve(H, unitary_op, tlist, args=args)
        sx = [expect(sigmax(), U * psi0) for U in output.states]
        sy = [expect(sigmay(), U * psi0) for U in output.states]
        sz = [expect(sigmaz(), U * psi0) for U in output.states]

    sx_analytic = np.zeros(np.shape(tlist))
    sy_analytic = np.array([-np.sin(delta * analytic_func(t, td_args))
                            for t in tlist])
    sz_analytic = np.array([np.cos(delta * analytic_func(t, td_args))
                            for t in tlist])

    np.testing.assert_allclose(sx, sx_analytic, atol=tol)
    np.testing.assert_allclose(sy, sy_analytic, atol=tol)
    np.testing.assert_allclose(sz, sz_analytic, atol=tol)


@pytest.mark.parametrize(['method'], params=all_ode_method, ids=all_ode_method)
def test_sesolve_method(method):
    """
    Compare integrated evolution with analytical result
    If U0 is not None then operator evo is checked
    Otherwise state evo
    """
    tol = 5e-3
    psi0 = basis(2, 0)
    tlist = np.linspace(0, 20, 200)
    args = {'alpha': 0.5}
    options = SolverOptions(mehtod=method)

    output = sesolve([[H1, 'exp(-alpha*t)']], psi0, tlist,
                     [sigmax(), sigmay(), sigmaz()],
                     args=args, options=options)
    sx, sy, sz = output.expect[0], output.expect[1], output.expect[2]

    sx_analytic = np.zeros(np.shape(tlist))
    sy_analytic = np.array([-np.sin(delta * ((1 - np.exp(-args['alpha'] * t))
                                             / args['alpha']))
                            for t in tlist])
    sz_analytic = np.array([np.cos(delta * ((1 - np.exp(-args['alpha'] * t))
                                            / args['alpha']))
                            for t in tlist])

    np.testing.assert_allclose(sx, sx_analytic, atol=tol)
    np.testing.assert_allclose(sy, sy_analytic, atol=tol)
    np.testing.assert_allclose(sz, sz_analytic, atol=tol)


eps = 0.2 * 2*np.pi
delta = 1.0 * 2*np.pi
w0 = 0.5*eps
w1 = 0.5*delta
H0 = w0*sigmaz()
H1 = w1*sigmax()

@pytest.mark.parametrize(['normalize'], [True, False], ids=['Normalized', ''])
@pytest.mark.parametrize(['H', 'args'],
    [pytest.param(H0 + H1,
                  {},
                  id='const_H'),
     pytest.param(lambda t, args: a * t * H0 + np.exp(-alpha * t) * H1,
                  {'a':a, 'alpha':alpha},
                  id='func_H'),
     pytest.param([[H0, lambda t, args: a*t], [H1, lambda t, args: np.cos(w_a*t)]],
                  {'a':a, 'w_a':w_a},
                  id='list_func_H'),
     pytest.param([H0, [H1, 'cos(w_a*t)']],
                  {'w_a':w_a},
                  id='list_str_H'),
])
def compare_evolution(H, normalize=False, args={}, tol=5e-5):
    """
    Compare integrated evolution of unitary operator with state evo
    """
    psi0 = basis(2, 0)
    tlist = np.linspace(0, 20, 200)

    U0 = qeye(2)
    options = SolverOptions(store_states=True, normalize_output=normalize)
    out_s = sesolve(H, psi0, tlist, [sigmax(), sigmay(), sigmaz()],
                    options=options,args=args)
    xs, ys, zs = out_s.expect[0], out_s.expect[1], out_s.expect[2]
    xss = [expect(sigmax(), U) for U in out_s.states]
    yss = [expect(sigmay(), U) for U in out_s.states]
    zss = [expect(sigmaz(), U) for U in out_s.states]

    assert (max(abs(xs - xss)) < tol)
    assert (max(abs(ys - yss)) < tol)
    assert (max(abs(zs - zss)) < tol)

    out_u = sesolve(H, U0, tlist, options=options, args=args)
    xu = [expect(sigmax(), U * psi0) for U in out_u.states]
    yu = [expect(sigmay(), U * psi0) for U in out_u.states]
    zu = [expect(sigmaz(), U * psi0) for U in out_u.states]

    assert (max(abs(xs - xu)) < tol)
    assert (max(abs(ys - yu)) < tol)
    assert (max(abs(zs - zu)) < tol)

def test_07_1_dynamic_args():
        "sesolve: state feedback"
        tol = 1e-3
        def f(t, args):
            return np.abs(args["state_vec"][1])

        H = [qeye(2), [destroy(2)+create(2), f]]
        res = sesolve(H, basis(2,1), tlist=np.linspace(0,10,11),
                      e_ops=[num(2)], args={"state_vec":basis(2,1)})
        assert_(max(abs(res.expect[0][5:])) < tol,
                msg="evolution with feedback not proceding as expected")

        def f(t, args):
            return np.sqrt(args["expect_op_0"])

        H = [qeye(2), [destroy(2)+create(2), f]]
        res = sesolve(H, basis(2,1), tlist=np.linspace(0,10,11),
                      e_ops=[num(2)], args={"expect_op_0":num(2)})
        assert_(max(abs(res.expect[0][5:])) < tol,
                msg="evolution with feedback not proceding as expected")
