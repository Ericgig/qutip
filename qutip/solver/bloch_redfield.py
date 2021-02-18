# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, QuSTaR
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

__all__ = ['brmesolve', 'BrMeSolver']

import numpy as np
import os
from time import time
import types
import warnings
import scipy.integrate
from ..core import (
    Qobj, spre, unstack_columns, stack_columns, expect, Cubic_Spline, QobjEvo, QobjEvoFunc
)
from ..core import data as _data
from .. import settings as qset
from ..core.cy.openmp.utilities import check_use_openmp
from ..ui.progressbar import BaseProgressBar, TextProgressBar
from .options import SolverOptions
from .result import Result
from .mesolve import MeSolver
from .br_codegen import bloch_redfield
# -----------------------------------------------------------------------------
# Solve the Bloch-Redfield master equation
#
def brmesolve(H, rho0, tlist, a_ops=[], e_ops=[], c_ops=[],
              args={}, use_secular=True, sec_cutoff=0.1,
              spectra_cb=None, options=None, feedback_args=None,
              progress_bar=None, _safe_mode=True, verbose=False):
    """
    Solves for the dynamics of a system using the Bloch-Redfield master equation,
    given an input Hamiltonian, Hermitian bath-coupling terms and their associated
    spectrum functions, as well as possible Lindblad collapse operators.

    For time-independent systems, the Hamiltonian must be given as a Qobj,
    whereas the bath-coupling terms (a_ops), must be written as a nested list
    of operator - spectrum function pairs, where the frequency is specified by
    the `w` variable.

    *Example*

        a_ops = [[a+a.dag(),lambda w: 0.2*(w>=0)]]

    For time-dependent systems, the Hamiltonian, a_ops, and Lindblad collapse
    operators (c_ops), can be specified in the QuTiP string-based time-dependent
    format.  For the a_op spectra, the frequency variable must be `w`, and the
    string cannot contain any other variables other than the possibility of having
    a time-dependence through the time variable `t`:

    *Example*

        a_ops = [[[a+a.dag(), 'exp(-t/2)'], '0.2*(w>=0)']]

    It is also possible to use Cubic_Spline objects for time-dependence.  In
    the case of a_ops, Cubic_Splines must be passed as a tuple:

    *Example*

        a_ops = [([a+a.dag(), g(t)], f(t,w)) ]

    where f(w) and g(t) are strings or Cubic_spline objects for the bath
    spectrum and time-dependence, respectively.

    Finally, if one has bath-couplimg terms of the form
    H = f(t)*a + conj[f(t)]*a.dag(), then the correct input format is

    *Example*

              a_ops = [ ([[a,g1(t)], [a.dag(),g2(t)]], f(t,w)),... ]

    where f(w) is the spectrum of the operators while g1(t) and g2(t)
    are the time-dependence of the operators `a` and `a.dag()`, respectively

    Parameters
    ----------
    H : Qobj / list
        System Hamiltonian given as a Qobj or
        nested list in string-based format.

    rho0: Qobj
        Initial density matrix or state vector (ket).

    tlist : array_like
        List of times for evaluating evolution

    a_ops : list
        Nested list of Hermitian system operators that couple to
        the bath degrees of freedom, along with their associated
        spectra.

    e_ops : list
        List of operators for which to evaluate expectation values.

    c_ops : list
        List of system collapse operators, or nested list in
        string-based format.

    args : dict
        Placeholder for future implementation, kept for API consistency.

    use_secular : bool {True}
        Use secular approximation when evaluating bath-coupling terms.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation.

    tol : float {qutip.setttings.atol}
        Tolerance used for removing small values after
        basis transformation.

    options : :class:`qutip.solver.SolverOptions`
        Options for the solver.

    progress_bar : BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------
    result: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains
        either an array of expectation values, for operators given in e_ops,
        or a list of states for the times specified by `tlist`.
    """
    use_brmesolve = bool(a_ops)

    if not use_brmesolve:
        return mesolve(H, rho0, tlist, e_ops=e_ops, args=args, options=options,
                       feedback_args=feedback_args, _safe_mode=_safe_mode)

    c_ops = c_ops if c_ops is not None else []
    solver = BrMeSolver(H, a_ops, c_ops, e_ops, use_secular, sec_cutoff,
                        options, times=tlist, args=args,
                        feedback_args=feedback_args, _safe_mode=_safe_mode)

    return solver.run(rho0, tlist, args)


def to_QEvo(op, args, times):
    if isinstance(op, QobjEvo):
        return op
    elif isinstance(op, (list, Qobj)):
        return QobjEvo(op, args=args, tlist=times)
    elif callable(op):
        return QobjEvoFunc(op, args=args)
    else:
        raise ValueError("Invalid time dependent format")


class BrMeSolver(MeSolver):
    def __init__(self, H, a_ops, c_ops=[], e_ops=None,
                 use_secular=True, sec_cutoff=0.0,
                 options=None, times=None, args=None,
                 feedback_args=None, _safe_mode=False):
        _time_start = time()
        self.stats = {}
        if e_ops is None:
            e_ops = []
        if options is None:
            options = SolverOptions()
        elif not isinstance(options, SolverOptions):
            raise ValueError("options must be an instance of "
                             "qutip.solver.SolverOptions")
        if args is None:
            args = {}
        if feedback_args is None:
            feedback_args = {}

        self._safe_mode = _safe_mode
        self._super = True
        self._state = None
        self._t = 0

        self.e_ops = e_ops
        self.options = options
        atol = options.ode["atol"]

        constant_system = True

        H = to_QEvo(H, args, times)
        constant_system = constant_system and H.const

        c_evos = []
        for op in c_ops:
            c_evos.append(to_QEvo(op, args, times))

        a_evos = []
        for (op, spec) in a_ops:
            if isinstance(spec, (tuple, list)):
                raise ValueError("a_ops format changed from v5, use: "
                                 "a_ops=[([op, g(t)], f(w))]")
            if isinstance(op, tuple):
                # TODO: Check if consistant with last version.
                evo = to_QEvo(op[0], args, times) + to_QEvo(op[1], args, times)
            else:
                evo = to_QEvo(op, args, times)
                if not evo(0).isherm:
                    raise ValueError("a_ops must be Hermitian")
            a_evos.append((evo, spec))

        self._system = bloch_redfield(H, a_evos, c_evos,
            use_secular=use_secular, sec_cutoff=sec_cutoff, atol=atol
        )

        self._evolver = self._get_evolver(options, args, feedback_args)
        self.stats["preparation time"] = time() - _time_start
        self.stats['solver'] = "Bloch Redfield Equation Evolution"
