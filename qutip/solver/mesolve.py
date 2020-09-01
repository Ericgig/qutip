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
"""
This module provides solvers for the Lindblad master equation and von Neumann
equation.
"""

__all__ = ['mesolve', 'MeSolver']

import numpy as np
from .. import ( Qobj, QobjEvo, isket, issuper, liouvillian, ket2dm)
from .solver import SolverOptions
from .sesolve import sesolve
from ._solverqevo import SolverQEvo
from .run import _driver_step, _driver_evolution
from .evolver import EvolverScipyZvode


# -----------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
#
def mesolve(H, rho0, tlist, c_ops=None, e_ops=None, args=None,
            feedback_args=None, options=None, _safe_mode=True):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the state vector or density matrix (`rho0`) using a given
    Hamiltonian or Liouvillian (`H`) and an optional set of collapse operators
    (`c_ops`), by integrating the set of ordinary differential equations
    that define the system. In the absence of collapse operators the system is
    evolved according to the unitary evolution of the Hamiltonian.

    The output is either the state vector at arbitrary points in time
    (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values.

    If either `H` or the Qobj elements in `c_ops` are superoperators, they
    will be treated as direct contributions to the total system Liouvillian.
    This allows the solution of master equations that are not in standard
    Lindblad form.

    **Time-dependent operators**

    For time-dependent problems, `H` and `c_ops` can be a specified in a
    nested-list format where each element in the list is a list of length 2,
    containing an operator (:class:`qutip.qobj`) at the first element and where
    the second element is either a string (*list string format*), a callback
    function (*list callback format*) that evaluates to the time-dependent
    coefficient for the corresponding operator, or a NumPy array (*list
    array format*) which specifies the value of the coefficient to the
    corresponding operator for each value of t in `tlist`.

    Alternatively, `H` (but not `c_ops`) can be a callback function with the
    signature `f(t, args) -> Qobj` (*callback format*), which can return the
    Hamiltonian or Liouvillian superoperator at any point in time.  If the
    equation cannot be put in standard Lindblad form, then this time-dependence
    format must be used.

    *Examples*

        H = [[H0, 'sin(w*t)'], [H1, 'sin(2*w*t)']]

        H = [[H0, f0_t], [H1, f1_t]]

        where f0_t and f1_t are python functions with signature f_t(t, args).

        H = [[H0, np.sin(w*tlist)], [H1, np.sin(2*w*tlist)]]

    In the *list string format* and *list callback format*, the string
    expression and the callback function must evaluate to a real or complex
    number (coefficient for the corresponding operator).

    In all cases of time-dependent operators, `args` is a dictionary of
    parameters that is used when evaluating operators. It is passed to the
    callback functions as their second argument.

    **Additional options**

    Additional options to mesolve can be set via the `options` argument, which
    should be an instance of :class:`qutip.solver.Options`. Many ODE
    integration options can be set this way, and the `store_states` and
    `store_final_state` options can be used to store states even though
    expectation values are requested via the `e_ops` argument.

    .. note::

        If an element in the list-specification of the Hamiltonian or
        the list of collapse operators are in superoperator form it will be
        added to the total Liouvillian of the problem without further
        transformation. This allows for using mesolve for solving master
        equations that are not in standard Lindblad form.

    .. note::

        On using callback functions: mesolve transforms all :class:`qutip.Qobj`
        objects to sparse matrices before handing the problem to the integrator
        function. In order for your callback function to work correctly, pass
        all :class:`qutip.Qobj` objects that are used in constructing the
        Hamiltonian via `args`. mesolve will check for :class:`qutip.Qobj` in
        `args` and handle the conversion to sparse matrices. All other
        :class:`qutip.Qobj` objects that are not passed via `args` will be
        passed on to the integrator in scipy which will raise a NotImplemented
        exception.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian, or a callback function for time-dependent
        Hamiltonians, or alternatively a system Liouvillian.

    rho0 : :class:`qutip.Qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : None / list of :class:`qutip.Qobj`
        single collapse operator, or list of collapse operators, or a list
        of Liouvillian superoperators.

    e_ops : None / list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

    options : None / :class:`qutip.SolverOptions`
        with options for the solver.

    progress_bar : None / BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        either an *array* `result.expect` of expectation values for the times
        specified by `tlist`, or an *array* `result.states` of state vectors or
        density matrices corresponding to the times in `tlist` [if `e_ops` is
        an empty list], or nothing if a callback function was given in place of
        operators for which to calculate the expectation values.

    """
    solver = MeSolver(H, c_ops, e_ops, options, tlist,
                      args, feedback_args, _safe_mode)

    return solver.run(rho0, tlist, args)

"""
    use_mesolve = (
        (c_ops and len(c_ops) > 0)
        or (not rho0.isket)
        or (isinstance(H, Qobj) and H.issuper)
        or (isinstance(H, QobjEvo) and H.cte.issuper)
        or (isinstance(H, list) and isinstance(H[0], Qobj) and H[0].issuper)
        or (not isinstance(H, (Qobj, QobjEvo))
            and callable(H)
            and H(0., args).issuper)
        or (not isinstance(H, (Qobj, QobjEvo))
            and callable(H))
    )

    if not use_mesolve:
        return sesolve(H, rho0, tlist, e_ops=e_ops, args=args, options=options,
                       progress_bar=progress_bar, _safe_mode=_safe_mode)
"""
class MeSolver:
    def __init__(self, H, c_ops, e_ops=None, options=None,
                 times=None, args=None, feedback_args=None,
                 _safe_mode=False):
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

        self.e_ops = e_ops
        self.options = options
        self._safe_mode = _safe_mode

        if isinstance(H, QobjEvo):
            pass
        elif isinstance(H, (list, Qobj)):
            H = QobjEvo(H, args=args, tlist=times)
        elif callable(H):
            raise NotImplementedError
        else:
            raise ValueError("Invalid Hamiltonian")

        c_evos = []
        for op in c_ops:
            if isinstance(H, QobjEvo):
                c_evos.append(op)
            elif isinstance(H, (list, Qobj)):
                c_evos.append(QobjEvo(op, args=args, tlist=times))
            elif callable(H):
                raise NotImplementedError
            else:
                raise ValueError("Invalid Hamiltonian")
        self.system = SolverQEvo(liouvillian(H, c_evos), None,
                                 args, feedback_args)

        self.evolver = EvolverScipyZvode(self.system, options)

    def run(self, rho0, tlist, args={}):
        if isket(rho0):
            rho0 = ket2dm(rho0)

        if self._safe_mode:
            v = rho0.full().ravel('F')
            self.system.mul_np_vec(0., v) + v

        if args:
            self.system.arguments(args)
        result = _driver_step(self.evolver, tlist, rho0,
                              self.options, self.e_ops, True)
        return result
