# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, QuSTaR.
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
import os
import numpy as np
from qutip.settings import settings as qset

from ..core import QobjEvoFunc, Qobj, QobjEvo, sprepost, liouvillian, Cubic_Spline, coefficient
from ..core.qobjevo import QobjEvoBase
from ._brtools import (bloch_redfield_tensor, CBR_RHS, Spectrum, Spectrum_Str,
                       Spectrum_array, Spectrum_func_t)
from ..core import data as _data


__all__ = ['bloch_redfield']

def _read_legacy_a_ops(a_op):
    constant = coefficient("1")
    parsed_a_op
    op, spec = a_op:
    if isinstance(op, Qobj):
        if isinstance(spec, str):
            parsed_a_op = (op, coefficient(spec, args={'w':0}))
        elif callable(spec):
            parsed_a_op =(op, Spectrum(const, spec))
        elif isinstance(spec, tuple):
            if isinstance(spec[0], str):
                freq_responce = coefficient(spec[0], args={'w':0})
            elif isinstance(spec[0], Cubic_Spline):
                freq_responce = coefficient(spec[0])
            else:
                raise Exception('Invalid bath-coupling specification.')
            if isinstance(spec[1], str):
                time_responce = coefficient(spec[1], args={'w':0})
            elif isinstance(spec[1], Cubic_Spline):
                time_responce = coefficient(spec[1])
            else:
                raise Exception('Invalid bath-coupling specification.')
            parsed_a_op = (op, Spectrum(time_coeff, freq_coeff)
    elif isinstance(op, tuple):
        qobj1, qobj2 = op
        if isinstance(spec[0], str):
            freq_responce = coefficient(spec[0], args={'w':0})
        elif isinstance(spec[0], Cubic_Spline):
            freq_responce = coefficient(spec[0])
        else:
            raise Exception('Invalid bath-coupling specification.')
        parsed_a_op =  (
            QobjEvo([[qobj1, spec[1]], [qobj1, spec[2]]]),
            Spectrum(const, spec)
        )

    return parsed_a_op


def bloch_redfield_tensor(object H, list a_ops, list c_ops=[],
                          bool use_secular=True, double sec_cutoff=0.1,
                          fock_basis=False, legacy_a_ops=False):
    """
    Calculates the Bloch-Redfield tensor for a system given
    a set of operators and corresponding spectral functions that describes the
    system's coupling to its environment.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        System Hamiltonian.

    a_ops : list of (a_op, spectra)
        Nested list of system operators that couple to the environment,
        and the corresponding bath spectra.

        a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
            The operator coupling to the environment. Must be hermitian.

        spectra : :class:`Coefficient`, callable
            The corresponding bath spectra.
            Can be a `Coefficient` using an 'w' args or a function of the
            frequence. The `SpectraCoefficient` can be used to use array based
            coefficient.
            Example:

            a_ops = [
                (a+a.dag(), coefficient('w>0', args={"w": 0})),
                (QobjEvo([b+b.dag(), f(t)]), g(w)),
                (c+c.dag(), SpectraCoefficient(coefficient(Cubic_Spline))),
            ]

    c_ops : list
        List of system collapse operators.

    use_secular : bool {True}
        Flag that indicates if the secular approximation should
        be used.

    sec_cutoff : float {0.1}
        Threshold for secular approximation.

    fock_basis : bool {False}
        Whether to return the tensor in the input basis or the diagonalized
        basis.

    legacy_a_ops : bool {False}
        Whether to use the v4's brmesolve's a_ops specification.

    sparse : bool {False}
        Whether to use the sparse eigensolver if

    Returns
    -------

    R, [evecs]: :class:`qutip.Qobj`, tuple of :class:`qutip.Qobj`
        If ``fock_basis``, return the Bloch Redfield tensor in the outside
        basis. Otherwise return the Bloch Redfield tensor in the diagonalized
        Hamiltonian basis and the eigenvectors of the Hamiltonian as hstacked
        column.
    """
    R = liouvillian(H, c_ops)
    diag = _EigenBasisTransform(H, sparse)

    if not fock_basis:
        R = diag.to_eigbasis(R)

    for a_op in a_ops:
        if legacy_a_ops:
            a_op = _read_legacy_a_ops(a_ops)
        R += brtensor(diag, a_op, use_secular, sec_cutoff, fock_basis)
    if not fock_basis:
        return R, diag.as_Qobj
    return R
