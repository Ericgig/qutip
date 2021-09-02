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

from . import Qobj, QobjEvo, liouvillian, Cubic_Spline, coefficient, sprepost
from ._brtools import SpectraCoefficient, _EigenBasisTransform
from ._brtensor import _BlochRedfieldElement


__all__ = ['bloch_redfield_tensor', 'brterm']


def _read_legacy_a_ops(a_op):
    constant = coefficient("1")
    parsed_a_op
    op, spec = a_op
    if isinstance(op, Qobj):
        if isinstance(spec, str):
            parsed_a_op = (op, coefficient(spec, args={'w':0}))
        elif callable(spec):
            parsed_a_op =(op, coefficient(lambda t, args: spec(args['w'])))
        elif isinstance(spec, tuple):
            if isinstance(spec[0], str):
                freq_responce = coefficient(spec[0], args={'w':0})
            elif isinstance(spec[0], Cubic_Spline):
                freq_responce = SpectraCoefficient(coefficient(spec[0]))
            else:
                raise Exception('Invalid bath-coupling specification.')
            if isinstance(spec[1], str):
                time_responce = coefficient(spec[1], args={'w':0})
            elif isinstance(spec[1], Cubic_Spline):
                time_responce = coefficient(spec[1])
            else:
                raise Exception('Invalid bath-coupling specification.')
            parsed_a_op = (op, freq_coeff * time_coeff)
    elif isinstance(op, tuple):
        qobj1, qobj2 = op
        if isinstance(spec[0], str):
            freq_responce = coefficient(spec[0], args={'w':0})
        elif isinstance(spec[0], Cubic_Spline):
            freq_responce = SpectraCoefficient(coefficient(spec[0]))
        else:
            raise Exception('Invalid bath-coupling specification.')
        parsed_a_op =  (
            QobjEvo([[qobj1, spec[1]], [qobj1, spec[2]]]),
            freq_responce
        )
    return parsed_a_op


def bloch_redfield_tensor(H, a_ops, c_ops=[],
                          use_secular=True, sec_cutoff=0.1,
                          fock_basis=False, legacy_a_ops=False, sparse=False):
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
    H_transform = _EigenBasisTransform(QobjEvo(H), sparse)

    if legacy_a_ops:
        a_ops = [_read_legacy_a_ops(a_op) for a_op in a_ops]

    if fock_basis:
        for a_op in a_ops:
            R += brterm(H_transform, *a_op, use_secular, sec_cutoff, fock_basis)
        return R
    else:
        # When the Hamiltonian is time-dependent, the transformation of `L` to
        # eigenbasis is not optimized.
        if isinstance(R, QobjEvo):
            # The `sprepost` will be computed 2 times for each parts of `R`.
            # Compressing the QobjEvo will lower the number of parts.
            R.compress()
        evec = H_transform.as_Qobj()
        R = sprepost(evec, evec.dag()) @ R @ sprepost(evec.dag(), evec)
        for a_op in a_ops:
            R += brterm(H_transform, *a_op,
                        use_secular, sec_cutoff, fock_basis)[0]
        return R, H_transform.as_Qobj()


def brterm(H, a_op, spectra, use_secular=True,
           sec_cutoff=0.1, fock_basis=False,
           sparse=False):
    """
    Calculates the contribution of one coupling operator to the Bloch-Redfield
    tensor.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        System Hamiltonian.

    a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        The operator coupling to the environment. Must be hermitian.

    spectra : :class:`Coefficient`
        The corresponding bath spectra.
        Must be a `Coefficient` using an 'w' args. The `SpectraCoefficient`
        can be used to use array based coefficient.

        Example:

            coefficient('w>0', args={"w": 0})
            SpectraCoefficient(coefficient(Cubic_Spline))

    use_secular : bool {True}
        Flag that indicates if the secular approximation should
        be used.

    sec_cutoff : float {0.1}
        Threshold for secular approximation.

    fock_basis : bool {False}
        Whether to return the tensor in the input basis or the diagonalized
        basis.

    sparse : bool {False}
        Whether to use the sparse eigensolver on the Hamiltonian.

    Returns
    -------

    R, [evecs]: :class:`~Qobj`, :class:`~QobjEvo` or tuple
        If ``fock_basis``, return the Bloch Redfield tensor in the outside
        basis. Otherwise return the Bloch Redfield tensor in the diagonalized
        Hamiltonian basis and the eigenvectors of the Hamiltonian as hstacked
        column. The tensors and, if given, evecs, will be :obj:`~QobjEvo` if
        the ``H`` and ``a_op`` is time dependent, :obj:`Qobj` otherwise.
    """
    if isinstance(H, Qobj):
        H = QobjEvo(H)

    if isinstance(H, _EigenBasisTransform):
        Hdiag = H
    else:
        Hdiag = _EigenBasisTransform(H, sparse=sparse)

    sec_cutoff = sec_cutoff if use_secular else np.inf
    R = QobjEvo(_BlochRedfieldElement(Hdiag, QobjEvo(a_op), spectra,
                sec_cutoff, not fock_basis))

    if Hdiag.isconstant and isinstance(a_op, Qobj):
        R = R(0)
    return R if fock_basis else (R, Hdiag.as_Qobj())
