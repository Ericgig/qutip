#!python
#cython: language_level=3
from qutip.matrix.cy.cdata cimport Cdata
cimport cython
cimport numpy as cnp
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs(
        double t,
        complex[::1] rho,
        Cdata mat):
    return mat.mulvec(rho)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args):
    H = H_func(t, args).data.cdata
    return -1j * H.mulvec(psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td_with_state(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args):

    H = H_func(t, psi, args).data.cdata
    return -1j * H.mulvec(psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rho_func_td(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] rho,
        object L0,
        object L_func,
        object args):
    cdef object L
    L = (L0 + L_func(t, args).data).cdata
    return L.mulvec(rho)
