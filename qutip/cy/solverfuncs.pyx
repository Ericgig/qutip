from csr_matrix cimport cy_csr_matrix, CSR_from_scipy
cimport cython
cimport numpy as cnp
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs(
        double t,
        complex[::1] rho,
        cy_csr_matrix mat):
    return mat.spmv(rho)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args):
    H = CSR_from_scipy(H_func(t, args).data)
    return -1j * H.spmv(psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_psi_func_td_with_state(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] psi,
        object H_func,
        object args):

    H = CSR_from_scipy(H_func(t, psi, args))
    return -1j * H.spmv(psi)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rho_func_td(
        double t,
        cnp.ndarray[complex, ndim=1, mode="c"] rho,
        object L0,
        object L_func,
        object args):
    cdef object L
    L = CSR_from_scipy(L0 + L_func(t, args).data)
    return L.spmv(rho)
