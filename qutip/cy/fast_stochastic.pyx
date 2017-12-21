import numpy as np
import scipy.sparse as sp
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.spmatfuncs cimport (spmv_csr)

include "parameters.pxi"

from qutip.cy.spmatfuncs import cy_expect_rho_vec_csr, cy_expect_rho_vec
from qutip.superoperator import mat2vec, vec2mat
#from qutip.cy.spmatfuncs import cy_expect_psi_csr, cy_expect_rho_vec
from qutip.qobj import Qobj


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_smesolve_fast_single_trajectory(int n, object sso):
    """
    Internal function. See smesolve.
    """
    cdef double dt = sso.dt
    cdef double[::1] times = sso.times
    cdef int d2_len = sso.d2_len
    cdef int N_substeps = sso.N_substeps
    cdef int N_store = sso.N_store

    cdef np.ndarray[complex, ndim=1] A_data
    cdef int[::1] A_ind
    cdef int[::1] A_ptr

    cdef np.ndarray[complex, ndim=1] AL_data
    cdef int[::1] AL_ind
    cdef int[::1] AL_ptr

    cdef object Ae

    cdef double tol
    cdef int A_size, sc_len

    cdef np.ndarray[complex, ndim=1] rho_t
    cdef np.ndarray[double, ndim=4] dW

    A_ops = sso.A_ops
    A_data = A_ops[0][0].data
    A_ind = A_ops[0][0].indices
    A_ptr = A_ops[0][0].indptr
    if sso.rhs == 30:
        # Milstein A
        AL_data = A_ops[0][1].data
        AL_ind = A_ops[0][1].indices
        AL_ptr = A_ops[0][1].indptr
    elif sso.rhs in (25,35):
        # Implicit A
        AL_data = A_ops[0][2].data
        AL_ind = A_ops[0][2].indices
        AL_ptr = A_ops[0][2].indptr
        Ae = A_ops[0][1]
        tol = sso.args['tol']
    else:
        A_size = A_ops[0][1]
    sc_len = len(A_ops)

    rho_t = mat2vec(sso.state0.full()).ravel()


    dims = sso.state0.dims

    expect = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)
    ss = np.zeros((len(sso.e_ops), sso.N_store), dtype=complex)

    # reseed the random number generator so that forked
    # processes do not get the same sequence of random numbers
    np.random.seed((n+1) * np.random.randint(0, 4294967295 // (sso.ntraj+1)))

    if sso.noise is None:
        if sso.generate_noise ==20:
            dW = _generate_noise_Milstein(sc_len, N_store, N_substeps,
                                    d2_len, dt)
        elif sso.generate_noise ==30:
            dW = _generate_noise_Taylor_15(sc_len, N_store, N_substeps,
                                    d2_len, dt)
        elif sso.homogeneous:
            dW = np.sqrt(dt) * np.random.randn(sc_len, N_store,
                                               N_substeps, d2_len)
        else:
            dW = np.zeros((len(A_ops), N_store, N_substeps, d2_len))
    else:
        dW = sso.noise[n]

    states_list = []
    measurements = np.zeros((len(times), len(sso.s_m_ops), d2_len),
                            dtype=complex)

    cdef int j, t_idx, e_idx
    cdef double t
    cdef np.ndarray[complex, ndim=1] rho_prev
    cdef int m_idx, dW_idx
    for t_idx, t in enumerate(times):
        if sso.s_e_ops:
            for e_idx, e in enumerate(sso.s_e_ops):
                s = cy_expect_rho_vec(e.data, rho_t, 0)
                expect[e_idx, t_idx] += s
                ss[e_idx, t_idx] += s ** 2
        if sso.store_states or not sso.s_e_ops:
            states_list.append(Qobj(vec2mat(rho_t), dims=dims))

        rho_prev = np.copy(rho_t)

        if sso.rhs == 10:
            for j in range(N_substeps):
                rho_t = _rhs_rho_euler_homodyne_fast(rho_t,
                        A_data, A_ind, A_ptr, A_size, dW[:, t_idx, j, 0])
        elif sso.rhs == 20:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_single_fast(rho_t,
                        A_data, A_ind, A_ptr, A_size, dW[:, t_idx, j, 0])
        elif sso.rhs == 21:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_two_fast(rho_t,
                        A_data, A_ind, A_ptr, A_size, dW[:, t_idx, j, 0])
        elif sso.rhs == 22:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_fast(rho_t,
                        A_data, A_ind, A_ptr, A_size, sc_len, dW[:, t_idx, j, 0])
        elif sso.rhs == 25:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_implicit(rho_t, A_data, A_ind, A_ptr,
                        Ae, AL_data, AL_ind, AL_ptr,
                        dt, dW[:, t_idx, j, 0], tol)
        elif sso.rhs == 30:
            for j in range(N_substeps):
                rho_t = _rhs_rho_taylor_15_one(rho_t, A_data, A_ind, A_ptr,
                        AL_data, AL_ind, AL_ptr, dt, dW[:, t_idx, j, 0])
        elif sso.rhs == 35:
            for j in range(N_substeps):
                rho_t = _rhs_rho_taylor_15_implicit(rho_t, A_data, A_ind, A_ptr,
                        Ae, AL_data, AL_ind, AL_ptr,
                        dt, dW[:, t_idx, j, 0], tol)
        elif sso.rhs == 40:
            for j in range(N_substeps):
                rho_t = _rhs_rho_pred_corr_homodyne_single(rho_t,
                        A_data, A_ind, A_ptr, A_size, dt, dW[:, t_idx, j, 0])

        elif sso.rhs == 110:
            for j in range(N_substeps):
                rho_t = _rhs_rho_euler_homodyne_fast_2(rho_t,
                        A_data, A_ind, A_ptr, A_size, dW[:, t_idx, j, 0])
        elif sso.rhs == 120:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_single_fast_2(rho_t,
                        A_data, A_ind, A_ptr, A_size, dW[:, t_idx, j, 0])
        elif sso.rhs == 121:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_two_fast_2(rho_t,
                        A_data, A_ind, A_ptr, A_size, dW[:, t_idx, j, 0])
        elif sso.rhs == 122:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_homodyne_fast_2(rho_t,
                        A_data, A_ind, A_ptr, A_size, sc_len, dW[:, t_idx, j, 0])
        elif sso.rhs == 125:
            for j in range(N_substeps):
                rho_t = _rhs_rho_milstein_implicit(rho_t, A_data, A_ind, A_ptr,
                        Ae, AL_data, AL_ind, AL_ptr,
                        dt, dW[:, t_idx, j, 0], tol)
        elif sso.rhs == 130:
            for j in range(N_substeps):
                rho_t = _rhs_rho_taylor_15_one(rho_t, A_data, A_ind, A_ptr,
                        AL_data, AL_ind, AL_ptr, dt, dW[:, t_idx, j, 0])
        elif sso.rhs == 135:
            for j in range(N_substeps):
                rho_t = _rhs_rho_taylor_15_implicit(rho_t, A_data, A_ind, A_ptr,
                        Ae, AL_data, AL_ind, AL_ptr,
                        dt, dW[:, t_idx, j, 0], tol)
        elif sso.rhs == 140:
            for j in range(N_substeps):
                rho_t = _rhs_rho_pred_corr_homodyne_single_2(rho_t,
                        A_data, A_ind, A_ptr, A_size, dt, dW[:, t_idx, j, 0])


        if sso.store_measurement:
            for m_idx, m in enumerate(sso.s_m_ops):
                for dW_idx, dW_factor in enumerate(sso.dW_factors):
                    if m[dW_idx]:
                        m_expt = cy_expect_rho_vec(m[dW_idx].data, rho_prev, 0)
                    else:
                        m_expt = 0
                    measurements[t_idx, m_idx, dW_idx] = m_expt + dW_factor * \
                        dW[m_idx, t_idx, :, dW_idx].sum() / (dt * N_substeps)

    if d2_len == 1:
        measurements = measurements.squeeze(axis=(2))

    return states_list, dW, measurements, expect, ss



@cython.wraparound(False)
@cython.boundscheck(False)
def dot(np.ndarray[double, ndim=1] V,
         np.ndarray[double, ndim=2]dV,
         np.ndarray[double, ndim=1]dW):
    cdef int i,j,i_max, j_max
    i_max = dV.shape[0]
    j_max = dV.shape[1]
    for i in range(i_max):
        for j in range(j_max):
            V[j] += dW[i]*dV[i,j]
    return V






#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef _rhs_rho_euler_homodyne_fast_2(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    Fast Euler-Maruyama for homodyne detection.
    """
    cdef np.ndarray[complex, ndim=2] d_vec = \
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[complex, ndim=1] e = \
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2)

    drho_t = d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    drho_t += (1.0 - np.inner(np.real(e), dW)) * rho_t
    return drho_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _rhs_rho_euler_homodyne_fast(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    Fast Euler-Maruyama for homodyne detection.
    """
    cdef int i, j, n_sc_A, l_rho
    l_rho = len(rho_t)
    cdef np.ndarray[complex, ndim=2] d_vec = \
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, l_rho)
    #cdef np.ndarray[complex, ndim=1] drho_t = d_vec[n_sc_A]
    n_sc_A = d_vec.shape[0]-1
    cdef np.ndarray[complex, ndim=1] e = np.zeros(n_sc_A, dtype=complex)
    for j in range(n_sc_A):
        for i in range(A_size):
            e[j] += d_vec[j,(A_size+1)*i]

    cdef double f = 1.0
    for i in range(n_sc_A):
        f -= np.real(e[i])*dW[i]

    for i in range(l_rho):
        rho_t[i] *=  f
        rho_t[i] += d_vec[n_sc_A,i]
        for j in range(n_sc_A):
            rho_t[i] += dW[j] * d_vec[j,i]
    return rho_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_single_fast(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    Fast Milstein for homodyne detection with 1 stochastic operator
    """

    cdef int i, j, n_sc_A, l_rho
    l_rho = len(rho_t)
    cdef np.ndarray[complex, ndim=2] d_vec = \
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(3, l_rho)
    n_sc_A = 2#d_vec.shape[0]-1
    cdef np.ndarray[double, ndim=1] e = np.zeros(n_sc_A)
    for j in range(n_sc_A):
        for i in range(A_size):
            e[j] += d_vec[j,(A_size+1)*i].real

    e[1] -= 2.0 * e[0] * e[0]

    cdef double f = 1.
    for i in range(n_sc_A):
        f -= e[i] * dW[i]
    dW[0] -= 2.0 * e[0] * dW[1]

    for i in range(l_rho):
        rho_t[i] *= f
        rho_t[i] += d_vec[n_sc_A,i]
        for j in range(n_sc_A):
            rho_t[i] += dW[j]* d_vec[j,i]

    return rho_t

@cython.boundscheck(False)
#@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_single_fast_2(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    Fast Milstein for homodyne detection with 1 stochastic operator
    """
    cdef np.ndarray[complex, ndim=2] d_vec = \
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))

    cdef np.ndarray[double, ndim=1] e = np.real(
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2))

    e[1] -= 2.0 * e[0] * e[0]

    cdef np.ndarray[complex, ndim=1] drho_t = - np.inner(e, dW) * rho_t
    dW[0] -= 2.0 * e[0] * dW[1]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    return rho_t + drho_t


#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_two_fast(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    fast Milstein for homodyne detection with 2 stochastic operators
    """
    cdef np.ndarray[complex, ndim=2] d_vec = \
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e = np.real(
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2))
    ee = e*1.0
    d_vec[-2] -= np.dot(e[:2][::-1], d_vec[:2])

    e[2:4] -= 2.0 * e[:2] * e[:2]
    e[4] -= 2.0 * e[1] * e[0]
    cdef double edw = np.inner(e, dW)*-1

    cdef np.ndarray[complex, ndim=1] drho_t = edw * rho_t
    dW[:2] -= 2.0 * e[:2] * dW[2:4]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    return rho_t + drho_t


#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_two_fast_2(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size,
                                np.ndarray[double, ndim=1] dW):
    """
    fast Milstein for homodyne detection with 2 stochastic operators
    """
    cdef np.ndarray[complex, ndim=2] d_vec = \
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e = np.real(
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2))
    ee = e*1.0
    d_vec[-2] -= np.dot(e[:2][::-1], d_vec[:2])

    e[2:4] -= 2.0 * e[:2] * e[:2]
    e[4] -= 2.0 * e[1] * e[0]
    cdef double edw = np.inner(e, dW)*-1
    #if not( abs(edw) < 1):
    #    print("e",e)
    #    print("ee",ee)
    #    print("a",A_data[0],A_data[1])
    #    print("d",len(d_vec))
    cdef np.ndarray[complex, ndim=1] drho_t = edw * rho_t
    dW[:2] -= 2.0 * e[:2] * dW[2:4]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])
    return rho_t + drho_t


@cython.boundscheck(False)
#@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_fast(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size, int sc_len,
                                np.ndarray[double, ndim=1] dW):
    """
    fast Milstein for homodyne detection with >2 stochastic operators
    """
    cdef int sc2_len = 2 * sc_len

    cdef np.ndarray[complex, ndim=2] d_vec = \
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e = np.real(d_vec[:-1].reshape(
        -1, A_size, A_size).trace(axis1=1, axis2=2))
    d_vec[sc2_len:-1] -= np.array(
        [e[m] * d_vec[n] + e[n] * d_vec[m]
         for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    e[sc_len:sc2_len] -= 2.0 * e[:sc_len] * e[:sc_len]
    e[sc2_len:] -= 2.0 * np.array(
        [e[n] * e[m] for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    cdef np.ndarray[complex, ndim=1] drho_t = - np.inner(e, dW) * rho_t
    dW[:sc_len] -= 2.0 * e[:sc_len] * dW[sc_len:sc2_len]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t

@cython.boundscheck(False)
#@cython.wraparound(False)
cdef _rhs_rho_milstein_homodyne_fast_2(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data,
                                int[::1] A_ind, int[::1] A_ptr,
                                int A_size, int sc_len,
                                np.ndarray[double, ndim=1] dW):
    """
    fast Milstein for homodyne detection with >2 stochastic operators
    """
    cdef int sc2_len = 2 * sc_len

    cdef np.ndarray[complex, ndim=2] d_vec = \
        spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e = np.real(d_vec[:-1].reshape(
        -1, A_size, A_size).trace(axis1=1, axis2=2))
    d_vec[sc2_len:-1] -= np.array(
        [e[m] * d_vec[n] + e[n] * d_vec[m]
         for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    e[sc_len:sc2_len] -= 2.0 * e[:sc_len] * e[:sc_len]
    e[sc2_len:] -= 2.0 * np.array(
        [e[n] * e[m] for (n, m) in np.ndindex(sc_len, sc_len) if n > m])

    cdef np.ndarray[complex, ndim=1] drho_t = - np.inner(e, dW) * rho_t
    dW[:sc_len] -= 2.0 * e[:sc_len] * dW[sc_len:sc2_len]

    drho_t += d_vec[-1]
    drho_t += np.dot(dW, d_vec[:-1])

    return rho_t + drho_t
# -----------------------------------------------------------------------------
# Taylor15 rhs functions for the stochastic master equation
#
cdef _rhs_rho_taylor_15_one(np.ndarray[complex, ndim=1] rho_t,
                            complex[::1] A_data,
                            int[::1] A_ind, int[::1] A_ptr,
                            complex[::1] AL_data,
                            int[::1] AL_ind, int[::1] AL_ptr,
                            double dt, np.ndarray[double, ndim=1] dW):
    """
    strong order 1.5 Tylor scheme for homodyne detection with 1 stochastic operator
    """

    #reusable operators and traces
    cdef np.ndarray[complex, ndim=1] a = spmv_csr(AL_data, AL_ind, AL_ptr, rho_t)
    cdef double e0 = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, rho_t, 1)
    cdef np.ndarray[complex, ndim=1] b = spmv_csr(A_data, A_ind, A_ptr, rho_t) - e0 * rho_t
    cdef double TrAb = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, b, 1)
    cdef np.ndarray[complex, ndim=1] Lb = spmv_csr(A_data, A_ind, A_ptr, b) - TrAb * rho_t - e0 * b
    cdef double TrALb = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, Lb, 1)
    cdef double TrAa = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, a, 1)

    #print(dt,dW[0],dW[2])
    #print(rho_t[0],rho_t[1],rho_t[2],rho_t[3])
    cdef np.ndarray[complex, ndim=1] drho_t1

    cdef np.ndarray[complex, ndim=1] drho_t = a * dt
    #print(drho_t[0],drho_t[1],drho_t[2],drho_t[3])
    drho_t += b * dW[0]
    drho_t += Lb * dW[1] # Milstein term

    # new terms:
    drho_t += spmv_csr(AL_data, AL_ind, AL_ptr, b) * dW[2]
    drho_t1 = spmv_csr(AL_data, AL_ind, AL_ptr, b)
    #print(b[0],b[1],b[2],b[3])
    #print(drho_t1[0],drho_t1[1],drho_t1[2],drho_t1[3])
    #print(" ")
    drho_t += (spmv_csr(A_data, A_ind, A_ptr, a) - TrAa * rho_t - e0 * a - TrAb * b) * dW[3]
    drho_t += spmv_csr(AL_data, AL_ind, AL_ptr, a) * (0.5 * dt*dt)
    drho_t += (spmv_csr(A_data, A_ind, A_ptr, Lb) - TrALb * rho_t - (2 * TrAb) * b - e0 * Lb) * dW[4]

    return rho_t + drho_t

#include _rhs_rho_Taylor_15_two#

# -----------------------------------------------------------------------------
# Implicit rhs functions for the stochastic master equation
#
cdef _rhs_rho_milstein_implicit(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data, int[::1] A_ind,
                                int[::1] A_ptr, object Ae,
                                complex[::1] AL_data, int[::1] AL_ind,
                                int[::1] AL_ptr, double dt,
                                np.ndarray[double, ndim=1] dW, double tol):
    """
    Drift implicit Milstein (theta = 1/2, eta = 0)
    Wang, X., Gan, S., & Wang, D. (2012).
    A family of fully implicit Milstein methods for stiff stochastic differential
    equations with multiplicative noise.
    BIT Numerical Mathematics, 52(3), 741–772.
    """

    #reusable operators and traces
    cdef np.ndarray[complex, ndim=1] a = spmv_csr(AL_data, AL_ind, AL_ptr, rho_t) * (0.5 * dt)
    cdef double e0 = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, rho_t, 1)
    cdef np.ndarray[complex, ndim=1] b = spmv_csr(A_data, A_ind, A_ptr, rho_t) - e0 * rho_t
    cdef double TrAb = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, b, 1)

    cdef np.ndarray[complex, ndim=1] drho_t = b * dW[0]
    drho_t += a
    drho_t += (spmv_csr(A_data, A_ind, A_ptr, b)  - TrAb * rho_t - e0 * b) * dW[1] # Milstein term
    drho_t += rho_t

    cdef np.ndarray[complex, ndim=1] v
    v, check = sp.linalg.bicgstab(Ae, drho_t, x0 = drho_t + a, tol=tol)

    return v

cdef _rhs_rho_taylor_15_implicit(np.ndarray[complex, ndim=1] rho_t,
                                complex[::1] A_data, int[::1] A_ind,
                                int[::1] A_ptr, object Ae,
                                complex[::1] AL_data, int[::1] AL_ind,
                                int[::1] AL_ptr, double dt,
                                np.ndarray[double, ndim=1] dW, double tol):
    """
    Drift implicit Taylor 1.5 (alpha = 1/2, beta = doesn't matter)
    Chaptert 12.2 Eq. (2.18) in Numerical Solution of Stochastic Differential Equations
    By Peter E. Kloeden, Eckhard Platen
    """

    #reusable operators and traces
    cdef np.ndarray[complex, ndim=1] a = spmv_csr(AL_data, AL_ind, AL_ptr, rho_t)
    cdef double e0 = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, rho_t, 1)
    cdef np.ndarray[complex, ndim=1] b = spmv_csr(A_data, A_ind, A_ptr, rho_t) - e0 * rho_t
    cdef double TrAb = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, b, 1)
    cdef np.ndarray[complex, ndim=1] Lb = spmv_csr(A_data, A_ind, A_ptr, b) - TrAb * rho_t - e0 * b
    cdef double TrALb = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, Lb, 1)
    cdef double TrAa = cy_expect_rho_vec_csr(A_data, A_ind, A_ptr, a, 1)

    cdef np.ndarray[complex, ndim=1] drho_t = b * dW[0]
    drho_t += Lb * dW[1] # Milstein term
    cdef np.ndarray[complex, ndim=1] xx0 = (drho_t + a * dt) + rho_t #starting vector for the linear solver (Milstein prediction)
    drho_t += (0.5 * dt) * a

    # new terms:
    drho_t += spmv_csr(AL_data, AL_ind, AL_ptr, b) * (dW[2] - 0.5*dW[0]*dt)
    drho_t += (spmv_csr(A_data, A_ind, A_ptr, a) - TrAa * rho_t - e0 * a - TrAb * b) * dW[3]

    drho_t += (spmv_csr(A_data, A_ind, A_ptr, Lb) - TrALb * rho_t - (2 * TrAb) * b - e0 * Lb) * dW[4]
    drho_t += rho_t

    cdef np.ndarray[complex, ndim=1] v

    v, check = sp.linalg.bicgstab(Ae, drho_t, x0 = xx0, tol=tol)

    return v

# -----------------------------------------------------------------------------
# Predictor Corrector rhs functions for the stochastic master equation
#
@cython.wraparound(False)
@cython.boundscheck(False)
cdef _rhs_rho_pred_corr_homodyne_single(np.ndarray[complex, ndim=1] rho_t,
                                        complex[::1] A_data,
                                        int[::1] A_ind, int[::1] A_ptr,
                                        int A_size, double dt,
                                        np.ndarray[double, ndim=1] dW):
    """
    1/2 predictor-corrector scheme for homodyne detection with 1 stochastic operator
    """

    cdef int i, n_sc_A, l_rho
    l_rho = len(rho_t)

    #predictor
    cdef np.ndarray[complex, ndim=2] d_vec
    cdef np.ndarray[complex, ndim=1] e = np.zeros(2, dtype=complex)
    cdef np.ndarray[double, ndim=1] er = np.zeros(2)
    cdef np.ndarray[complex, ndim=1] a_pred, b_pred, a_corr, b_corr, pred_rho_t

    d_vec = spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(3, l_rho)
    n_sc_A = 2
    for i in range(A_size):
        e[0] += d_vec[0,(A_size+1)*i]
        e[1] += d_vec[1,(A_size+1)*i]

    a_pred = d_vec[n_sc_A]*1.0
    pred_rho_t = a_pred*1.0
    b_pred = - e[0] * rho_t

    for i in range(l_rho):
        b_pred[i] += d_vec[0,i]

        a_pred[i] -= ((d_vec[1,i] - e[1] * rho_t[i]) - (2.0 * e[0]) * b_pred[i]) * (0.5 * dt)

        pred_rho_t[i] += b_pred[i] * dW[0]
        pred_rho_t[i] += rho_t[i]

    #corrector
    d_vec = spmv_csr(A_data, A_ind, A_ptr, pred_rho_t).reshape(3, l_rho)
    for i in range(A_size):
        er[0] += d_vec[0,(A_size+1)*i].real
        er[1] += d_vec[1,(A_size+1)*i].real

    a_corr = d_vec[n_sc_A]
    b_corr = - er[0] * pred_rho_t

    for i in range(l_rho):
        b_corr[i] += d_vec[0,i]

        a_corr[i] -= ((d_vec[1,i] - er[1] * pred_rho_t[i]) - (2.0 * er[0]) * b_corr[i]) * (0.5 * dt)
        a_corr[i] += a_pred[i]
        a_corr[i] *= 0.5

        b_corr[i] += b_pred[i]
        b_corr[i] *= 0.5 * dW[0]

        rho_t[i] += a_corr[i]
        rho_t[i] += b_corr[i]

    return rho_t
# -----------------------------------------------------------------------------
# Predictor Corrector rhs functions for the stochastic master equation
#
cdef _rhs_rho_pred_corr_homodyne_single_2(np.ndarray[complex, ndim=1] rho_t,
                                        complex[::1] A_data,
                                        int[::1] A_ind, int[::1] A_ptr,
                                        int A_size, double dt,
                                        np.ndarray[double, ndim=1] dW):
    """
    1/2 predictor-corrector scheme for homodyne detection with 1 stochastic operator
    """

    #predictor
    cdef np.ndarray[complex, ndim=2] d_vec = spmv_csr(A_data, A_ind, A_ptr, rho_t).reshape(-1, len(rho_t))
    cdef np.ndarray[double, ndim=1] e = np.real(
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2))

    cdef np.ndarray[complex, ndim=1] a_pred = np.copy(d_vec[-1])
    cdef np.ndarray[complex, ndim=1] b_pred = - e[0] * rho_t
    b_pred += d_vec[0]

    cdef np.ndarray[complex, ndim=1] pred_rho_t = np.copy(a_pred)
    pred_rho_t += b_pred * dW[0]
    pred_rho_t += rho_t

    a_pred -= ((d_vec[1] - e[1] * rho_t) - (2.0 * e[0]) * b_pred) * (0.5 * dt)

    #corrector
    d_vec = spmv_csr(A_data, A_ind, A_ptr, pred_rho_t).reshape(-1, len(rho_t))
    e = np.real(
        d_vec[:-1].reshape(-1, A_size, A_size).trace(axis1=1, axis2=2))

    cdef np.ndarray[complex, ndim=1] a_corr = d_vec[-1]
    cdef np.ndarray[complex, ndim=1] b_corr = - e[0] * pred_rho_t
    b_corr += d_vec[0]

    a_corr -= ((d_vec[1] - e[1] * pred_rho_t) - (2.0 * e[0]) * b_corr) * (0.5 * dt)
    a_corr += a_pred
    a_corr *= 0.5

    b_corr += b_pred
    b_corr *= 0.5 * dW[0]

    cdef np.ndarray[complex, ndim=1] corr_rho_t = a_corr
    corr_rho_t += b_corr
    corr_rho_t += rho_t

    return corr_rho_t


cdef _generate_noise_Milstein(int sc_len, int N_store,
                int N_substeps, int d2_len, double dt):
    """
    generate noise terms for the fast Milstein scheme
    """
    cdef np.ndarray[double, ndim=4] dW_temp = np.sqrt(dt) * \
        np.random.randn(sc_len, N_store, N_substeps, 1)
    cdef np.ndarray[double, ndim=4] noise
    if sc_len == 1:
        noise = np.vstack([dW_temp, 0.5 * (dW_temp * dW_temp - dt * \
                          np.ones((sc_len, N_store, N_substeps, 1)))])
    else:
        noise = np.vstack(
            [dW_temp,
             0.5 * (dW_temp * dW_temp -
                    dt * np.ones((sc_len, N_store, N_substeps, 1)))] + \
            [[dW_temp[n] * dW_temp[m]
              for (n, m) in np.ndindex(sc_len, sc_len) if n > m]])

    return noise

cdef _generate_noise_Taylor_15(int sc_len, int N_store,
                int N_substeps, int d2_len, double dt):
    """
    generate noise terms for the strong Taylor 1.5 scheme
    """
    cdef np.ndarray[double, ndim=4] U1 = np.random.randn(sc_len, N_store, N_substeps, 1)
    cdef np.ndarray[double, ndim=4] U2 = np.random.randn(sc_len, N_store, N_substeps, 1)
    cdef np.ndarray[double, ndim=4] dW = U1 * np.sqrt(dt)
    cdef np.ndarray[double, ndim=4] dZ = 0.5 * dt**(3./2) * (U1 + 1./np.sqrt(3) * U2)
    cdef np.ndarray[double, ndim=4] noise
    if sc_len == 1:
        noise = np.vstack([ dW, 0.5 * (dW * dW - dt), dZ, dW * dt - dZ, 0.5 * (1./3. * dW**2 - dt) * dW ])

    elif sc_len == 2:
        noise = np.vstack([ dW, 0.5 * (dW**2 - dt), dZ, dW * dt - dZ, 0.5 * (1./3. * dW**2 - dt) * dW]
                    + [[dW[n] * dW[m] for (n, m) in np.ndindex(sc_len, sc_len) if n < m]]  # Milstein
                    + [[0.5 * dW[n] * (dW[m]**2 - dt) for (n, m) in np.ndindex(sc_len, sc_len) if n != m]])

    #else:
        #noise = np.vstack([ dW, 0.5 * (dW**2 - dt), dZ, dW * dt - dZ, 0.5 * (1./3. * dW**2 - dt) * dW]
                    #+ [[dW[n] * dW[m] for (n, m) in np.ndindex(sc_len, sc_len) if n > m]]  # Milstein
                    #+ [[0.5 * dW[n] * (dW[m]**2 - dt) for (n, m) in np.ndindex(sc_len, sc_len) if n != m]]
                    #+ [[dW[n] * dW[m] * dW[k] for (n, m, k) in np.ndindex(sc_len, sc_len, sc_len) if n>m>k]])
    else:
        raise Exception("too many stochastic operators")

    return noise
