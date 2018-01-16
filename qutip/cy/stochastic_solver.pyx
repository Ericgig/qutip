

#from qutip.cy.spmatfuncs cimport (spmv_csr)
#import scipy.sparse as sp
#from qutip.cy.spmatfuncs import cy_expect_rho_vec_csr, cy_expect_rho_vec
#from qutip.superoperator import mat2vec, vec2mat
#from qutip.cy.spmatfuncs import cy_expect_psi_csr, cy_expect_rho_vec
#from qutip.qobj import Qobj
import time

import numpy as np
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.td_qobj_cy cimport cy_qobj
from qutip.qobj import Qobj
from qutip.superoperator import vec2mat
include "parameters.pxi"
include "complex_math.pxi"

import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.linalg.cython_blas cimport zaxpy, zdotu, zdotc
from scipy.linalg.cython_blas cimport dznrm2 as raw_dznrm2


cdef int ONE = 1

cpdef void axpy(complex a, complex[::1] x, complex[::1] y):
    cdef int l = x.shape[0]
    zaxpy(&l, &a, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

cpdef complex dot(complex[::1] x,complex[::1] y):
    cdef int l = x.shape[0]
    return zdotu(&l, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

cpdef complex dotc(complex[::1] x,complex[::1] y):
    cdef int l = x.shape[0]
    return zdotc(&l, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

cpdef double dznrm2(complex[::1] vec):
    cdef int l = vec.shape[0]
    return raw_dznrm2(&l, <complex*>&vec[0], &ONE)

cpdef void normalize_inplace(complex[::1] vec):
    cdef int i, l = vec.shape[0]
    cdef double norm = dznrm2(vec)
    for i in range(l):
        vec[i] /= norm


"""
Solver:
  order 0.5
    euler-maruyama     50
  order 1.0
    platen            100
    pred_corr         101
    milstein          102
    milstein-imp      103
    pred_corr(2)      104
    runge-kutta       105
    platen(2)         106
  order 1.5
    platen1.5         150
    taylor1.5         152
    taylor1.5-imp     153
  order 2.0
    pred_corr2.0      201
    taylor2.0         202
    taylor2.0-imp     203
"""

cdef class ssolvers:
    cdef int l_vec, N_ops
    cdef int solver#, noise_type
    cdef int N_step, N_substeps, N_dw
    cdef int normalize
    cdef double dt
    cdef int noise_type
    cdef double[:, :, :, ::1] custom_noise
    cdef double[::1] dW_factor
    cdef unsigned int[::1] seed
    cdef object generate_noise

    cdef object debug

    def __init__(self):
        self.l_vec = 0
        self.N_ops = 0
        self.solver = 0

    def set_solver(self, sso):
        self.debug = sso.debug
        self.solver = sso.solver_code
        self.dt = sso.dt
        self.N_substeps = sso.nsubsteps
        if self.solver in [103, 153, 154]:
            self.set_implicit(sso)
        self.normalize = sso.normalize and not sso.me
        self.N_step = len(sso.times)
        self.N_dw = len(sso.sops)
        if self.solver in [150, 151, 152, 153, 154]:
            self.N_dw *= 2
        self.noise_type = sso.noise_type
        self.dW_factor = np.array(sso.dW_factors,dtype=np.float64)
        if self.noise_type == 2:
            self.generate_noise = sso.generate_noise
        elif self.noise_type == 1:
            self.custom_noise = sso.noise
        elif self.noise_type == 0:
            self.seed = sso.noise

    cdef double[:, :, ::1] make_noise(self, int n):
        if self.noise_type == 0:
            np.random.seed(self.seed[n])
            return np.random.randn(self.N_step, self.N_substeps, self.N_dw) *\
                                   np.sqrt(self.dt)
        elif self.noise_type == 1:
            return self.custom_noise[n,:,:,:]
        elif self.noise_type == 2:
            return self.generate_noise((self.N_step, self.N_substeps, self.N_dw),
                                        self.dt, n)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cy_sesolve_single_trajectory(self, int n, sso):
        cdef double[::1] times = sso.times
        cdef complex[::1] rho_t
        cdef double t
        cdef int m_idx, t_idx, e_idx
        cdef double[:, :, ::1] noise = self.make_noise(n)

        rho_t = sso.rho0.copy()
        dims = sso.state0.dims

        expect = np.zeros((len(sso.ce_ops), len(sso.times)), dtype=complex)
        ss = np.zeros((len(sso.ce_ops), len(sso.times)), dtype=complex)
        measurements = np.zeros((len(times), len(sso.cm_ops)), dtype=complex)
        states_list = []
        for t_idx, t in enumerate(times):
            if sso.ce_ops:
                for e_idx, e in enumerate(sso.ce_ops):
                    s = e.compiled_Qobj.expect(t, rho_t, 0)
                    expect[e_idx, t_idx] = s
                    ss[e_idx, t_idx] = s ** 2
            if sso.store_states or not sso.ce_ops:
                if sso.me:
                    states_list.append(Qobj(vec2mat(np.asarray(rho_t)),
                        dims=dims))
                else:
                    states_list.append(Qobj(np.asarray(rho_t), dims=dims))

            if sso.store_measurement:
                for m_idx, m in enumerate(sso.cm_ops):
                    m_expt = m.compiled_Qobj.expect(t, rho_t, 0)
                    measurements[t_idx, m_idx] = m_expt + self.dW_factor[m_idx] * \
                        sum(noise[t_idx, :, m_idx]) / (self.dt * self.N_substeps)

            rho_t = self.run(t, self.dt, noise[t_idx, :, :],
                             rho_t, self.N_substeps)

        if sso.method == 'heterodyne':
            measurements = measurements.reshape(len(times),len(sso.cm_ops)//2,2)

        return states_list, noise, measurements, expect, ss

    cdef complex[::1] run(self, double t, double dt, double[:, ::1] noise,
                          complex[::1] vec, int N_substeps):
        cdef complex[::1] out = np.zeros(self.l_vec, dtype=complex)
        cdef int i
        if self.solver == 50:
            for i in range(N_substeps):
                self.euler(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 100:
            for i in range(N_substeps):
                self.platen(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 101:
            for i in range(N_substeps):
                self.pred_corr(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 102:
            for i in range(N_substeps):
                self.milstein(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 103:
            for i in range(N_substeps):
                self.milstein_imp(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 104:
            for i in range(N_substeps):
                self.pred_corr_a(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 105:
            for i in range(N_substeps):
                self.platen_m(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 150:
            for i in range(N_substeps):
                self.platen15(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 151:
            for i in range(N_substeps):
                self.taylor15(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 152:
            for i in range(N_substeps):
                self.taylor15_1(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 153:
            for i in range(N_substeps):
                self.taylor15_1_imp(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 154:
            for i in range(N_substeps):
                self.taylor15_imp(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        if self.normalize:
            normalize_inplace(vec)
        return vec

    # Dummy functions
    cdef void d1(self, double t, complex[::1] v, complex[::1] out):
        pass

    cdef void d2(self, double t, complex[::1] v, complex[:, ::1] out):
        pass

    cdef void d2d2p(self, double t, complex[::1] v,
                    complex[:, ::1] out, complex[:, :, ::1] out2):
        pass

    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                       complex[::1] out, np.ndarray[complex, ndim=1] guess):
        pass

    cdef void taylor_15_1(self, double t, complex[::1] rho,
                           complex[:, ::1] out):
        pass

    cdef void derivatives(self, double t, int deg, complex[::1] rho,
                              complex[::1] a, complex[:, ::1] b,
                              complex[:, :, ::1] Lb, complex[:,::1] La,
                              complex[:, ::1] L0b, complex[:, :, :, ::1] LLb,
                              complex[::1] L0a):
        pass

    def set_implicit(self, sso):
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void euler(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        cdef int i, j
        cdef complex[:, ::1] d2 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        for j in range(self.l_vec):
            out[j] = vec[j]
        self.d1(t, vec, out)
        self.d2(t, vec, d2)
        for i in range(self.N_ops):
            for j in range(self.l_vec):
                out[j] += d2[i,j] * noise[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void platen(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        """
        # Platen rhs function for both master eq and schrodinger eq.
        # dV = -iH* (V+Vt)/2 * dt + (d1(V)+d1(Vt))/2 * dt
        #      + (2*d2_i(V)+d2_i(V+)+d2_i(V-))/4 * dW_i
        #      + (d2_i(V+)-d2_i(V-))/4 * (dW_i**2 -dt) * dt**(-.5)

        # Vt = V -iH*V*dt + d1*dt + d2_i*dW_i
        # V+/- = V -iH*V*dt + d1*dt +/- d2_i*dt**.5
        """

        cdef int i, j
        cdef double sqrt_dt = np.sqrt(dt)
        cdef double sqrt_dt_inv = 0.25/sqrt_dt
        cdef double dw, dw2

        cdef complex[::1] d1 = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] Vp = np.empty(self.l_vec, dtype=complex)
        cdef complex[::1] Vm = np.empty(self.l_vec, dtype=complex)
        cdef complex[::1] Vt = np.empty(self.l_vec, dtype=complex)
        cdef complex[:, ::1] d2 = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] d2p = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] d2m = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        self.d1(t, vec, d1)
        for j in range(self.l_vec):
          d1[j] += vec[j]
          Vp[j] = d1[j]
          Vm[j] = d1[j]
          Vt[j] = d1[j]
          out[j] = 0.5* d1[j]

        self.d2(t, vec, d2)
        for i in range(self.N_ops):
          for j in range(self.l_vec):
            Vp[j] += d2[i,j] * sqrt_dt
            Vm[j] -= d2[i,j] * sqrt_dt
            Vt[j] += d2[i,j] * noise[i]

        d1 = np.zeros(self.l_vec, dtype=complex)
        self.d1(t, Vt, d1)  #  t+dt
        for j in range(self.l_vec):
          out[j] += 0.5* (d1[j] + vec[j])

        self.d2(t, Vp, d2p)  #  t+dt
        self.d2(t, Vm, d2m)  #  t+dt
        for i in range(self.N_ops):
          dw = noise[i] * 0.25
          dw2 =  sqrt_dt_inv * (noise[i]*noise[i] - dt)
          for j in range(self.l_vec):
            out[j] += dw * (d2[i,j] *2 + d2p[i,j] + d2m[i,j])
            out[j] += dw2 * (d2p[i,j] - d2m[i,j])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void platen_m(self, double t, double dt, double[:] noise,
                       complex[::1] vec, complex[::1] out):
        """
        Platen rhs function for both master eq and schrodinger eq.
        dV = -iH* (V+Vt)/2 * dt + (d1(V)+d1(Vt))/2 * dt
             + (2*d2_i(V)+d2_i(V+)+d2_i(V-))/4 * dW_i
             + (d2_i(V+)-d2_i(V-))/4 * (dW_i**2 -dt) * dt**(-.5)

        Vt = V -iH*V*dt + d1*dt + d2_i*dW_i
        V+/- = V -iH*V*dt + d1*dt +/- d2_i*dt**.5
        """

        cdef int i, j, k
        cdef double sqrt_dt = np.sqrt(dt)
        cdef double sqrt_dt_inv = 0.25/sqrt_dt
        cdef double dw, dw2

        cdef complex[::1] d1 = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] Vt = np.empty(self.l_vec, dtype=complex)
        cdef complex[:, ::1] d2 = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] Vm = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] Vp = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] d2p
        cdef complex[:, ::1] d2m

        self.d1(t, vec, d1)
        self.d2(t, vec, d2)
        for j in range(self.l_vec):
            # out[j] = vec[j] + d1[j]
            # d1[j] += vec[j]
            d1[j] += vec[j]
            Vt[j] = d1[j]
            out[j] = 0.5* d1[j]

        for i in range(self.N_ops):
          for j in range(self.l_vec):
            Vp[i,j] += d1[j]
            Vm[i,j] += d1[j]
            Vp[i,j] += d2[i,j] * sqrt_dt
            Vm[i,j] -= d2[i,j] * sqrt_dt
            Vt[j] += d2[i,j] * noise[i]

        d1 = np.zeros(self.l_vec, dtype=complex)
        self.d1(t, Vt, d1)  #  t+dt
        for j in range(self.l_vec):
          out[j] += 0.5* (d1[j] + vec[j])

        for i in range(self.N_ops):
          for j in range(self.l_vec):
            out[j] += d2[i,j] * noise[i]

        for i in range(self.N_ops):
          d2p = np.zeros((self.N_ops, self.l_vec), dtype=complex)
          d2m = np.zeros((self.N_ops, self.l_vec), dtype=complex)
          self.d2(t, Vp[i,:], d2p)
          self.d2(t, Vm[i,:], d2m)
          for j in range(self.N_ops):
            if i == j:
              dw2 = sqrt_dt_inv * (noise[i]*noise[i] - dt)
            else:
              dw2 = sqrt_dt_inv * noise[i] * noise[j]
            for k in range(self.l_vec):
              out[k] += dw2 * (d2p[j,k] - d2m[j,k])

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void pred_corr(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        # a=0. b=0.5
        cdef int i, j, k
        cdef complex[::1] euler = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] a_pred = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] b_pred = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] b_corr = np.zeros((self.l_vec), dtype=complex)
        cdef complex[:, ::1] d2 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] dd2 = np.zeros((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)
        cdef double dt_2 = dt*0.5
        self.d1(t, vec, a_pred)
        self.d2d2p(t, vec, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                b_pred[j] = d2[i,j] * noise[i]

        for j in range(self.l_vec):
            euler[j] = a_pred[j] + b_pred[j] + vec[j]

        for i in range(self.N_ops):
            for k in range(self.l_vec):
                a_pred[k] -= dd2[i,i,k] * dt_2

        d2 = np.zeros((self.N_ops, self.l_vec), dtype=complex)
        self.d2(t + dt, euler, d2)

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                b_corr[j] = d2[i,j] * noise[i]

        for j in range(self.l_vec):
            out[j] = vec[j] + a_pred[j] + (b_pred[j] + b_corr[j]) * 0.5

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void pred_corr_a(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        # a=0.5, b=0.5
        cdef int i, j, k
        cdef complex[::1] euler = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] a_pred = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] b_pred = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] a_corr = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] b_corr = np.zeros((self.l_vec), dtype=complex)
        cdef complex[:, ::1] d2 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] dd2 = np.zeros((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)
        cdef double dt_2 = dt*0.5
        self.d1(t, vec, a_pred)
        self.d2d2p(t, vec, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                b_pred[j] = d2[i,j] * noise[i]

        for j in range(self.l_vec):
            euler[j] = a_pred[j] + b_pred[j] + vec[j]

        for i in range(self.N_ops):
            for k in range(self.l_vec):
                a_pred[k] -= dd2[i,i,k] * dt_2

        d2 = np.zeros((self.N_ops, self.l_vec), dtype=complex)
        dd2 = np.zeros((self.N_ops, self.N_ops, self.l_vec), dtype=complex)
        self.d1(t+dt, euler, a_corr)
        self.d2d2p(t+dt, euler, d2, dd2)

        for i in range(self.N_ops):
            for k in range(self.l_vec):
                a_corr[k] -= dd2[i,i,k] * dt_2

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                b_corr[j] = d2[i,j] * noise[i]

        for j in range(self.l_vec):
            out[j] = vec[j] + (a_pred[j] + a_corr[j] +
                               b_pred[j] + b_corr[j]) * 0.5

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void milstein(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        """
        Milstein rhs function for both master eq and schrodinger eq.

        Slow but should be valid for non-commuting operators since computing
            both i x j and j x i. Need to be commuting anyway

        dV = -iH*V*dt + d1*dt + d2_i*dW_i
             + 0.5*d2_i' d2_j*(dW_i*dw_j -dt*delta_ij)
        """
        cdef int i, j, k
        cdef double dw
        cdef complex[:, ::1] d2 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] dd2 = np.zeros((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)
        for j in range(self.l_vec):
            out[j] = vec[j]
        self.d1(t, vec, out)
        self.d2d2p(t, vec, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                out[j] += d2[i,j] * noise[i]

        for i in range(self.N_ops):
            for j in range(i,self.N_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j])
                for k in range(self.l_vec):
                    out[k] += dd2[i,j,k] * dw

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void milstein_imp(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef int i, j, k
        cdef double dw
        cdef np.ndarray[complex, ndim=1] guess = np.zeros((self.l_vec, ),
                                                          dtype=complex)
        cdef np.ndarray[complex, ndim=1] a = np.zeros((self.l_vec, ),
                                                      dtype=complex)
        cdef np.ndarray[complex, ndim=1] dvec = np.zeros((self.l_vec, ),
                                                         dtype=complex)
        cdef complex[:, ::1] d2 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] dd2 = np.zeros((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)

        self.d1(t, vec, a)
        self.d2d2p(t, vec, d2, dd2)

        for j in range(self.l_vec):
            dvec[j] = a[j]*0.5 + vec[j]

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                dvec[j] += d2[i,j] * noise[i]

        for i in range(self.N_ops):
            for j in range(self.N_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j])
                for k in range(self.l_vec):
                    dvec[k] += dd2[i,j,k] * dw

        for j in range(self.l_vec):
            guess[j] = dvec[j] + a[j]*0.5

        self.implicit(t+dt, dvec, out, guess)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void taylor15_1(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef int i
        cdef complex[:, ::1] dvec = np.zeros((7, self.l_vec),
                                                          dtype=complex)
        cdef double dw = noise[0]
        # The dt of dz is included in the d1 part (Ldt) and the noise (dt**.5)
        cdef double dz = 0.5 *(dw + 1./np.sqrt(3) * noise[1])

        self.taylor_15_1(t, vec, dvec)
        for i in range(self.l_vec):
            out[i] = vec[i] + dvec[0,i]
        axpy(dw, dvec[1,:], out)
        axpy(0.5*(dw*dw-dt), dvec[2,:], out)
        axpy(dz, dvec[3,:], out)
        axpy(dw-dz , dvec[4,:], out)
        axpy(0.5, dvec[5,:], out)
        axpy(0.5 * (0.3333333333333333 * dw * dw - dt) * dw,
             dvec[6,:], out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void taylor15(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef complex[::1] a = np.zeros((self.l_vec), dtype=complex)
        cdef complex[:, ::1] b = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] Lb = np.zeros((self.N_ops, self.N_ops,
                                               self.l_vec), dtype=complex)
        cdef complex[:, ::1] L0b = np.zeros((self.N_ops,
                                             self.l_vec), dtype=complex)
        cdef complex[:,::1] La = np.zeros((self.N_ops,
                                           self.l_vec), dtype=complex)
        cdef complex[:, :, :, ::1] LLb = np.zeros((self.N_ops, self.N_ops,
                                        self.N_ops, self.l_vec), dtype=complex)
        cdef complex[::1] L0a = np.zeros((self.l_vec), dtype=complex)
        self.derivatives(t, 2, vec, a, b, Lb, La, L0b, LLb, L0a)

        cdef int i,j,k
        cdef double[::1] dz, dw
        dw = np.zeros(self.N_ops)
        dz = np.zeros(self.N_ops)
        # The dt of dz is included in the d1 part (Ldt) and the noise (dt**.5)
        for i in range(self.N_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.N_ops])

        for i in range(self.l_vec):
            out[i] = a[i] + vec[i]
        axpy(0.5*self.debug[0], L0a, out)

        for i in range(self.N_ops):
            axpy(dw[i], b[i,:], out)
            axpy(0.5*(dw[i]*dw[i]-dt)*self.debug[1], Lb[i,i,:], out)
            axpy(dz[i]*self.debug[0], La[i,:], out)
            axpy(dw[i]-dz[i]*self.debug[0] , L0b[i,:], out)
            axpy(0.5 * (0.3333333333333333 * dw[i] * dw[i] - dt) * dw[i]*self.debug[0],
                        LLb[i,i,i,:], out)
            for j in range(i+1,self.N_ops):
                axpy((dw[i]*dw[j]*self.debug[1]), Lb[i,j,:], out)
                axpy(0.5*(dw[j]*dw[j]-dt)*dw[i]*self.debug[0], LLb[i,j,j,:], out)
                axpy(0.5*(dw[i]*dw[i]-dt)*dw[j]*self.debug[0], LLb[i,i,j,:], out)
                for k in range(j+1,self.N_ops):
                    axpy(dw[i]*dw[j]*dw[k]*self.debug[0], LLb[i,j,k,:], out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void taylor15_1_imp(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef int i
        cdef np.ndarray[complex, ndim=1] d1 = np.zeros(self.l_vec,
                                                       dtype=complex)
        cdef np.ndarray[complex, ndim=1] guess = np.zeros((self.l_vec, ),
                                                          dtype=complex)
        cdef np.ndarray[complex, ndim=1] vec_t = np.zeros((self.l_vec, ),
                                                         dtype=complex)
        cdef complex[:, ::1] dvec = np.zeros((7, self.l_vec),
                                                          dtype=complex)
        cdef double dw = noise[0]
        # The dt of dz is included in the d1 part (Ldt) and the noise (dt**.5)
        cdef double dz = 0.5 *(dw + 1./np.sqrt(3) * noise[1])

        self.taylor_15_1(t, vec, dvec)
        for i in range(self.l_vec):
            vec_t[i] = vec[i] + dvec[0,i]*0.5
        axpy(dw, dvec[1,:], vec_t)
        axpy(0.5*(dw*dw-dt), dvec[2,:], vec_t)
        axpy(dz-0.5*dw, dvec[3,:], vec_t)
        #self.d1(t, dvec[1,:], d1)
        #axpy(-0.5*dw, d1, vec_t)
        axpy(dw-dz, dvec[4,:], vec_t)
        axpy(0.5 * (0.3333333333333333 * dw * dw - dt) * dw,
             dvec[6,:], vec_t)
        for i in range(self.l_vec):
            guess[i] = vec_t[i] + dvec[0,i]*0.5

        self.implicit(t+dt, vec_t, out, guess)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void taylor15_imp(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef complex[::1] a = np.zeros((self.l_vec), dtype=complex)
        cdef complex[:, ::1] b = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] Lb = np.zeros((self.N_ops, self.N_ops,
                                               self.l_vec), dtype=complex)
        cdef complex[:, ::1] L0b = np.zeros((self.N_ops,
                                             self.l_vec), dtype=complex)
        cdef complex[:,::1] La = np.zeros((self.N_ops,
                                           self.l_vec), dtype=complex)
        cdef complex[:, :, :, ::1] LLb = np.zeros((self.N_ops, self.N_ops,
                                        self.N_ops, self.l_vec), dtype=complex)
        cdef complex[::1] L0a = np.zeros((self.l_vec), dtype=complex)
        cdef np.ndarray[complex, ndim=1] guess = np.zeros((self.l_vec, ),
                   dtype=complex)
        cdef np.ndarray[complex, ndim=1] vec_t = np.zeros((self.l_vec, ),
                  dtype=complex)
        self.derivatives(t, 2, vec, a, b, Lb, La, L0b, LLb, L0a)

        cdef int i,j,k
        cdef double[::1] dz, dw
        dw = np.zeros(self.N_ops)
        dz = np.zeros(self.N_ops)
        # The dt of dz is included in the d1 part (Ldt) and the noise (dt**.5)
        for i in range(self.N_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.N_ops])

        for i in range(self.l_vec):
            vec_t[i] = a[i]*0.5 + vec[i]
        #axpy(0.5, L0a, vec_t)

        for i in range(self.N_ops):
            axpy(dw[i], b[i,:], vec_t)
            axpy(0.5*(dw[i]*dw[i]-dt), Lb[i,i,:], vec_t)
            axpy(dz[i]-dw[i]*0.5, La[i,:], vec_t)
            axpy(dw[i]-dz[i] , L0b[i,:], vec_t)
            axpy(0.5 * (0.3333333333333333 * dw[i] * dw[i] - dt) * dw[i],
                        LLb[i,i,i,:], vec_t)
            for j in range(i+1,self.N_ops):
                axpy((dw[i]*dw[j]), Lb[i,j,:], vec_t)
                axpy(0.5*self.debug[0]*(dw[j]*dw[j]-dt)*dw[i], LLb[i,j,j,:], vec_t)
                axpy(0.5*self.debug[0]*(dw[i]*dw[i]-dt)*dw[j], LLb[i,i,j,:], vec_t)
                for k in range(j+1,self.N_ops):
                    axpy(dw[i]*dw[j]*dw[k], LLb[i,j,k,:], vec_t)

        for i in range(self.l_vec):
            guess[i] = vec_t[i] + a[i]*0.5

        self.implicit(t+dt, vec_t, out, guess)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void platen15(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        """
        Chaptert 11.2 Eq. (2.13)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef int i, j, k
        cdef double sqrt_dt = np.sqrt(dt)
        cdef double sqrt_dt_inv = 1./sqrt_dt
        cdef double ddz, ddw, ddd
        cdef double[::1] dz, dw
        dw = np.zeros(self.N_ops)
        dz = np.zeros(self.N_ops)
        for i in range(self.N_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.N_ops])

        cdef complex[::1] d1 = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] d1p = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] d1m = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] V = np.zeros(self.l_vec, dtype=complex)
        cdef complex[:, ::1] d2 = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] dd2 = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] d2p = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] d2m = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] d2pp = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] d2mm = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] v2p = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, ::1] v2m = np.zeros((self.N_ops, self.l_vec),
                                            dtype=complex)
        cdef complex[:, :, ::1] p2p = np.zeros((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)
        cdef complex[:, : ,::1] p2m = np.zeros((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)


        cdef complex[:, :, ::1] ddd2 = np.zeros((self.N_ops, self.N_ops,
                                                  self.l_vec), dtype=complex)
        self.d2d2p(t, vec, d2, ddd2)

        self.d1(t, vec, d1)
        #self.d2(t, vec, d2)
        self.d2(t + dt, vec, dd2)

        cdef complex[::1] temp = np.zeros(self.l_vec, dtype=complex)

        # Euler part
        for j in range(self.l_vec):
            out[j] = d1[j] + vec[j]

        for i in range(self.N_ops):
            axpy(dw[i], d2[i,:], out)

        # ~Milstein part
        axpy(1., vec, V)
        axpy(1./self.N_ops, d1, V)
        for i in range(self.N_ops):
            axpy(1., V, v2p[i,:])
            axpy(sqrt_dt, d2[i,:], v2p[i,:])
            axpy(1., V, v2m[i,:])
            axpy(-sqrt_dt, d2[i,:], v2m[i,:])

        for i in range(self.N_ops):
            d2p = np.zeros((self.N_ops, self.l_vec), dtype=complex)
            d2m = np.zeros((self.N_ops, self.l_vec), dtype=complex)
            self.d2(t, v2p[i,:], d2p)
            self.d2(t, v2m[i,:], d2m)
            ddw = (dw[i]*dw[i]-dt)*0.25/sqrt_dt  # 1.0
            axpy( ddw, d2p[i,:], out)
            axpy(-ddw, d2m[i,:], out)

        for i in range(self.N_ops):
            d2p = np.zeros((self.N_ops, self.l_vec), dtype=complex)
            self.d2(t, v2p[i,:], d2p)
            for j in range(self.N_ops):
                axpy(      1., v2p[i,:], p2p[i,j,:])
                axpy( sqrt_dt, d2p[j,:], p2p[i,j,:])
                axpy(      1., v2p[i,:], p2m[i,j,:])
                axpy(-sqrt_dt, d2p[j,:], p2m[i,j,:])

        axpy(-0.5*(self.N_ops)*self.debug[1], d1, out)
        for i in range(self.N_ops):
            ddz = dz[i]*0.5/sqrt_dt  # 1.5
            ddd = 0.25*(dw[i]*dw[i]/3-dt)*dw[i]/dt  # 1.5
            d1p = np.zeros((self.l_vec), dtype=complex)
            d1m = np.zeros((self.l_vec), dtype=complex)
            d2m = np.zeros((self.N_ops, self.l_vec), dtype=complex)
            d2p = np.zeros((self.N_ops, self.l_vec), dtype=complex)
            d2pp = np.zeros((self.N_ops, self.l_vec), dtype=complex)
            d2mm = np.zeros((self.N_ops, self.l_vec), dtype=complex)
            self.d1(t + dt/self.N_ops, v2p[i,:], d1p)
            self.d1(t + dt/self.N_ops, v2m[i,:], d1m)
            self.d2(t, v2p[i,:], d2p)
            self.d2(t, v2m[i,:], d2m)
            self.d2(t, p2p[i,i,:], d2pp)
            self.d2(t, p2m[i,i,:], d2mm)

            axpy(0.25*self.debug[1], d1p, out)
            axpy(0.25*self.debug[1], d1m, out)

            axpy( ddz, d1p, out)
            axpy(-ddz, d1m, out)

            axpy((dw[i]-dz[i])*self.debug[0], dd2[i,:], out)
            axpy((dz[i]-dw[i])*self.debug[0], d2[i,:], out)

            axpy( ddd, d2pp[i,:], out)
            axpy(-ddd, d2mm[i,:], out)
            axpy(-ddd, d2p[i,:], out)
            axpy( ddd, d2m[i,:], out)

            for j in range(self.N_ops):
                ddw = 0.5*(dw[j]-dz[j]) # 1.5
                axpy(ddw, d2p[j,:], out)
                axpy(-2*ddw, d2[j,:], out)
                axpy(ddw, d2m[j,:], out)

            for j in range(i+1,self.N_ops):
                ddw = 0.5*(dw[i]*dw[j])/sqrt_dt # 1.0
                axpy( ddw, d2p[j,:], out)
                axpy(-ddw, d2m[j,:], out)

            for j in range(i+1,self.N_ops):
                ddw = 0.25*(dw[j]*dw[j]-dt)*dw[i]/dt*self.debug[2] # 1.5
                d2pp = np.zeros((self.N_ops, self.l_vec), dtype=complex)
                d2mm = np.zeros((self.N_ops, self.l_vec), dtype=complex)
                self.d2(t, p2p[j,i,:], d2pp)
                self.d2(t, p2m[j,i,:], d2mm)
                axpy( ddw, d2pp[j,:], out)
                axpy(-ddw, d2mm[j,:], out)
                axpy(-ddw, d2p[j,:], out)
                axpy( ddw, d2m[j,:], out)

                for k in range(j+1,self.N_ops):
                    ddw = 0.5*dw[i]*dw[j]*dw[k]/dt*self.debug[2] # 1.5
                    axpy( ddw, d2pp[k,:], out)
                    axpy(-ddw, d2mm[k,:], out)
                    axpy(-ddw, d2p[k,:], out)
                    axpy( ddw, d2m[k,:], out)

            for j in range(i):
                ddw = 0.25*(dw[j]*dw[j]-dt)*dw[i]/dt*self.debug[2] # 1.5
                d2pp = np.zeros((self.N_ops, self.l_vec), dtype=complex)
                d2mm = np.zeros((self.N_ops, self.l_vec), dtype=complex)
                self.d2(t, p2p[j,i,:], d2pp)
                self.d2(t, p2m[j,i,:], d2mm)
                axpy( ddw, d2pp[j,:], out)
                axpy(-ddw, d2mm[j,:], out)
                axpy(-ddw, d2p[j,:], out)
                axpy( ddw, d2m[j,:], out)

    def checks(self, double t, double dt, complex[::1] vec):
        cdef complex[::1] a = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] a1 = np.zeros((self.l_vec), dtype=complex)
        cdef complex[:, ::1] b = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, ::1] b1 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, ::1] b2 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] Lb = np.zeros((self.N_ops, self.N_ops,
                    self.l_vec), dtype=complex)
        cdef complex[:, :, ::1] Lb2 = np.zeros((self.N_ops, self.N_ops,
                                               self.l_vec), dtype=complex)
        cdef complex[:, ::1] L0b = np.zeros((self.N_ops,
                                             self.l_vec), dtype=complex)
        cdef complex[:,::1] La = np.zeros((self.N_ops,
                                           self.l_vec), dtype=complex)
        cdef complex[:, :, :, ::1] LLb = np.zeros((self.N_ops, self.N_ops,
                                        self.N_ops, self.l_vec), dtype=complex)
        cdef complex[::1] L0a = np.zeros((self.l_vec), dtype=complex)

        cdef complex[:, ::1] dvec = np.zeros((7, self.l_vec), dtype=complex)

        self.d1(t, vec, a)
        self.d2(t, vec, b)
        self.d2d2p(t, vec, b1, Lb)
        self.derivatives(t, 2, vec, a1, b2, Lb2, La, L0b, LLb, L0a)
        self.taylor_15_1(t, vec, dvec)

        print("a[0]",a[0],a1[0],dvec[0,0])
        print("a[1]",a[1],a1[1],dvec[0,1])
        for i in range(self.N_ops):
            print(i)
            print("b[0]",b[i,0],b1[i,0],b2[i,0],dvec[1,0])
            print("b[1]",b[i,1],b1[i,1],b2[i,1],dvec[1,1])
        for i in range(self.N_ops):
          for j in range(self.N_ops):
            print(i,j)
            print("Lb[0]",Lb[i,j,0],Lb[i,j,0],dvec[2,0])
            print("Lb[1]",Lb[i,j,1],Lb[i,j,1],dvec[2,1])
        for i in range(self.N_ops):
            print(i)
            print("La[0]",La[i,0],dvec[3,0])
            print("La[1]",La[i,1],dvec[3,1])
            print("L0b[0]",L0b[i,0],dvec[4,0])
            print("L0b[1]",L0b[i,1],dvec[4,1])
        for i in range(self.N_ops):
          for j in range(self.N_ops):
            for k in range(self.N_ops):
              print(i,j,k)
              print("LLb[0]",LLb[i,j,k,0],dvec[6,0])
              print("LLb[1]",LLb[i,j,k,1],dvec[6,1])
        print("L0a[0]",L0a[0],dvec[5,0])
        print("L0a[1]",L0a[1],dvec[5,1])

cdef class sse(ssolvers):
    cdef cy_qobj L
    cdef object c_ops
    cdef object cdc_ops
    cdef object cpcd_ops
    cdef object imp
    cdef double tol, imp_t

    def set_data(self, L, c_ops):
        self.l_vec = L.cte.shape[0]
        self.N_ops = len(c_ops)
        self.L = L.compiled_Qobj
        self.c_ops = []
        self.cdc_ops = []
        self.cpcd_ops = []
        for i, op in enumerate(c_ops):
            self.c_ops.append(op[0].compiled_Qobj)
            self.cdc_ops.append(op[1].compiled_Qobj)
            self.cpcd_ops.append(op[2].compiled_Qobj)

    def implicit_op(self, vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.l_vec, dtype=complex)
        self.d1(self.imp_t, vec, out)
        cdef int i
        for i in range(self.l_vec):
            out[i]=vec[i] - 0.5* out[i]
        return out

    def set_implicit(self, sso):
        self.tol = sso.tol
        self.imp = LinearOperator( (self.l_vec,self.l_vec),
                                  matvec=self.implicit_op, dtype=complex)

    cdef void d1(self, double t, complex[::1] vec, complex[::1] out):
        self.L._rhs_mat(t, &vec[0], &out[0])
        cdef int i
        cdef complex e
        cdef cy_qobj c_op
        cdef complex[::1] temp = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] temp2 = np.zeros(self.l_vec, dtype=complex)
        for i in range(self.N_ops):
            c_op = self.cpcd_ops[i]
            e = c_op._expect_mat(t, &vec[0], 0)
            if self.debug[3]:
                temp2 = np.zeros(self.l_vec, dtype=complex)
                c_op = self.cdc_ops[i]
                c_op._rhs_mat(t, &vec[0], &temp2[0])
            temp = np.zeros(self.l_vec, dtype=complex)
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &temp[0])
            for j in range(self.l_vec):
                out[j] += -0.125 * e * e * vec[j] * self.dt +\
                          0.5 * e * temp[j] * self.dt +\
                          temp2[j]

    cdef void d2(self, double t, complex[::1] vec, complex[:, ::1] out):
        cdef int i, k
        cdef cy_qobj c_op
        cdef complex expect
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &out[i,0])
            c_op = self.cpcd_ops[i]
            expect = c_op._expect_mat(t, &vec[0], 0)
            for k in range(self.l_vec):
                out[i,k] -= 0.5 * expect * vec[k]

    cdef void d2d2p(self, double t, complex[::1] vec,
                    complex[:, ::1] d2_out, complex[:, :, ::1] dd2_out):
        cdef int i, j, k
        cdef cy_qobj c_op
        cdef complex expect2
        cdef double[::1] expect = np.zeros((self.N_ops,))#, dtype=complex)
        cdef complex[::1] temp = np.zeros((self.l_vec,), dtype=complex)
        cdef complex[::1] temp2 = np.zeros((self.l_vec,), dtype=complex)
        cdef complex[:, ::1] AV = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &AV[i,0])
            expect[i] = 2*real(dotc(vec, AV[i,:]))
            for k in range(self.l_vec):
                d2_out[i,k] = AV[i,k] - 0.5 * expect[i] * vec[k]

        for i in range(self.N_ops):
            for j in range(i, self.N_ops):
                expect2 = 0.
                c_op = self.c_ops[i]
                c_op._rhs_mat(t, &d2_out[j,0], &dd2_out[i,j,0])
                for k in range(self.l_vec):
                    temp[k] = conj(d2_out[j,k])
                    temp2[k] = 0.
                c_op._rhs_mat(t, &temp[0], &temp2[0])
                expect2 = dotc(vec, dd2_out[i,j,:]) + dot(d2_out[j,:], AV[i,:]) +\
                          conj(dotc(d2_out[j,:], AV[i,:]) + dotc(vec, temp2))
                for k in range(self.l_vec):
                    dd2_out[i, j, k] -= expect2 * vec[k] * 0.5
                    dd2_out[i, j, k] -= expect[i] * d2_out[j,k] * 0.5

    cdef void taylor_15_1(self, double t, complex[::1] vec,
                           complex[:, ::1] out):
        # drho = a (rho, t) * dt + b (rho, t) * dW
        # rho_(n+1) = rho_n + adt + bdW + 0.5bb'((dW)^2 - dt) +  - Milstein terms
        # + ba'dZ + (ab'+bbb"*0.5)(dWdt - dZ) + 0.5[(da/dt) + aa']dt^2 + 0.5bb'bb'(1/3(dW)^2 - dt)dW

        cdef int i, j, k
        cdef double dt = self.dt
        cdef cy_qobj c_op, cdc_op
        cdef complex e, de_b, de_bb, de_a, dde_bb
        cdef complex e_real

        cdef complex[::1] Cvec = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] Cb = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] temp = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] temp2 = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] dbbb = np.zeros(self.l_vec, dtype=complex)  #b"bb

        c_op = self.c_ops[0]
        cdc_op = self.cdc_ops[0]  # <==== In H

        # a
        self.L._rhs_mat(t, &vec[0], &out[0,0])
        if self.debug[3]:
            cdc_op._rhs_mat(t, &vec[0], &out[0,0])  # <==== In H
        c_op._rhs_mat(t, &vec[0], &Cvec[0])
        e = dotc(vec,Cvec)
        e_real = real(e)
        axpy(-0.5 * e_real * e_real * dt, vec, out[0,:])
        axpy(e_real * dt, Cvec, out[0,:])

        # b
        axpy(1., Cvec, out[1,:])
        axpy(-e_real, vec, out[1,:])

        #bb'
        c_op._rhs_mat(t, &out[1,0], &Cb[0])
        for k in range(self.l_vec):
            temp[k] = conj(out[1,k])
        c_op._rhs_mat(t, &temp[0], &temp2[0])
        de_b = dotc(vec, Cb) + dot(out[1,:], Cvec) + \
               conj(dotc(out[1,:], Cvec) + dotc(vec, temp2))
        dde_bb = 2*(dot(out[1,:], Cb) +  conj(dotc(out[1,:], temp2)))
        axpy(1., Cb, out[2,:])
        axpy(-e_real, out[1,:], out[2,:])
        axpy(-de_b*0.5, vec, out[2,:])

        #ba'
        self.L._rhs_mat(t, &out[1,0], &out[3,0])
        if self.debug[3]:
            cdc_op._rhs_mat(t, &out[1,0], &out[3,0])  # <==== In H
        axpy(-0.5 * e_real * e_real * dt, out[1,:], out[3,:])
        axpy(-0.5 * e_real * de_b * dt, vec, out[3,:])
        axpy(e_real * dt, Cb, out[3,:])
        axpy(0.5 * de_b * dt, Cvec, out[3,:])

        #ab' + db/dt + bbb"/2
        c_op._rhs_mat(t, &out[0,0], &out[4,0])
        for k in range(self.l_vec):
            temp[k] = conj(out[0,k])
            temp2[k] = 0.
        c_op._rhs_mat(t, &temp[0], &temp2[0])
        de_a = dotc(vec, out[4,:]) + dot(out[0,:], Cvec) + \
               conj(dotc(out[0,:], Cvec) + dotc(vec, temp2))
        axpy(-e_real, out[0,:], out[4,:])
        axpy(-de_a*0.5, vec, out[4,:])

        temp = np.zeros(self.l_vec, dtype=complex)
        c_op._rhs_mat(t + self.dt, &vec[0], &temp[0])
        e = dotc(vec,temp)
        axpy(1., temp, out[4,:])
        axpy(-real(e), vec, out[4,:])
        axpy(-1., out[1,:], out[4,:])

        axpy(-de_b, out[1,:], dbbb)
        axpy(-dde_bb*0.5, vec, dbbb)
        axpy(0.5 *dt, dbbb, out[4,:])

        #da/dt + aa' + bba"
        #self.L._rhs_mat(t + dt, &vec[0], &out[5,0])
        self.d1(t + dt, vec, out[5,:])
        axpy(-1.0, out[0,:], out[5,:])
        self.L._rhs_mat(t, &out[0,0], &out[5,0])
        if self.debug[3]:
            cdc_op._rhs_mat(t, &out[0,0], &out[5,0])  # <==== In H
        temp = np.zeros(self.l_vec, dtype=complex)
        c_op._rhs_mat(t, &out[0,0], &temp[0])
        axpy(-0.5 * e_real * e_real * dt, out[0,:], out[5,:])
        axpy(-0.5 * e_real * de_a * dt, vec, out[5,:])
        axpy(e_real * dt, temp, out[5,:])
        axpy(0.5 * de_a * dt, Cvec, out[5,:])

        axpy(-0.125 * (2*e_real * dde_bb + de_b * de_b) * dt* dt, vec, out[5,:])
        axpy(-0.5 * e_real * de_b * dt* dt, out[1,:], out[5,:])
        axpy(0.25 * dde_bb * dt* dt, Cvec, out[5,:])
        axpy(0.5*de_b * dt* dt, Cb, out[5,:])

        #b(bb"+b'b')
        c_op._rhs_mat(t, &out[2,0], &out[6,0])
        for k in range(self.l_vec):
            temp[k] = conj(out[2,k])
            temp2[k] = 0.
        c_op._rhs_mat(t, &temp[0], &temp2[0])
        de_bb = dotc(vec, out[6,:]) + dot(out[2,:], Cvec) + \
                conj(dotc(out[2,:], Cvec) + dotc(vec, temp2))
        axpy(-e_real, out[2,:], out[6,:])
        axpy(-de_bb*0.5, vec, out[6,:])

        axpy(1.0, dbbb, out[6,:])

    cdef void derivatives(self, double t, int deg, complex[::1] vec,
                          complex[::1] a, complex[:, ::1] b,
                          complex[:, :, ::1] Lb, complex[:,::1] La,
                          complex[:, ::1] L0b, complex[:, :, :, ::1] LLb,
                          complex[::1] L0a):
        """
        combinaisons of a and b derivative for m sc_ops up to order dt**1.5
        dY ~ a dt + bi dwi
                                         deg   use        noise
        b[i.:]       bi                   >=0   d2 euler   dwi
        a[:]         a                    >=0   d1 euler   dt
        Lb[i,i,:]    bi'bi                >=1   milstein   (dwi^2-dt)/2
        Lb[i,j,:]    bj'bi                >=1   milstein   dwi*dwj

        L0b[i,:]     ab' +db/dt +bbb"/2   >=2   taylor15   dwidt-dzi
        La[i,:]      ba'                  >=2   taylor15   dzi
        LLb[i,i,i,:] bi(bibi"+bi'bi')     >=2   taylor15   (dwi^2/3-dt)dwi/2
        LLb[i,j,j,:] bi(bjbj"+bj'bj')     >=2   taylor15   (dwj^2-dt)dwj/2
        LLb[i,j,k,:] bi(bjbk"+bj'bk')     >=2   taylor15   dwi*dwj*dwk
        L0a[:]       aa' +da/dt +bba"/2    2    taylor15   dt^2/2
        """
        cdef int i, j, k, l
        cdef double dt = self.dt
        cdef cy_qobj c_op, cdc_op
        cdef complex e, de_bb
        cdef complex[::1] e_real = np.zeros((self.N_ops), dtype=complex)
        cdef complex[:, ::1] de_b = np.zeros((self.N_ops, self.N_ops), dtype=complex)
        cdef complex[::1] de_a = np.zeros((self.N_ops), dtype=complex)
        cdef complex[:, :, ::1] dde_bb = np.zeros((self.N_ops, self.N_ops, self.N_ops), dtype=complex)

        cdef complex[:, ::1] Cvec = np.zeros((self.N_ops, self.l_vec), dtype=complex)
        cdef complex[:, :, ::1] Cb = np.zeros((self.N_ops, self.N_ops, self.l_vec), dtype=complex)
        cdef complex[::1] temp = np.zeros(self.l_vec, dtype=complex)
        cdef complex[::1] temp2 = np.zeros(self.l_vec, dtype=complex)

        # a b
        self.L._rhs_mat(t, &vec[0], &a[0])
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &Cvec[i,0])
            e = dotc(vec,Cvec[i,:])
            e_real[i] = real(e)
            axpy(1., Cvec[i,:], b[i,:])
            axpy(-e_real[i], vec, b[i,:])
            if self.debug[3]:
                cdc_op = self.cdc_ops[i]
                cdc_op._rhs_mat(t, &vec[0], &a[0])  # <==== In H
            axpy(-0.5 * e_real[i] * e_real[i] * dt, vec, a[:])
            axpy(e_real[i] * dt, Cvec[i,:], a[:])

        #Lb bb'
        for i in range(self.N_ops):
          c_op = self.c_ops[i]
          for j in range(self.N_ops):
            c_op._rhs_mat(t, &b[j,0], &Cb[i,j,0])
            for k in range(self.l_vec):
                temp[k] = conj(b[j,k])
                temp2[k] = 0.
            c_op._rhs_mat(t, &temp[0], &temp2[0])
            de_b[i,j] = (dotc(vec, Cb[i,j,:]) + dot(b[j,:], Cvec[i,:]) + \
                        conj(dotc(b[j,:], Cvec[i,:]) + dotc(vec, temp2))) * 0.5
            axpy(1., Cb[i,j,:], Lb[i,j,:])
            axpy(-e_real[i], b[j,:], Lb[i,j,:])
            axpy(-de_b[i,j], vec, Lb[i,j,:])

            for k in range(self.N_ops):
              dde_bb[i,j,k] += (dot(b[j,:], Cb[i,k,:]) + \
                                dot(b[k,:], Cb[i,j,:]) + \
                                conj(dotc(b[k,:], temp2)))*.5
              dde_bb[i,k,j] += conj(dotc(b[k,:], temp2))*.5


        #L0b La LLb
        for i in range(self.N_ops):
            #ba'
            self.L._rhs_mat(t, &b[i,0], &La[i,0])
            for j in range(self.N_ops):
                cdc_op = self.cdc_ops[j]
                cdc_op._rhs_mat(t, &b[i,0], &La[i,0])  # <==== In H
                axpy(-0.5 * e_real[j] * e_real[j] * dt, b[i,:], La[i,:])
                axpy(-e_real[j] * de_b[i,j] * dt, vec, La[i,:])
                axpy(e_real[j] * dt, Cb[i,j,:], La[i,:])
                axpy(de_b[i,j] * dt, Cvec[i,:], La[i,:])

            #ab' + db/dt + bbb"/2
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &a[0], &L0b[i,0])
            for k in range(self.l_vec):
                temp[k] = conj(a[k])
                temp2[k] = 0.
            c_op._rhs_mat(t, &temp[0], &temp2[0])
            de_a[i] = (dotc(vec, L0b[i,:]) + dot(a, Cvec[i,:]) + \
                      conj(dotc(a, Cvec[i,:]) + dotc(vec, temp2))) * 0.5
            axpy(-e_real[i], a, L0b[i,:])
            axpy(-de_a[i], vec, L0b[i,:])

            temp = np.zeros(self.l_vec, dtype=complex)
            c_op._rhs_mat(t + self.dt, &vec[0], &temp[0])
            e = dotc(vec,temp)
            axpy(1., temp, L0b[i,:])
            axpy(-real(e), vec, L0b[i,:])
            axpy(-1., b[i,:], L0b[i,:])

            for j in range(self.N_ops):
                axpy(-de_b[i,j]*dt, b[j,:], L0b[i,:])
                axpy(-dde_bb[i,j,j]*dt, vec, L0b[i,:])

            #b(bb"+b'b')
            for j in range(i,self.N_ops):
                for k in range(j, self.N_ops):
                    c_op._rhs_mat(t, &Lb[j,k,0], &LLb[i,j,k,0])
                    for l in range(self.l_vec):
                        temp[l] = conj(Lb[j,k,l])
                        temp2[l] = 0.
                    c_op._rhs_mat(t, &temp[0], &temp2[0])
                    de_bb = (dotc(vec, LLb[i,j,k,:]) + \
                             dot(Lb[j,k,:], Cvec[i,:]) + \
                             conj(dotc(Lb[j,k,:], Cvec[i,:]) +\
                             dotc(vec, temp2)))*0.5
                    axpy(-e_real[i], Lb[j,k,:], LLb[i,j,k,:])
                    axpy(-de_bb, vec, LLb[i,j,k,:])

                    axpy(-dde_bb[i,j,k], vec, LLb[i,j,k,:])
                    axpy(-de_b[i,j], b[k,:], LLb[i,j,k,:])
                    axpy(-de_b[i,k], b[j,:], LLb[i,j,k,:])

        #da/dt + aa' + bba"
        self.d1(t + dt, vec, L0a)
        axpy(-1.0, a, L0a)
        self.L._rhs_mat(t, &a[0], &L0a[0])
        for j in range(self.N_ops):
            c_op = self.c_ops[j]
            if self.debug[3]:
              cdc_op = self.cdc_ops[j]
              cdc_op._rhs_mat(t, &a[0], &L0a[0])  # <==== In H
            temp = np.zeros(self.l_vec, dtype=complex)
            c_op._rhs_mat(t, &a[0], &temp[0])
            axpy(-0.5 * e_real[j] * e_real[j] * dt, a[:], L0a[:])
            axpy(-e_real[j] * de_a[j] * dt, vec, L0a[:])
            axpy(e_real[j] * dt, temp, L0a[:])
            axpy(de_a[j] * dt, Cvec[j,:], L0a[:])
            for i in range(self.N_ops):
                axpy(-0.5*(e_real[i] * dde_bb[i,j,j] +
                        de_b[i,j] * de_b[i,j]) * dt * dt, vec, L0a[:])
                axpy(-e_real[i] * de_b[i,j] * dt * dt, b[j,:], L0a[:])
                axpy(0.5*dde_bb[i,j,j] * dt * dt, Cvec[i,:], L0a[:])
                axpy(de_b[i,j] * dt * dt, Cb[i,j,:], L0a[:])

    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                                        complex[::1] out,
                                        np.ndarray[complex, ndim=1] guess):
        # np.ndarray to memoryview is OK but not the reverse
        # scipy function only take np array, not memoryview
        self.imp_t = t
        spout, check = sp.linalg.bicgstab(self.imp,
                                        dvec, x0 = guess, tol=self.tol)
        cdef int i
        for i in range(self.l_vec):
            out[i]=spout[i]

cdef class sme(ssolvers):
    cdef cy_qobj L
    cdef object imp
    cdef object c_ops
    cdef int N_root
    cdef double tol

    def set_data(self, L, c_ops):
        self.l_vec = L.cte.shape[0]
        self.N_ops = len(c_ops)
        self.L = L.compiled_Qobj
        self.c_ops = []
        self.N_root = np.sqrt(self.l_vec)
        for i, op in enumerate(c_ops):
            self.c_ops.append(op.compiled_Qobj)

    def set_implicit(self, sso):
        self.tol = sso.tol
        self.imp = 1 - sso.LH * 0.5
        self.imp.compile()

    cdef complex expect(self, complex[::1] rho):
        cdef complex e = 0.
        cdef int k
        for k in range(self.N_root):
            e += rho[k*(self.N_root+1)]
        return e

    cdef void d1(self, double t, complex[::1] rho, complex[::1] out):
        self.L._rhs_mat(t, &rho[0], &out[0])

    cdef void d2(self, double t, complex[::1] rho, complex[:, ::1] out):
        cdef int i, k
        cdef cy_qobj c_op
        cdef complex expect
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &rho[0], &out[i,0])
            expect = 0.
            for k in range(self.N_root):
                expect += out[i, k*(self.N_root+1)]
            for k in range(self.l_vec):
                out[i,k] -= expect * rho[k]

    cdef void d2d2p(self, double t, complex[::1] rho,
                    complex[:, ::1] d2_out, complex[:, :, ::1] dd2_out):
        cdef int i, j, k
        cdef cy_qobj c_op
        cdef complex[::1] expect = np.zeros((self.N_ops,), dtype=complex)
        cdef complex expect2
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &rho[0], &d2_out[i,0])
            for k in range(self.N_root):
                expect[i] += d2_out[i, k*(self.N_root+1)]
            for k in range(self.l_vec):
                d2_out[i,k] -= expect[i] * rho[k]

        for i in range(self.N_ops):
            for j in range(i, self.N_ops):
                expect2 = 0.
                c_op = self.c_ops[i]
                c_op._rhs_mat(t, &d2_out[j,0], &dd2_out[i,j,0])
                for k in range(self.N_root):
                    expect2 += dd2_out[i, j, k*(self.N_root+1)]
                for k in range(self.l_vec):
                    dd2_out[i, j, k] -= expect2 * rho[k]
                    dd2_out[i, j, k] -= expect[i] * d2_out[j,k]

    cdef void taylor_15_1(self, double t, complex[::1] rho,
                           complex[:, ::1] out):
        # drho = a (rho, t) * dt + b (rho, t) * dW
        # rho_(n+1) = rho_n + adt + bdW + 0.5bb'((dW)^2 - dt) +  - Milstein terms
        # + ba'dZ + (ab'+bbb"*0.5)(dWdt - dZ) + 0.5[(da/dt) + aa']dt^2 + 0.5bb'bb'(1/3(dW)^2 - dt)dW

        cdef int i, j, k
        cdef cy_qobj c_op
        cdef complex trAp, trApp, trAa, trAb, trAbb
        cdef complex[::1] temp = np.zeros((self.l_vec, ), dtype=complex)
        c_op = self.c_ops[0]
        # a
        self.L._rhs_mat(t, &rho[0], &out[0,0])
        # b
        c_op._rhs_mat(t, &rho[0], &out[1,0])
        trAp = self.expect(out[1,:])
        axpy(-trAp, rho, out[1,:])
        #bb'
        c_op._rhs_mat(t, &out[1,0], &out[2,0])
        trAb = self.expect(out[2,:])
        axpy(-trAp, out[1,:], out[2,:])
        axpy(-trAb, rho, out[2,:])
        #ba' + db/dt
        self.L._rhs_mat(t, &out[1,0], &out[3,0])
        #ab' + bbb"/2 + db/dt
        c_op._rhs_mat(t + self.dt, &rho[0], &out[4,0])
        trApp = self.expect(out[4,:])
        axpy(-trApp, rho, out[4,:])
        axpy(-1, out[1,:], out[4,:])
        c_op._rhs_mat(t, &out[0,0], &temp[0])
        trAa = self.expect(temp)
        axpy(1., temp, out[4,:])
        axpy(-trAp, out[0,:], out[4,:])
        axpy(-trAa, rho, out[4,:])
        axpy(-trAb*self.dt, out[1,:], out[4,:])  # L contain dt
        #da/dt + aa'+ba"/2
        self.L._rhs_mat(t+self.dt, &rho[0], &out[5,0])
        axpy(-1, out[0,:], out[5,:])
        self.L._rhs_mat(t, &out[0,0], &out[5,0])
        #b(bb"+b'b')
        c_op._rhs_mat(t, &out[2,0], &out[6,0])
        trAbb = self.expect(out[6,:])
        axpy(-trAp, out[2,:], out[6,:])
        axpy(-trAbb, rho, out[6,:])
        axpy(-trAb*2, out[1,:], out[6,:])

    cdef void derivatives(self, double t, int deg, complex[::1] rho,
                          complex[::1] a, complex[:, ::1] b,
                          complex[:, :, ::1] Lb, complex[:,::1] La,
                          complex[:, ::1] L0b, complex[:, :, :, ::1] LLb,
                          complex[::1] L0a):
        """
        combinaisons of a and b derivative for m sc_ops up to order dt**1.5
        dY ~ a dt + bi dwi
                                         deg   use        noise
        b[i.:]       bi                   >=0   d2 euler   dwi
        a[:]         a                    >=0   d1 euler   dt
        Lb[i,i,:]    bi'bi                >=1   milstein   (dwi^2-dt)/2
        Lb[i,j,:]    bj'bi                >=1   milstein   dwi*dwj

        L0b[i,:]     ab' +db/dt +bbb"/2   >=2   taylor15   dwidt-dzi
        La[i,:]      ba'                  >=2   taylor15   dzi
        LLb[i,i,i,:] bi(bibi"+bi'bi')     >=2   taylor15   (dwi^2/3-dt)dwi/2
        LLb[i,j,j,:] bi(bjbj"+bj'bj')     >=2   taylor15   (dwj^2-dt)dwj/2
        LLb[i,j,k,:] bi(bjbk"+bj'bk')     >=2   taylor15   dwi*dwj*dwk
        L0a[:]       aa' +da/dt +bba"/2    2    taylor15   dt^2/2
        """
        cdef int i, j, k
        cdef cy_qobj c_op
        cdef cy_qobj c_opj
        cdef complex trApp, trAbb, trAa
        cdef complex[::1] trAp = np.zeros((self.N_ops, ), dtype=complex)
        cdef complex[:, ::1] trAb = np.zeros((self.N_ops, self.N_ops), dtype=complex)
        cdef complex[::1] temp = np.zeros((self.l_vec, ), dtype=complex)

        # a
        self.L._rhs_mat(t, &rho[0], &a[0])

        # b
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            # bi
            c_op._rhs_mat(t, &rho[0], &b[i,0])
            trAp[i] = self.expect(b[i,:])
            axpy(-trAp[i], rho, b[i,:])

        # Libj = bibj', i<=j
        # sc_ops must commute (Libj = Ljbi)
        if deg >= 1:
          for i in range(self.N_ops):
            c_op = self.c_ops[i]
            for j in range(i, self.N_ops):
                c_op._rhs_mat(t, &b[j,0], &Lb[i,j,0])
                trAb[i,j] = self.expect(Lb[i,j,:])
                axpy(-trAp[j], b[i,:], Lb[i,j,:])
                axpy(-trAb[i,j], rho, Lb[i,j,:])

        # L0b La LLb
        if deg >= 2:
          for i in range(self.N_ops):
            c_op = self.c_ops[i]
            # Lia = bia'
            self.L._rhs_mat(t, &b[i,0], &La[i,0])

            # L0bi = abi' + dbi/dt + Sum_j bjbjbi"/2
            # db/dt
            c_op._rhs_mat(t + self.dt, &rho[0], &L0b[i,0])
            trApp = self.expect(L0b[i,:])
            axpy(-trApp, rho, L0b[i,:])
            axpy(-1, b[i,:], L0b[i,:])
            # ab'
            temp = np.zeros((self.l_vec, ), dtype=complex)
            c_op._rhs_mat(t, &a[0], &temp[0])
            trAa = self.expect(temp)
            axpy(1., temp, L0b[i,:])
            axpy(-trAp[i], a[:], L0b[i,:])
            axpy(-trAa, rho, L0b[i,:])
            # bbb" : trAb[i,j] only defined for j>=i
            for j in range(i):
                axpy(-trAb[j,i]*self.dt, b[j,:], L0b[i,:])  # L contain dt
            for j in range(i,self.N_ops):
                axpy(-trAb[i,j]*self.dt, b[j,:], L0b[i,:])  # L contain dt

            # LLb
            # LiLjbk = bi(bj'bk'+bjbk"), i<=j<=k
            # sc_ops must commute (LiLjbk = LjLibk = LkLjbi)
            for j in range(i,self.N_ops):
              for k in range(j,self.N_ops):
                c_op._rhs_mat(t, &Lb[j,k,0], &LLb[i,j,k,0])
                trAbb = self.expect(LLb[i,j,k,:])
                axpy(-trAp[i], Lb[j,k,:], LLb[i,j,k,:])
                axpy(-trAbb, rho, LLb[i,j,k,:])
                axpy(-trAb[i,k], b[j,:], LLb[i,j,k,:])
                axpy(-trAb[i,j], b[k,:], LLb[i,j,k,:])

        # L0a = a'a + da/dt + bba"/2  (a" = 0)
        if deg == 2:
            self.L._rhs_mat(t, &a[0], &L0a[0])
            self.L._rhs_mat(t+self.dt, &rho[0], &L0a[0])
            axpy(-1, a, L0a)


    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                                        complex[::1] out,
                                        np.ndarray[complex, ndim=1] guess):
        # np.ndarray to memoryview is OK but not the reverse
        # scipy function only take np array, not memoryview
        spout, check = sp.linalg.bicgstab(self.imp(t, data=1),
                                        dvec, x0 = guess, tol=self.tol)
        cdef int i
        for i in range(self.l_vec):
            out[i]=spout[i]
