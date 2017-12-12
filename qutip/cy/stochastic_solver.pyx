

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

import scipy.sparse as sp
from scipy.linalg.cython_blas cimport zaxpy

"""
Solver:
  order 0.5
    euler-maruyama     50
  order 1.0
    platen            100
    pred_corr         101
    milstein          102
    milstein-imp      103
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
    cdef double dt
    cdef int noise_type
    cdef double[:,:,:,::1] custom_noise
    cdef double[::1] dW_factor
    cdef unsigned int[::1] seed
    cdef object generate_noise

    def __init__(self):
        self.l_vec = 0
        self.N_ops = 0
        self.solver = 0

    def set_solver(self, sso):
        self.solver = sso.solver_code
        self.dt = sso.dt
        self.N_substeps = sso.nsubsteps
        if self.solver % 10 == 3:
            self.set_implicit(sso)
        self.N_step = len(sso.times)
        self.N_dw = len(sso.sops)
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
                states_list.append(Qobj(vec2mat(rho_t), dims=dims))
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

        if self.solver == 100:
            for i in range(N_substeps):
                self.platen(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        if self.solver == 101:
            for i in range(N_substeps):
                self.pred_corr(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        if self.solver == 102:
            for i in range(N_substeps):
                self.milstein(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        if self.solver == 103:
            for i in range(N_substeps):
                self.milstein_imp(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

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
        Platen rhs function for both master eq and schrodinger eq.
        dV = -iH* (V+Vt)/2 * dt + (d1(V)+d1(Vt))/2 * dt
             + (2*d2_i(V)+d2_i(V+)+d2_i(V-))/4 * dW_i
             + (d2_i(V+)-d2_i(V-))/4 * (dW_i**2 -dt) * dt**(-.5)

        Vt = V -iH*V*dt + d1*dt + d2_i*dW_i
        V+/- = V -iH*V*dt + d1*dt +/- d2_i*dt**.5

        Not validated for time-dependent operators
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
        self.d1(t, Vt, d1)
        for j in range(self.l_vec):
          out[j] += 0.5* (d1[j] + vec[j])

        self.d2(t, Vp, d2p)
        self.d2(t, Vm, d2m)
        for i in range(self.N_ops):
          dw = noise[i] * 0.25
          dw2 =  sqrt_dt_inv * (noise[i]*noise[i] - dt)
          for j in range(self.l_vec):
            out[j] += dw * (d2[i,j] *2 + d2p[i,j] + d2m[i,j])
            out[j] += dw2 * (d2p[i,j] - d2m[i,j])

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void pred_corr(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
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

        self.d1(t, vec, a_pred)
        self.d2d2p(t, vec, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                b_pred[j] = d2[i,j] * noise[i]

        for j in range(self.l_vec):
            euler[j] = a_pred[j] + b_pred[j] + vec[j]

        for i in range(self.N_ops):
            for j in range(self.N_ops):
                for k in range(self.l_vec):
                    a_pred[k] -= 0.5 * dd2[i,j,k] * dt*0

        d2 = np.zeros((self.N_ops, self.l_vec), dtype=complex)
        dd2 = np.zeros((self.N_ops, self.N_ops, self.l_vec), dtype=complex)
        self.d1(t, euler, a_corr)
        self.d2d2p(t, euler, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.N_ops):
                for k in range(self.l_vec):
                    a_corr[k] -= 0.5 * dd2[i,j,k] * dt*0

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                b_corr[j] = d2[i,j] * noise[i]

        for j in range(self.l_vec):
            out[j] = vec[j] + (a_pred[j] + a_corr[j] + b_pred[i] + b_corr[i]*0) * 0.5

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
                    dw = (noise[i] * noise[j]) #* 0.5
                for k in range(self.l_vec):
                    out[k] += dd2[i,j,k] * dw *0

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void milstein_imp(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        """
        Milstein rhs function for both master eq and schrodinger eq.

        Slow but should be valid for non-commuting operators since computing
            both i x j and j x i. Need to be commuting anyway

        dV = -iH*V*dt + d1*dt + d2_i*dW_i
             + 0.5*d2_i(d2_j(V))*(dW_i*dw_j -dt*delta_ij)
        """
        cdef int i, j, k
        cdef double dw
        cdef np.ndarray[complex, ndim=1] guess = np.zeros((self.l_vec, ),
                                                          dtype=complex)
        cdef np.ndarray[complex, ndim=1] a = np.zeros((self.l_vec, ),
                                                      dtype=complex)
        cdef np.ndarray[complex, ndim=1] dvec = np.zeros((self.l_vec, ),
                                                         dtype=complex)
        cdef complex[:, ::1] d2 = np.empty((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] dd2 = np.empty((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)

        self.d1(t, vec, a)
        self.d2d2p(t, vec, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                dvec[j] += d2[i,j] * noise[i]

        for j in range(self.l_vec):
            dvec[j] += a[j] + vec[j]
            guess[j] = dvec[j] + a[j]

        for i in range(self.N_ops):
            for j in range(self.N_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j]) * 0.5
                for k in range(self.l_vec):
                    dvec[k] += dd2[i,j,k] * dw

        self.implicit(t+dt, dvec, out, guess)


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
        self.imp = 1 - sso.LH * (sso.dt * 0.5)
        self.imp.compile()

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
                    dd2_out[i, j, k] -= expect[i] * d2_out[j,0]

    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                                        complex[::1] out,
                                        np.ndarray[complex, ndim=1] guess):
        # np.ndarray to memoryview is OK but not the reverse
        # scipy function only take np array, not memoryview
        out, check = sp.linalg.bicgstab(self.imp(t, data=1),
                                        dvec, x0 = guess, tol=self.tol)
