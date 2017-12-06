

#from qutip.cy.spmatfuncs cimport (spmv_csr)
#import scipy.sparse as sp
#from qutip.cy.spmatfuncs import cy_expect_rho_vec_csr, cy_expect_rho_vec
#from qutip.superoperator import mat2vec, vec2mat
#from qutip.cy.spmatfuncs import cy_expect_psi_csr, cy_expect_rho_vec
#from qutip.qobj import Qobj


import numpy as np
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.td_qobj_cy cimport cy_qobj
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
    #cdef object generate_noise
    #cdef double[:,:,:] noise

    def set_solver(self, sso):
        self.solver = sso.solver_code
        if self.solver % 10 == 3:
            # implicit solver
            self.set_implicit(sso)
        """self.noise_type = sso.noise_type
        if noise_type == 3:
            self.generate_noise = sso.generate_noise
        if noise_type == 2:
            self.noise = sso.noise
        if noise_type == 1:
            np.random.seed(sso.noise)
            self.noise = np.random.random(len(sso.times,sso.nsubsteps,
                                          sso.dw_len))
        if noise_type == 0:
            np.random.seed()
            self.noise = np.random.random(len(sso.times,sso.nsubsteps,
                                          sso.dw_len))"""




    def run(self, double t, double dt, double[::1] noise,
            complex[::1] vec, int N_substeps):
        cdef complex[::1] out = np.zeros(self.l_vec)
        cdef int i
        if self.solver == 50:
            for i in range(N_substeps):
                self.euler(t, dt, noise, vec, out)
                out, vec = vec, out

        if self.solver == 100:
            for i in range(N_substeps):
                self.platen(t, dt, noise, vec, out)
                out, vec = vec, out

        if self.solver == 101:
            for i in range(N_substeps):
                self.pred_corr(t, dt, noise, vec, out)
                out, vec = vec, out

        if self.solver == 102:
            for i in range(N_substeps):
                self.milstein(t, dt, noise, vec, out)
                out, vec = vec, out

        if self.solver == 103:
            for i in range(N_substeps):
                self.milstein_imp(t, dt, noise, vec, out)
                out, vec = vec, out

        return vec


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void euler(self, double t, double dt, double[:] noise,
                    complex[:] vec, complex[:] out):
        cdef int i, j
        cdef complex[:, ::1] d2 = np.empty((self.N_ops, self.l_vec),
                                           dtype=complex)
        for j in range(self.l_vec):
            out[j] = 0.
        self.d1(t, vec, out)
        self.d2(t, vec, d2)
        for i in range(self.N_ops):
            for j in range(self.l_vec):
                out[j] += d2[i,j] * noise[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void platen(self, double t, double dt, double[:] noise,
                    complex[:] vec, complex[:] out):
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

        self.d1(t, Vt, d1)
        for j in range(self.l_vec):
          out[j] += 0.5* d1[j]

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
                    complex[:] vec, complex[:] out):

        cdef int i, j, k
        cdef complex[::1] euler = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] a_pred = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] b_pred = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] a_corr = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] b_corr = np.zeros((self.l_vec), dtype=complex)
        cdef complex[:, ::1] d2 = np.empty((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] dd2 = np.empty((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)

        self.d1(t, vec, a_pred)
        self.d2d2p(t, vec, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                b_pred[j] = d2[i,j] * noise[i]

        for j in range(self.l_vec):
            euler[j] = a_pred[j] + b_pred[j]

        for i in range(self.N_ops):
            for j in range(self.N_ops):
                for k in range(self.l_vec):
                    a_pred[k] -= 0.5 * dd2[i,j,k]

        self.d1(t, euler, a_corr)
        self.d2d2p(t, euler, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.N_ops):
                for k in range(self.l_vec):
                    a_corr[k] -= 0.5 * dd2[i,j,k]

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                b_corr[j] = d2[i,j] * noise[i]

        for j in range(self.l_vec):
            out[j] = (a_pred[j] + a_corr[j] + b_pred[i,j] + b_corr[i,j]) * 0.5

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void milstein(self, double t, double dt, double[:] noise,
                    complex[:] vec, complex[:] out):
        """
        Milstein rhs function for both master eq and schrodinger eq.

        Slow but should be valid for non-commuting operators since computing
            both i x j and j x i. Need to be commuting anyway

        dV = -iH*V*dt + d1*dt + d2_i*dW_i
             + 0.5*d2_i(d2_j(V))*(dW_i*dw_j -dt*delta_ij)
        """
        cdef int i, j, k
        cdef double dw
        cdef complex[:, ::1] d2 = np.empty((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] dd2 = np.empty((self.N_ops, self.N_ops,
                                                self.l_vec), dtype=complex)
        for j in range(self.l_vec):
            out[j] = 0.
        self.d1(t, vec, out)
        self.d2d2p(t, vec, d2, dd2)

        for i in range(self.N_ops):
            for j in range(self.l_vec):
                out[j] += d2[i,j] * noise[i]

        for i in range(self.N_ops):
            for j in range(self.N_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j]) * 0.5
                for k in range(self.l_vec):
                    out[k] += dd2[i,j,k] * dw

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void milstein_imp(self, double t, double dt, double[:] noise,
                           complex[:] vec, complex[:] out):
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
            dvec[j] += a[j]
            guess[j] = dvec[j] + a[j]

        for i in range(self.N_ops):
            for j in range(self.N_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j]) * 0.5
                for k in range(self.l_vec):
                    dvec[k] += dd2[i,j,k] * dw

        self.implicit(dvec, out, guess)


"""cdef class sse(ssolvers):
    pass"""

cdef class sme(ssolvers):
      cdef cy_qobj L
      cdef object imp
      cdef cy_qobj** c_ops
      #cdef cy_qobj c_op
      cdef int N_root
      cdef double tol

      def __init__(self):
          self.N_ops = 0
          self.c_ops = <cy_qobj**> PyDataMem_NEW(0 * sizeof(cy_qobj*))

      def __del__(self):
          for i in range(self.N_ops):
              PyDataMem_FREE(self.c_ops[i])
          PyDataMem_FREE(self.c_ops)

      def set_data(self, L, c_ops):
          self.l_vec = L.cte.shape[0]
          self.N_ops = len(c_ops)
          self.L = L.compiled_Qobj
          PyDataMem_FREE(self.c_ops)
          self.c_ops = <cy_qobj**> PyDataMem_NEW(self.N_ops * sizeof(cy_qobj*))
          for i, op in enumerate(c_ops):
              self.c_ops[i] = <cy_qobj*> op.compiled_Qobj

      def set_implicit(self, sso):
          self.tol = sso.tol
          self.imp = 1 - sso.LH * (dt *0.5)
          self.imp.compile()

      cdef void d1(self, double t, complex[::1] rho, complex[::1] out):
          self.L.rhs_mat(t, rho, out)

      cdef void d2(self, double t, complex[::1] rho, complex[:, ::1] out):
          cdef int i, k
          cdef complex[::1] expect = np.zeros((self.N_ops,), dtype=complex)
          cdef complex expect
          for i in range(self.N_ops):
              self.c_ops[i].rhs_mat(t, &rho[0], &out[i,0])
              expect = 0.
              for k in range(self.N_root):
                  expect += out[i, k*(self.N_root+1)]
              for k in range(self.l_vec):
                  out[i,k] -= expect * rho[k]

      cdef void d2d2p(self, double t, complex[::1] rho,
                      complex[:, ::1] d2_out, complex[:, :, ::1] dd2_out):
          cdef int i, j, k
          cdef complex[::1] expect = np.zeros((self.N_ops,), dtype=complex)
          cdef complex expect2
          for i in range(self.N_ops):
              self.c_ops[i].rhs_mat(t, &rho[0], &d2_out[i,0])
              for k in range(self.N_root):
                  expect[i] += d2_out[i, k*(self.N_root+1)]
              for k in range(self.l_vec):
                  d2_out[i,k] -= expect * rho[k]

          for i in range(self.N_ops):
              for j in range(i, self.N_ops):
                  expect2 = 0.
                  self.c_ops[i].rhs_mat(t, &d2_out[j,0], &dd2_out[i,j,0])
                  for k in range(self.N_root):
                      expect2 += dd2_out[i, j, k*(self.N_root+1)]
                  for k in range(self.l_vec):
                      dd2_out[i, j, k] -= expect2 * rho[k]
                      dd2_out[i, j, k] -= expect[i] * d2_out[j,0]

      cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                                          complex[::1] out
                                          np.ndarray[complex, ndim=1] guess):
          # np.ndarray to memoryview is OK but not the reverse
          # scipy function only take np array, not memoryview
          out, check = sp.linalg.bicgstab(self.imp(t, data=1),
                                          dvec, x0 = guess, tol=self.tol)
      """cdef void d2_united(self, double t, complex[::1] rho, complex[:, ::1] out):
          self.c_op.rhs_mat(t, &rho, &out[0])
          for i in range(self.N_ops):
              j = i*l_vec
              expect = 0.
              for k in range(N_root):
                  expect += out[j+k*(N_root+1)]
              for k in range(l_vec):
                  out[j+k] += expect * rho[k]"""
