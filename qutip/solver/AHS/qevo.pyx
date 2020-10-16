#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdvision=True

from qutip.solver._solverqevo cimport SolverQEvo
from qutip.solver.AHS.matmul_ahs2 cimport *
from qutip.core.data cimport Data, Dense
from libc.math cimport round
from qutip.core.cy.cqobjevo cimport LTYPE, CSR_TYPE, Dense_TYPE, CSC_TYPE
cdef extern from "<complex>" namespace "std" nogil:
    # abs is templated such that Cython treats std::abs as complex->complex
    double real(double complex x)
    double imag(double complex x)
import numpy as np
import time
from qutip.core.data.base import idxint_dtype

cdef class AHS_config:
    def __repr__(self):
        out = ""
        out += "tols:" + str(self.atol) +" "+ str(self.rtol) +" "+ str(self.safety_rtol) + "\n"
        out += ("pads:") + str(self.padding) +" "+ str(self.safety_pad) + "\n"
        out += ("state:") + str(self.limits[0]) +" "+ str(self.limits[1]) +" "+ str(self.extra_padding) + "\n"
        out += ("passing:") + str(self.passed) + "\n"
        return out

cdef class SolverQEvoAHS(SolverQEvo):
    def __init__(self, base, options, dict args, dict feedback):
        limits = np.zeros(2, dtype=idxint_dtype)
        cdef idxint N = base.shape[1]
        self.base = base.compiled_qobjevo
        self.set_feedback(feedback, args, base.cte.issuper,
                          options['feedback_normalize'])
        self.collapse = []

        self.super = self.base.issuper
        self.layer_type = self.base.layer_type

        self.config = AHS_config()
        self.config.padding = 5            # options.func["padding"]
        self.config.safety_pad = 3         # options.func["padding"]
        self.config.extra_padding = 1.
        self.config.rtol = options["rtol"] # options.func["rtol"]
        self.config.atol = options["atol"] # options.func["atol"]
        self.config.safety_rtol = options["rtol"]/100
        limits[0] = 0
        limits[1] = N if not self.super else (<idxint> N**0.5)
        self.config.limits = limits
        self.config.np_array = limits
        print(self.config, N, <idxint> N**0.5, self.super)
        time.sleep(0.2)

    cpdef bint resize(self, Dense state):
        if self.super:
            print("dm")
            time.sleep(0.2)
            self.get_idx_dm(state)
        else:
            print("ket")
            time.sleep(0.2)
            self.get_idx_ket(state)
        return self.config.passed

    cpdef idxint[::1] get_idx_ket(self, Dense state):
        # this and get_idx_dm could be merged if a vector of probabilities
        # was available, but making it would require more work and probably
        # end up slower
        cdef double tol, max_prob, safe_tol
        cdef idxint ii, N=state.shape[0], found=0, eff_padding
        if self.config.rtol != 0:
            for ii in range(N):
                if max_prob < real(state.data[ii]):
                    max_prob = real(state.data[ii])
                if max_prob < imag(state.data[ii]):
                    max_prob = imag(state.data[ii])
        tol = max_prob * self.config.rtol + self.config.atol
        safe_tol = max_prob * self.config.safety_rtol + self.config.atol
        print(tol, safe_tol)
        print(self.config)
        time.sleep(0.5)

        self.config.passed = True
        for ii in range(self.config.safety_pad):
            if (
                (
                 (real(state.data[self.config.limits[0] + ii]) > safe_tol or
                  imag(state.data[self.config.limits[0] + ii]) > safe_tol
                 ) and not self.config.limits[0] == 0
                ) or (
                 (real(state.data[self.config.limits[1] - ii - 1]) > safe_tol or
                  imag(state.data[self.config.limits[1] - ii - 1]) > safe_tol
                 ) and not self.config.limits[1] == N
                )
            ):
                self.config.passed = False
                self.config.extra_padding *= 1.5
                break

        if self.config.passed:
            self.config.extra_padding = (2 + self.config.extra_padding) / 3

        print(self.config)
        time.sleep(0.5)

        for ii in range(state.shape[0]):
            if real(state.data[ii]) > tol or imag(state.data[ii])> tol:
                found = 1
                self.config.limits[1] = ii
            elif not found:
                self.config.limits[0] = ii
        eff_padding = <idxint> round(self.config.padding*self.config.extra_padding)
        self.config.limits[0] = max(0, self.config.limits[0] - eff_padding + 1)
        self.config.limits[1] = min(N, self.config.limits[1] + eff_padding + 1)
        return self.config.limits

    cpdef idxint[::1] get_idx_dm(self, Dense state):
        cdef double tol, max_prob, safe_tol
        cdef idxint ii, N=state.shape[0], found=0, eff_padding
        if self.config.rtol != 0:
            for ii in range(N):
                if max_prob < real(state.data[ii*(N+1)]):
                    max_prob = real(state.data[ii*(N+1)])
        tol = max_prob * self.config.rtol + self.config.atol
        safe_tol = max_prob * self.config.safety_rtol + self.config.atol

        self.config.passed = True
        for ii in range(self.config.safety_pad):
            if (
                (
                 not self.config.limits[0] == 0 and
                 real(state.data[self.config.limits[0] + ii]) > safe_tol
                ) or (
                 not self.config.limits[1] == N and
                 real(state.data[self.config.limits[1] - ii - 1]) > safe_tol
                )
            ):
                self.config.passed = False
                self.config.extra_padding *= 1.5
                break

        if self.config.passed:
            self.config.extra_padding = (2 + self.config.extra_padding) / 3

        for ii in range(state.shape[0]):
            if real(state.data[ii*(N+1)]) > tol:
                found = 1
                self.config.limits[1] = ii
            elif not found:
                self.config.limits[0] = ii
        eff_padding = <idxint> round(self.config.padding*self.config.extra_padding)
        self.config.limits[0] = max(0, self.config.limits[0] - eff_padding + 1)
        self.config.limits[1] = min(N, self.config.limits[1] + eff_padding + 1)
        return self.config.limits

    cdef Dense mul_dense(self, double t, Dense vec, Dense out):
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)

        cdef size_t i
        self.base._factor(t)
        self.mul_ahs(self.base.constant, vec, 1, out)
        for i in range(self.n_ops):
            self.mul_ahs(<Data> self.base.ops[i], vec,
                         self.base.coefficients[i], out)
        return out

    cdef void mul_ahs(self, Data mat,  Dense vec, double complex a, Dense out):
        if self.super:
            if self.layer_type == CSR_TYPE:
                matmul_trunc_dm_csr_dense( mat, vec, self.config.limits, 1, out)
            elif self.layer_type == CSC_TYPE:
                matmul_trunc_dm_csc_dense( mat, vec, self.config.limits, 1, out)
            elif self.layer_type == Dense_TYPE:
                matmul_trunc_dm_dense( mat, vec, self.config.limits, 1, out)
        else:
            if self.layer_type == CSR_TYPE:
                matmul_trunc_ket_dense( mat, vec, self.config.limits, 1, out)
            elif self.layer_type == CSC_TYPE:
                matmul_trunc_ket_csc_dense( mat, vec, self.config.limits, 1, out)
            elif self.layer_type == Dense_TYPE:
                matmul_trunc_ket_dense( mat, vec, self.config.limits, 1, out)
