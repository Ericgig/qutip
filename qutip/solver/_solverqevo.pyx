#cython: language_level=3

from qutip.core.cy.cqobjevo cimport CQobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data.base cimport idxint
cimport qutip.core.data as _data
import qutip.core.data as data
from qutip import Qobj, spre
cimport cython
from qutip.solver._feedback cimport *
from qutip.core.data.base import idxint_dtype
from libc.math cimport round

cdef class SolverQEvo:
    def __init__(self, base, options, dict args, dict feedback):
        self.base = base.compiled_qobjevo
        self.set_feedback(feedback, args, base.cte.issuper,
                          options['feedback_normalize'])
        self.collapse = []

    def jac_np_vec(self, t, vec):
        return self.jac_data(t).as_array()

    cdef _data.Data jac_data(self, double t):
        if self.has_dynamic_args:
            raise NotImplementedError("jacobian not available with feedback")
        return self.base.call(t, data=True)

    def mul_np_vec(self, t, vec):
        cdef int i, row, col
        cdef _data.Dense state = _data.dense.fast_from_numpy(vec)
        _data.column_unstack_dense(state, self.base.shape[1], inplace=True)
        cdef _data.Dense out = _data.dense.zeros(state.shape[0],
                                state.shape[1], state.fortran)
        out = self.mul_dense(t, state, out)
        _data.column_stack_dense(out, inplace=True)
        return out.as_ndarray().ravel()

    cdef _data.Data mul_data(self, double t, _data.Data vec):
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)
        return self.base.matmul(t, vec)

    cdef _data.Dense mul_dense(self, double t, _data.Dense vec, _data.Dense out):
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)
        return self.base.matmul_dense(t, vec, out)

    def set_feedback(self, dict feedback, dict args, bint issuper, bint norm):
        # Move elsewhere and op should be a dimensions object when available
        self.args = args
        self.dynamic_arguments = []
        for key, val in feedback.items():
            if val in [Qobj, "Qobj", "qobj", "state"]:
                self.dynamic_arguments.append(QobjFeedback(key, args[key],
                                                           norm))
            elif isinstance(val, Qobj):
                self.dynamic_arguments.append(ExpectFeedback(key, val,
                                                             issuper, norm))
            elif val in ["collapse"]:
                if not isinstance(args[key], list):
                    args[key] = []
                self.collapse = args[key]
                self.dynamic_arguments.append(CollapseFeedback(key, args[key]))
            else:
                raise ValueError("unknown feedback type")
        self.has_dynamic_args = bool(self.dynamic_arguments)

    cdef void apply_feedback(self, double t, _data.Data matrix) except *:
        cdef Feedback feedback
        for dyn_args in self.dynamic_arguments:
            feedback = <Feedback> dyn_args
            val = feedback._call(t, matrix)
            self.args[feedback.key] = val
        for i in range(self.base.n_ops):
            (<Coefficient> self.base.coeff[i]).arguments(self.args)

    cpdef void arguments(self, dict args):
        self.args = args
        for i in range(self.base.n_ops):
            (<Coefficient> self.base.coeff[i]).arguments(self.args)


from qutip.core.data.matmul_ahs cimport *

cdef class SolverQEvoAHS(SolverQEvo):
    def __init__(self, base, options, dict args, dict feedback):
        self.base = base.compiled_qobjevo
        self.set_feedback(feedback, args, base.cte.issuper,
                          options['feedback_normalize'])
        self.collapse = []

        self.super = base.issuper
        self.padding = 5 # options.func["padding"]
        self.extra_padding = 1
        self.failed_step_tol = 100
        self.state_rtol = options["rtol"] # options.func["rtol"]
        self.state_atol = options["atol"] # options.func["atol"]
        self.type = self.base.layer_type

    def resize(_data.Dense state):
        if self.super:
            used_idx = get_idx_dm(
                state,
                self.state_atol,
                self.state_rtol,
                round(self.padding*self.extra_padding)
            )
        else:
            used_idx = get_idx_ket(
                state,
                self.state_atol,
                self.state_rtol,
                round(self.padding*self.extra_padding)
            )
        return True # TODO: support failing step 

    cdef _data.Dense mul_dense(self, double t, _data.Dense vec, _data.Dense out):
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)

        cdef size_t i
        self.base._factor(t)
        self.mul_ahs(self.base.constant, vec, 1, out)
        for i in range(self.n_ops):
            self.mul_ahs(<Data> self.base.ops[i], matrix,
                         self.base.coefficients[i], out)
        return out

    cdef void mul_ahs(_data.Data mat,  _data.Dense vec, double complex a, _data.Dense out):
        if self.super:
            if self.type == CSR_TYPE:
                matmul_trunc_dm_csr_dense( mat, vec, self.used_idx, 1, out)
            elif self.type == CSC_TYPE:
                matmul_trunc_dm_csc_dense( mat, vec, self.used_idx, 1, out)
            elif self.type == Dense_TYPE:
                matmul_trunc_dm_dense( mat, vec, self.used_idx, 1, out)
        else:
            if self.type == CSR_TYPE:
                matmul_trunc_ket_dense( mat, vec, self.used_idx, 1, out)
            elif self.type == CSC_TYPE:
                matmul_trunc_ket_csc_dense( mat, vec, self.used_idx, 1, out)
            elif self.type == Dense_TYPE:
                matmul_trunc_ket_dense( mat, vec, self.used_idx, 1, out)
