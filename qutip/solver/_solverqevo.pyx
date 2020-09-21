#cython: language_level=3

from qutip.core.cy.cqobjevo cimport CQobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data.base cimport idxint
cimport qutip.core.data as _data
import qutip.core.data as data
from qutip import Qobj, spre
cimport cython
from ._feedback cimport *


cdef class SolverQEvo:
    def __init__(self, base, options, dict args, dict feedback):
        self.base = base.compiled_qobjevo
        self.set_feedback(feedback, args, base.cte.issuper)
        self.collapse = []

    def mul_np_vec(self, t, vec):
        cdef int i, row, col
        cdef _data.Dense state = _data.dense.fast_from_numpy(vec)
        _data.column_unstack_dense(state, self.base.shape[1], inplace=True)
        cdef _data.Dense out = _data.dense.zeros(state.shape[0],
                                state.shape[1],
                                state.fortran)
        self.mul_data(t, state, out)
        _data.column_stack_dense(out, inplace=True)
        return out.as_ndarray().ravel()

    cdef void mul_data(self, double t, _data.Data vec, _data.Data out):
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)
        self.base.matmul(t, vec, out)

    def set_feedback(self, dict feedback, dict args, bint issuper):
        # Move elsewhere and op should be a dimensions object when available
        self.args = args
        self.dynamic_arguments = []
        for key, val in feedback.items():
            if val in [Qobj, "Qobj", "qobj", "state"]:
                self.dynamic_arguments.append(QobjFeedback(key, args[key]))
            elif isinstance(val, Qobj):
                self.dynamic_arguments.append(ExpectFeedback(key, val, issuper))
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
