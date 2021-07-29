#cython: language_level=3
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data cimport Data

cdef class SpectraCoefficient(Coefficient):
    cdef Coefficient coeff_t
    cdef Coefficient coeff_w
    cdef double w

cpdef Data matmul_var(Data left, Data right, int transleft, int transright,
                     double complex alpha=*, Data out=*)

cdef class _EigenBasisTransform:
    cdef:
        QobjEvo _oper
        int size
        bint isconstant
        double _t
        object _eigvals  # np.ndarray
        Data _evecs, _inv
        object _skew
        double _dw_min

    cpdef object diagonal(self, double t)
    cpdef Data evecs(self, double t)
    cpdef Data inv(self, double t)
    cdef void _compute_eigen(self, double t) except *
    cpdef Data S_converter(self, double t)
    cpdef Data S_converter_inverse(self, double t)
    cpdef Data to_eigbasis(self, double t, Data fock)
    cpdef Data from_eigbasis(self, double t, Data eig)
    cpdef object skew(self, double t)
    cpdef double dw_min(self, double t)
