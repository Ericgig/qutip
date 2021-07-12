

cdef Data matmul_var(Data left, Data right, int transleft, int transright,
                     double complex alpha=*, Data out=*)

cdef class _DiagonalizedOperator:
    cdef:
        QobjEvo _oper
        int size
        bint isconstant, _isherm
        double _t
        object _eigvals  # np.ndarray
        Data _evecs, _inv

    cpdef object diagonal(self, double t)
    cpdef Data evecs(self, double t)
    cpdef Data inv(self, double t)
    cdef void _compute_eigen(self, double t) except *
    cdef Data _S_conv(self, double t)
    cdef Data _S_conv_inv(self, double t)
    cpdef void to_eigbasis(self, Data fock)
    cpdef void from_eigbasis(self, Data eig)

cdef class _DiagonalizedOperatorHermitian(_DiagonalizedOperator):
    double[:, :] _skew
    double _dw_min
    cdef double[:, :] skew(self, double t)
    cdef double dw_min(self, double t)
