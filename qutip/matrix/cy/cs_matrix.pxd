#!python
#cython: language_level=3
import numpy as np
cimport numpy as np
cimport numpy as cnp
cimport cython
from qutip.matrix.cy.cdata cimport Cdata
import qutip.settings as qset
from libcpp cimport bool

cdef class cy_cs_matrix(Cdata):
    cdef:
        double complex * data
        int * indices
        int * indptr
        int nnz
        # int ncols
        # int nrows
        int nptrs
        int is_set
        int max_length
        int numpy_lock
        int is_csr

    cdef void init(self, int nnz, int nrows, int ncols = *, int nptrs = *,
                        int max_length = *, int init_zeros = *, int csr = *)

    cdef void free(self)

    cdef void copy_cs(self, cy_cs_matrix mat)

    #def as_vecs(self)

    cpdef _shallow_get_state(self)

    cpdef _shallow_set_state(self, state)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # method returning info about the matrix

    #def bandwidth(self)

    #def profile(self)

    #def profile_full(self)

    cpdef double one_norm(self)

    cpdef double inf_norm(self)

    cpdef isdiag(self)

    cpdef isherm(self, double tol=*)

    cpdef complex trace(self)

    cpdef cnp.ndarray[complex, ndim=1, mode='c'] get_diag(self, int k=*)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # inplace modification methods
    cpdef void transpose(self)

    cpdef void adjoint(self)

    cpdef void sparse_permute(self,
            cnp.ndarray[int, ndim=1] rperm,
            cnp.ndarray[int, ndim=1] cperm)

    cpdef void sparse_reverse_permute(self,
            cnp.ndarray[int, ndim=1] rperm,
            cnp.ndarray[int, ndim=1] cperm)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # internal method call bu other method

    cdef void _shorten(self, int N)

    cdef void _sort_indices(self)

    cdef double _max_sum_main(self)

    cdef double _max_sum_sec(self)

    cdef void _zcs_trans_core(self, cy_cs_matrix out) nogil

    cdef void _zcs_adjoint_core(self, cy_cs_matrix out) nogil
