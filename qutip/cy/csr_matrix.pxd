#!python
#cython: language_level=3
import numpy as np
cimport numpy as np
cimport numpy as cnp
cimport cython
np.import_array()
from qutip.fastsparse import fast_csr_matrix
import qutip.settings as qset
from libcpp cimport bool
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from libc.stdlib cimport div, ldiv
from libc.math cimport abs, fabs, sqrt

include "parameters.pxi"




cdef class cy_csr_matrix:
    cdef:
        double complex * data
        int * indices
        int * indptr
        int nnz
        int nrows
        int ncols
        int is_set
        int max_length
        int numpy_lock


    cdef void init(self, int nnz, int nrows, int ncols = *,
                        int max_length = *, int init_zeros = *) #

    cdef void copy_CSR(self, cy_csr_matrix mat)

    cpdef cy_csr_matrix copy(self) ##

    cdef void free(self) #

    cdef void _shorten(self, int N)

    #def object to_scipy(self)

    #def object scipy(self) ##

    #def csr(self)

    cdef void _sort_indices(self)

    cpdef void _coo_indices(self, int[::1] rows, int[::1] cols) #

    cpdef void _from_coo_indices(self, int[::1] rows, int[::1] cols) #

    cpdef reshape(self, int new_rows, int new_cols) ##

    #def _sparse_bandwidth(self) ##

    #def _sparse_profile(self) ##

    cpdef void _sparse_permute(self,
            cnp.ndarray[ITYPE_t, ndim=1] rperm,
            cnp.ndarray[ITYPE_t, ndim=1] cperm)

    cpdef _sparse_reverse_permute(self,
            cnp.ndarray[ITYPE_t, ndim=1] rperm,
            cnp.ndarray[ITYPE_t, ndim=1] cperm)

    cpdef _isdiag(self) ##

    cpdef cnp.ndarray[complex, ndim=1, mode='c'] _csr_get_diag(self, int k=*) ##

    #def unit_row_norm(self) ##

    cpdef double zcsr_one_norm(self) ##

    cpdef double zcsr_inf_norm(self) ##

    cpdef bool cy_tidyup(self, double atol) ##

    cpdef void transpose(self) ##

    cdef void _zcsr_trans_core(self, cy_csr_matrix out) nogil #

    cpdef void adjoint(self) ##

    cdef void _zcsr_adjoint_core(self, cy_csr_matrix out) nogil #

    #def isherm(self, double tol = qset.atol) ##

    #def proj(self) ##

    #def trace(self, bool isherm) ##

    cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv(self, complex[::1] vec)

    cpdef complex expect_rho_vec(self, complex[::1] vec)

    cpdef complex expect_psi_vec(self, complex[::1] vec)


cpdef cy_csr_matrix CSR_from_scipy(object A, copy=*) ##

cpdef cy_csr_matrix identity_CSR(unsigned int nrows) ##
