#!python
#cython: language_level=3
import numpy as np
cimport numpy as np
cimport numpy as cnp
cimport cython
cimport qutip.matrix.cy.cs_matrix
import qutip.settings as qset
from libcpp cimport bool
from qutip.qdata import _qdata, qdata

cdef class cy_csr_matrix(cy_cs_matrix):
    def __init__(self, object data=None):
        if data is None:
            cy_cs_matrix.__init__(self)
            return
        elif not isinstance(data, _qdata):
            data = qdata(data)


    cpdef cy_csr_matrix copy(self) ##

    #def object to_qdata(self)

    #def object qdata(self) ##

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # method returning info about the matrix
    cpdef double one_norm(self) ##

    cpdef double inf_norm(self) ##

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # act on dense vector/matrix method
    cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv(self, complex[::1] vec)

    cpdef void spmvpy(self, complex[::1] vec, complex[::1] out, complex alpha)

    cpdef void spmmpy(self, cnp.ndarray[complex, ndim=2] mat,
                            cnp.ndarray[complex, ndim=2] out,
                            complex alpha)

    cpdef void spmmcpy(self, complex[:, ::1] mat, complex[:, ::1] out,  complex alpha)

    cpdef void spmmfpy(self, complex[::1, :] mat,  complex[::1, :] out, complex alpha)

    cpdef cnp.ndarray[complex, ndim=2] spmm(self, cnp.ndarray[complex, ndim=2] mat)

    cpdef cnp.ndarray[complex, ndim=2] spmmf(self, complex[::1, :] mat)

    cpdef cnp.ndarray[complex, ndim=2] spmmc(self, complex[:, ::1] mat)

    #cpdef complex expect_rho_vec(self, complex[::1] vec)

    #cpdef complex expect_psi_vec(self, complex[::1] vec)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # inplace modification methods

    cpdef unit_row_norm(self) ##

    cpdef reshape(self, int new_rows, int new_cols) ##

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # new matrix creating methods

    cpdef cy_csr_matrix proj(self) ##

    #def ptrace(self, sel)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # internal method call bu other method

    cdef cy_csr_matrix _ptrace_core_dense(self, int[:, ::1] tensor_table, int num_sel_dims)

    cdef cy_csr_matrix _ptrace_core_sp(self, int[:, ::1] tensor_table)

    cdef void _coo_indices(self, int[::1] rows, int[::1] cols) #

    cdef void _from_coo_indices(self, int[::1] rows, int[::1] cols) #


cpdef cy_csr_matrix csr_from_scipy(object A, copy=*)

cpdef cy_csr_matrix csr_from_dense(complex[:, :] mat)

cpdef cy_csr_matrix csr_from_scipy_coo(object A)

cpdef cy_csr_matrix identity_csr(unsigned int nrows)

cpdef object csr_qmatrix_from_cdata(cy_csr_matrix cdata)
