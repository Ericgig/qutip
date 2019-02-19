#!python
#cython: language_level=3

cimport numpy as cnp

cdef class Cdata:
  cdef:
    int nrows
    int ncols

  cpdef complex expect_psi_vec(self, complex[::1] psi)

  cpdef complex expect_rho_vec(self, complex[::1] rho)


  cpdef cnp.ndarray[complex, ndim=1, mode="c"] matvec(self, complex[::1] vec)

  cpdef void matvecpy(self, complex[::1] vec, complex[::1] out, complex alpha)


  cpdef cnp.ndarray[complex, ndim=2] matmat(self, cnp.ndarray[complex, ndim=2] mat)

  cpdef cnp.ndarray[complex, ndim=2] matmatf(self, complex[::1, :] mat)

  cpdef cnp.ndarray[complex, ndim=2] matmatc(self, complex[:, ::1] mat)

  cpdef void matmatpy(self, cnp.ndarray[complex, ndim=2] mat,
                            cnp.ndarray[complex, ndim=2] out,
                            complex alpha)

  cpdef void matmatpyf(self, complex[::1, :] mat,  complex[::1, :] out, complex alpha)

  cpdef void matmatpyc(self, complex[:, ::1] mat, complex[:, ::1] out,  complex alpha)

  cpdef void matmat_as_vec_f(self, complex[::1] vec, complex[::1] out, complex alpha)
