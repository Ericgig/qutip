#!python
#cython: language_level=3

cimport numpy as cnp
import numpy as np

cdef class Cdata:
  cpdef complex expect_psi_vec(self, complex[::1] psi):
      return 0.+0.j

  cpdef complex expect_rho_vec(self, complex[::1] rho):
      return 0.+0.j


  cpdef cnp.ndarray[complex, ndim=1, mode="c"] matvec(self, complex[::1] vec):
      return np.zeros(0, dtype=complex)


  cpdef void matvecpy(self, complex[::1] vec, complex[::1] out, complex alpha):
      pass


  cpdef cnp.ndarray[complex, ndim=2] matmat(self, cnp.ndarray[complex, ndim=2] mat):
      return np.zeros((0,0), dtype=complex)

  cpdef cnp.ndarray[complex, ndim=2] matmatf(self, complex[::1, :] mat):
      return np.zeros((0,0), dtype=complex)

  cpdef cnp.ndarray[complex, ndim=2] matmatc(self, complex[:, ::1] mat):
      return np.zeros((0,0), dtype=complex)

  cpdef void matmatpy(self, cnp.ndarray[complex, ndim=2] mat,
                            cnp.ndarray[complex, ndim=2] out,
                            complex alpha):
      pass

  cpdef void matmatpyf(self, complex[::1, :] mat,  complex[::1, :] out, complex alpha):
      pass

  cpdef void matmatpyc(self, complex[:, ::1] mat, complex[:, ::1] out,  complex alpha):
      pass

  cpdef void matmat_as_vec_f(self, complex[::1] vec, complex[::1] out, complex alpha):
      pass
