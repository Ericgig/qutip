



















cdef class Cdata:
  cpdef complex expect_psi_vec(complex[::1] psi):
      raise NotImplementedError
      return 0.0 +0j

  cpdef complex expect_rho_vec(complex[::1] rho):
      raise NotImplementedError
      return 0.0 +0j

  cpdef void matvec(complex[::1] vec, complex[::1] out):
      raise NotImplementedError

  cpdef void matvecpy(complex[::1] vec, complex[::1] out, complex alpha):
      raise NotImplementedError
