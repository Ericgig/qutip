# distutils: language = c++

import numpy as np
cimport numpy as np

from qutip.cy.td_qobj_cy cimport cy_qobj#, cy_td_qobj, cy_cte_qobj

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)


"""cdef extern from "src/dopri5td.cpp":
    cdef cppclass ode:
        ode()
        ode(int*, double*, void (*_H)(double, complex *, complex *))
        int step(double, double, complex*, complex*)
        double integrate(double, double, double, complex*, double*)"""

cdef cy_qobj compiled_cte_cy

cdef void _rhs(double t, complex* vec, complex* out):
    global compiled_cte_cy
    compiled_cte_cy._rhs_mat(t, vec, out)

cdef extern from "/home/eric/algo/qutip/qutip/qutip/cy/src/dopri5td.cpp":
    cdef cppclass ode:
        ode()
        ode(int*, double*, void (*_H)(double, complex *, complex *))
        int step(double, double, complex*, complex*)
        double integrate(double, double, double, complex*, double*)
        int len()
        int debug(complex * , complex *, double* )

cdef class ode_td_dopri:
    cdef ode* cobj

    #def __init__(self, int l, rhs_function, config):
    def __init__(self, int l, H, config):
        global compiled_cte_cy
        _y1 = np.zeros(l,dtype=complex)
        cdef int[::1] int_option = np.zeros(2,dtype=np.intc)
        cdef double[::1] double_option = np.zeros(5,dtype=np.double)
        #cdef void* rhs_ptr = PyLong_AsVoidPtr(rhs_function)
        compiled_cte_cy = H.compiled_Qobj
        cdef void (*rhs)(double, complex*, complex*)
        rhs = <void (*)(double, complex*, complex*)> _rhs

        int_option[0]=l
        int_option[1]=config.norm_steps

        double_option[0]=config.options.atol
        double_option[1]=config.options.rtol
        double_option[2]=config.options.min_step
        double_option[3]=config.options.max_step
        double_option[4]=config.norm_tol

        self.cobj = new ode(<int*>&int_option[0],
                            <double*>&double_option[0],
                            rhs)
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')

    def __del__(self):
        del self.cobj

    cpdef double integrate(self, double _t_in, double _t_target, double rand, \
                           complex[::1] _psi, double[::1] _err):
        return self.cobj.integrate(_t_in, _t_target, rand,
                                   <complex*>&_psi[0],
                                   <double*>&_err[0])

    def test1(self):
        global compiled_cte_cy
        print(compiled_cte_cy.call(0,data=1))

    def test(self, double t, complex[::1] vec, complex[::1] out):
        global compiled_cte_cy
        compiled_cte_cy._rhs_mat(t, <complex*>&vec[0], <complex*>&out[0])


    def debug(self):
      l = self.cobj.len()
      cdef complex[::1] derr_in = np.zeros(l,dtype=complex)
      cdef complex[::1] derr_out = np.zeros(l,dtype=complex)
      cdef double[::1] opt = np.zeros(8)
      print("step_limit", self.cobj.debug(<complex*>&derr_in[0],
                                          <complex*>&derr_out[0],
                                          <double*>&opt[0]))
      for i in range(l):
          print(derr_in[i], derr_out[i])
      for i in range(8):
          print(opt[i])
