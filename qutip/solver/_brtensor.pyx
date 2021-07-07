cimport cython
import qutip.core.data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy._element cimport _BaseElement
from qutip.core.diagoper cimport _DiagonalizedOperatorHermitian
from cython.parallel import prange
cimport openmp



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR _br_term_sparse(Data A, double[:, ::1] skew, double[:, ::1] spectrum,
                          bint use_secular, double cutoff):

    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k # matrix indexing variables
    cdef double complex elem, ac_elem, bd_elem
    cdef double complex[:,:] A_mat
    cdef vector[idxint] coo_rows, coo_cols
    cdef vector[double complex] coo_data

    if type(A) is Dense:
        A_mat = A.as_ndarray()
    else:
        A_mat = A.to_array()

    for a in prange(nrows, nogil=True, schedule='dynamic'):
      for b in range(nrows):
        for c in range(nrows):
          for d in range(nrows):
            if (not use_secular) or (
                fabs(skew[a,b]] - skew[c,d]) < cutoff
            ):
                elem = (A_mat[a, c] * A_mat[d, b]) * 0.5
                elem *= (spectrum[c, a] + spectrum[d, b])

                if a == c:
                    ac_elem = 0
                    for k in range(nrows):
                        ac_elem += A_mat[d, k] * A_mat[k, b] * spectrum[d, k]
                    elem -= 0.5 * ac_elem

                if b == d:
                    bd_elem = 0
                    for k in range(nrows):
                        bd_elem += A_mat[a, k] * A_mat[k, c] * spectrum[c, k]
                    elem -= 0.5 * bd_elem

                if elem != 0:
                    coo_rows.push_back(a * nrows + b)
                    coo_cols.push_back(c * nrows + d)
                    coo_data.push_back(elem)

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size())












cdef class _BlochRedfieldElement(_BaseElement):
    cdef _DiagonalizedOperatorHermitian H
    cdef QobjEvo a_ops
    cdef Spectrum spectra
    cdef bint use_secular
    cdef double sec_cutoff, atol
    cdef size_t nrows

    cdef object np_datas
    cdef double[:, ::1] skew
    cdef double[:, ::1] spectrum

    cdef readonly Dense evecs, out, eig_vec, temp, op_eig

    def __init__(self, H, a_op, spectra, use_secular, sec_cutoff, atol):
        if isinstance(H, _DiagonalizedOperator):
            self.H = H
        else:
            self.H = _DiagonalizedOperatorHermitian(H)
        self.a_op = a_op
        self.nrows = a_op.shape[0]
        self.shape = (_mul_checked(self.nrows, self.nrows),
                      _mul_checked(self.nrows, self.nrows))
        self.dims = [a_op.dims, a_op.dims]

        self.spectra = spectra
        self.sec_cutoff = sec_cutoff
        self.atol = atol

        # Allocate some array
        # Let numpy manage memory
        self.np_datas = np.zeros((self.nrows, self.nrows), dtype=np.float64)
        self.spectrum = self.np_datas

    cpdef Data data(self, double t):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self.H.dw_min(t)
        cdef double[:, :] skew = self.H.skew(t)
        self._compute_spectrum(t)

        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = _br_term(A_eig, self.skew, self.spectrum, cutoff)
        return self.H.to_fockbasis(t, BR_eig)

    cpdef object qobj(self, double t):
        return Qobj(self.data(t), dims=self.dims, type="super",
                     copy=False, superrep="super")

    cpdef double complex coeff(self, double t) except *:
        return 1.

    cdef _compute_spectrum(self, double t):
        for col in range(self.nrows):
            for row in range(self.nrows):
                self.spectrum[row, col] = \
                    self.spectra._call_t(t, self.skew[row, col])

    cdef Data matmul_data_t(_BaseElement self, double t, Data state,
                            Data out=None):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self.H.dw_min(t)
        cdef double[:, :] skew = self.H.skew(t)
        self._compute_spectrum(t)

        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        state_eig = self.H.to_eigbasis(t, state)
        BR_eig = _br_term(A_eig, self.skew, self.spectrum, cutoff)
        out_eig = _data.matmul(BR_eig, state_eig)
        return _data.add(self.H.to_fockbasis(t, out_eig), out)

    def linear_map(self, f, anti=False):
        return _MapElement(self, [f])

    def replace_arguments(self, args, cache=None):
        if cache is None:
            return _FuncElement(self._func, {**self._args, **args})
        for old, new in cache:
            if old is self:
                return new
        new = _FuncElement(self._func, {**self._args, **args})
        cache.append((self, new))
        return new
