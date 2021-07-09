cimport cython
import qutip.core.data as _data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy._element cimport _BaseElement
from qutip.core.diagoper cimport _DiagonalizedOperatorHermitian
from cython.parallel import prange
cimport openmp
from libcpp.vector cimport vector
from libc.float cimport DBL_MAX


def make_spectra(f, args):
    """
        # wrapper to use Coefficient for spectrum function in string format
        # What was supported:
        #     time independent brsolve:
        #         function(w : float) -> float
        #     time dependent brsolve:
        #         str expression depending on `t` and `w`
        #         3 strings: depending on `w`, `t` and `t`.
        #         Pair of Cubic_Spline, of depending on `t` and one of `w`.
    """
    try:
        coeff = coefficient(f, args={**args, **{"w":0}})
        return Spectrum_coeff(coeff)
    except ValueError:
        pass
    if callable(f):
        return Spectrum(f)


cdef class Spectrum:
    cdef object _func

    def __init__(self, func):
        self._func = func

    cpdef _call_t(self, double t, double w):
        return self._func(w)

    def __call__(self, double w):
        return self._func(w)


cdef class Spectrum_coeff(Spectrum):
    # wrapper to use Coefficient for spectrum function in string format
    cdef Coefficient _coeff

    def __init__(self, coeff):
        self._coeff = coeff

    cpdef _call_t(self, double t, double w):
        self._coeff.arguments({"w":w})
        return real(self._coeff._call(t))

    def __call__(self, double w):
        self._coeff.arguments({"w":w})
        return real(self._coeff._call(0))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Data _br_term_data(Data A, double[:, ::1] skew, double[:, ::1], spectrum,
                         double cutoff):
    # TODO: need point multiply to as dispatch to implement...
    pass


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Dense _br_term_dense(Data A, double[:, ::1] skew, double[:, ::1] spectrum,
                           double cutoff):

    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k # matrix indexing variables
    cdef double complex elem, ac_elem, bd_elem
    cdef double complex[:,:] A_mat, ac_term, bd_term
    cdef object np2term
    cdef Dense out
    cdef double complex[::1, :] out_array

    if type(A) is Dense:
        A_mat = A.as_ndarray()
    else:
        A_mat = A.to_array()

    out = _data.dense.zeros(nrows*nrows, nrows*nrows)
    out_array = out.as_ndarray()

    np2term = np.zeros((nrows, nrows, 2), dtype=np.complex128)
    ac_term = np2term[:, :, 0]
    bd_term = np2term[:, :, 1]

    for a in range(nrows):
        for b in range(nrows):
            if fabs(skew[a,b]) < cutoff:
                for k in range(nrows):
                    ac_elem[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[a, k]
                    bd_elem[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[b, k]

    for a in prange(nrows, nogil=True, schedule='dynamic'):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    if fabs(skew[a, b] - skew[c, d]) < cutoff:
                        elem = (A_mat[a, c] * A_mat[d, b]) * 0.5
                        elem *= (spectrum[c, a] + spectrum[d, b])

                    if a == c:
                        elem -= 0.5 * ac_elem[d, b]

                    if b == d:
                        elem -= 0.5 * bd_elem[a, c]

                    out_array[a * nrows + b, c * nrows + d] = elem
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR _br_term_sparse(Data A, double[:, ::1] skew, double[:, ::1] spectrum,
                          double cutoff):

    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k # matrix indexing variables
    cdef double complex elem, ac_elem, bd_elem
    cdef double complex[:,:] A_mat, ac_term, bd_term
    cdef double[:,:] spectrum
    cdef object np2term
    cdef vector[idxint] coo_rows, coo_cols
    cdef vector[double complex] coo_data

    if type(A) is Dense:
        A_mat = A.as_ndarray()
    else:
        A_mat = A.to_array()

    np2term = np.zeros((nrows, nrows, 2), dtype=np.complex128)
    ac_term = np2term[:, :, 0]
    bd_term = np2term[:, :, 1]

    for a in range(nrows):
        for b in range(nrows):
            if fabs(skew[a,b]) < cutoff:
                for k in range(nrows):
                    ac_elem[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[a, k]
                    bd_elem[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[b, k]

    for a in range(nrows):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    if fabs(skew[a,b] - skew[c,d]) < cutoff:
                        elem = (A_mat[a, c] * A_mat[d, b]) * 0.5
                        elem *= (spectrum[c, a] + spectrum[d, b])

                        if a == c:
                            elem -= 0.5 * ac_elem[d, b]

                        if b == d:
                            elem -= 0.5 * bd_elem[a, c]

                        if elem != 0:
                            coo_rows.push_back(a * nrows + b)
                            coo_cols.push_back(c * nrows + d)
                            coo_data.push_back(elem)

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size()
    )


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
        self.sec_cutoff = sec_cutoff if use_secular else np.inf
        self.atol = atol

        # Allocate some array
        # Let numpy manage memory
        self.np_datas = np.zeros((self.nrows, self.nrows), dtype=np.float64)
        self.spectrum = self.np_datas

    cdef _compute_spectrum(self, double t):
        self.skew = self.H.skew(t)
        for col in range(self.nrows):
            for row in range(self.nrows):
                self.spectrum[row, col] = \
                    self.spectra._call_t(t, self.skew[row, col])
        return self.H.dw_min(t)

    cdef Data _br_term(Data A_eig, double cutoff):
        # TODO: better swpping.
        # dense is in parallel,
        # sparse can be great when cutoff is small and output is sparse.
        # Data could be good for gpu implemenations.
        if False and type(A_eig) is not Dense:
            return _br_term_data(A_eig, self.skew, self.spectrum, cutoff)
        elif self.sec_cutoff >= DBL_MAX:
            return _br_term_dense(A_eig, self.skew, self.spectrum, cutoff)
        else:
            return _br_term_sparse(A_eig, self.skew, self.spectrum, cutoff)

    cpdef Data data(self, double t):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)

        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = self._br_term(A_eig, cutoff)
        return self.H.to_fockbasis(t, BR_eig)

    cpdef object qobj(self, double t):
        return Qobj(self.data(t), dims=self.dims, type="super",
                    copy=False, superrep="super")

    cpdef double complex coeff(self, double t) except *:
        return 1.

    cdef Data matmul_data_t(_BaseElement self, double t, Data state,
                            Data out=None):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)

        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        state_eig = self.H.to_eigbasis(t, state)
        BR_eig = self._br_term(A_eig, cutoff)
        out_eig = _data.matmul(BR_eig, state_eig)
        return _data.add(self.H.to_fockbasis(t, out_eig), out)

    def linear_map(self, f, anti=False):
        return _MapElement(self, [f])

    def replace_arguments(self, args, cache=None):
        if cache is None:
            return _BlochRedfieldElement(
                self.H, QobjEvo(self.a_op, args=args), self.spectra,
                True, self.sec_cutoff, self.atol
            )
        H = None
        for old, new in cache:
            if old is self:
                return new
            if old is self.H:
                H = new
        if H is None:
            H_new = type(self.H)(QobjEvo(self.H._oper, args=args),
                                 type(self.H._oper) is CSR)
        new = _BlochRedfieldElement(
            H_new, QobjEvo(self.a_op, args=args), self.spectra,
            True, self.sec_cutoff, self.atol
        )
        cache.append((self, new))
        cache.append((self.H, H_new))
        return new
