#cython: language_level=3
cimport cython
from libc.math cimport fabs
import qutip.core.data as _data
from qutip.core.data cimport Dense, CSR, Data, idxint, csr
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.cy._element cimport _BaseElement, _MapElement
from qutip.solver._brtools cimport SpectraCoefficient, _EigenBasisTransform
from qutip import Qobj
import numpy as np

from cython.parallel import prange
cimport openmp
from libcpp.vector cimport vector
from libc.float cimport DBL_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Data _br_term_data(Data A, double[:, ::1] spectrum,
                         double[:, ::1] skew, double cutoff):
    # TODO: need point multiply to as dispatch to implement...
    # TODO: cutoff not applied.
    raise NotImplementedError
    """
    cdef Data B, C, AB, id
    cdef double complex cutoff_arr
    cdef int nrow = A.shape[0]
    B = _data.element_wise_multiply(
        A,
        _data.to(A.__class__, _data.Dense(spectrum, copy=False))
    )
    cdef Data out = _data.kron(A.transpose(), B)
    out = _data.add(_data.kron(B.transpose(), A), out)
    cutoff_arr = np.zeros((nrow*nrow, nrow*nrow), dtype=np.complex128)
    if cutoff != DBL_MAX:
        for a in prange(nrows, nogil=True, schedule='dynamic'):
            for b in range(nrows):
                for c in range(nrows):
                    for d in range(nrows):
                        if fabs(skew[a, b] - skew[c, d]) < cutoff:
                            cutoff_arr[a * nrows + b, c * nrows + d] = 1.
    C = A.__class__(_data.Dense(cutoff_arr, copy=False))
    AB = _data.matmul(A, B)
    id = _data.identity[A.__class__](*A.shape)
    out = _data.add(out, _data.kron(AB.transpose(), id), 0.5)
    out = _data.add(out, _data.kron(id, AB), 0.5)
    return _data.element_wise_multiply(out, C)
    """


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Dense _br_term_dense(Data A, double[:, ::1] spectrum,
                           double[:, ::1] skew, double cutoff):

    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k # matrix indexing variables
    cdef double complex elem
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
            if fabs(skew[a, b]) < cutoff:
                for k in range(nrows):
                    ac_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[a, k]
                    bd_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[b, k]

    for a in range(nrows): # prange(nrows, nogil=True, schedule='dynamic'):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    elem = 0.
                    if fabs(skew[a, b] - skew[c, d]) < cutoff:
                        elem = (A_mat[a, c] * A_mat[d, b] * 0.5 *
                                (spectrum[c, a] + spectrum[d, b]))

                    if a == c:
                        elem = elem - 0.5 * ac_term[d, b]

                    if b == d:
                        elem = elem - 0.5 * bd_term[a, c]

                    out_array[a * nrows + b, c * nrows + d] = elem
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR _br_term_sparse(Data A, double[:, ::1] spectrum,
                          double[:, ::1] skew, double cutoff):

    cdef size_t nrows = A.shape[0]
    cdef size_t a, b, c, d, k # matrix indexing variables
    cdef double complex elem, ac_elem, bd_elem
    cdef double complex[:,:] A_mat, ac_term, bd_term
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
                    ac_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[a, k]
                    bd_term[a, b] += A_mat[a, k] * A_mat[k, b] * spectrum[b, k]

    for a in range(nrows):
        for b in range(nrows):
            for c in range(nrows):
                for d in range(nrows):
                    if fabs(skew[a,b] - skew[c,d]) < cutoff:
                        elem = (A_mat[a, c] * A_mat[d, b]) * 0.5
                        elem *= (spectrum[c, a] + spectrum[d, b])

                        if a == c:
                            elem -= 0.5 * ac_term[d, b]

                        if b == d:
                            elem -= 0.5 * bd_term[a, c]

                        if elem != 0:
                            coo_rows.push_back(a * nrows + b)
                            coo_cols.push_back(c * nrows + d)
                            coo_data.push_back(elem)

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size()
    )


cdef class _BlochRedfieldElement(_BaseElement):
    cdef _EigenBasisTransform H
    cdef QobjEvo a_ops
    cdef Coefficient spectra
    cdef bint use_secular
    cdef double sec_cutoff, atol
    cdef size_t nrows

    cdef object np_datas
    cdef double[:, ::1] skew
    cdef double[:, ::1] spectrum

    cdef readonly Dense evecs, out, eig_vec, temp, op_eig

    def __init__(self, H, a_op, spectra, sec_cutoff):
        if isinstance(H, _EigenBasisTransform):
            self.H = H
        else:
            self.H = _EigenBasisTransform(H)
        self.a_op = a_op
        self.nrows = a_op.shape[0]
        self.shape = (self.nrows * self.nrows, self.nrows * self.nrows)
        self.dims = [a_op.dims, a_op.dims]

        self.spectra = spectra
        self.sec_cutoff = sec_cutoff # if use_secular else np.inf

        # Allocate some array
        # Let numpy manage memory
        self.np_datas = np.zeros((self.nrows, self.nrows), dtype=np.float64)
        self.spectrum = self.np_datas

    cdef _compute_spectrum(self, double t):
        cdef Coefficient spec
        self.skew = self.H.skew(t)
        for col in range(self.nrows):
            for row in range(self.nrows):
                spec = self.spectra.replace_arguments(w=self.skew[row, col])
                self.spectrum[row, col] = spec(t)
        return self.H.dw_min(t)

    cdef Data _br_term(self, Data A_eig, double cutoff):
        # TODO: better swapping.
        # dense can run in parallel,
        # sparse can be great when cutoff is small and output is sparse.
        # Data could be good for gpu implemenations.
        if False and type(A_eig) is not Dense:
            return _br_term_data(A_eig, self.spectrum, self.skew, cutoff)
        elif self.sec_cutoff >= DBL_MAX:
            return _br_term_dense(A_eig, self.spectrum, self.skew, cutoff)
        else:
            return _br_term_sparse(A_eig, self.spectrum, self.skew, cutoff)

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

    cdef Data matmul_data_t(self, double t, Data state, Data out=None):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)
        cdef Data A_eig, state_eig, BR_eig, out_eig

        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        state_eig = self.H.to_eigbasis(t, state)
        BR_eig = self._br_term(A_eig, cutoff)
        out_eig = _data.matmul(BR_eig, state_eig)
        if out is None:
            return self.H.to_fockbasis(t, out_eig)
        return _data.add(self.H.to_fockbasis(t, out_eig), out)

    def linear_map(self, f, anti=False):
        return _MapElement(self, [f])

    def replace_arguments(self, args, cache=None):
        if cache is None:
            return _BlochRedfieldElement(
                self.H,
                QobjEvo(self.a_op, args=args),
                self.spectra,
                self.sec_cutoff
            )
        H = None
        for old, new in cache:
            if old is self:
                return new
            if old is self.H:
                H = new
        if H is None:
            H = type(self.H)(QobjEvo(self.H._oper, args=args),
                                 type(self.H._oper) is CSR)
        new = _BlochRedfieldElement(
            H, QobjEvo(self.a_op, args=args), self.spectra, self.sec_cutoff
        )
        cache.append((self, new))
        cache.append((self.H, H))
        return new


cdef class _BlochRedfieldEigenElement(_BlochRedfieldElement):
    cpdef Data data(self, double t):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)
        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = self._br_term(A_eig, cutoff)
        return BR_eig

    cdef Data matmul_data_t(self, double t, Data state_eig, Data out=None):
        cdef size_t i
        cdef double cutoff = self.sec_cutoff * self._compute_spectrum(t)

        A_eig = self.H.to_eigbasis(t, self.a_op._call(t))
        BR_eig = self._br_term(A_eig, cutoff)
        out_eig = _data.matmul(BR_eig, state_eig)
        if out is None:
            return out_eig
        return _data.add(out_eig, out)


def brtensor(H, a_op, spectra, use_secular=True,
             sec_cutoff=0.1, fock_basis=False,
             sparse=False):
    """
    Calculates the Bloch-Redfield tensor for a system given
    a set of operators and corresponding spectral functions that describes the
    system's coupling to its environment.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        System Hamiltonian.

    a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        The operator coupling to the environment. Must be hermitian.

    spectra : :class:`Coefficient`
        The corresponding bath spectra.
        Must be a `Coefficient` using an 'w' args. The `SpectraCoefficient`
        can be used to use array based coefficient.

        Example:

            coefficient('w>0', args={"w": 0})
            SpectraCoefficient(coefficient(Cubic_Spline))

    use_secular : bool {True}
        Flag that indicates if the secular approximation should
        be used.

    sec_cutoff : float {0.1}
        Threshold for secular approximation.

    fock_basis : bool {False}
        Whether to return the tensor in the input basis or the diagonalized
        basis.

    Returns
    -------

    R, [evecs]: :class:`qutip.Qobj`, tuple of :class:`qutip.Qobj`
        If ``fock_basis``, return the Bloch Redfield tensor in the outside
        basis. Otherwise return the Bloch Redfield tensor in the diagonalized
        Hamiltonian basis and the eigenvectors of the Hamiltonian as hstacked
        column.
    """

    if isinstance(H, Qobj):
        H = QobjEvo(H)
        any_Qevo = False
    else:
        any_Qevo = True

    if isinstance(H, _EigenBasisTransform):
        Hdiag = H
    else:
        Hdiag = _EigenBasisTransform(H, sparse=sparse)

    sec_cutoff = sec_cutoff if use_secular else np.inf
    R = QobjEvo.__new__(QobjEvo)
    R.dims = [H.dims, H.dims]
    R.shape = (H.shape[0]**2, H.shape[0]**2)
    R._issuper = True
    R.elements = []
    any_Qevo = any_Qevo or isinstance(a_op, QobjEvo)
    if fock_basis:
        R.elements = [
            _BlochRedfieldElement(Hdiag, QobjEvo(a_op), spectra, sec_cutoff)
        ]
    else:
        R.elements = [
            _BlochRedfieldEigenElement(Hdiag, QobjEvo(a_op), spectra, sec_cutoff)
        ]
    if Hdiag.isconstant and isinstance(a_op, Qobj):
        R = R(0)
    return R
