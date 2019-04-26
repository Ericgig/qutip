#!python
#cython: language_level=3
import numpy as np
cimport numpy as np
cimport numpy as cnp
cimport cython
np.import_array()
from qutip.matrix.cy.cdata cimport Cdata
import qutip.settings as qset
from libcpp cimport bool
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from libc.stdlib cimport div, ldiv
from libc.math cimport abs, fabs, sqrt

# Todo:
# proper numpy_lock Checks
# ctypedef size for indices and derived variables (template?)
# factory: mixed type
# ptrace
# PyDataMem_NEW remove?  (cnp.import_array() is needed)

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

cdef extern from "stdlib.h":
    ctypedef struct div_t:
        int quot
        int rem

cdef extern from "stdlib.h":
    ctypedef struct ldiv_t:
        long quot
        long rem

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

cdef extern from "<complex>" namespace "std" nogil:
    #double abs(double complex x)
    double real(double complex x)
    double imag(double complex x)
    double complex conj(double complex x)

cdef extern from "<complex>" namespace "std" nogil:
    double cabs "abs" (double complex x)

cdef inline int int_max(int x, int y):
    return x ^ ((x ^ y) & -(x < y))

#Struct used for CSR indices sorting
cdef struct _data_ind_pair:
    double complex data
    int ind

ctypedef _data_ind_pair data_ind_pair
ctypedef int (*cfptr)(data_ind_pair, data_ind_pair)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int ind_sort(data_ind_pair x, data_ind_pair y):
    return x.ind < y.ind

cdef void raise_error_cs(int E):
    if E == -1:
        raise MemoryError('Could not allocate memory.')
    elif E == -2:
        raise Exception('Error manipulating Matrix structure.')
    elif E == -3:
        raise Exception('Matrix is not initialized.')
    elif E == -4:
        raise Exception('NumPy already has lock on data.')
    elif E == -5:
        raise Exception('Cannot expand data structures past max_length.')
    elif E == -6:
        raise Exception('Matrix cannot be expanded.')
    elif E == -7:
        raise Exception('Data length cannot be larger than max_length')
    else:
        raise Exception('Error in Cython code.')

cdef class cy_cs_matrix(Cdata):
    def __init__(self):
        self.is_set = 0
        self.nnz = 0
        self.nrows = 0
        self.ncols = 0
        self.nptrs = 0
        self.max_length = 0
        self.numpy_lock = 0
        self.data = NULL
        self.indices = NULL
        self.indptr = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void init(self, int nnz, int nrows, int ncols = 0, int nptrs = 0,
                        int max_length = 0, int init_zeros = 1, int csr = 1):
            """
            Initialize CS_Matrix struct. Matrix is assumed to be square with
            shape nrows x nrows.  Manually set mat.ncols otherwise

            Parameters
            ----------
            nnz : int
                Length of data and indices arrays. Also number of nonzero elements
            nrows : int
                Number of rows in matrix. Also gives length
                of indptr array (nrows+1).
            ncols : int (default = 0)
                Number of cols in matrix. Default is ncols = nrows.
            nptrs : int
                Number of elements in the indptr vector -1
                nrows for csr, ncols for csc
            max_length : int (default = 0)
                Maximum length of data and indices arrays.  Used for resizing.
                Default value of zero indicates no resizing.
            csr : int (bool)
                True if csr
            """
            if max_length == 0:
                max_length = nnz
            if ncols == 0:
                ncols = nrows
            if nptrs == 0:
                if csr:
                    nptrs = nrows
                else:
                    nptrs = ncols
            if nnz > max_length:
                raise_error_cs(-7)
            if init_zeros:
                self.indices = <int *>PyDataMem_NEW_ZEROED(nnz, sizeof(int))
                self.indptr = <int *>PyDataMem_NEW_ZEROED((nptrs+1), sizeof(int))
                self.data = <double complex *>PyDataMem_NEW_ZEROED(nnz, sizeof(double complex))
            else:
                self.indices = <int *>PyDataMem_NEW(nnz * sizeof(int))
                self.indptr = <int *>PyDataMem_NEW((nptrs+1) * sizeof(int))
                self.data = <double complex *>PyDataMem_NEW(nnz * sizeof(double complex))
            if self.data == NULL:
                raise_error_cs(-1)
            self.nnz = nnz
            self.nrows = nrows
            self.ncols = ncols
            self.nptrs = nptrs
            self.is_set = 1
            self.max_length = max_length
            self.numpy_lock = 0
            self.is_csr = csr != 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void free(self):
        """
        Manually free CS_Matrix data structures if
        data is not locked by NumPy.
        """
        if not self.numpy_lock and self.is_set:
            if self.data != NULL:
                PyDataMem_FREE(self.data)
            if self.indices != NULL:
                PyDataMem_FREE(self.indices)
            if self.indptr != NULL:
                PyDataMem_FREE(self.indptr)
            self.is_set = 0
        else:
            raise_error_cs(-2)

    def display(self):
        """For debug purpose"""
        print(self.nnz, self.nrows, self.ncols, self.nptrs)
        print(self.is_set, self.max_length, self.numpy_lock, self.is_csr)
        print(self.indices[0], self.data[0])
        print(self.indptr[0], self.indptr[self.nptrs])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void copy_cs(self, cy_cs_matrix mat):
        """
        Copy a cs_matrix.
        """
        cdef size_t kk
        if not mat.is_set:
            raise_error_cs(-3)
        elif self.is_set:
            self.free()
        self.init(mat.nnz, mat.nrows, mat.ncols, mat.nptrs,
                  mat.max_length, 0, mat.is_csr)
        # We cannot use memcpy here since there are issues with
        # doing so on Win with the GCC compiler
        for kk in range(mat.nnz):
            self.data[kk] = mat.data[kk]
            self.indices[kk] = mat.indices[kk]
        for kk in range(mat.nptrs+1):
            self.indptr[kk] = mat.indptr[kk]

    def __dealloc__(self):
        if not self.numpy_lock:
            if self.data != NULL:
                PyDataMem_FREE(self.data)
            if self.indices != NULL:
                PyDataMem_FREE(self.indices)
            if self.indptr != NULL:
                PyDataMem_FREE(self.indptr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def as_vecs(self):
        """
        Return a copy as numpy vector making the csr or csc matrix.
        The life of the scipy object is independant of this object.
        """
        cdef int i
        cdef cnp.ndarray[complex, ndim=1] _data
        cdef cnp.ndarray[int, ndim=1] _ind, _ptr
        if self.is_set:
            _data = np.empty(self.nnz, dtype=complex)
            _ind = np.empty(self.nnz, dtype=np.int32)
            for i in range(self.nnz):
                _data[i] = self.data[i]
                _ind[i] = self.indices[i]
            _ptr = np.empty(self.nptrs+1, dtype=np.int32)
            for i in range(self.nptrs+1):
                _ptr[i] = self.indptr[i]
            return (_data, _ind, _ptr)
        else:
            raise_error_cs(-3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _shallow_get_state(self):
        """
        Converts a cs sparse matrix to a tuples for pickling.
        A pointer to data is passed instead of the data itself.
        """
        longp_data = PyLong_FromVoidPtr(<void *>self.data)
        longp_indices = PyLong_FromVoidPtr(<void *>self.indices)
        longp_indptr = PyLong_FromVoidPtr(<void *>self.indptr)
        return (longp_data,  longp_indices,  longp_indptr,
                self.nrows, self.ncols, self.nnz, self.max_length,
                self.is_set, self.numpy_lock, self.nptrs, self.is_csr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _shallow_set_state(self, state):
        """
        Converts back a cs sparse matrix from a tuples for pickling.
        No copy of the data, pointer are passed.
        """
        self.data = <complex*>PyLong_AsVoidPtr(state[0])
        self.indices = <int*>PyLong_AsVoidPtr(state[1])
        self.indptr = <int*>PyLong_AsVoidPtr(state[2])
        self.nrows = state[3]
        self.ncols = state[4]
        self.nnz = state[5]
        self.max_length = state[6]
        self.is_set = state[7]
        self.numpy_lock = state[8]
        self.nptrs = state[9]
        self.is_csr = state[10]

    ###################################################################
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def bandwidth(self):
        """
        Calculates the max (mb), lower(lb), and upper(ub) bandwidths of a
        csr_matrix.
        """
        cdef int ldist
        cdef int lb = -self.nptrs
        cdef int ub = -self.nptrs
        cdef int mb = 0
        cdef size_t ii, jj

        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii + 1]):
                ldist = ii - self.indices[jj]
                lb = int_max(lb, ldist)
                ub = int_max(ub, -ldist)
        mb = ub + lb + 1
        return mb, lb, ub

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def profile(self):
        cdef int ii, jj, temp, ldist=0
        cdef int pro = 0
        for ii in range(self.nptrs):
            temp = 0
            for jj in range(self.indptr[ii], self.indptr[ii + 1]):
                ldist = self.indices[jj] - ii
                # faster if sorted
                temp = int_max(temp, ldist)
            pro += temp
        return pro

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def profile_full(self):
        cdef int ii, jj, temp, ldist=0
        cdef int pro = 0, proT = 0
        cdef int l_other = self.ncols if self.is_csr else self.nrows
        cdef int[::1] m_other = np.zeros(l_other)
        for ii in range(self.nptrs):
            temp = 0
            for jj in range(self.indptr[ii], self.indptr[ii + 1]):
                ldist = self.indices[jj] - ii
                m_other[self.indices[jj]] = ii
                temp = int_max(temp, ldist)
            pro += temp
        for ii in range(l_other):
            proT += int_max(0, m_other[ii]-ii)
        return pro, proT

    cpdef double one_norm(self):
        if self.is_csr:
            return self._max_sum_sec()
        else:
            return self._max_sum_main()

    cpdef double inf_norm(self):
        if self.is_csr:
            return self._max_sum_main()
        else:
            return self._max_sum_sec()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef isdiag(self):
        cdef int row, num_elems
        for row in range(self.nrows):
            num_elems = self.indptr[row+1] - self.indptr[row]
            if num_elems > 1:
                return 0
            elif num_elems == 1:
                if self.indices[self.indptr[row]] != row:
                    return 0
        return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef isherm(self, double tol = qset.atol):
        """
        Determines if a given input sparse CSR matrix is Hermitian
        to within a specified floating-point tolerance.

        Parameters
        ----------
        A : csr_matrix
            Input sparse matrix.
        tol : float (default is atol from settings)
            Desired tolerance value.

        Returns
        -------
        isherm : int
            One if matrix is Hermitian, zero otherwise.

        Notes
        -----
        This implimentation is esentially an adjoint calulation
        where the data and indices are not stored, but checked
        elementwise to see if they match those of the input matrix.
        Thus we do not need to build the actual adjoint.  Here we
        only need a temp array of output indptr.
        """

        cdef int k, nxt, isherm = 1
        cdef size_t ii, jj
        cdef complex tmp, tmp2

        if self.nrows != self.ncols:
            return 0

        cdef int * out_ptr = <int *>PyDataMem_NEW_ZEROED(self.nptrs+1, sizeof(int))

        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj] + 1
                out_ptr[k] += 1

        for ii in range(self.nptrs):
            out_ptr[ii+1] += out_ptr[ii]

        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj]
                nxt = out_ptr[k]
                out_ptr[k] += 1
                #structure test
                if self.indices[nxt] != ii:
                    isherm = 0
                    break
                tmp = conj(self.data[jj])
                tmp2 = self.data[nxt]
                #data test
                if abs(tmp-tmp2) > tol:
                    isherm = 0
                    break
            else:
                continue
            break

        PyDataMem_FREE(out_ptr)
        return isherm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex trace(self):
        cdef size_t ii, jj
        cdef complex tr = 0

        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                if self.indices[jj] == ii:
                    tr += self.data[jj]
                    break
        return tr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[complex, ndim=1, mode='c'] get_diag(self, int k=0):
        # Only work properly for square matrix
        # may pad zeros for for rectangle cases
        cdef size_t row, jj
        cdef int abs_k = abs(k)
        cdef int start, stop
        cdef cnp.ndarray[complex, ndim=1, mode='c'] out = np.zeros(self.nptrs-abs_k, dtype=complex)
        k = k if self.is_csr else -k

        if k >= 0:
            start = 0
            stop = self.nptrs-abs_k
        else: #k < 0
            start = abs_k
            stop = self.nptrs

        for row in range(start, stop):
            for jj in range(self.indptr[row], self.indptr[row+1]):
                if self.indices[jj]-k == row:
                    out[row-start] = self.data[jj]
                    break
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void transpose(self):
        """
        Adjoint of a sparse matrix in CSR format.
        """
        if self.numpy_lock:
            raise_error_cs(-4)
        cdef cy_cs_matrix B = cy_cs_matrix()
        B.init(self.nnz, self.ncols, self.nrows,
               nptrs=self.ncols if self.is_csr else self.nrows,
               max_length=self.nnz, init_zeros=1, csr=self.is_csr)

        self._zcs_trans_core(B)

        self.free()
        self.ncols = B.ncols
        self.nrows = B.nrows
        self.nptrs = B.nptrs
        self.is_csr = B.is_csr
        self.is_set = 1
        self.data = B.data
        self.indptr = B.indptr
        self.indices = B.indices
        B.numpy_lock = 1  # stop __dealloc__ from freeing ptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void adjoint(self):
        """
        Adjoint of a sparse matrix in CSR format.
        """
        if self.numpy_lock:
            raise_error_cs(-4)

        cdef cy_cs_matrix B = cy_cs_matrix()
        B.init(self.nnz, self.ncols, self.nrows,
               nptrs=self.ncols if self.is_csr else self.nrows,
               max_length=self.nnz, init_zeros=1, csr=self.is_csr)

        self._zcs_adjoint_core(B)

        self.free()
        self.ncols = B.ncols
        self.nrows = B.nrows
        self.nptrs = B.nptrs
        self.is_csr = B.is_csr
        self.is_set = 1
        self.data = B.data
        self.indptr = B.indptr
        self.indices = B.indices
        B.numpy_lock = 1  # stop __dealloc__ from freeing ptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void sparse_permute(self,
            cnp.ndarray[int, ndim=1] rperm,
            cnp.ndarray[int, ndim=1] cperm):
        """
        Permutes the rows and columns of a sparse CSR matrix according to
        the permutation arrays rperm and cperm, respectively.
        Here, the permutation arrays specify the new order of the rows and columns.
        i.e. [0,1,2,3,4] -> [3,0,4,1,2].
        """
        cdef int ii, jj, kk, k0
        cdef double complex * new_data = <double complex *>PyDataMem_NEW_ZEROED(self.nnz, sizeof(double complex))
        cdef int * new_idx = <int *>PyDataMem_NEW_ZEROED(self.nnz, sizeof(int))
        cdef int * new_ptr = <int *>PyDataMem_NEW_ZEROED((self.nrows+1), sizeof(int))
        cdef cnp.ndarray[int, ndim=1] perm_r
        cdef cnp.ndarray[int, ndim=1] perm_c
        cdef cnp.ndarray[int, ndim=1] inds

        if self.is_csr != 1:
            # csr and csc computation are the same but row/col are inverted
            rperm, cperm = cperm, rperm

        if rperm.shape[0] != 0:
            inds = np.argsort(rperm).astype(np.int32)
            perm_r = np.arange(rperm.shape[0], dtype=np.int32)[inds]

            for jj in range(self.nptrs):
                ii = perm_r[jj]
                new_ptr[ii + 1] = self.indptr[jj + 1] - self.indptr[jj]

            for jj in range(self.nptrs):
                new_ptr[jj + 1] = new_ptr[jj+1] + new_ptr[jj]

            for jj in range(self.nptrs):
                k0 = new_ptr[perm_r[jj]]
                for kk in range(self.indptr[jj], self.indptr[jj + 1]):
                    new_idx[k0] = self.indices[kk]
                    new_data[k0] = self.data[kk]
                    k0 = k0 + 1

        if cperm.shape[0] != 0:
            inds = np.argsort(cperm).astype(np.int32)
            perm_c = np.arange(cperm.shape[0], dtype=np.int32)[inds]
            for jj in range(self.nnz):
                new_idx[jj] = perm_c[new_idx[jj]]

        if self.numpy_lock:
            for jj in range(self.nnz):
                self.data[jj] = new_data[jj]
                self.indices[jj] = new_idx[jj]
            for jj in range(self.nptrs):
                self.indptr[jj] = new_ptr[jj]
            PyDataMem_FREE(new_data)
            PyDataMem_FREE(new_idx)
            PyDataMem_FREE(new_ptr)
        else:
            PyDataMem_FREE(self.data)
            PyDataMem_FREE(self.indices)
            PyDataMem_FREE(self.indptr)
            self.data = new_data
            self.indices = new_idx
            self.indptr = new_ptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void sparse_reverse_permute(self,
            cnp.ndarray[int, ndim=1] rperm,
            cnp.ndarray[int, ndim=1] cperm):
        """
        Reverse permutes the rows and columns of a sparse CSR or CSC matrix
        according to the original permutation arrays rperm and cperm, respectively.
        """
        cdef int ii, jj, kk, k0
        cdef double complex * new_data = <double complex *>PyDataMem_NEW_ZEROED(self.nnz, sizeof(double complex))
        cdef int * new_idx = <int *>PyDataMem_NEW_ZEROED(self.nnz, sizeof(int))
        cdef int * new_ptr = <int *>PyDataMem_NEW_ZEROED((self.nrows+1), sizeof(int))

        if self.is_csr != 1:
            # csr and csc computation are the same but row/col are inverted
            rperm, cperm = cperm, rperm

        if rperm.shape[0] != 0:
            for jj in range(self.nrows):
                ii = rperm[jj]
                new_ptr[ii + 1] = self.indptr[jj + 1] - self.indptr[jj]

            for jj in range(self.nrows):
                new_ptr[jj + 1] = new_ptr[jj + 1] + new_ptr[jj]

            for jj in range(self.nrows):
                k0 = new_ptr[rperm[jj]]
                for kk in range(self.indptr[jj], self.indptr[jj + 1]):
                    new_idx[k0] = self.indices[kk]
                    new_data[k0] = self.data[kk]
                    k0 = k0 + 1

        if cperm.shape[0] > 0:
            for jj in range(self.nnz):
                new_idx[jj] = cperm[new_idx[jj]]

        if self.numpy_lock:
            for jj in range(self.nnz):
                self.data[jj] = new_data[jj]
                self.indices[jj] = new_idx[jj]
            for jj in range(self.nptrs):
                self.indptr[jj] = new_ptr[jj]
            PyDataMem_FREE(new_data)
            PyDataMem_FREE(new_idx)
            PyDataMem_FREE(new_ptr)
        else:
            PyDataMem_FREE(self.data)
            PyDataMem_FREE(self.indices)
            PyDataMem_FREE(self.indptr)
            self.data = new_data
            self.indices = new_idx
            self.indptr = new_ptr

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # private fucntions
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _shorten(self, int N):
        """
        Shortends the length of CSR data and indices arrays.
        """
        if (not self.numpy_lock) and self.is_set:
            self.data = <double complex *>PyDataMem_RENEW(self.data, N * sizeof(double complex))
            self.indices = <int *>PyDataMem_RENEW(self.indices, N * sizeof(int))
            self.nnz = N
        else:
            if self.numpy_lock:
                raise_error_cs(-4)
            elif not self.is_set:
                raise_error_cs(-3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _sort_indices(self):
        """
        Sorts the indices of a CS_Matrix inplace.
        """
        cdef size_t ii, jj
        cdef vector[data_ind_pair] pairs
        cdef cfptr cfptr_ = &ind_sort
        cdef int row_start, row_end, length

        for ii in range(self.nptrs):
            row_start = self.indptr[ii]
            row_end = self.indptr[ii+1]
            length = row_end - row_start
            pairs.resize(length)

            for jj in range(length):
                pairs[jj].data = self.data[row_start+jj]
                pairs[jj].ind = self.indices[row_start+jj]

            sort(pairs.begin(), pairs.end(), cfptr_)

            for jj in range(length):
                self.data[row_start+jj] = pairs[jj].data
                self.indices[row_start+jj] = pairs[jj].ind

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _max_sum_sec(self):
        cdef int k, other_shape
        cdef size_t ii, jj
        other_shape = self.ncols if self.is_csr else self.nrows
        cdef double * col_sum = <double *>PyDataMem_NEW_ZEROED(other_shape, sizeof(double))
        cdef double max_col = 0
        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj]
                col_sum[k] += cabs(self.data[jj])
        for ii in range(other_shape):
            if col_sum[ii] > max_col:
                max_col = col_sum[ii]
        PyDataMem_FREE(col_sum)
        return max_col

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _max_sum_main(self):
        cdef int k
        cdef size_t ii, jj
        cdef double * row_sum = <double *>PyDataMem_NEW_ZEROED(self.nptrs, sizeof(double))
        cdef double max_row = 0
        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                row_sum[ii] += cabs(self.data[jj])
        for ii in range(self.nptrs):
            if row_sum[ii] > max_row:
                max_row = row_sum[ii]
        PyDataMem_FREE(row_sum)
        return max_row

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void _zcs_trans_core(self, cy_cs_matrix out):# nogil:
        cdef int k, nxt, other_shape
        cdef size_t ii, jj
        other_shape = self.ncols if self.is_csr else self.nrows

        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj] + 1
                out.indptr[k] += 1

        for ii in range(other_shape):
            out.indptr[ii+1] += out.indptr[ii]

        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj]
                nxt = out.indptr[k]
                out.data[nxt] = self.data[jj]
                out.indices[nxt] = ii
                out.indptr[k] = nxt + 1

        for ii in range(other_shape,0,-1):
            out.indptr[ii] = out.indptr[ii-1]

        out.indptr[0] = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void _zcs_adjoint_core(self, cy_cs_matrix out):# nogil:
        cdef int k, nxt, other_shape
        cdef size_t ii, jj
        other_shape = self.ncols if self.is_csr else self.nrows

        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj] + 1
                out.indptr[k] += 1

        for ii in range(other_shape):
            out.indptr[ii+1] += out.indptr[ii]

        for ii in range(self.nptrs):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj]
                nxt = out.indptr[k]
                out.data[nxt] = conj(self.data[jj])
                out.indices[nxt] = ii
                out.indptr[k] = nxt + 1

        for ii in range(other_shape,0,-1):
            out.indptr[ii] = out.indptr[ii-1]

        out.indptr[0] = 0
