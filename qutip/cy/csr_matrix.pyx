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

# Todo:
# proper numpy_lock Checks
# ctypedef size for indices and derived variables (template?)
# factory: mixed type
# ptrace
# PyDataMem_NEW remove?  (cnp.import_array() is needed)

#include "parameters.pxi"
DTYPE = np.float64
ITYPE = np.int32
CTYPE = np.complex128
CTYPE = np.int64

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

cdef extern from "src/zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows)


cdef inline void spmvpy(complex * data, int * ind, int * ptr,
            complex * vec,
            complex a,
            complex * out,
            unsigned int nrows):

    zspmvpy(data, ind, ptr, vec, a, out, nrows)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _spmm_c_py(complex* data, int* ind, int* ptr,
            complex* mat, complex a, complex* out,
            unsigned int sp_rows, unsigned int nrows, unsigned int ncols):
    """
    sparse*dense "C" ordered.
    """
    cdef int row, col, ii, jj, row_start, row_end
    for row from 0 <= row < sp_rows :
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            for col in range(ncols):
                out[row * ncols + col] += a*data[jj]*mat[ind[jj] * ncols + col]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _spmm_f_py(complex* data, int* ind, int* ptr,
            complex* mat, complex a, complex* out,
            unsigned int sp_rows, unsigned int nrows, unsigned int ncols):
    """
    sparse*dense "F" ordered.
    """
    cdef int col
    for col in range(ncols):
        spmvpy(data, ind, ptr, mat+nrows*col, a, out+sp_rows*col, sp_rows)


# val in vec in pure cython
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _in(int val, int[::1] vec):
    cdef int ii
    for ii in range(vec.shape[0]):
        if val == vec[ii]:
            return 1
    return 0

# indices determining function for ptrace
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void i2_k_t(int N,
                 int[:, ::1] tensor_table,
                 int[::1] out):
    cdef int ii, t1, t2
    out[0] = 0
    out[1] = 0
    for ii in range(rho2ten.shape[0]):
        t1 = tensor_table[0, ii]
        t2 = N / t1
        N = N % t1
        out[0] += tensor_table[1, ii] * t2
        out[1] += tensor_table[2, ii] * t2

cdef void raise_error_CSR(int E):
    if E == -1:
        raise MemoryError('Could not allocate memory.')
    elif E == -2:
        raise Exception('Error manipulating CSR_Matrix structure.')
    elif E == -3:
        raise Exception('CSR_Matrix is not initialized.')
    elif E == -4:
        raise Exception('NumPy already has lock on data.')
    elif E == -5:
        raise Exception('Cannot expand data structures past max_length.')
    elif E == -6:
        raise Exception('CSR_Matrix cannot be expanded.')
    elif E == -7:
        raise Exception('Data length cannot be larger than max_length')
    else:
        raise Exception('Error in Cython code.')


cdef class cy_csr_matrix:

    def __init__(self):
        self.is_set = 0
        self.nnz = 0
        self.nrows = 0
        self.ncols = 0
        self.max_length = 0
        self.numpy_lock = 0
        self.data = NULL
        self.indices = NULL
        self.indptr = NULL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void init(self, int nnz, int nrows, int ncols = 0,
                        int max_length = 0, int init_zeros = 1):
            """
            Initialize CSR_Matrix struct. Matrix is assumed to be square with
            shape nrows x nrows.  Manually set mat.ncols otherwise

            Parameters
            ----------
            mat : CSR_Matrix *
                Pointer to struct.
            nnz : int
                Length of data and indices arrays. Also number of nonzero elements
            nrows : int
                Number of rows in matrix. Also gives length
                of indptr array (nrows+1).
            ncols : int (default = 0)
                Number of cols in matrix. Default is ncols = nrows.
            max_length : int (default = 0)
                Maximum length of data and indices arrays.  Used for resizing.
                Default value of zero indicates no resizing.
            """
            if max_length == 0:
                max_length = nnz
            if nnz > max_length:
                raise_error_CSR(-7)
            if init_zeros:
                self.indices = <int *>PyDataMem_NEW_ZEROED(nnz, sizeof(int))
                self.indptr = <int *>PyDataMem_NEW_ZEROED((nrows+1), sizeof(int))
                self.data = <double complex *>PyDataMem_NEW_ZEROED(nnz, sizeof(double complex))
            else:
                self.indices = <int *>PyDataMem_NEW(nnz * sizeof(int))
                self.indptr = <int *>PyDataMem_NEW((nrows+1) * sizeof(int))
                self.data = <double complex *>PyDataMem_NEW(nnz * sizeof(double complex))
            if self.data == NULL:
                raise_error_CSR(-1)
            self.nnz = nnz
            self.nrows = nrows
            if ncols == 0:
                self.ncols = nrows
            else:
                self.ncols = ncols
            self.is_set = 1
            self.max_length = max_length
            self.numpy_lock = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void copy_CSR(self, cy_csr_matrix mat):
        """
        Copy a CSR_Matrix.
        """
        cdef size_t kk
        if not mat.is_set:
            raise_error_CSR(-3)
        elif self.is_set:
            self.free()
        self.init(mat.nnz, mat.nrows, mat.nrows, mat.max_length)
        # We cannot use memcpy here since there are issues with
        # doing so on Win with the GCC compiler
        for kk in range(mat.nnz):
            self.data[kk] = mat.data[kk]
            self.indices[kk] = mat.indices[kk]
        for kk in range(mat.nrows+1):
            self.indptr[kk] = mat.indptr[kk]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cy_csr_matrix copy(self):
        """
        Return a copy.
        """
        cdef cy_csr_matrix out = cy_csr_matrix()
        out.copy_CSR(self)
        return out

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
    cdef void free(self):
        """
        Manually free CSR_Matrix data structures if
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
            raise_error_CSR(-2)

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
                raise_error_CSR(-4)
            elif not self.is_set:
                raise_error_CSR(-3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def to_scipy(self):
        """
        Converts a CSR_Matrix struct to a SciPy csr_matrix class object.
        The NumPy arrays are generated from the pointers, and the lifetime
        of the pointer memory is tied to that of the NumPy array
        (i.e. automatic garbage cleanup.)
        """
        cdef np.npy_intp dat_len, ptr_len
        cdef np.ndarray[complex, ndim=1] _data
        cdef np.ndarray[int, ndim=1] _ind, _ptr
        if (not self.numpy_lock) and self.is_set:
            dat_len = self.nnz
            ptr_len = self.nrows+1
            _data = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_COMPLEX128, self.data)
            PyArray_ENABLEFLAGS(_data, np.NPY_OWNDATA)

            _ind = np.PyArray_SimpleNewFromData(1, &dat_len, np.NPY_INT32, self.indices)
            PyArray_ENABLEFLAGS(_ind, np.NPY_OWNDATA)

            _ptr = np.PyArray_SimpleNewFromData(1, &ptr_len, np.NPY_INT32, self.indptr)
            PyArray_ENABLEFLAGS(_ptr, np.NPY_OWNDATA)
            self.numpy_lock = 1
            return fast_csr_matrix((_data, _ind, _ptr), shape=(self.nrows, self.ncols))
        else:
            if self.numpy_lock:
                raise_error_CSR(-4)
            elif not self.is_set:
                raise_error_CSR(-3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def scipy(self):
        """
        Return a copy as a SciPy csr_matrix class object.
        The life of the scipy object is independant of this object.
        """
        cdef int i
        cdef np.ndarray[complex, ndim=1] _data
        cdef np.ndarray[int, ndim=1] _ind, _ptr
        if self.is_set:
            _data = np.empty(self.nnz, dtype=complex)
            _ind = np.empty(self.nnz, dtype=np.int32)
            for i in range(self.nnz):
                _data[i] = self.data[i]
                _ind[i] = self.indices[i]
            _ptr = np.empty(self.nrows+1, dtype=np.int32)
            for i in range(self.nrows+1):
                _ptr[i] = self.indptr[i]
            return fast_csr_matrix((_data, _ind, _ptr), shape=(self.nrows, self.ncols))
        else:
            raise_error_CSR(-3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def csr(self):
        """
        Return a copy as a SciPy csr_matrix class object.
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
            _ptr = np.empty(self.nrows+1, dtype=np.int32)
            for i in range(self.nrows+1):
                _ptr[i] = self.indptr[i]
            return (_data, _ind, _ptr)
        else:
            raise_error_CSR(-3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _sort_indices(self):
        """
        Sorts the indices of a CSR_Matrix inplace.
        """
        cdef size_t ii, jj
        cdef vector[data_ind_pair] pairs
        cdef cfptr cfptr_ = &ind_sort
        cdef int row_start, row_end, length

        for ii in range(self.nrows):
            row_start = self.indptr[ii]
            row_end = self.indptr[ii+1]
            length = row_end - row_start
            pairs.resize(length)

            for jj in range(length):
                pairs[jj].data = self.data[row_start+jj]
                pairs[jj].ind = self.indices[row_start+jj]

            sort(pairs.begin(),pairs.end(),cfptr_)

            for jj in range(length):
                self.data[row_start+jj] = pairs[jj].data
                self.indices[row_start+jj] = pairs[jj].ind

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void _coo_indices(self, int[::1] rows, int[::1] cols):
        """
        out: cols, rows of the data
        """
        cdef int k1, k2
        cdef size_t jj, kk
        for kk in range(self.nnz):
            cols[kk] = self.indices[kk]
        for kk in range(0,self.nrows):
            k1 = self.indptr[kk+1]
            k2 = self.indptr[kk]
            for jj in range(k2, k1):
                rows[jj] = kk

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    cpdef void _from_coo_indices(self, int[::1] rows, int[::1] cols):
        """
        rebuild ind and ptr from cols and rows
        data is inplace
        """
        cdef size_t kk
        cdef int i, j, init, inext, jnext, ipos, nn
        cdef complex val, val_next
        cdef int * work = <int *>PyDataMem_NEW_ZEROED(self.nrows+1, sizeof(int))
        # Determine output indptr array
        for kk in range(self.nnz):
            i = rows[kk]
            work[i+1] += 1
        work[0] = 0
        self.indptr[0] = 0
        for kk in range(self.nrows):
            work[kk+1] += work[kk]
            self.indptr[kk+1] = work[kk+1]
        init = 0
        while init < self.nnz:
            if (rows[init] < 0):
                init += 1
                continue
            val = self.data[init]
            i = rows[init]
            j = cols[init]
            rows[init] = -1
            while 1:
                ipos = work[i]
                val_next = self.data[ipos]
                inext = rows[ipos]
                jnext = cols[ipos]

                self.data[ipos] = val
                self.indices[ipos] = j
                rows[ipos] = -1
                work[i] += 1
                if inext < 0:
                    break
                val = val_next
                i = inext
                j = jnext
            init += 1

        #Free working array
        PyDataMem_FREE(work)

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    cpdef reshape(self, int new_rows, int new_cols):
        """
        Reshapes a complex CSR matrix.

        Parameters
        ----------
        A : fast_csr_matrix
            Input CSR matrix.
        new_rows : int
            Number of rows in reshaped matrix.
        new_cols : int
            Number of cols in reshaped matrix.

        Returns
        -------
        out : fast_csr_matrix
            Reshaped CSR matrix.

        Notes
        -----
        This routine does not need to make a temp. copy of the matrix.
        """
        if self.numpy_lock:
            raise Exception

        cdef int[::1] rows = np.empty(self.nnz, dtype=np.int32)
        cdef int[::1] cols = np.empty(self.nnz, dtype=np.int32)
        cdef div_t new_inds
        cdef size_t kk

        if (self.nrows * self.ncols) != (new_rows * new_cols):
            raise Exception('Total size of array must be unchanged.')

        self._coo_indices(rows, cols)
        for kk in range(self.nnz):
            new_inds = div(self.ncols*rows[kk]+cols[kk], new_cols)
            rows[kk] = new_inds.quot
            cols[kk] = new_inds.rem

        self.nrows = new_rows
        self.ncols = new_cols
        self.indptr = <int *>PyDataMem_RENEW(self.indptr, (new_rows+1) * sizeof(int))
        self._from_coo_indices(rows, cols)
        self._sort_indices()

    ###################################################################
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _sparse_bandwidth(self):
        """
        Calculates the max (mb), lower(lb), and upper(ub) bandwidths of a
        csr_matrix.
        """
        cdef int ldist
        cdef int lb = -self.nrows
        cdef int ub = -self.nrows
        cdef int mb = 0
        cdef size_t ii, jj

        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii + 1]):
                ldist = ii - self.indices[jj]
                lb = int_max(lb, ldist)
                ub = int_max(ub, -ldist)
        mb = ub + lb + 1
        return mb, lb, ub


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _sparse_profile(self):
        cdef int ii, jj, temp, ldist=0
        cdef LTYPE_t pro = 0
        for ii in range(self.nrows):
            temp = 0
            for jj in range(self.indptr[ii], self.indptr[ii + 1]):
                ldist = self.indices[jj] - ii
                temp = int_max(temp, ldist)
            pro += temp
        return pro


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void _sparse_permute(self,
            cnp.ndarray[ITYPE_t, ndim=1] rperm,
            cnp.ndarray[ITYPE_t, ndim=1] cperm):
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
        cdef cnp.ndarray[ITYPE_t, ndim=1] perm_r
        cdef cnp.ndarray[ITYPE_t, ndim=1] perm_c
        cdef cnp.ndarray[ITYPE_t, ndim=1] inds

        if rperm.shape[0] != 0:
            inds = np.argsort(rperm).astype(ITYPE)
            perm_r = np.arange(rperm.shape[0], dtype=ITYPE)[inds]

            for jj in range(self.nrows):
                ii = perm_r[jj]
                new_ptr[ii + 1] = self.indptr[jj + 1] - self.indptr[jj]

            for jj in range(self.nrows):
                new_ptr[jj + 1] = new_ptr[jj+1] + new_ptr[jj]

            for jj in range(self.nrows):
                k0 = new_ptr[perm_r[jj]]
                for kk in range(self.indptr[jj], self.indptr[jj + 1]):
                    new_idx[k0] = self.indices[kk]
                    new_data[k0] = self.data[kk]
                    k0 = k0 + 1

        if cperm.shape[0] != 0:
            inds = np.argsort(cperm).astype(ITYPE)
            perm_c = np.arange(cperm.shape[0], dtype=ITYPE)[inds]
            for jj in range(self.nnz):
                new_idx[jj] = perm_c[new_idx[jj]]

        PyDataMem_FREE(self.data)
        PyDataMem_FREE(self.indices)
        PyDataMem_FREE(self.indptr)
        self.data = new_data
        self.indices = new_idx
        self.indptr = new_ptr


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _sparse_reverse_permute(self,
            cnp.ndarray[ITYPE_t, ndim=1] rperm,
            cnp.ndarray[ITYPE_t, ndim=1] cperm):
        """
        Reverse permutes the rows and columns of a sparse CSR or CSC matrix
        according to the original permutation arrays rperm and cperm, respectively.
        """
        cdef int ii, jj, kk, k0
        cdef double complex * new_data = <double complex *>PyDataMem_NEW_ZEROED(self.nnz, sizeof(double complex))
        cdef int * new_idx = <int *>PyDataMem_NEW_ZEROED(self.nnz, sizeof(int))
        cdef int * new_ptr = <int *>PyDataMem_NEW_ZEROED((self.nrows+1), sizeof(int))

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

        PyDataMem_FREE(self.data)
        PyDataMem_FREE(self.indices)
        PyDataMem_FREE(self.indptr)
        self.data = new_data
        self.indices = new_idx
        self.indptr = new_ptr


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _isdiag(self):

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
    cpdef cnp.ndarray[complex, ndim=1, mode='c'] get_diag(self, int k=0):

        cdef size_t row, jj
        cdef int abs_k = abs(k)
        cdef int start, stop
        cdef cnp.ndarray[complex, ndim=1, mode='c'] out = np.zeros(self.nrows-abs_k, dtype=complex)

        if k >= 0:
            start = 0
            stop = self.nrows-abs_k
        else: #k < 0
            start = abs_k
            stop = self.nrows

        for row in range(start, stop):
            for jj in range(self.indptr[row], self.indptr[row+1]):
                if self.indices[jj]-k == row:
                    out[row-start] = self.data[jj]
                    break
        return out


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def unit_row_norm(self):
        cdef size_t row, ii
        cdef double total
        for row in range(self.nrows):
            total = 0
            for ii in range(self.indptr[row], self.indptr[row+1]):
                total += real(self.data[ii]) * real(self.data[ii]) + \
                         imag(self.data[ii]) * imag(self.data[ii])
            total = sqrt(total)
            for ii in range(self.indptr[row], self.indptr[row+1]):
                self.data[ii] /= total


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double zcsr_one_norm(self):
        cdef int k
        cdef size_t ii, jj
        cdef double * col_sum = <double *>PyDataMem_NEW_ZEROED(self.ncols, sizeof(double))
        cdef double max_col = 0
        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj]
                col_sum[k] += cabs(self.data[jj])
        for ii in range(self.ncols):
            if col_sum[ii] > max_col:
                max_col = col_sum[ii]
        PyDataMem_FREE(col_sum)
        return max_col


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double zcsr_inf_norm(self):
        cdef int k
        cdef size_t ii, jj
        cdef double * row_sum = <double *>PyDataMem_NEW_ZEROED(self.nrows, sizeof(double))
        cdef double max_row = 0
        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                row_sum[ii] += cabs(self.data[jj])
        for ii in range(self.nrows):
            if row_sum[ii] > max_row:
                max_row = row_sum[ii]
        PyDataMem_FREE(row_sum)
        return max_row


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bool cy_tidyup(self, double atol):
        """
        Performs an in-place tidyup of CSR matrix data
        """
        cdef size_t kk
        cdef double re, im
        cdef bool re_flag, im_flag, out_flag = 0
        for kk in range(self.nnz):
            re_flag = 0
            im_flag = 0
            re = real(self.data[kk])
            im = imag(self.data[kk])
            if fabs(re) < atol:
                re = 0
                re_flag = 1
            if fabs(im) < atol:
                im = 0
                im_flag = 1

            if re_flag or im_flag:
                self.data[kk] = re + 1j*im

            if re_flag and im_flag:
                out_flag = 1
        return out_flag

    ############################################################

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void transpose(self):
        """
        Adjoint of a sparse matrix in CSR format.
        """
        cdef cy_csr_matrix B = cy_csr_matrix()
        B.init(self.nnz, self.ncols, self.nrows)

        self._zcsr_trans_core(B)

        self.free()
        self.ncols = B.ncols
        self.nrows = B.nrows
        self.is_set = 1
        self.data = B.data
        self.indptr = B.indptr
        self.indices = B.indices
        B.numpy_lock = 1  # stop __dealloc__ from freeing ptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _zcsr_trans_core(self, cy_csr_matrix out) nogil:
        cdef int k, nxt
        cdef size_t ii, jj

        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj] + 1
                out.indptr[k] += 1

        for ii in range(self.ncols):
            out.indptr[ii+1] += out.indptr[ii]

        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj]
                nxt = out.indptr[k]
                out.data[nxt] = self.data[jj]
                out.indices[nxt] = ii
                out.indptr[k] = nxt + 1

        for ii in range(self.ncols,0,-1):
            out.indptr[ii] = out.indptr[ii-1]

        out.indptr[0] = 0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void adjoint(self):
        """
        Adjoint of a sparse matrix in CSR format.
        """
        cdef cy_csr_matrix B = cy_csr_matrix()
        B.init(self.nnz, self.ncols, self.nrows)

        self._zcsr_adjoint_core(B)

        self.free()
        self.ncols = B.ncols
        self.nrows = B.nrows
        self.is_set = 1
        self.data = B.data
        self.indptr = B.indptr
        self.indices = B.indices
        B.numpy_lock = 1  # stop __dealloc__ from freeing ptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _zcsr_adjoint_core(self, cy_csr_matrix out) nogil:

        cdef int k, nxt
        cdef size_t ii, jj

        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj] + 1
                out.indptr[k] += 1

        for ii in range(self.ncols):
            out.indptr[ii+1] += out.indptr[ii]

        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj]
                nxt = out.indptr[k]
                out.data[nxt] = conj(self.data[jj])
                out.indices[nxt] = ii
                out.indptr[k] = nxt + 1

        for ii in range(self.ncols,0,-1):
            out.indptr[ii] = out.indptr[ii-1]

        out.indptr[0] = 0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def isherm(self, double tol = qset.atol):
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

        cdef int * out_ptr = <int *>PyDataMem_NEW_ZEROED(self.ncols+1, sizeof(int))

        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                k = self.indices[jj] + 1
                out_ptr[k] += 1

        for ii in range(self.nrows):
            out_ptr[ii+1] += out_ptr[ii]

        for ii in range(self.nrows):
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
    cpdef cy_csr_matrix proj(self):
        """
        Computes the projection operator
        from a given ket or bra vector
        in CSR format.

        This is ~3x faster than doing the
        conjugate transpose and sparse multiplication
        directly.  Also, does not need a temp matrix.
        """
        cdef int offset = 0, new_idx, count, change_idx
        cdef size_t jj, kk
        cdef cy_csr_matrix out = cy_csr_matrix()
        out.init(self.nnz**2, self.nrows)

        if self.ncols == 1: # is_ket:
            #Compute new ptrs and inds
            for jj in range(self.nrows):
                out.indptr[jj] = self.indptr[jj]*self.nnz
                if self.indptr[jj+1] != self.indptr[jj]:
                    new_idx = jj
                    for kk in range(self.nnz):
                        out.indices[offset+kk*self.nnz] = new_idx
                    offset += 1
            #set nnz in new ptr
            out.indptr[self.nrows] = self.nnz**2

            #Compute the data
            for jj in range(self.nnz):
                for kk in range(self.nnz):
                    out.data[jj*self.nnz+kk] = self.data[jj]*conj(self.data[kk])
        else:
            count = self.nnz**2
            new_idx = self.nrows
            for kk in range(self.nnz-1,-1,-1):
                for jj in range(self.nnz-1,-1,-1):
                    out.indices[offset+jj] = self.indices[jj]
                    out.data[kk*self.nnz+jj] = conj(self.data[kk])*self.data[jj]
                offset += self.nnz
                change_idx = self.indices[kk]
                while new_idx > change_idx:
                    out.indptr[new_idx] = count
                    new_idx -= 1
                count -= self.nnz

        return out

    @cython.boundscheck(False)
    @cython.wraparound(False) ##
    cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv(
            self,
            complex[::1] vec):
        """
        Sparse matrix, dense vector multiplication.
        Here the vector is assumed to have one-dimension.
        Matrix must be in CSR format and have complex entries.

        Parameters
        ----------
        op : csr matrix
        vec : array
            Dense vector for multiplication.  Must be one-dimensional.

        Returns
        -------
        out : array
            Returns dense array.

        """
        cdef cnp.ndarray[complex, ndim=1, mode="c"] out = np.zeros((op.nrows), dtype=np.complex)
        zspmvpy(self.data, self.indices, self.indptr, &vec[0], 1.0, &out[0], self.nrows)
        return out


    @cython.boundscheck(False)
    @cython.wraparound(False) ##
    cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmvpy(
            self,
            complex[::1] vec, complex[::1] out, complex alpha):
        """
        Sparse matrix, dense vector multiplication.
        Here the vector is assumed to have one-dimension.
        Matrix must be in CSR format and have complex entries.

        Parameters
        ----------
        op : csr matrix
        vec : array
            Dense vector for multiplication.  Must be one-dimensional.

        Returns
        -------
        out : array
            Returns dense array.

        """
        zspmvpy(self.data, self.indices, self.indptr, &vec[0], alpha, &out[0], self.nrows)

    @cython.boundscheck(False)
    @cython.wraparound(False) ##
    cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmvpy(
            self,
            complex[::1] vec, c):
        """
        Sparse matrix, dense vector multiplication.
        Here the vector is assumed to have one-dimension.
        Matrix must be in CSR format and have complex entries.

        Parameters
        ----------
        op : csr matrix
        vec : array
            Dense vector for multiplication.  Must be one-dimensional.

        Returns
        -------
        out : array
            Returns dense array.

        """
        cdef cnp.ndarray[complex, ndim=1, mode="c"] out = np.zeros((op.nrows), dtype=np.complex)
        zspmvpy(self.data, self.indices, self.indptr, &vec[0], 1.0, &out[0], self.nrows)
        return out

    cpdef cnp.ndarray[complex, ndim=2] spmm(
            self, cnp.ndarray[complex, ndim=2] mat):
    if mat.flags["F_CONTIGUOUS"]:
        return self.spmmf(mat)
    else:
        return self.spmmc(mat)

    cpdef cnp.ndarray[complex, ndim=2] spmmf(self, cnp.ndarray[complex, ndim=2] mat):
        cdef cnp.ndarray[complex, ndim=2, mode="fortran"] out = \
                          np.zeros((sp_rows, ncols), dtype=complex, order="F")
        _spmm_f_py(self.data, self.indices, self.indptr,
                   &mat[0,0], (1.0+0j), &out[0,0],
                   self.nrows, self.ncols, mat.shape[1])
        return out

    cpdef cnp.ndarray[complex, ndim=2] spmmc(self, cnp.ndarray[complex, ndim=2] mat):
        cdef cnp.ndarray[complex, ndim=2, mode="c"] out = \
                          np.zeros((sp_rows, ncols), dtype=complex)
        _spmm_c_py(self.data, self.indices, self.indptr,
                   &mat[0,0], (1.0+0j), &out[0,0],
                   self.nrows, self.ncols, mat.shape[1])
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex expect_rho_vec(self, complex[::1] rho_vec):
        cdef size_t row
        cdef int jj, row_start, row_end
        cdef int num_rows = rho_vec.shape[0]
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0

        for row from 0 <= row < num_rows by n+1:
            row_start = self.indptr[row]
            row_end = self.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.data[jj]*rho_vec[self.indices[jj]]

        return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex expect_psi_vec(self, complex[::1] vec):
        cdef size_t row, jj
        cdef int nrows = vec.shape[0]
        cdef complex expt = 0, temp, cval

        for row in range(nrows):
            cval = conj(vec[row])
            temp = 0
            for jj in range(self.indptr[row], self.indptr[row+1]):
                temp += self.data[jj]*vec[self.indices[jj]]
            expt += cval*temp

        return expt

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def trace(self, bool isherm=0):
        #cdef complex[::1] data = A.data
        #cdef int[::1] ind = A.indices
        #cdef int[::1] ptr = A.indptr
        #cdef int nrows = ptr.shape[0]-1
        cdef size_t ii, jj
        cdef complex tr = 0

        for ii in range(self.nrows):
            for jj in range(self.indptr[ii], self.indptr[ii+1]):
                if self.indices[jj] == ii:
                    tr += self.data[jj]
                    break
        if imag(tr) == 0 or isherm:
            return real(tr)
        else:
            return tr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef object ptrace(self, obejct dims, object sel): # work for N<= 26 on 16G Ram
        cdef int[::1] _sel
        cdef cy_csr_matrix _oper
        cdef size_t ii
        cdef size_t factor_keep = 1, factor_trace = 1, factor_tensor = 1
        cdef int[:, ::1] tensor_table = np.zeros((3, num_dims), dtype=np.int32)
        cdef int[::1] drho = np.asarray(dims[0], dtype=np.int32).ravel()
        cdef int num_dims = drho.shape[0]

        if isinstance(sel, int):
            _sel = np.array([sel], dtype=np.int32)
        else:
            _sel = np.asarray(sel, dtype=np.int32)

        for ii in range(_sel.shape[0]):
            if _sel[ii] < 0 or _sel[ii] >= num_dims:
                raise TypeError("Invalid selection index in ptrace.")

        if np.prod(self.ncols) == 1:
            _oper = self.proj()
        else:
            _oper = self

        for ii in range(num_dims-1,-1,-1):
            tensor_table[0, d] = factor_tensor
            factor_tensor *= drho[d]
            if _in(ii, _sel):
                tensor_table[1, ii] = factor_keep
                factor_keep *= drho[ii]
            else:
                tensor_table[2, ii] = factor_trace
                factor_trace *= drho[ii]

        dims_kept0 = np.asarray(rho.dims[0]).take(_sel).tolist()
        rho1_dims = [dims_kept0, dims_kept0]

        # Try to evaluate how sparse the result will be.
        if factor_keep*factor_keep > _oper.nnz:
            return _oper._ptrace_core_sp(tensor_table), rho1_dims
        else:
            return _oper._ptrace_core_dense(tensor_table, factor_keep), rho1_dims

    cdef cy_csr_matrix _ptrace_core_sp(self, int[:, ::1] tensor_table):
        cdef int p = 0, nnz, ii
        cdef int[::1] pos_c = np.empty(2, dtype=np.int32)
        cdef int[::1] pos_r = np.empty(2, dtype=np.int32)
        cdef int[::1] col = np.empty(_oper.nnz, dtype=np.int32)
        cdef int[::1] row = np.empty(_oper.nnz, dtype=np.int32)
        cdef cnp.ndarray[complex, ndim=1, mode='c'] new_data = np.zeros(cco.nnz, dtype=complex)
        cdef cnp.ndarray[int, ndim=1, mode='c'] new_col = np.zeros(cco.nnz, dtype=np.int32)
        cdef cnp.ndarray[int, ndim=1, mode='c'] new_row = np.zeros(cco.nnz, dtype=np.int32)

        nnz = _oper.nnz
        _coo_indices(_oper, row, col)

        for ii in range(nnz):
            i2_k_t(col[ii], tensor_table, pos_c)
            i2_k_t(row[ii], tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                new_data[p] = self.data[i]
                new_col[p] = (pos_c[0])
                new_row[p] = (pos_r[0])
                p += 1

        # Here there can be redundance in (new_col, new_row) pairs.
        # scipy coo_matrix does the sum but _from_coo_indices does not.
        cdef object out = coo_matrix((new_data, [new_col, new_row])).tocsr()
        return CSR_from_scipy(out, False)

    cdef cy_csr_matrix _ptrace_core_dense(self, int[:, ::1] tensor_table, int num_sel_dims):
        cdef int nnz, ii
        cdef int[::1] pos_c = np.empty(2, dtype=np.int32)
        cdef int[::1] pos_r = np.empty(2, dtype=np.int32)
        cdef int[::1] col = np.empty(_oper.nnz, dtype=np.int32)
        cdef int[::1] row = np.empty(_oper.nnz, dtype=np.int32)

        cdef complex[:, ::1] data = np.zeros((num_sel_dims, num_sel_dims),
                                              dtype=complex)
        nnz = _oper.nnz
        _coo_indices(_oper, row, col)

        for ii in range(nnz):
            i2_k_t(col[ii], tensor_table, pos_c)
            i2_k_t(row[ii], tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                data[pos_r[0], pos_c[0]] += self.data[p]
                p += 1

        return dense2D_to_CSR(data)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_csr_matrix CSR_from_scipy(object A, copy=False):
    """
    Converts a SciPy CSR sparse matrix to a
    CSR_Matrix struct.
    """
    cdef complex[::1] data = A.data
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr = A.indptr
    cdef int i
    cdef cy_csr_matrix mat = cy_csr_matrix.__new__(cy_csr_matrix)
    mat.nrows = A.shape[0]
    mat.ncols = A.shape[1]
    mat.nnz = ptr[mat.nrows]
    mat.max_length = mat.nnz
    mat.is_set = 1
    if copy:
        mat.indices = <int *>PyDataMem_NEW(mat.nnz * sizeof(int))
        mat.indptr = <int *>PyDataMem_NEW((mat.nrows+1) * sizeof(int))
        mat.data = <double complex *>PyDataMem_NEW(mat.nnz * sizeof(double complex))
        for i in range(mat.nnz):
            mat.indices[i] = ind[i]
            mat.data[i] = data[i]
        for i in range(mat.nrows+1):
            mat.indptr[i] = ptr[i]
        mat.numpy_lock = 0
    else:
        mat.data = &data[0]
        mat.indices = &ind[0]
        mat.indptr = &ptr[0]
        mat.numpy_lock = 1
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_csr_matrix dense2D_to_CSR(complex[:, :] mat):
    """
    Converts a dense complex ndarray to a CSR matrix struct.

    Parameters
    ----------
    mat : ndarray
        Input complex ndarray

    Returns
    -------
    out : cy_csr_matrix
        Output matrix as cy_csr_matrix.
    """
    cdef int nnz = 0
    cdef size_t ii, jj
    cdef cy_csr_matrix out = cy_csr_matrix
    cdef int nrows = mat.shape[0], ncols = mat.shape[1]
    out.init(nrows*ncols, nrows, ncols, nrows*ncols)

    for ii in range(nrows):
        for jj in range(ncols):
            if mat[ii,jj] != 0:
                out.indices[nnz] = jj
                out.data[nnz] = mat[ii,jj]
                nnz += 1
        out.indptr[ii+1] = nnz

    if nnz < (nrows*ncols):
        out._shorten(nnz)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_csr_matrix CSR_from_scipy_coo(object A):
    """
    Converts a SciPy COO sparse matrix to a
    cy_csr_matrix object.
    """
    cdef cy_csr_matrix mat = cy_csr_matrix.__new__(cy_csr_matrix)
    mat.nrows = A.shape[0]
    mat.ncols = A.shape[1]
    mat.nnz = A.data.shape[0]
    mat.max_length = mat.nnz
    mat.is_set = 1
    mat.numpy_lock = 0
    mat.indices = <int *>PyDataMem_NEW(mat.nnz * sizeof(int))
    mat.indptr = <int *>PyDataMem_NEW((mat.nrows+1) * sizeof(int))
    mat.data = <double complex *>PyDataMem_NEW(mat.nnz * sizeof(double complex))
    for i in range(mat.nnz):
        mat.data[i] = A.data[i]
    mat._from_coo_indices(A.row, A.col)
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_csr_matrix identity_CSR(unsigned int nrows):
    cdef size_t kk
    cdef cy_csr_matrix mat = cy_csr_matrix.__new__(cy_csr_matrix)
    mat.init(nrows, nrows, nrows, 0, 0)
    for kk in range(nrows):
        mat.data[kk] = 1
        mat.indices[kk] = kk
        mat.indptr[kk] = kk
    mat.indptr[nrows] = nrows
    return mat
