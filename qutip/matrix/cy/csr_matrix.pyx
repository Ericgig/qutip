#!python
#cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()

from qutip.fastsparse import fast_csr_matrix
import qutip.settings as qset
from libcpp cimport bool
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
from libc.stdlib cimport div, ldiv
from libc.math cimport abs, fabs, sqrt

# Todo:
# ctypedef size for indices and derived variables (template?)
# factory: mixed type

#include "parameters.pxi"
DTYPE = np.float64
ITYPE = np.int32
CTYPE = np.complex128
CTYPE = np.int64
cdef extern from "stdlib.h":
  ctypedef struct ldiv_t:
    long quot
    long rem

cdef extern from "stdlib.h":
    ctypedef struct div_t:
        int quot
        int rem

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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _in(int val, int[::1] vec):
    # val in vec in pure cython
    cdef int ii
    for ii in range(vec.shape[0]):
        if val == vec[ii]:
            return 1
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _i2_k_t(int N,
                 int[:, ::1] tensor_table,
                 int[::1] out):
    # indices determining function for ptrace
    cdef int ii, t1, t2
    out[0] = 0
    out[1] = 0
    for ii in range(rho2ten.shape[0]):
        t1 = tensor_table[0, ii]
        t2 = N / t1
        N = N % t1
        out[0] += tensor_table[1, ii] * t2
        out[1] += tensor_table[2, ii] * t2

cdef void raise_error_csr(int E):
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


cdef class cy_csr_matrix(cy_cs_matrix):
    def __cinit__(self):
        self.is_csr = 1

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
    cpdef cy_csr_matrix copy(self):
        """
        Return a copy.
        """
        cdef cy_csr_matrix out = cy_csr_matrix()
        out.copy_cs(self)
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def to_qdata(self):
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
            return csr_qmatrix_from_cdata(self)
        else:
            if self.numpy_lock:
                raise_error_csr(-4)
            elif not self.is_set:
                raise_error_csr(-3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def qdata(self):
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
            return csr_qmatrix_from_data((_data, _ind, _ptr),
                                         shape=(self.nrows, self.ncols))
        else:
            raise_error_csr(-3)

    cpdef double one_norm(self):
        return self._max_sum_sec()

    cpdef double inf_norm(self):
        return self._max_sum_main()



    cpdef cnp.ndarray[complex, ndim=1, mode="c"] spmv(self, complex[::1] vec):
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

    cpdef void spmvpy(self, complex[::1] vec, complex[::1] out, complex alpha):
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

    cpdef cnp.ndarray[complex, ndim=2] spmm(self, cnp.ndarray[complex, ndim=2] mat):
        if mat.flags["F_CONTIGUOUS"]:
            return self.spmmf(mat)
        else:
            return self.spmmc(mat)

    cpdef cnp.ndarray[complex, ndim=2] spmmf(self, complex[::1, :] mat):
        cdef cnp.ndarray[complex, ndim=2, mode="fortran"] out = \
                          np.zeros((sp_rows, ncols), dtype=complex, order="F")
        _spmm_f_py(self.data, self.indices, self.indptr,
                   &mat[0,0], (1.0+0j), &out[0,0],
                   self.nrows, self.ncols, mat.shape[1])
        return out

    cpdef cnp.ndarray[complex, ndim=2] spmmc(self, complex[:, ::1] mat):
        cdef cnp.ndarray[complex, ndim=2, mode="c"] out = \
                          np.zeros((sp_rows, ncols), dtype=complex)
        _spmm_c_py(self.data, self.indices, self.indptr,
                   &mat[0,0], (1.0+0j), &out[0,0],
                   self.nrows, self.ncols, mat.shape[1])
        return out

    cpdef void spmmpy(self, cnp.ndarray[complex, ndim=2] mat,
                            cnp.ndarray[complex, ndim=2] out,
                            complex alpha):
        """
        Sparse matrix, dense matrix multiplication
        out = out + alpha * self@mat

        Parameters
        ----------
        mat : array
            Dense matrix for multiplication.

        out : array
            Returns dense array.

        alpha : complex
            factor to the multiplication

        """
        if mat.flags["F_CONTIGUOUS"]:
            self.spmmfpy(mat, out, alpha)
        else:
            self.spmmcpy(mat, out, alpha)

    cpdef void spmmfpy(self, complex[::1, :] mat,  complex[::1, :] out, complex alpha):
        _spmm_f_py(self.data, self.indices, self.indptr,
                   &mat[0,0], alpha, &out[0,0],
                   self.nrows, self.ncols, mat.shape[1])

    cpdef void spmmcpy(self, complex[:, ::1] mat, complex[:, ::1] out,  complex alpha):
        _spmm_c_py(self.data, self.indices, self.indptr,
                   &mat[0,0], alpha, &out[0,0],
                   self.nrows, self.ncols, mat.shape[1])

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
    @cython.cdivision(True)
    cpdef unit_row_norm(self):
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
            raise_error_csr(-4)

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
    @cython.wraparound(False)
    @cython.cdivision(True)
    def ptrace(self, dims, sel): # work for N<= 26 on 16G Ram
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
            return csr_qmatrix_from_cdata(_oper._ptrace_core_sp(tensor_table)), rho1_dims
        else:
            return csr_qmatrix_from_cdata(_oper._ptrace_core_dense(tensor_table, factor_keep)), rho1_dims

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef cy_csr_matrix _ptrace_core_sp(self, int[:, ::1] tensor_table):
        cdef int p = 0, nnz, ii
        cdef int[::1] pos_c = np.empty(2, dtype=np.int32)
        cdef int[::1] pos_r = np.empty(2, dtype=np.int32)
        cdef int[::1] col = np.empty(self.nnz, dtype=np.int32)
        cdef int[::1] row = np.empty(self.nnz, dtype=np.int32)
        cdef cnp.ndarray[complex, ndim=1, mode='c'] new_data = np.zeros(cco.nnz, dtype=complex)
        cdef cnp.ndarray[int, ndim=1, mode='c'] new_col = np.zeros(cco.nnz, dtype=np.int32)
        cdef cnp.ndarray[int, ndim=1, mode='c'] new_row = np.zeros(cco.nnz, dtype=np.int32)

        nnz = self.nnz
        _coo_indices(self, row, col)

        for ii in range(nnz):
            _i2_k_t(col[ii], tensor_table, pos_c)
            _i2_k_t(row[ii], tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                new_data[p] = self.data[i]
                new_col[p] = (pos_c[0])
                new_row[p] = (pos_r[0])
                p += 1

        # Here there can be redundance in (new_col, new_row) pairs.
        # scipy coo_matrix does the sum but _from_coo_indices does not.
        cdef object out = coo_matrix((new_data, [new_col, new_row])).tocsr()
        return csr_from_scipy(out, False)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef cy_csr_matrix _ptrace_core_dense(self, int[:, ::1] tensor_table, int num_sel_dims):
        cdef int nnz, ii
        cdef int[::1] pos_c = np.empty(2, dtype=np.int32)
        cdef int[::1] pos_r = np.empty(2, dtype=np.int32)
        cdef int[::1] col = np.empty(self.nnz, dtype=np.int32)
        cdef int[::1] row = np.empty(self.nnz, dtype=np.int32)
        cdef complex[:, ::1] data = np.zeros((num_sel_dims, num_sel_dims),
                                              dtype=complex)
        nnz = self.nnz
        _coo_indices(self, row, col)
        for ii in range(nnz):
            _i2_k_t(col[ii], tensor_table, pos_c)
            _i2_k_t(row[ii], tensor_table, pos_r)
            if pos_c[1] == pos_r[1]:
                data[pos_r[0], pos_c[0]] += self.data[p]
                p += 1
        return dense_to_csr(data)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _coo_indices(self, int[::1] rows, int[::1] cols):
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _from_coo_indices(self, int[::1] rows, int[::1] cols):
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


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # method from cdata for name compatibility
    cpdef cnp.ndarray[complex, ndim=1, mode="c"] matvec(self, complex[::1] vec):
        return self.spmv(vec)

    cpdef void matvecpy(self, complex[::1] vec, complex[::1] out, complex alpha):
        self.spmvpy(vec, out, alpha)


    cpdef cnp.ndarray[complex, ndim=2] matmat(self, cnp.ndarray[complex, ndim=2] mat):
        return self.spmm(mat)

    cpdef cnp.ndarray[complex, ndim=2] matmatf(self, complex[::1, :] mat):
        return self.spmmf(mat)

    cpdef cnp.ndarray[complex, ndim=2] matmatc(self, complex[:, ::1] mat):
        return self.spmmc(mat)

    cpdef void matmatpy(self, cnp.ndarray[complex, ndim=2] mat,
                              cnp.ndarray[complex, ndim=2] out,
                              complex alpha):
        self.spmmpy(mat, out, alpha)

    cpdef void matmatpyf(self, complex[::1, :] mat,  complex[::1, :] out, complex alpha):
        self.spmmfpy(mat, out, alpha)

    cpdef void matmatpyc(self, complex[:, ::1] mat, complex[:, ::1] out,  complex alpha):
        self.spmmcpy(mat, out, alpha)

    cpdef void matmat_as_vec_f(self, complex[::1] vec, complex[::1] out, complex alpha):
        # shortcut function for solver, only work for square self and mat
        _spmm_f_py(self.data, self.indices, self.indptr,
                   &mat[0], alpha, &out[0],
                   self.nrows, self.ncols, self.ncols)




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_csr_matrix csr_from_scipy(object A, copy=False):
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
cpdef cy_csr_matrix csr_from_scipy_coo(object A):
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
cpdef cy_csr_matrix identity_csr(unsigned int nrows):
    cdef size_t kk
    cdef cy_csr_matrix mat = cy_csr_matrix.__new__(cy_csr_matrix)
    mat.init(nrows, nrows, nrows, 0, 0)
    for kk in range(nrows):
        mat.data[kk] = 1
        mat.indices[kk] = kk
        mat.indptr[kk] = kk
    mat.indptr[nrows] = nrows
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object csr_qmatrix_from_cdata(cy_csr_matrix cdata):
    ...


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cy_csr_matrix dense_to_csr(complex[:, :] mat):
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
