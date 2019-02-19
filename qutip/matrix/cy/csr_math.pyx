#!python
#cython: language_level=3
from qutip.matrix.cy.csr_matrix cimport cy_csr_matrix
import numpy as np
cimport numpy as np
import qutip.settings as qset
cimport cython
cimport libc.math
from libcpp cimport bool

np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double         real(double complex)
    double         imag(double complex)
    double         abs(double complex)

@cython.overflowcheck(True)
cdef _safe_multiply(int A, int B):
    """
    Computes A*B and checks for overflow.
    """
    cdef int C = A*B
    return C


@cython.boundscheck(False)
@cython.wraparound(False) ##
cpdef cy_csr_matrix zcsr_add(cy_csr_matrix A, cy_csr_matrix B, double complex alpha = 1):
    """
    Adds two sparse CSR matries. Like SciPy, we assume the worse case
    for the fill A.nnz + B.nnz.
    out = A+alpha*B
    """
    cdef int worse_fill = A.nnz + B.nnz
    cdef int nnz, i
    cdef cy_csr_matrix out = cy_csr_matrix()
    #Both matrices are zero mats
    if A.nnz == 0 and B.nnz == 0:
        out.init(0, A.nrows, A.ncols, 0, 1)
        return out
    #A is the zero matrix
    elif A.nnz == 0:
        out.copy_CSR(B)
        for i in range(B.nnz):
            out.data[i] *= alpha
        return out
    #B is the zero matrix
    elif B.nnz == 0:
        out.copy_CSR(A)
        return out
    # Out CSR_Matrix
    out.init(worse_fill, A.nrows, A.ncols, worse_fill)

    nnz = _zcsr_add_core(A.data, A.indices, A.indptr,
                         B.data, B.indices, B.indptr,
                         alpha,
                         out.data, out.indices, out.indptr,
                         A.nrows, A.ncols)
    #Shorten data and indices if needed
    if out.nnz > nnz:
        out._shorten(nnz)
    return out


@cython.boundscheck(False)
@cython.wraparound(False) #
cdef int _zcsr_add_core(double complex * Adata, int * Aind, int * Aptr,
                        double complex * Bdata, int * Bind, int * Bptr,
                        double complex alpha,
                        double complex * Cdata, int * Cind, int * Cptr,
                        int nrows, int ncols) nogil:

    cdef int j1, j2, kc = 0
    cdef int ka, kb, ka_max, kb_max
    cdef size_t ii
    cdef double complex tmp
    Cptr[0] = 0
    if alpha != 1:
        for ii in range(nrows):
            ka = Aptr[ii]
            kb = Bptr[ii]
            ka_max = Aptr[ii+1]-1
            kb_max = Bptr[ii+1]-1
            while (ka <= ka_max) or (kb <= kb_max):
                if ka <= ka_max:
                    j1 = Aind[ka]
                else:
                    j1 = ncols+1

                if kb <= kb_max:
                    j2 = Bind[kb]
                else:
                    j2 = ncols+1

                if j1 == j2:
                    tmp = Adata[ka] + alpha*Bdata[kb]
                    if tmp != 0:
                        Cdata[kc] = tmp
                        Cind[kc] = j1
                        kc += 1
                    ka += 1
                    kb += 1
                elif j1 < j2:
                    Cdata[kc] = Adata[ka]
                    Cind[kc] = j1
                    ka += 1
                    kc += 1
                elif j1 > j2:
                    Cdata[kc] = alpha*Bdata[kb]
                    Cind[kc] = j2
                    kb += 1
                    kc += 1

            Cptr[ii+1] = kc
    else:
        for ii in range(nrows):
            ka = Aptr[ii]
            kb = Bptr[ii]
            ka_max = Aptr[ii+1]-1
            kb_max = Bptr[ii+1]-1
            while (ka <= ka_max) or (kb <= kb_max):
                if ka <= ka_max:
                    j1 = Aind[ka]
                else:
                    j1 = ncols+1

                if kb <= kb_max:
                    j2 = Bind[kb]
                else:
                    j2 = ncols+1

                if j1 == j2:
                    tmp = Adata[ka] + Bdata[kb]
                    if tmp != 0:
                        Cdata[kc] = tmp
                        Cind[kc] = j1
                        kc += 1
                    ka += 1
                    kb += 1
                elif j1 < j2:
                    Cdata[kc] = Adata[ka]
                    Cind[kc] = j1
                    ka += 1
                    kc += 1
                elif j1 > j2:
                    Cdata[kc] = Bdata[kb]
                    Cind[kc] = j2
                    kb += 1
                    kc += 1

            Cptr[ii+1] = kc
    return kc


@cython.boundscheck(False)
@cython.wraparound(False) ##
cpdef cy_csr_matrix zcsr_mult(cy_csr_matrix A, cy_csr_matrix B, int sorted = 1):
    cdef int Annz = A.nnz
    cdef int Bnnz = B.nnz
    cdef int nrows = A.nrows
    cdef int ncols = B.ncols
    cdef cy_csr_matrix out = cy_csr_matrix()
    cdef int nnz

    #Both matrices are zero mats
    if Annz == 0 or Bnnz == 0:
        out.init(0,nrows,ncols,0,1)
        return out
    nnz = _zcsr_mult_pass1(A.data, A.indices, A.indptr,
                           B.data, B.indices, B.indptr,
                           nrows, ncols)

    if nnz == 0:
        out.init(0,nrows,ncols,0,1)
        return out

    out.init(nnz, nrows, ncols)
    _zcsr_mult_pass2(A.data, A.indices, A.indptr,
                     B.data, B.indices, B.indptr,
                     out, nrows, ncols)

    #Shorten data and indices if needed
    if out.nnz > out.indptr[out.nrows]:
        out._shorten(out.indptr[out.nrows])

    if sorted:
        out._sort_indices()
    return out


@cython.boundscheck(False)
@cython.wraparound(False) ##
cdef int _zcsr_mult_pass1(double complex * Adata, int * Aind, int * Aptr,
                     double complex * Bdata, int * Bind, int * Bptr,
                     int nrows, int ncols):

    cdef int j, k, nnz = 0
    cdef size_t ii,jj,kk
    #Setup mask array
    cdef int * mask = <int *>PyDataMem_NEW(ncols*sizeof(int))
    for ii in range(ncols):
        mask[ii] = -1
    #Pass 1
    for ii in range(nrows):
        for jj in range(Aptr[ii], Aptr[ii+1]):
            j = Aind[jj]
            for kk in range(Bptr[j], Bptr[j+1]):
                k = Bind[kk]
                if mask[k] != ii:
                    mask[k] = ii
                    nnz += 1
    PyDataMem_FREE(mask)
    return nnz


@cython.boundscheck(False)
@cython.wraparound(False) ##
cdef void _zcsr_mult_pass2(double complex * Adata, int * Aind, int * Aptr,
                           double complex * Bdata, int * Bind, int * Bptr,
                           cy_csr_matrix C,
                           int nrows, int ncols):

    cdef int head, length, temp, j, k, nnz = 0
    cdef size_t ii,jj,kk
    cdef double complex val
    cdef double complex * sums = <double complex *>PyDataMem_NEW_ZEROED(ncols, sizeof(double complex))
    cdef int * nxt = <int *>PyDataMem_NEW(ncols*sizeof(int))
    for ii in range(ncols):
        nxt[ii] = -1

    C.indptr[0] = 0
    for ii in range(nrows):
        head = -2
        length = 0
        for jj in range(Aptr[ii], Aptr[ii+1]):
            j = Aind[jj]
            val = Adata[jj]
            for kk in range(Bptr[j], Bptr[j+1]):
                k = Bind[kk]
                sums[k] += val*Bdata[kk]
                if nxt[k] == -1:
                    nxt[k] = head
                    head = k
                    length += 1

        for jj in range(length):
            if sums[head] != 0:
                C.indices[nnz] = head
                C.data[nnz] = sums[head]
                nnz += 1
            temp = head
            head = nxt[head]
            nxt[temp] = -1
            sums[temp] = 0

        C.indptr[ii+1] = nnz

    #Free temp arrays
    PyDataMem_FREE(sums)
    PyDataMem_FREE(nxt)


@cython.boundscheck(False)
@cython.wraparound(False) ##
cpdef cy_csr_matrix zcsr_kron(cy_csr_matrix A, cy_csr_matrix B):
    """
    Computes the kronecker product between two complex
    sparse matrices in CSR format.
    """
    cdef int out_nnz = _safe_multiply(A.nnz, B.nnz)
    cdef int rows_out = A.nrows * B.nrows
    cdef int cols_out = A.ncols * B.ncols

    cdef cy_csr_matrix out = cy_csr_matrix()
    out.init(out_nnz, rows_out, cols_out)

    _zcsr_kron_core(A.data, A.indices, A.indptr,
                    B.data, B.indices, B.indptr,
                    out, A.nrows, B.nrows, B.ncols)
    return out


@cython.boundscheck(False)
@cython.wraparound(False) #
cdef void _zcsr_kron_core(double complex * dataA, int * indsA, int * indptrA,
                          double complex * dataB, int * indsB, int * indptrB,
                          cy_csr_matrix out,
                          int rowsA, int rowsB, int colsB) nogil:
    cdef size_t ii, jj, ptrA, ptr
    cdef int row = 0
    cdef int ptr_start, ptr_end
    cdef int row_startA, row_endA, row_startB, row_endB, distA, distB, ptrB

    for ii in range(rowsA):
        row_startA = indptrA[ii]
        row_endA = indptrA[ii+1]
        distA = row_endA - row_startA

        for jj in range(rowsB):
            row_startB = indptrB[jj]
            row_endB = indptrB[jj+1]
            distB = row_endB - row_startB

            ptr_start = out.indptr[row]
            ptr_end = ptr_start + distB

            out.indptr[row+1] = out.indptr[row] + distA * distB
            row += 1

            for ptrA in range(row_startA, row_endA):
                ptrB = row_startB
                for ptr in range(ptr_start, ptr_end):
                    out.indices[ptr] = indsA[ptrA] * colsB + indsB[ptrB]
                    out.data[ptr] = dataA[ptrA] * dataB[ptrB]
                    ptrB += 1

                ptr_start += distB
                ptr_end += distB


@cython.boundscheck(False)
@cython.wraparound(False) ##
def zcsr_inner(cy_csr_matrix A, cy_csr_matrix B):
    """
    Computes the inner-product <A|B> between ket-ket,
    or bra-ket vectors in sparse CSR format.
    """

    cdef double complex inner = 0
    cdef size_t jj, kk
    cdef int a_idx, b_idx

    if A.nrows == 1:
      for kk in range(A.nnz):
          a_idx = A.indices[kk]
          if (B.indptr[a_idx+1]-B.indptr[a_idx]) != 0:
              inner += A.data[kk]*B.data[B.indptr[a_idx]]

          """for kk in range(a_ind.shape[0]):
              a_idx = a_ind[kk]
              for jj in range(B.nrows):
                  if (b_ptr[jj+1]-b_ptr[jj]) != 0:
                      if jj == a_idx:
                          inner += a_data[kk]*b_data[b_ptr[jj]]
                          break"""
    else:
        for kk in range(B.nrows):
            a_idx = A.indptr[kk]
            b_idx = B.indptr[kk]
            if (A.indptr[kk+1]-a_idx) != 0:
                if (B.indptr[kk+1]-b_idx) != 0:
                    inner += conj(A.data[a_idx])*B.data[b_idx]

    return inner


@cython.boundscheck(False)
@cython.wraparound(False) ##
cpdef double complex zcsr_mat_elem(cy_csr_matrix A,
                                   cy_csr_matrix left,
                                   cy_csr_matrix right):
    """
    Computes the matrix element for an operator A and left and right vectors.
    right must be a ket, but left can be a ket or bra vector.  If left
    is bra then bra_ket = 1, else set bra_ket = 0.
    """
    cdef int j, go, head=0
    cdef size_t ii, jj, kk
    cdef double complex cval=0, row_sum, mat_elem=0

    for ii in range(A.nrows):
        row_sum = 0
        go = 0
        if left.nrows == 1:
            for kk in range(head, left.nnz):
                if left.indices[kk] == ii:
                    cval = left.data[kk]
                    head = kk
                    go = 1
        else:
            if (left.indptr[ii] - left.indptr[ii+1]) != 0:
                cval = conj(left.data[left.indptr[ii]])
                go = 1

        if go:
            for jj in range(A.indptr[ii], A.indptr[ii+1]):
                j = A.indices[jj]
                if (right.indptr[j] - right.indptr[j+1]) != 0:
                    row_sum += A.data[jj]*right.data[right.indptr[j]]
            mat_elem += cval*row_sum

    return mat_elem


##
def zcsr_expect_ket(cy_csr_matrix A, cy_csr_matrix B, int isherm):
    if isherm:
        return real(_zcsr_expect_ket_core(A, B))
    else:
        return _zcsr_expect_ket_core(A, B)


@cython.boundscheck(False)
@cython.wraparound(False) ###
cpdef complex _zcsr_expect_ket_core(cy_csr_matrix A, cy_csr_matrix B):
    cdef int j
    cdef size_t ii, jj
    cdef double complex cval=0, row_sum, expt = 0
    for ii in range(A.nrows):
        if (B.indptr[ii+1] - B.indptr[ii]) != 0:
            cval = conj(B.data[B.indptr[ii]])
            row_sum = 0
            for jj in range(A.indptr[ii], A.indptr[ii+1]):
                j = A.indices[jj]
                if (B.indptr[j+1] - B.indptr[j]) != 0:
                    row_sum += A.data[jj]*B.data[B.indptr[j]]
            expt += cval*row_sum
    return expt

##
def zcsr_expect_rho_csr(cy_csr_matrix super_op,
                        cy_csr_matrix rho,
                        int isherm):
    if isherm:
        return real(_zcsr_expect_rho_csr_core(super_op, rho))
    else:
        return _zcsr_expect_rho_csr_core(super_op, rho)


@cython.boundscheck(False)
@cython.wraparound(False) ##
cpdef complex _zcsr_expect_rho_csr_core(cy_csr_matrix op,
                                        cy_csr_matrix rho):
    cdef size_t row
    cdef int jj, row_start, row_end, rho_i
    cdef int n = rho.ncols
    cdef int num_rows = n*n
    cdef complex dot = 0.0

    cdef int[::1] rho_rows = np.empty(rho.nnz, dtype=np.int32)
    cdef int[::1] rho_cols = np.empty(rho.nnz, dtype=np.int32)
    rho._coo_indices(rho_rows, rho_cols)
    for jj in range(rho.nnz):
        rho_rows[jj] = n*rho_rows[jj]+rho_cols[jj]

    for row from 0 <= row < num_rows by n+1:
        row_start = op.indptr[row]
        row_end = op.indptr[row+1]
        rho_i = 0
        for jj from row_start <= jj < row_end:
            while rho_i < rho.nnz and op.indices[jj] > rho_rows[rho_i]:
                rho_i += 1
            if rho_i >= rho.nnz:
                break
            if op.indices[jj] == rho_rows[rho_i]:
                dot += op.data[jj]*conj(rho.data[rho_i])
    return dot


def zcsr_spmm_tr(cy_csr_matrix op1, cy_csr_matrix op2, int herm=0): ##
    if herm == 0:
        return _zcsr_spmm_tr_core(op1, op2)
    else:
        return real(_zcsr_spmm_tr_core(op1, op2))


@cython.boundscheck(False)
@cython.wraparound(False) ##
cdef complex _zcsr_spmm_tr_core(cy_csr_matrix op1, cy_csr_matrix op2):
    cdef size_t row
    cdef complex tr = 0.0
    cdef int col1, row1_idx_start, row1_idx_end
    cdef int col2, row2_idx_start, row2_idx_end
    cdef int num_rows = op1.nrows

    for row in range(num_rows):
        row1_idx_start = op1.indptr[row]
        row1_idx_end = op1.indptr[row + 1]
        for row1_idx from row1_idx_start <= row1_idx < row1_idx_end:
            col1 = op1.indices[row1_idx]

            row2_idx_start = op2.indptr[col1]
            row2_idx_end = op2.indptr[col1 + 1]
            for row2_idx from row2_idx_start <= row2_idx < row2_idx_end:
                col2 = op2.indices[row2_idx]

                if col2 == row:
                    tr += op1.data[row1_idx] * op2.data[row2_idx]
                    break
    return tr
