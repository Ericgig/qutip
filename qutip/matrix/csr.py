import numpy as np
import operator
from scipy.sparse import (_sparsetools, isspmatrix, isspmatrix_csr, isspmatrix_coo,
                          csr_matrix, coo_matrix, csc_matrix, dia_matrix)
from scipy.sparse.sputils import (upcast, upcast_char, to_native, isdense, isshape,
                      getdtype, isscalarlike, IndexMixin, get_index_dtype)
from scipy.sparse.base import spmatrix, isspmatrix, SparseEfficiencyWarning

from warnings import warn
from qutip.matrix.qdata import _qdata
from qutip.matrix.sparse import *
import qutip.settings as settings
from qutip.matrix.cy.utils import cy_tidyup
from qutip.cy.openmp.utilities import use_openmp
if settings.has_openmp:
    from qutip.cy.openmp.omp_utils import omp_tidyup


class csr_qmatrix(csr_matrix, _qdata):
    """
    A subclass of scipy.sparse.csr_matrix that skips the data format
    checks that are run everytime a new csr_matrix is created.
    """
    def __init__(self, args=None, shape=None, dtype=None, copy=False):
        if args is None: #Build zero matrix
            if shape is None:
                raise Exception('Shape must be given when building zero matrix.')
            self.data = np.array([], dtype=complex)
            self.indices = np.array([], dtype=np.int32)
            self.indptr = np.zeros(shape[0]+1, dtype=np.int32)
            self._shape = tuple(int(s) for s in shape)

        elif isinstance(args, tuple):
            if args[0].shape[0] and args[0].dtype != complex:
                raise TypeError('fast_csr_matrix allows only complex data.')
            if args[1].shape[0] and args[1].dtype != np.int32:
                raise TypeError('fast_csr_matrix allows only int32 indices.')
            if args[2].shape[0] and args[1].dtype != np.int32:
                raise TypeError('fast_csr_matrix allows only int32 indptr.')
            self.data = np.array(args[0], dtype=complex, copy=copy)
            self.indices = np.array(args[1], dtype=np.int32, copy=copy)
            self.indptr = np.array(args[2], dtype=np.int32, copy=copy)
            if shape is None:
                self._shape = tuple([len(self.indptr)-1]*2)
            else:
                self._shape = tuple(int(s) for s in shape)

        elif isspmatrix_csr(args):
            self.data = np.array(args.data, dtype=complex, copy=copy)
            self.indices = np.array(args.indices, dtype=np.int32, copy=copy)
            self.indptr = np.array(args.indptr, dtype=np.int32, copy=copy)
            if shape is None:
                self._shape = args.shape
            else:
                self._shape = tuple(int(s) for s in shape)

        else:
            raise ValueError("Unknown input format")

        self.dtype = complex
        self.maxprint = 50
        self.format = 'csr'
        self._isherm = None
        self._isdiag = None
        self._cdata = None

    def copy(self):
        return csr_qmatrix(self, copy=True)
        # probably a few us faster but worth it?
        # out = csr_qmatrix.__new__(csr_qmatrix)
        # out.__dict__ = self.__dict__.copy()
        # out.data = self.data.copy()
        # out.indices = self.indices.copy()
        # out.indptr = self.indptr.copy()
        # out._cdata = None
        # return out


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # scipy method overload
    def _binopt(self, other, op):
        """
        Do the binary operation fn to two sparse matrices using
        csr_qmatrix only when other is also a csr_qmatrix.
        """
        other_csr_qmatrix = isinstance(other, csr_qmatrix)
        self_sp = csr_matrix(self, copy=False)
        out = csr_matrix._binopt(self_sp, other, op)
        if other_csr_qmatrix and out.data.dtype != np.bool_:
            out = csr_qmatrix(out)
        return out

    def _mul_sparse_matrix(self, other):
        """
        Do the sparse matrix mult returning fast_csr_matrix only
        when other is also fast_csr_matrix.
        """
        if isinstance(other, csr_qmatrix):
            # use our product
            return zcsr_mult(cy_csr_matrix(self),
                             cy_csr_matrix(other), sorted=1)
        return csr_matrix._mul_sparse_matrix(self, other, op)

    def _with_data(self,data,copy=True):
        """
        Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        # We need this just in case something like abs(data) gets called
        # does nothing if data.dtype is complex.
        data = np.asarray(data, dtype=complex)
        if copy:
            return csr_qmatrix((data, self.indices.copy(), self.indptr.copy()),
                                   shape=self.shape, dtype=data.dtype)
        else:
            return csr_qmatrix((data, self.indices, self.indptr),
                                   shape=self.shape, dtype=data.dtype)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Qutip methods
    def transpose(self, axis=None, copy=True):
        return self.trans(copy=copy)

    def trans(self, axis=None, copy=True):
        """
        Returns the transpose of the matrix, keeping
        it in csr_qdata format.
        """
        if copy:
            ccdata = self.cdata.copy()
            ccdata.transpose()
            return ccdata.to_qdata()
        self.cdata.transpose()
        return self

    def adjoint(self, copy=True):
        """
        Returns the conj transpose of the matrix, keeping
        it in csr_qdata format.
        """
        if copy:
            ccdata = self.cdata.copy()
            ccdata.adjoint()
            return ccdata.to_qdata()
        else:
            self.cdata.adjoint()
            return self

    def tidyup(self, atol=settings.auto_tidyup_atol):
        """
        Removes small elements from the quantum object.
        """
        if self.nnz:
            #This does the tidyup and returns True if
            #The sparse data needs to be shortened
            if use_openmp() and self.nnz > 500:
                if omp_tidyup(self.data, atol, self.nnz, settings.num_cpus):
                    self.eliminate_zeros()
            else:
                if cy_tidyup(self.data, atol, self.nnz):
                    self.eliminate_zeros()
        return self

    @property
    def cdata(self):
        if self._cdata is None:
            self._cdata = cy_csr_matrix(self)
        return self._cdata

    @cdata.setter
    def cdata(self, _):
        # Need to keep both object using the same data array
        pass

    def norm(self, norm):
        """
        Norm of a quantum object.
        """
        if norm == 'fro':
            return sp_fro_norm(self)
        elif norm == 'one':
            return sp_one_norm(self)
        elif norm == 'inf':
            return sp_inf_norm(self)
        elif norm == 'max':
            return sp_max_norm(self)
        elif norm == 'l2':
            return sp_L2_norm(self)

    def eigs(self, isherm, vecs=True, sparse=False, sort='low',
                               eigvals=0, tol=0, maxiter=100000):
        """
        eigenvalue and eigstates
        """
        return sp_eigs(self, isherm, vecs=vecs, sparse=sparse, sort=sort,
                                   eigvals=eigvals, tol=tol, maxiter=maxiter)

    def proj(self):
        """
        self*self.dag
        """
        return self.cdata.proj().to_qdata()

    def trace(self):
        """
        tr(self)
        """
        return self.cdata.trace()

    def expm(self, method):
        """
        Matrix exponential of quantum operator.
        """
        return sp_expm(self, sparse=(method=="sparse"))

    def ptrace(self, selection):
        """
        partial trance
        """
        data, dims = self.cdata.ptrace(selection)
        return data.to_qdata(), dims

    def expect_rho_vec(self, vec):
        """
        tr(self*vec2mat(vec))
        """
        return self.cdata.expect_rho_vec(vec)

    def expect_psi_vec(self, vec):
        """
        vec.dag * self * vec
        """
        return self.cdata.expect_psi_vec(vec)

    def mul_vec(self, in_):
        """
        out = self * in
        used in solver iterations: should be fast
        """
        return self.cdata.spmv(in_)

    def mul_vec_py(self, in_, out, alpha):
        """ out = out + alpha * self * in
        used in solver iterations: should be fast
        """
        self.cdata.spmvpy(in_, out, alpha)

    def mul_mat(self, in_):
        """
        out = self @ in
        used in solver iterations: should be fast
        """
        return self.cdata.spmm(in_)

    def mul_mat_py(self, in_, out, alpha):
        """ out = out + alpha * self @ in
        used in solver iterations: should be fast
        """
        self.cdata.spmmpy(in_, out, alpha)

    def unit_row_norm(self):
        """ normalize each row
        """
        self.cdata.unit_row_norm(self)

    def get_diag(self, L):
        """ same as diagonal
        """
        self.cdata._get_diag(L)

    def isherm(self):
        if self._isherm is not None:
            return self._isherm
        self._isherm = self.cdata.isherm()
        return self._isherm

    def isdiag(self):
        if self._isdiag is not None:
            return self._isdiag
        self._isdiag = self.cdata.isdiag()
        return self._isdiag


def csr_qmatrix_identity(N):
    """Generates a sparse identity matrix in
    csr qdata format.
    """
    data = np.ones(N, dtype=complex)
    ind = np.arange(N, dtype=np.int32)
    ptr = np.arange(N+1, dtype=np.int32)
    ptr[-1] = N
    return csr_qmatrix_from_data((data,ind,ptr), shape=(N,N))

def csr_qmatrix_from_data(data, shape, copy=False):
    return csr_qmatrix(data, shape=shape, copy=copy)

def csr_qmatrix_from_csr(A, copy=False):
    return csr_qmatrix(A, copy=copy)

def csr_qmatrix_from_coo(A):
    if isspmatrix_coo(A):
        return csr_from_scipy_coo(A).to_qdata()
    raise ValueError("Wrong input matrix format")

def csr_qmatrix_from_dense(data):
    if not isinstance(data, np.ndarray):
        raise ValueError("Wrong input matrix format")
    if len(data.shape) == 1:
        data = data.reshape((1,-1))
    if len(data.shape) == 2:
        return csr_from_dense(data).to_qdata()

def csr_qmatrix_from_sparse(A):
    csr = sp.csr_matrix(A)
    return csr_qmatrix_from_csr(csr)


from qutip.matrix.cy.csr_matrix import cy_csr_matrix, csr_from_dense, csr_from_scipy_coo
from qutip.matrix.cy.csr_math import zcsr_mult
