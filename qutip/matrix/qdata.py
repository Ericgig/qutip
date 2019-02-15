# base class for all Qobj data, mainly here as a template to know what to do
# when adding a new format. Instanting it would not result in much.
class _qdata:
    def __init__(self):
        self.format = "None"
        self.shape = (0,0)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # method defined in numpy/scipy
    # we can reuse but:
    # 1. scipy sparse can change the type to csr during computation.
    # 2. there may be speed gain since we can skip some safety checks

    def __mul__(self, other):
        raise NotImplementedError("dummy class")

    def __rmul__(self, other):
        raise NotImplementedError("dummy class")

    def __add__(self, other):
        raise NotImplementedError("dummy class")

    def __radd__(self, other):
        raise NotImplementedError("dummy class")

    def __sub__(self, other):
        raise NotImplementedError("dummy class")

    def __rsub__(self, other):
        raise NotImplementedError("dummy class")

    def __div__(self, other):
        raise NotImplementedError("dummy class")

    def __neg__(self, other):
        raise NotImplementedError("dummy class")

    def __pow__(self, other):
        raise NotImplementedError("dummy class")

    def __abs__(self, other):
        raise NotImplementedError("dummy class")

    def __str__(self, other):
        raise NotImplementedError("dummy class")

    def __getitem__(self, key):
        raise NotImplementedError("dummy class")

    def __setitem__(self, key):
        raise NotImplementedError("dummy class")

    def transpose(self, axes=None, copy=False):
        """ transpose, 'axes' is for numpy compatibility
        """
        raise NotImplementedError("dummy class")

    def conj(self, copy=True):
        """ return conj of self
        """
        raise NotImplementedError("dummy class")

    def toarray(self, order=None, out=None):
        """ dense 2d numpy array
        """
        raise NotImplementedError("dummy class")

    def diagonal(self, k=0):
        """ diagonal k
        """
        raise NotImplementedError("dummy class")

    def dot(self, other):
        """ self * other
        """
        raise self * other

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    # Qutip method
    def adjoint(self):
        """ return dag of self
        scipy/numpy as getH
        """
        raise NotImplementedError("dummy class")

    def norm(self, norm):
        """
        Norm of a quantum object.
        """
        raise NotImplementedError("dummy class")

    def eigs(self):
        """ eigenvalue and eigstates
        """
        raise NotImplementedError("dummy class")

    def proj(self):
        """
        self*self.dag
        """
        raise NotImplementedError("dummy class")

    def trace(self):
        """
        tr(self)
        """
        raise NotImplementedError("dummy class")

    def expm(self, ):
        """
        exp(self)
        """
        raise NotImplementedError("dummy class")

    def ptrace(self, selection):
        """
        partial trace
        """
        raise NotImplementedError("dummy class")

    def tidyup(self, atol=settings.auto_tidyup_atol):
        """
        clean near 0
        """
        raise NotImplementedError("dummy class")

    def expect_rho_vec(self, vec):
        """
        tr(self*vec2mat(vec))
        """
        raise NotImplementedError("dummy class")

    def expect_psi_vec(self, vec):
        """
        vec.dag * self * vec
        """
        raise NotImplementedError("dummy class")

    def mul_vec(self, in ):
        """ out = self * in
        used in solver iterations: should be fast
        """
        raise NotImplementedError("dummy class")

    def mul_vec_py(self, in, out, alpha):
        """ out = out + alpha * self * in
        used in solver iterations: should be fast
        """
        raise NotImplementedError("dummy class")

    def unit_row_norm(self):
        """
        normalize each row
        """
        raise NotImplementedError("dummy class")

    def get_diag(self, L):
        """
        same as diagonal, but for square only
        """
        raise NotImplementedError("dummy class")

    @property
    def cdata(self):
        """
        return and instance of cdata for the object
        """
        raise NotImplementedError("dummy class")
