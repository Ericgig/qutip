from ._base import Qobj, _QobjBuilder, _require_equal_type
from ._operator import Operator
import qutip
from ..dimensions import enumerate_flat, collapse_dims_super, Dimensions
import numpy as np
from ...settings import settings
from .. import data as _data
from typing import Sequence
from qutip.typing import LayerType, DimensionLike


__all__ = ["KrausMap"]


class SuperOperator(Qobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type not in ["super"]:
            raise ValueError(
                f"Expected super operator dimensions, but got {self._dims.type}"
            )

    @property
    def issuper(self) -> bool:
        return True

    @property
    def isherm(self) -> bool:
        if self._flags.get("isherm", None) is None:
            self._flags["isherm"] = _data.isherm(self._data)
        return self._flags["isherm"]

    @isherm.setter
    def isherm(self, isherm: bool):
        self._flags["isherm"] = isherm

    @property
    def isunitary(self) -> bool:
        if self._flags.get("isunitary", None) is None:
            if not self.isoper or self._data.shape[0] != self._data.shape[1]:
                self._flags["isunitary"] = False
            else:
                cmp = _data.matmul(self._data, self._data.adjoint())
                iden = _data.identity_like(cmp)
                self._flags["isunitary"] = _data.iszero(
                    _data.sub(cmp, iden), tol=settings.core['atol']
                )
        return self._flags["isunitary"]

    @isunitary.setter
    def isunitary(self, isunitary: bool):
        self._flags["isunitary"] = isunitary

    @property
    def ishp(self) -> bool:
        if self._flags.get("ishp", None) is None:
            try:
                J = qutip.to_choi(self)
                self._flags["ishp"] = J.isherm
            except:
                self._flags["ishp"] = False

        return self._flags["ishp"]

    @property
    def iscp(self) -> bool:
        if self._flags.get("iscp", None) is None:
            # We can test with either Choi or chi, since the basis
            # transformation between them is unitary and hence preserves
            # the CP and TP conditions.
            J = self if self.superrep in ('choi', 'chi') else qutip.to_choi(self)
            # If J isn't hermitian, then that could indicate either that J is not
            # normal, or is normal, but has complex eigenvalues.  In either case,
            # it makes no sense to then demand that the eigenvalues be
            # non-negative.
            self._flags["iscp"] = (
                J.isherm
                and np.all(J.eigenenergies() >= -settings.core['atol'])
            )
        return self._flags["iscp"]

    @property
    def istp(self) -> bool:
        if self._flags.get("istp", None) is None:
            # Normalize to a super of type choi or chi.
            # We can test with either Choi or chi, since the basis
            # transformation between them is unitary and hence
            # preserves the CP and TP conditions.
            if self.issuper and self.superrep in ('choi', 'chi'):
                qobj = self
            else:
                qobj = qutip.to_choi(self)
            # Possibly collapse dims.
            if any([len(index) > 1
                    for super_index in qobj.dims
                    for index in super_index]):
                qobj = Qobj(qobj.data,
                            dims=collapse_dims_super(qobj.dims),
                            superrep=qobj.superrep,
                            copy=False)
            # We use the condition from John Watrous' lecture notes,
            # Tr_1(J(Phi)) = identity_2.
            # See: https://cs.uwaterloo.ca/~watrous/LectureNotes.html,
            # Theory of Quantum Information (Fall 2011), theorem 5.4.
            tr_oper = qobj.ptrace([0])
            self._flags["istp"] = np.allclose(
                tr_oper.full(),
                np.eye(tr_oper.shape[0]),
                atol=settings.core['atol']
            )
        return self._flags["istp"]

    @property
    def iscptp(self) -> bool:
        if (
            self._flags.get("iscp", None) is None
            and self._flags.get("istp", None) is None
        ):
            reps = ('choi', 'chi')
            q_oper = qutip.to_choi(self) if self.superrep not in reps else self
            self._flags["iscp"] = q_oper.iscp
            self._flags["istp"] = q_oper.istp
        return self.iscp and self.istp

    def dnorm(self, B: Qobj = None) -> float:
        """Calculates the diamond norm, or the diamond distance to another
        operator.

        Parameters
        ----------
        B : :class:`.Qobj` or None
            If B is not None, the diamond distance d(A, B) = dnorm(A - B)
            between this operator and B is returned instead of the diamond norm.

        Returns
        -------
        d : float
            Either the diamond norm of this operator, or the diamond distance
            from this operator to B.

        """
        return qutip.dnorm(self, B)

    def __call__(self, other: Qobj) -> Qobj:
        """
        Acts this Qobj on another Qobj either by left-multiplication,
        or by vectorization and devectorization, as
        appropriate.
        """
        if not isinstance(other, Qobj):
            raise TypeError("Only defined for quantum objects.")
        if other.type not in ["ket", "oper"]:
            raise TypeError(self.type + " cannot act on " + other.type)
        if other.isket:
            other = other.proj()
        return qutip.vector_to_operator(self @ qutip.operator_to_vector(other))

    # Method from operator are merged on case per case basis

    def _warn(f, name=None):
        import warnings
        def func(*a, **kw):
            warnings.warn(f"SuperOperator.{name} used")
            return f(*a, **kw)
        return func

    # Matrix operations, should we support them?
    # Can't be easily applied on kraus map
    # __pow__ = Operator.__pow__
    # expm = Operator.expm
    # logm = Operator.logm
    # cosm = Operator.cosm
    # cosm = Operator.cosm
    sqrtm = _warn(Operator.sqrtm, "sqrtm")  # Fidelity used for choi
    # inv = Operator.inv

    # These depend on the matrix representation.
    tr = Operator.tr
    # diag = Operator.diag

    eigenstates = Operator.eigenstates  # Could be modified to return dm
    eigenenergies = Operator.eigenenergies
    # groundstate = Operator.groundstate  # Useful for super operator?

    def dual_chan(self) -> Qobj:
        """Dual channel of quantum object representing a completely positive
        map.
        """
        # Uses the technique of Johnston and Kribs (arXiv:1102.0948), which
        # is only valid for completely positive maps.
        if not self.iscp:
            raise ValueError("Dual channels are only implemented for CP maps.")
        J = qutip.to_choi(self)
        tensor_idxs = enumerate_flat(J.dims)
        J_dual = qutip.tensor_swap(
            J,
            *(
                list(zip(tensor_idxs[0][1], tensor_idxs[0][0])) +
                list(zip(tensor_idxs[1][1], tensor_idxs[1][0]))
            )
        ).trans()
        J_dual.superrep = 'choi'
        return J_dual

    def to_choi(self):
        return qutip.to_choi(self)

    def to_super(self):
        return qutip.to_super(self)

    def to_kraus(self):
        if self.ishp:
            return qutip.to_kraus(self)
        else:
            from qutip.core.superop_reps import _generalized_kraus
            pp = [(u, v.dag()) for u, v in _generalized_kraus(self.to_choi())]
            return KrausMap.generalizedKraus(general_terms=pp)


class KrausMap(SuperOperator):
    """
    Parameter
    ---------
    arg : list of Qobj

    """
    @staticmethod
    def _check_oper(oper, dims):
        if not isinstance(oper, Qobj) and oper.isoper:
            raise TypeError("KrausMap componants must be operators")
        if oper._dims != dims:
            raise TypeError(
                "KrausMap operators must all have the same dimensions"
            )

    def __init__(self, arg, dims=None, copy=False, **_):
        self._data = None
        self._kmap = []
        self._hpmap = []
        self._genmap = []
        dims_ = None
        for oper in arg:
            if dims_ is None:
                dims_ = oper._dims
            KrausMap._check_oper(oper, dims_)
            self._kmap.append(oper.copy() if copy else oper)

        if dims_ is not None:
            self._dims = Dimensions.from_prepost(dims_, dims_.transpose())
            if dims is not None and Dimensions(dims) != self._dims:
                raise ValueError(
                    "Provided dimensions do not match operators"
                )
        elif dims is not None:
            # Dimensions but no operators, this result in a map that always
            # a zeros state.
            self._dims = Dimensions(dims)
        else:
            raise ValueError("Missing information to initialise KrausMap")

        self._flags = {
            "ishp": True,
            "isherm": False,
            "isunitary": False,
        }
        self.superrep = "kraus"

    @classmethod
    def generalizedKraus(
        cls,
        kraus_terms: Sequence[Operator] = (),
        hp_terms: Sequence[Operator] = (),
        general_terms: Sequence[tuple[Operator, Operator]] = (),
        copy: bool = False,
    ) -> "KrausMap":
        """
        Create a generalized Kraus Map.

            G(rho) -> sum_i A[i] @ rho @ B[i]

        Internaly hermiticity preserving terms are kept appart:

            G(rho) ->
                sum_i K[i] @ rho @ K[i].dag()
                + sum_i (H[i] @ rho + rho @ H[i].dag())
                + sum_i A[i] @ rho @ B[i]

        Parameters
        ----------
        kraus_terms: list of Qobj
            Terms in the form: K[i] @ rho @ K[i].dag()

        hp_terms: list of Qobj
            Terms in the form: H[i] @ rho + rho @ H[i].dag()

        general_terms: list of tuple of Qobj, Qobj
            Terms in the form: A[i] @ rho @ B[i]
        """
        out = KrausMap.__new__(KrausMap)
        dims_pre = None
        dims_post = None
        self._data = None
        self._kmap = []
        self._hpmap = []
        self._genmap = []

        for oper in kraus_terms:
            if dims_pre is None:
                dims_pre = oper._dims
                dims_post = oper._dims.tr()
            KrausMap._check_oper(oper, dims_pre)
            self._kmap.append(oper.copy() if copy else oper)

        for oper in hp_terms:
            if dims_pre is None:
                dims_pre = oper._dims
                dims_post = oper._dims.tr()
            KrausMap._check_oper(oper, dims_pre)
            out._hpmap.append(oper.copy() if copy else oper)

        for pre, post in general_terms:
            if dims_pre is None:
                dims_pre = pre._dims
                dims_post = post._dims
            KrausMap._check_oper(pre, dims_pre)
            KrausMap._check_oper(post, dims_post)
            out._genmap.append(
                (pre.copy(), post.copy())
                if copy else (pre, post)
            )

        if dims_pre is None:
            raise ValueError(
                "Can't initialise general kraus map from no operators"
            )

        self._dims = Dimensions.from_prepost(dims_pre, dims_post)
        self._flags = {
            "isherm": False,
            "isunitary": False,
        }
        self.superrep = "kraus"

    def _pre_dims(self):
        return Dimensions(self._dims[1].oper[0], self._dims[0].oper[0])

    def _post_dims(self):
        return Dimensions(self._dims[0].oper[1], self._dims[1].oper[1])

    def __len__(self):
        return len(self._kmap) + len(self._hpmap) + len(self._genmap)

    def __call__(self, other: Qobj) -> Qobj:
        if not isinstance(other, Qobj):
            raise TypeError("Only defined for quantum objects.")
        if (self.type, other.type) not in _CALL_ALLOWED:
            raise TypeError(self.type + " cannot act on " + other.type)
        if other.isket:
            other = other.proj()
        out = qzeros(self._dims[1])
        for oper in self._kmap:
            out += oper @ other @ oper.dag()
        for oper in self._hpmap:
            out += oper @ other + other @ oper.dag()
        for pre, post in self._genmap:
            out += pre @ other @ post
        return out

    @property
    def data(self):
        raise ValueError("data not defined")

    @property
    def shape(self):
        return self._pre_dims().shape

    def dtype(self):
        raise NotImplementedError

    def to(self, data_type: LayerType, copy: bool=False) -> Qobj:
        raise NotImplementedError

    @_require_equal_type
    def __add__(self, other):
        if other == 0:
            return self.copy()
        return KrausMap.generalizedKraus(
            kraus_terms = self._kmap + other._kmap,
            hp_terms = self._hpmap + other._hpmap,
            general_terms = self._genmap + other._genmap,
        )

    @_require_equal_type
    def __sub__(self, other):
        if other == 0:
            return self.copy()
        return self + other * -1

    def __neg__(self):
        return other * -1

    def __mul__(self, other):
        if isinstance(other, Qobj):
            return self.__matmul__(other)

        if other == 0:
            return KrausMap([], dims=self._dims)

        try:
            if other.real == other and other >= 0:
                sq = other.real ** 0.5
                kmap = [u * sq for u in self._kmap]
                genmap = []
            else:
                kmap = []
                genmap = [(u * other, u.dag()) for u in self._kmap]
            if other.real == other:
                hpmap = [u * other for u in self._hpmap]
            else:
                hpmap = []
                genmap += [
                    (u * other, qutip.qeye_like(u))
                    for u in self._hpmap
                ]
                genmap += [
                    (qutip.qeye_like(u), u.dag() * other)
                    for u in self._hpmap
                ]
            genmap += [
                (u, v * other)
                for u, v in self._genmap
            ]
        except TypeError:
            return NotImplemented

        return KrausMap.generalizedKraus(
            kraus_terms = kmap,
            hp_terms = hpmap,
            general_terms = genmap,
        )

    def __matmul__(self, other):
        return NotImplemented

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Qobj) or self._dims != other._dims:
            return False
        return NotImplementedError

    def __and__(self, other):
        raise NotImplementedError

    def dag(self):
        raise NotImplementedError

    def conj(self):
        raise NotImplementedError

    def trans(self):
        raise NotImplementedError

    def copy(self):
        return KrausMap.generalizedKraus(
            kraus_terms=self._kmap,
            hp_terms=self._hpmap,
            general_terms=self._genmap,
            copy=True
        )

    def __str__(self):
        out = self._str_header()
        if self._genmap or self._hpmap:
            type = "Generalized " + out
        return out

    def _repr_latex_(self):
        return self.__str__()

    def __getitem__(self, ind):
        raise NotImplementedError

    def full(self):
        shape = self.shape
        if not (self._genmap or self._hpmap):
            out = np.zeros((len(self._kmap), *shape), dtype=complex)
            for i, oper in enumerate(self._kmap):
                out[i] = oper.full()
        else:
            nk = len(self._kmap)
            nhp = len(self._hpmap)
            ng = len(self._genmap)
            out = np.zeros((nk + 2 * nhp + ng, 2, *shape), dtype=complex)
            for i, oper in enumerate(self._kmap):
                out[i, 0] = oper.full()
                out[i, 1] = oper.dag().full()
            for i, oper in enumerate(self._hpmap):
                out[2 * i + nk, 0] = oper.full()
                out[2 * i + nk, 1] = qutip.qeye(self._post_dims())
                out[2 * i + nk, 0] = qutip.qeye(self._pre_dims())
                out[2 * i + nk, 1] = oper.dag().full()
            for i, (u, v) in enumerate(self._genmap):
                out[i + nk + 2 * nhp, 0] = u.full()
                out[i + nk + 2 * nhp, 1] = v.full()
        return out

    def data_as(self):
        raise NotImplementedError

    def data_as(self):
        raise NotImplementedError

    def contract(self):
        raise NotImplementedError

    def permute(self):
        raise NotImplementedError

    def tidyup(self):
        raise NotImplementedError

    def norm(self):
        raise NotImplementedError

    def unit(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def overlap(self):
        raise NotImplementedError

    def ptrace(self):
        raise NotImplementedError

    def purity(self):
        raise NotImplementedError

    def to_super(self):
        out = 0
        for oper in self._kmap:
            out += qutip.to_super(oper)
        for oper in self._hpmap:
            out += qutip.spre(oper)
            out += qutip.spost(oper.dag())
        for pre, post in self._genmap:
            out += qutip.sprepost(pre, post)
        out.superrep = "super"
        return out

    def to_choi(self):
        out = 0
        print(len(self._kmap), len(self._hpmap), len(self._genmap))
        for oper in self._kmap:
            vec = qutip.operator_to_vector(oper)
            out += vec @ vec.dag()
        for oper in self._hpmap:
            ones = qeye_like(oper)
            out += qutip.operator_to_vector(oper) @ qutip.operator_to_vector(ones).dag()
            out += qutip.operator_to_vector(ones) @ qutip.operator_to_vector(oper).dag()
        for pre, post in self._genmap:
            out += qutip.operator_to_vector(pre) @ qutip.operator_to_vector(post).trans()
        print(out)
        out.superrep = "choi"
        return out

    @property
    def iscp(self) -> bool:
        if self._flags.get("iscp", None) is None:
            if not self._hpmap and not self._genmap:
                self._flags["iscp"] = True
            else:
                choi = self.to_choi()
                self._flags["iscp"] = choi.iscp
        return self._flags["iscp"]

    @property
    def ishp(self) -> bool:
        if self._flags.get("ishp", None) is None:
            if not self._genmap:
                self._flags["ishp"] = True
            else:
                choi = self.to_choi()
                self._flags["ishp"] = choi.ishp
        return self._flags["ishp"]


_QobjBuilder.qobjtype_to_class["super"] = SuperOperator
