from ._operator import Operator
from ._base import Qobj, _QobjBuilder
import qutip
from ..dimensions import enumerate_flat, collapse_dims_super
import numpy as np
from ...settings import settings


class SuperOperator(Operator):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type not in ["super"]:
            raise ValueError(
                f"Expected super operator dimensions, but got {self._dims.type}"
            )

    @property
    def isoper(self) -> bool:
        """Indicates if the Qobj represents an operator."""
        return False

    @property
    def issuper(self) -> bool:
        """Indicates if the Qobj represents a superoperator."""
        return True

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
            KrausMap._check_oper(operm, dims_)
            self._kmap.append(oper.copy() if copy else oper)

        if dims_ is not None:
            self._dims = Dimensions.from_prepost(dims_, dims_.tr())
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

        self._flags = {"ishp": True}
        self.superrep = "kraus"

    @classmethod
    def generalizedKraus(
        cls,
        kraus_terms: Iterator[Operator] = (),
        hp_terms: Iterator[Operator] = (),
        general_terms: Iterator[tuple[Operator, Operator]] = (),
        copy: bool = False,
    ) -> KrausMap:
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
        self.superrep = "generalized kraus"

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
        for pre, post in self._kmap:
            out += pre @ other @ post()
        return out

    @property
    def data(self):
        raise ValueError("data not defined")


_QobjBuilder.qobjtype_to_class["super"] = SuperOperator
