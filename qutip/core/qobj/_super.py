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
        """Indicates if the Qobj represents a superoperator."""
        return self._dims.type == "scalar"

    @property
    def issuper(self) -> bool:
        """Indicates if the Qobj represents a superoperator."""
        return True

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
        J_dual = qutip.tensor_swap(J, *(
                list(zip(tensor_idxs[0][1], tensor_idxs[0][0])) +
                list(zip(tensor_idxs[1][1], tensor_idxs[1][0]))
        )).trans()
        J_dual.superrep = 'choi'
        return J_dual


_QobjBuilder.qobjtype_to_class["super"] = SuperOperator
