from __future__ import annotations  # Required for Sphinx to follow
                                    # autodoc_type_aliases

from qutip.typing import LayerType, ElementType, QobjEvoLike
from qutip.core.qobj import Qobj
from qutip.core.data import Data
from qutip.core.coefficient import Coefficient
from numbers import Number
from numpy.typing import ArrayLike
from typing import Any, overload, Callable


class QobjEvo:
    dims: list
    isbra: bool
    isconstant: bool
    isket: bool
    isoper: bool
    isoperbra: bool
    isoperket: bool
    issuper: bool
    num_elements: int
    shape: tuple[int, int]
    superrep: str
    type: str
    def __init__(
        self,
        Q_object: QobjEvoLike,
        args: dict[str, Any] = None,
        *,
        copy: bool = True,
        compress: bool = True,
        function_style: str = None,
        tlist: ArrayLike = None,
        order: int = 3,
        boundary_conditions: tuple | str = None,
    ) -> None: ...
    @overload
    def arguments(self, new_args: dict[str, Any]) -> None: ...
    @overload
    def arguments(self, **new_args) -> None: ...
    def compress(self) -> QobjEvo: ...
    def tidyup(self, atol: Number) -> QobjEvo: ...
    def copy(self) -> QobjEvo: ...
    def conj(self) -> QobjEvo: ...
    def dag(self) -> QobjEvo: ...
    def trans(self) -> QobjEvo: ...
    def to(self, data_type: LayerType) -> QobjEvo: ...
    def linear_map(self, op_mapping: Callable[[Qobj], Qobj]) -> QobjEvo: ...
    def expect(self, t: Number, state: Qobj, check_real: bool = True) -> Number: ...
    def expect_data(self, t: Number, state: Data) -> Number: ...
    def matmul(self, t: Number, state: Qobj) -> Qobj: ...
    def matmul_data(self, t: Number, state: Data, out: Data = None) -> Data: ...
    def to_list(self) -> list[ElementType]: ...
    def __add__(self, other: QobjEvo | Qobj | Number) -> QobjEvo: ...
    def __iadd__(self, other: QobjEvo | Qobj | Number) -> QobjEvo: ...
    def __radd__(self, other: QobjEvo | Qobj | Number) -> QobjEvo: ...
    def __sub__(self, other: QobjEvo | Qobj | Number) -> QobjEvo: ...
    def __isub__(self, other: QobjEvo | Qobj | Number) -> QobjEvo: ...
    def __rsub__(self, other: QobjEvo | Qobj | Number) -> QobjEvo: ...
    def __and__(self, other: Qobj | QobjEvo) -> QobjEvo: ...
    def __rand__(self, other: Qobj | QobjEvo) -> QobjEvo: ...
    def __call__(self, t: float, **new_args) -> Qobj: ...
    def __matmul__(self, other: Qobj | QobjEvo) -> QobjEvo: ...
    def __imatmul__(self, other: Qobj | QobjEvo) -> QobjEvo: ...
    def __rmatmul__(self, other: Qobj | QobjEvo) -> QobjEvo: ...
    def __mul__(self, other: Number | Coefficient) -> QobjEvo: ...
    def __imul__(self, other: Number | Coefficient) -> QobjEvo: ...
    def __rmul__(self, other: Number | Coefficient) -> QobjEvo: ...
    def __truediv__(self, other : Number) -> QobjEvo: ...
    def __idiv__(self, other : Number) -> QobjEvo: ...
    def __neg__(self) -> QobjEvo: ...
    def __reduce__(self): ...
