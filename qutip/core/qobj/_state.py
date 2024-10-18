from ._base import Qobj, _QobjBuilder
from .. import data as _data


__all__ = []

class _StateQobj(Qobj):
    def proj(self) -> Qobj:
        """Form the projector from a given ket or bra vector.

        Parameters
        ----------
        Q : :class:`.Qobj`
            Input bra or ket vector

        Returns
        -------
        P : :class:`.Qobj`
            Projection operator.
        """
        dims = ([self._dims[0], self._dims[0]] if self.isket
                else [self._dims[1], self._dims[1]])
        return Qobj(_data.project(self._data),
                    dims=dims,
                    isherm=True,
                    copy=False)

    @property
    def ishp(self) -> bool:
        return False

    @property
    def iscp(self) -> bool:
        return False

    @property
    def istp(self) -> bool:
        return False

    @property
    def iscptp(self) -> bool:
        return False

    @property
    def isherm(self) -> bool:
        return False

    @property
    def isunitary(self) -> bool:
        return False

    @property
    def isoper(self) -> bool:
        return False

    @property
    def issuper(self) -> bool:
        return False

    @property
    def isoperket(self) -> bool:
        return False

    @property
    def isoperbra(self) -> bool:
        return False


class Bra(_StateQobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type != "bra":
            raise ValueError(
                f"Expected bra dimensions, but got {self._dims.type}"
            )

    @property
    def isbra(self) -> bool:
        return True

    @property
    def isket(self) -> bool:
        return False


class Ket(_StateQobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type != "ket":
            raise ValueError(
                f"Expected ket dimensions, but got {self._dims.type}"
            )

    @property
    def isbra(self) -> bool:
        return False

    @property
    def isket(self) -> bool:
        return True


class OperKet(_StateQobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type != "operator-ket":
            raise ValueError(
                f"Expected ket dimensions, but got {self._dims.type}"
            )

    @property
    def isbra(self) -> bool:
        return False

    @property
    def isket(self) -> bool:
        return False

    @property
    def isoperket(self) -> bool:
        return True

    @property
    def isoperbra(self) -> bool:
        return False


class OperBra(_StateQobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type != "operator-bra":
            raise ValueError(
                f"Expected ket dimensions, but got {self._dims.type}"
            )

    @property
    def isbra(self) -> bool:
        return False

    @property
    def isket(self) -> bool:
        return False

    @property
    def isoperket(self) -> bool:
        return False

    @property
    def isoperbra(self) -> bool:
        return True


_QobjBuilder.qobjtype_to_class["ket"] = Ket
_QobjBuilder.qobjtype_to_class["bra"] = Bra
_QobjBuilder.qobjtype_to_class["operator-ket"] = OperKet
_QobjBuilder.qobjtype_to_class["operator-bra"] = OperBra
