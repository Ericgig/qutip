from .qobj_base import Qobj

class _StateQobj(Qobj):
    _auto_dm = ["ptrace",]

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
    def __init__(self, data, dims, copy):
        super.__init__(data, dims, copy)
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
    def __init__(self, data, dims, copy):
        super.__init__(data, dims, copy)
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
