from typing import Any
from dataclasses import dataclass
import numpy as np

from qutip.core.data import Data
from qutip.core import data as _data
from qutip.core.data.permute import get_permutations
from qutip.core.cy._element import _BaseElement
from qutip.core.dimensions import Dimensions

from qutip.core._qobj.utils import *

__all__ = []


def _prepare_meta(modes, dimension):
    """
    Compute the size of modes before and after to insert matrix into.
    If the opertor apply to multiples modes which are not continous and ordered
    compute the permutations indexes.
    """
    start = min(modes)
    end = max(modes)

    before = 1
    after = 1
    hilbert_space = dimension[1].flat()
    N = len(hilbert_space)
    if dimension.issuper:
        hilbert_space = hilbert_space[N//2:] + hilbert_space[:N//2]

    for size in hilbert_space[:start]:
        before *= size
    for size in hilbert_space[end + 1:]:
        after *= size

    if len(modes) == 1 or np.all(np.diff(np.array(modes)) == 1):
        # All modes are consecutive, no permutation needed.
        return before, after, None, None, None

    hilbert_out = dimension[0].flat()  # Rectangular operator supported.
    if dimension.issuper:
        hilbert_out = hilbert_out[N//2:] + hilbert_out[:N//2]
    internal_modes = [mode - start for mode in modes]
    internal_hilbert_in = hilbert_space[start : end + 1]
    internal_hilbert_out = hilbert_out[start : end + 1]

    N = len(internal_hilbert_in)
    order = list(internal_modes)
    not_modes = set(range(N)) - set(internal_modes)

    in_sizes = []
    out_sizes = []
    for mode in internal_modes:
        in_sizes.append(internal_hilbert_in[mode])
        out_sizes.append(internal_hilbert_out[mode])
        order.append(mode)

    for i in not_modes:
        mode_size = internal_hilbert_in[i]
        out_side *= mode_size
        in_sizes.append(mode_size)
        out_sizes.append(mode_size)
        order.append(i)

    order = np.argsort(order)

    return before, after, in_sizes, out_sizes, order


def _reorder(
    oper: Data,
    in_sizes: list,
    out_sizes: list,
    order: list,
):
    """
    Permute the operator so it's mode are ordered and continuous.
    Insert identity mode if needed.
    """
    extra_N = np.prod(in_sizes) // oper.shape[1]
    oper = _data.kron(oper, _data.identity(extra_N))
    perm_row = get_permutations(out_sizes, order)
    perm_col = get_permutations(in_sizes, order)

    return _data.permute.indices(oper, perm_row, perm_col)


def _insert(before: int, oper: Data, after: int):
    """
    Insert a Data object into a larger one with kron product.
    """
    if before > 1:
        oper = _data.kron(_data.identity(before), oper)
    if after > 1:
        oper = _data.kron(oper, _data.identity(after))
    return oper


@dataclass(frozen=True)
class _ProdTerm:
    """
    Elementary operator inserted in a larger quantum object.

    ``operator`` is the matrix that is first transformed by ``transform`` then
    applied on the modes of a quantum system of the given dimensions.
    """
    # The operator is normaly Data, but QobjEvo's Elements are supported for
    # function callback.
    # Any is for Operator without circular import (TODO: Fix)
    # But anything that can be converted to a matrix could be supported.
    operator: Data | _BaseElement | Any
    modes: tuple[int]
    dimension: Dimensions
    transform: Transform

    def to_data(self, t=None) -> Data:
        """
        Return as a single Data object on the local modes.
        Apply the transformation and resolve callback.
        """
        from .oper import Operator
        if isinstance(self.operator, Data):
            oper = self.operator
        elif isinstance(self.operator, _BaseElement):
            oper = _data.mul(self.operator.data(t), self.operator.coeff(t))
        elif isinstance(self.operator, Operator):
            oper = self.operator.to_data(t)
        return apply_transform(self.operator, self.transform)

    def expand_data(self, t=None) -> Data:
        """
        Return the local operator inserted in the larger hilbert space.
        """
        N = len(self.dimension[0].flat())
        in_space = [self.dimension[1].flat()[mode] for mode in self.modes]
        out_space = [self.dimension[0].flat()[mode] for mode in self.modes]

        super_element = False  # Is it already in super representation?
        if self.dimension.issuper:
            super_element = not (
                all(mode < N // 2 for mode in self.modes)
                or all(mode >= N // 2 for mode in self.modes)
            )

        if self.dimension.issuper and not super_element:
            # Super representation stack in F order, but tensor in C order.
            # Rearange everything in effective C order.
            modes = [((mode + N // 2) % N) for mode in self.modes]
            in_space = in_space[N//2:] + in_space[:N//2]
            out_space = out_space[N//2:] + out_space[:N//2]
        else:
            modes = list(self.modes)

        oper = self.to_data(t)

        before, after, in_sizes, out_sizes, order = _prepare_meta(
            modes, self.dimension
        )
        if order is not None:
            oper = _reorder(oper, in_sizes, out_sizes, order)

        return _insert(before, oper, after)

    def __hash__(self):
        return hash(
            (
                id(self.operator),  # Update when implementing hash for Data
                tuple(self.modes),
                self.dimension,
                self.transform,
            )
        )

    def _add_transform(self, transform_map):
        dims = self.dimension
        if transform_map in [trans_transform, adjoint_transform]:
            dims = Dimensions([dims[1], dims[0]])
        return _ProdTerm(
            self.operator,
            self.modes,
            dims,
            transform_map[self.transform],
        )
