from typing import Any
from dataclasses import dataclass
import numpy as np

from qutip.core.data import Data
from qutip.core import data as _data
from qutip.core.data.permute import get_permutations

from qutip.core._qobj.utils import Transform, apply_transform

__all__ = []


def _prepare_meta(modes, hilbert_space, space_in, space_out, issuper):
    hilbert_out = list(hilbert_space)
    if space_in != space_out:
        for i, mode in enumerate(modes):
            hilbert_out[mode] = space_out[i]

    start = min(modes)
    end = max(modes)

    before = 1
    after = 1

    if not issuper:
        internal_modes = [mode - start for mode in modes]
        internal_hilbert_in = hilbert_space[start : end + 1]
        internal_hilbert_out = hilbert_out[start : end + 1]
        for size in hilbert_space[:start]:
            before *= size
        for size in hilbert_space[end + 1:]:
            after *= size
    else:
        internal_modes = [mode for mode in modes]
        internal_hilbert_in = hilbert_space
        internal_hilbert_out = hilbert_out

    if len(modes) == 1 or np.all(np.diff(np.array(internal_modes)) == 1):
        return hilbert_out, None, None, None, before, after

    N = len(internal_hilbert_in)
    order = list(internal_modes)
    not_modes = set(range(N)) - set(internal_modes)

    super_element = False
    if issuper:
        super_element = not (
            all(mode < N // 2 for mode in internal_modes)
            or all(mode >= N // 2 for mode in internal_modes)
        )

    if issuper and not super_element:
        internal_modes = [((mode + N // 2) % N) for mode in internal_modes]

    in_sizes = []
    out_sizes = []
    for mode in internal_modes:
        in_sizes.append(internal_hilbert_in[mode])
        out_sizes.append(internal_hilbert_out[mode])
        order.append(i)

    for i in not_modes:
        if issuper:
            mode_size = internal_hilbert_in[(i + N // 2) % N]
        else:
            mode_size = internal_hilbert_in[i]
        out_side *= mode_size
        in_sizes.append(mode_size)
        out_sizes.append(mode_size)
        order.append(i)

    order = np.argsort(order)

    return hilbert_out, in_sizes, out_sizes, order, before, after


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

    operator: Data
    modes: tuple[int]
    in_space: list
    out_space: list
    transform: Transform

    def to_data(self, t=None) -> Data:
        return apply_transform(self.operator, self.transform)

    def expand_data(
        self, t=None, hilbert_space: list = None, issuper: bool = None
    ) -> Data:
        N = len(hilbert_space)
        super_element = False
        if issuper:
            super_element = not (
                all(mode < N // 2 for mode in self.modes)
                or all(mode >= N // 2 for mode in self.modes)
            )

        print("super_element", super_element)

        if super_element:
            return self.expand_data_2(t, hilbert_space, issuper)

        out_hilbert, in_sizes, out_sizes, order, before, after = _prepare_meta(
            self.modes, hilbert_space, self.in_space, self.out_space, issuper
        )
        if order is not None:
            oper = _reorder(self.to_data(t), in_sizes, out_sizes, order)
        else:
            oper = self.to_data(t)

        print(out_hilbert, hilbert_space, self.modes, issuper)
        print(self.in_space, self.out_space)
        print(before, oper.shape, after)
        print(_insert(before, oper, after).shape)
        return out_hilbert, _insert(before, oper, after)

    def expand_data_2(
        self, t=None, hilbert_space: list = None, issuper: bool = None
    ) -> Data:
        in_hilbert = list(hilbert_space)
        out_hilbert = in_hilbert.copy()
        in_sizes = self.in_space.copy()
        out_sizes = self.out_space.copy()
        N = len(hilbert_space)

        if self.in_space != self.out_space:
            for i, mode in enumerate(self.modes):
                out_hilbert[mode] = out_sizes[i]

        super_element = False
        if issuper:
            super_element = not (
                all(mode < N // 2 for mode in self.modes)
                or all(mode >= N // 2 for mode in self.modes)
            )

        if issuper and not super_element:
            modes = [((mode + N // 2) % N) for mode in self.modes]
        else:
            modes = list(self.modes)

        order = list(modes)
        modes = set(modes)
        not_modes = set(range(N)) - modes

        oper = apply_transform(self.operator, self.transform)
        out_side = 1

        for i in not_modes:
            if issuper:
                mode_size = in_hilbert[(i + N // 2) % N]
            else:
                mode_size = in_hilbert[i]
            out_side *= mode_size
            in_sizes.append(mode_size)
            out_sizes.append(mode_size)
            order.append(i)

        if len(in_hilbert) > len(modes):
            oper = _data.kron(oper, _data.identity(out_side))

        order = np.argsort(order)
        perm_row = get_permutations(out_sizes, order)
        perm_col = get_permutations(in_sizes, order)

        return out_hilbert, _data.permute.indices(oper, perm_row, perm_col)

    def __hash__(self):
        return hash(
            (
                id(self.operator),
                self.modes,
                tuple(self.in_space),
                tuple(self.out_space),
                self.transform,
            )
        )
