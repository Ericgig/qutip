from typing import NamedTuple, Any
import itertools
import numpy as np

from qutip.core.data import Data
from qutip.core import data as _data
from qutip.core.data.permute import get_permutations
from qutip.settings import settings
from qutip.typing import LayerType

from .utils import (
    Transform, conj_transform, trans_transform, adjoint_transform,
    _compare_hilbert
)

__all__ = ["Operator"]


class ProdTerm(NamedTuple):
    operator: Any
    modes: tuple[int]
    in_space: Dimensions
    out_space: Dimensions
    transform: Transform


class Term(NamedTuple):
    prod_terms: tuple[ProdTerm]
    factor: complex


def _read_dims(shape, dimension):
    if dimension is None:
        return Dimensions([[shape[0]], [shape[1]]])
    dimensions = Dimensions(dimension)
    if shape is not None and shape != dimensions.shape:
        raise ValueError("given dimensions and shape do not match")
    return dimensions


class Operator:
    """
    Symbolic layer between Data and Qobj.

    The operator is composed by a sum of terms, with each term composed of
    multiple product:

    op = sum_i term[i].factor * prod(term[i].prod_terms)

    prod(term[i].prod_terms) =
        expand_operator(oper[N-1]^trans[N-1], hilbert[N-1], hilbert_dims) @
        ...
        expand_operator(oper[1]^trans[1], hilbert[1], hilbert_dims) @
        expand_operator(oper[0]^trans[0], hilbert[0], hilbert_dims)

    This object always has (partial) knowledge of the full hilbert space.
    There is no dual representation, super operator will have the hilbert space
    doubled, with the operation being applied to the right first and
    transposed.
    # TODO: How super / Rectangular are supported?
    """
    terms: list
    _dims: Dimensions
    shape: tuple[int, int]

    def __init__(
        self,
        arg=None,
        mode=None,
        shape=None,
        dimension=None,
    ):
        self.terms = []
        if isinstance(mode, int):
            mode = (mode,)
        elif mode is None:
            mode = (0,)

        dimension = _read_dims(shape, dimension)

        if arg is None:
            if dimension is None:
                raise ValueError(
                    "Either hilbert_dims or shape must be provided "
                    "for empty Operator."
                )
            self._dims = dimension

        elif isinstance(arg, Data):
            if dimension is None:
                if mode != (0,):
                    raise ValueError(
                        "Dimensions are needed when creating an operator acting"
                        " on multiples modes."
                    )
                self._dims = _read_dims(arg.shape, None)
            else:
                if dimension.mode_shape(mode) != arg.shape:
                    raise ValueError("Data shape does not match hilbert space")
                self._dims = dimension

            self.terms.append(
                Term([ProdTerm(
                    arg,
                    mode or (0,),
                    self._dims[1],
                    self._dims[0],
                    Transform.DIRECT
                )], 1.+0j)
            )

        else:
            raise TypeError(f"{type(arg)} not supported.")

    @property
    def hilbert_space(self):
        if self._dims.issquare and not self._dims.issuper:
            return self._dims[0]
        elif self._dims.issquare and self._dims[0].oper.issquare:
            return self._dims[0].oper._dims[0]
        elif not self._dims.issuper:
            return (self._dims[0], self._dims[1])
        else:
            return (
                (self._dims[0].oper[0], self._dims[0].oper[1]),
                (self._dims[1].oper[0], self._dims[1].oper[1])
            )

    @property
    def left_hilbert_space(self):
        if self._dims.issquare and not self._dims.issuper:
            return self._dims[0].as_list()
        elif self._dims.issquare and self._dims[0].oper.issquare:
            return self._dims[0].oper._dims[0].as_list() * 2
        elif not self._dims.issuper:
            return self._dims[0].as_list()
        else:
            return self._dims[0].oper[0].as_list() + self._dims[0].oper[1].as_list()

    @property
    def right_hilbert_space(self):
        if self._dims.issquare and not self._dims.issuper:
            return self._dims[0]
        elif self._dims.issquare and self._dims[0].oper.issquare:
            return self._dims[0].oper._dims[0]
        elif not self._dims.issuper:
            return self._dims[1].as_list()
        else:
            return self._dims[1].oper[0].as_list() + self._dims[1].oper[1].as_list()

    @property
    def shape(self):
        return self._dims.shape

    @property
    def nnz(self):
        """
        Approximate number of non-zeros elements.
        """
        # TODO:
        return self.shape[0] * self.shape[1]

    def _dtype(self):
        """
        Guess best dtype when merging.
        """
        # TODO:
        return _data.Dia

    def _term_to_data(self, term, dtype):
        """
        Convert a term to a Data object.
        """
        max_modes = 0
        modes_affected = [0] * len(self.right_hilbert_space)

        for pterm in term.prod_terms
            _, modes, _ = pterm
            max_modes = max(max_modes, len(modes))
            for mode in modes:
                modes_affected[mode] += 1

        if max_modes == 0:
            # Empty prod term:
            if not self._dims.issquare:
                raise ValueError("Empty, non-square...")
            out = _data.identity[dtype](self._dims.shape[0], scale=term.factor)

        elif max_modes == 1:
            opers = [None] * len(modes_affected)
            for pterm in term.prod_terms
                oper, modes, transform = pterm
                mode = modes[0]
                if opers[mode] is None:
                    opers[mode] = _apply_transform(oper, transform)
                else:
                    opers[mode] = _apply_transform(oper, transform) @ opers[mode]

            out = _data.mul(oper[0], term.factor)
            for oper in opers[1:]:
                out = _data.kron(out, oper)

        else:
            # TODO: To optimize
            # - It is not always needed to expand to the full space before the
            # product.
            N = term.prod_terms[0].in_space.size
            out = _data.identity[dtype](N)

            for prod_term in term.prod_terms:
                oper, modes, in_space, out_space, transform = pterm
                oper = _apply_transform(oper, transform)

                in_hilbert = in_space.flat()
                out_hilbert = out_space.flat()
                idxs = list(range(len(hilbert)))
                in_sizes = []
                out_sizes = []
                for mode in modes:
                    in_sizes.append(in_hilbert[mode])
                    out_sizes.append(out_hilbert[mode])

                for i in reversed(sorted(modes)):
                    del idxs[i]

                for i in idxs:
                    N = in_hilbert[i]
                    oper = _data.kron(oper, _data.identity(N))
                    in_sizes.append(N)
                    out_sizes.append(N)

                order = np.argsort(list(prod_term.hilbert) + idxs)
                perm_row = get_permutations(out_sizes, order)
                perm_col = get_permutations(in_sizes, order)

                out = _data.permute.indices(oper, perm_row, perm_col) @ out

        return out

    def to_data(self, dtype: LayerType = None):
        """
        Convert to a single Data instance.

        Parameters
        ----------
        dtype: LayerType, optional
            Format of the output Data instance.
        """
        if dtype is None:
            dtype = self._dtype

        oper = self._term_to_data(self.terms[0], dtype)

        for term in self.terms[1:]:
            out = out + self._term_to_data(term, dtype)
        return out

    def to_array(self):
        """
        Convert to a numpy array.
        """
        return self.to_data().to_array()

    def conj(self):
        new = Operator(dimension=self._dims)
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.append(ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.in_space,
                    pterm.out_space,
                    conj_transform[pterm.transform],
                ))
            new.terms.append(Term(tuple(pterms), term.factor.conjugate()))
        return new

    def transpose(self):
        new = Operator(dimension=self._dims)
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                pterms.append(ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.out_space,
                    pterm.in_space,
                    trans_transform[pterm.transform],
                ))
            new.terms.append(Term(tuple(pterms), term.factor))
        return new

    def adjoint(self):
        new = Operator(dimension=self._dims)
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                pterms.append(ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.out_space,
                    pterm.in_space,
                    adjoint_transform[pterm.transform],
                ))
            new.terms.append(Term(tuple(pterms), term.factor.conjugate()))
        return new

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented

        if self._dims != other._dims:
            raise ValueError("Incompatible dimensions.")
        new = Operator(dimensions=self._dims)
        new.terms = self.terms + other.terms
        return new

    def __sub__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented

        if self._dims != other._dims:
            raise ValueError("Incompatible dimensions.")
        # TODO: cancellation detection.
        # With this A - A would not result in `0`, but in 2 opposite tests.
        return self + -other

    def __mul__(self, other):
        new = Operator(dimensions=self._dims)
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms:
                copy_term.prod_terms.append(ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.in_space,
                    pterm.out_space,
                    pterm.transform,
                ))
            new.terms.append(Term(tuple(pterms), term.factor * other))
        return new

    def __truediv__(self, other):
        return self * (1 / other)

    def __matmul__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented

        out_dims = self._dims @ other._dims
        new = Operator(dimensions=out_dims)

        for term_left, term_right in itertools.product(left.terms, right.terms):
            new.terms.append(
                Term(
                    term_right.prod_terms + term_left.prod_terms,
                    term_left.factor * term_right.factor,
                )
            )

        return new

    def __and__(left, right):
        N = len(left.hilbert_dims)
        dims = [[left._dims[0],  right._dims[0]], [left._dims[1],  right._dims[1]]]

        left_ext = Operator(dimensions=dims)
        for term in left.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.prod_terms.append(ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.in_space,
                    pterm.out_space,
                    pterm.transform,
                ))
            left_ext.terms.append(Term(tuple(pterms), factor=term.factor))

        right_shifted = Operator(dimensions=dims)
        for term in right.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.prod_terms.append(ProdTerm(
                    pterm.operator,
                    tuple(i + N for i in pterm.modes),
                    pterm.in_space,
                    pterm.out_space,
                    pterm.transform,
                ))
            right_shifted.terms.append(Term(tuple(pterms), factor=term.factor))

        return left_ext @ right_shifted
