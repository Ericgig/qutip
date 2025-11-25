from typing import NamedTuple, Any
from types import NoneType
from dataclasses import dataclass
import itertools
import numpy as np

from qutip.core.data import Data
from qutip.core import data as _data
from qutip.core.dimensions import Space, Dimensions
from qutip.core.data.permute import get_permutations
from qutip.core.coefficient import Coefficient, coefficient
from qutip.settings import settings
from qutip.typing import LayerType

from qutip.core._qobj.utils import (
    Transform,
    conj_transform,
    trans_transform,
    adjoint_transform,
    apply_transform
)

__all__ = ["Operator"]


@dataclass(frozen=True)
class _ProdTerm:
    """
    Elementary operator inserted in a larger quantum object.

    ``operator`` is the matrix that is first transformed by ``transform`` then
    applied on the modes of a quantum system of the given dimensions.
    """
    operator: Any  # Data or Element?
    modes: tuple[int]
    in_space: list
    out_space: list
    transform: Transform

    def to_data(self, hilbert_space: list, issuper: bool) -> Data:
        in_hilbert = hilbert_space.copy()
        in_sizes = self.in_space
        out_sizes = self.out_space
        N = len(hilbert_space)

        if self.in_space == self.out_space:
            out_hilbert = in_hilbert
        else:
            out_hilbert = hilbert_space.copy()
            for i, mode in enumerate(self.modes):
                out_hilbert[mode] = out_sizes[i]

        super_element = False
        if issuper:
            super_element = not (
                all(mode < N//2 for mode in self.modes)
                or all(mode >= N//2 for mode in self.modes)
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
                mode_size = in_hilbert[(i + N//2) % N]
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
        return hash((
            id(self.operator), self.modes, tuple(self.in_space),
            tuple(self.out_space), self.transform
        ))


@dataclass(frozen=True)
class _Term:
    prod_terms: list[_ProdTerm]
    hilbert_space: Dimensions
    factor: complex

    def to_data(self, t: float, dtype):
        """
        Convert a term to a Data object.
        """
        max_modes = 0
        modes_affected = [0] * len(self.hilbert_space[1].flat())

        for pterm in self.prod_terms:
            modes = pterm.modes
            max_modes = max(max_modes, len(modes))
            for mode in modes:
                modes_affected[mode] += 1

        if max_modes == 0:
            # Empty prod term:
            if not self.hilbert_space.issquare:
                raise ValueError("Empty, non-square...")
            out = _data.identity[dtype](
                self.hilbert_space.shape[0], scale=self.factor(t)
            )

        elif max_modes == 1:
            opers = [None] * len(self.hilbert_space[1])
            for pterm in self.prod_terms:
                oper = apply_transform(pterm.operator, pterm.transform)
                mode = pterm.modes[0]
                if opers[mode] is not None:
                    opers[mode] = oper @ opers[mode]
                else:
                    opers[mode] = oper

            for i in range(len(opers)):
                if opers[i] is not None:
                    continue
                opers[i] = _data.identity[dtype](self.hilbert_space[0].flat()[i])

            if not self.hilbert_space.issuper:
                out = _data.mul(opers[0], self.factor(t))
                for oper in opers[1:]:
                    out = _data.kron(out, oper)
            else:
                N = len(self.hilbert_space[1]) // 2
                pre = opers[0]
                post = _data.mul(opers[N], self.factor(t))
                for i in range(1, N):
                    pre = _data.kron(pre, opers[i])
                    post = _data.kron(post, opers[i + N])
                # superrep = super: inverted post.T & pre
                # The transpose already accounted for in pterm.transform
                out = _data.kron(post, pre)

        else:
            # TODO: To optimize
            # It's not always needed to expand to the full space before the
            #   product.
            space = self.hilbert_space[1]
            out = _data.identity[dtype](space.size, scale=self.factor(t))
            hilbert = space.flat()
            for prod_term in self.prod_terms:
                hilbert, oper = prod_term.to_data(
                    hilbert, self.hilbert_space.issuper
                )
                out = oper @ out

        return out

    def _compressed(self):
        """
        Reorder and merge prod_terms.
        Return a new _Term.
        """
        if len(self.prod_terms) == 0:
            return self

        # make_layer = lambda : [[] for _ in range(len(self.hilbert_space))]
        # commute = lambda layer, modes: not any(layer[m] for m in modes)
        # compatible = lambda layer, modes: (
        #     layer[modes[0]]
        #     and layer[modes[0]][0].modes == modes
        # )

        N = len(self.hilbert_space[1].flat())
        layers = []
        front = {mode: -1 for mode in range(N)}

        for term in self.prod_terms:
            modes = term.modes
            target_layer_idx = max(front[mode] for mode in modes)

            can_merge = False
            if target_layer_idx >= 0:
                layer = layers[target_layer_idx]
                if modes in layer:
                    can_merge = True

            if can_merge:
                layers[target_layer_idx][modes].append(pterm)
            else:
                target_layer_idx += 1
                if target_layer_idx == len(layers):
                    layers.append({})
                layers[target_layer_idx][modes] = [pterm]
                for m in modes:
                    frontier[m] = new_idx

        # Now we have each layer[n][modes] is a list of operator that can act
        # on the same modes and can be merged together.

        pterm = []
        for layer in layers:
            done_modes = set()
            for mode, terms in enumerate(layer):
                if mode in done_modes:
                    continue
                if not terms:
                    continue
                if len(terms) == 1:
                    pterm.append(terms[0])
                    continue

                oper = apply_transform(term[0].operator, term[0].transform)
                done_modes |= set(terms[0].modes)
                # TODO: We probably should not always merge. `ket @ bra` in dense format
                # in probably faster as 2 operations. etc.
                for term in terms[1:]:
                    oper = apply_transform(pterm.operator, pterm.transform) @ oper
                pterm.append(_ProdTerm(
                    oper,
                    terms[0].modes,
                    terms[0].in_space,
                    terms[-1].out_space,
                    Transform.DIRECT,
                ))
        return _Term(tuple(pterm), self.hilbert_space, self.factor)


def _read_dims(shape, dimension):
    if dimension is None and shape is None:
        return None
    if dimension is None:
        return Dimensions([[shape[0]], [shape[1]]])
    dimensions = Dimensions(dimension)
    if shape is not None and shape != dimensions.shape:
        raise ValueError("given dimensions and shape do not match")
    if dimensions[0].flat() != dimensions[1].flat():
        N = max(len(dimensions[0].flat()), len(dimensions[1].flat()))
        dimensions = Dimensions([
            dimensions[0].expand_scalar_dims(N),
            dimensions[1].expand_scalar_dims(N),
        ])

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
    """
    terms: list
    _dims: Dimensions
    shape: tuple[int, int]

    def __init__(
        self,
        arg=None,
        modes=None,
        shape=None,
        dimension=None,
    ):
        self.terms = []
        if isinstance(modes, int):
            modes = (modes,)
        elif modes is None:
            modes = (0,)

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
                if modes != (0,):
                    raise ValueError(
                        "Dimensions are needed when creating an operator "
                        "acting on multiples modes."
                    )
                self._dims = _read_dims(arg.shape, None)
            else:
                if dimension._shape_modes(modes) != arg.shape:
                    raise ValueError("Data shape does not match hilbert space")
                self._dims = dimension

            self.terms.append(
                _Term((_ProdTerm(
                    arg,
                    modes,
                    self._dims[1]._extract_modes(modes).as_list(),
                    self._dims[0]._extract_modes(modes).as_list(),
                    Transform.DIRECT
                ),), self._dims, coefficient(1.))
            )

        else:
            raise TypeError(f"{type(arg)} not supported.")

    @classmethod
    def identity(self, hilbert_space):
        out = Operator(dimension=[hilbert_space] * 2)
        out.terms.append(_Term((), out._dims, coefficient(1)))
        return out

    @classmethod
    def from_qobj(self, qobj):
        return Operator(
            qobj.data,
            dimension=qobj._dims,
            modes=tuple(range(len(qobj._dims[0].as_list())))
        )

    @property
    def hilbert_space(self) -> Space | NoneType:
        if self._dims.issquare and not self._dims.issuper:
            return self._dims[0]
        elif self._dims.issquare and self._dims[0].oper.issquare:
            return self._dims[0].oper[0]
        else:
            return None

    @property
    def dims(self):
        return self._dims.as_list()

    @property
    def shape(self):
        return self._dims.shape

    @property
    def isconstant(self):
        return all(
            isinstance(term.factor, ConstantCoefficient)
            for term in self.terms
        )

    def to_data(self, t=0, dtype: LayerType = None):
        """
        Convert to a single Data instance.

        Parameters
        ----------
        dtype: LayerType, optional
            Format of the output Data instance.
        """
        if dtype is None:
            # Lowest priority in many operation
            dtype = _data.Dia

        if len(self.terms) == 0:
            return _data.zeros[dtype](*self.shape)

        out = self.terms[0].to_data(t, dtype)

        for term in self.terms[1:]:
            out = out + term.to_data(t, dtype)
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
                pterms.append(_ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.in_space.as_list(),
                    pterm.out_space.as_list(),
                    conj_transform[pterm.transform],
                ))
            new.terms.append(
                _Term(tuple(pterms), self._dims, term.factor.conj())
            )
        return new

    def transpose(self):
        new = Operator(dimension=Dimensions(self._dims[0], self._dims[1]))
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.out_space.as_list(),
                    pterm.in_space.as_list(),
                    trans_transform[pterm.transform],
                ))
            new.terms.append(_Term(tuple(pterms), new._dims, term.factor))
        return new

    def adjoint(self):
        new = Operator(dimension=Dimensions(self._dims[0], self._dims[1]))
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.out_space.as_list(),
                    pterm.in_space.as_list(),
                    adjoint_transform[pterm.transform],
                ))
            new.terms.append(
                _Term(tuple(pterms), new._dims, term.factor.conj())
            )
        return new

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented

        if self._dims != other._dims:
            raise ValueError("Incompatible dimensions.")
        new = Operator(dimension=self._dims)
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
        if isinstance(other, (int, float, complex)):
            other = coefficient(other)
        if not isinstance(other, Coefficient):
            return NotImplemented
        new = Operator(dimension=self._dims)
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.in_space.as_list(),
                    pterm.out_space.as_list(),
                    pterm.transform,
                ))
            new.terms.append(
                _Term(tuple(pterms), self._dims, term.factor * other)
            )
        return new

    def __truediv__(self, other):
        return self * (1 / other)

    def __matmul__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented

        out_dims = self._dims @ other._dims
        new = Operator(dimension=out_dims)

        # TODO: Term contraction missing

        for term_l, term_r in itertools.product(self.terms, other.terms):
            new.terms.append(
                _Term(
                    term_r.prod_terms + term_l.prod_terms,
                    out_dims,
                    term_l.factor * term_r.factor,
                )
            )

        new.merge_terms()
        return new

    def _tensor_super(left, right):
        N = len(left.hilbert_space)
        dims = [
            [left._dims[0],  right._dims[0]],
            [left._dims[1],  right._dims[1]]
        ]

        left_ext = Operator(dimension=dims)
        for term in left.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.in_space.as_list(),
                    pterm.out_space.as_list(),
                    pterm.transform,
                ))
            left_ext.terms.append(
                _Term(tuple(pterms), dims, factor=term.factor)
            )

        right_shifted = Operator(dimension=dims)
        for term in right.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    tuple(i + N for i in pterm.modes),
                    pterm.in_space.as_list(),
                    pterm.out_space.as_list(),
                    pterm.transform,
                ))
            right_shifted.terms.append(
                _Term(tuple(pterms), dims, factor=term.factor)
            )

        return left_ext @ right_shifted

    def __and__(left, right):
        """
        `kron` operation.
        """
        if left._dims.issuper and right._dims.issuper:
            return _tensor_super(left, right)
        if left._dims.issuper or right._dims.issuper:
            raise TypeError("Can't compound normal and super space together.")
        N = len(left.hilbert_space)
        dims = [
            [left._dims[0],  right._dims[0]],
            [left._dims[1],  right._dims[1]]
        ]

        left_ext = Operator(dimension=dims)
        for term in left.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    pterm.modes,
                    pterm.in_space.as_list(),
                    pterm.out_space.as_list(),
                    pterm.transform,
                ))
            left_ext.terms.append(
                _Term(tuple(pterms), dims, factor=term.factor)
            )

        right_shifted = Operator(dimension=dims)
        for term in right.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    tuple(i + N for i in pterm.modes),
                    pterm.in_space.as_list(),
                    pterm.out_space.as_list(),
                    pterm.transform,
                ))
            right_shifted.terms.append(
                _Term(tuple(pterms), dims, factor=term.factor)
            )

        return left_ext @ right_shifted

    def spre(self):
        if self._dims.issuper:
            raise TypeError("Already a superoperator")
        if not self._dims.issquare:
            raise TypeError("spost only defined for square dimensions")

        super_op = Operator(dimension=[self._dims, self._dims])
        for term in self.terms:
            super_op.terms.append(
                _Term(term.prod_terms, super_op._dims, factor=term.factor)
            )

        return super_op

    def spost(self):
        if self._dims.issuper:
            raise TypeError("Already a superoperator")
        if not self._dims.issquare:
            raise TypeError("spost only defined for square dimensions")

        super_op = Operator(dimension=[self._dims, self._dims])
        N = len(self.hilbert_space)
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    tuple(i + N for i in pterm.modes),
                    pterm.out_space.as_list(),
                    pterm.in_space.as_list(),
                    trans_transform[pterm.transform],
                ))
            super_op.terms.append(
                _Term(tuple(pterms), super_op._dims, factor=term.factor)
            )

        return super_op

    def sprepost(self, post):
        if self._dims.issuper or post._dims.issuper:
            raise TypeError("Already a superoperator")

        out_dims = Dimensions([
            Dimensions([self._dims[0], post._dims[1]]),
            Dimensions([self._dims[1], post._dims[0]]),
        ])

        pre_part = Operator(dimension=out_dims)
        post_part = Operator(dimension=out_dims)
        N = len(self.hilbert_space)
        for term in self.terms:
            pre_part.terms.append(
                _Term(term.prod_terms, factor=term.factor)
            )
        for term in post.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                pterms.append(_ProdTerm(
                    pterm.operator,
                    tuple(i + N for i in pterm.modes),
                    pterm.out_space.as_list(),
                    pterm.in_space.as_list(),
                    trans_transform[pterm.transform],
                ))
            post_part.terms.append(
                _Term(tuple(pterms), out_dims, factor=term.factor)
            )

        return pre_part @ post_part

    def merge_terms(self):
        grouped_terms = {}

        for term in self.terms:
            signature = []
            for pterm in term.prod_terms:
                signature.append(hash(pterm))

            key = tuple(signature)

            if key in grouped_terms:
                term, factor = grouped_terms[key]
                grouped_terms[key] = (term, factor + term.factor)
            else:
                grouped_terms[key] = (term, term.factor)

        new_terms = []
        for term, factor in grouped_terms.values():
            if (
                isinstance(factor, ConstantCoefficient)
                and abs(factor(0)) > settings.core["atol"]
            ):
                continue
            new_terms.append(_Term(term.prod_terms, self._dims, factor))

        self.terms = new_terms
