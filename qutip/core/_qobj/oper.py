from typing import NamedTuple, Any
from types import NoneType
from dataclasses import dataclass
import itertools
import numpy as np

from qutip.core.data import Data
from qutip.core import CoreOptions
from qutip.core import data as _data
from qutip.core.dimensions import Space, Dimensions
from qutip.core.data.permute import get_permutations
from qutip.core.coefficient import (
    Coefficient,
    coefficient,
    ConstantCoefficient,
)
from qutip.settings import settings
from qutip.typing import LayerType
from ._pterm import _ProdTerm

from qutip.core._qobj.utils import (
    Transform,
    conj_transform,
    trans_transform,
    adjoint_transform,
    apply_transform,
)

__all__ = ["Operator"]


class _Term:
    prod_terms: tuple[_ProdTerm]
    hilbert_space: tuple[int]
    issuper: bool
    factor: Coefficient

    def __init__(self, prod_terms, hilbert_space, issuper, factor):
        self.prod_terms = tuple(prod_terms)
        self.hilbert_space = tuple(hilbert_space)
        self.issuper = bool(issuper)
        self.factor = coefficient(factor)

    def to_data(self, t: float, dtype):
        """
        Convert a term to a Data object.
        """
        max_modes = 0
        modes_affected = [0] * len(self.hilbert_space)

        for pterm in self.prod_terms:
            modes = pterm.modes
            max_modes = max(max_modes, len(modes))
            for mode in modes:
                modes_affected[mode] += 1

        if max_modes == 0:
            out = _data.identity[dtype](
                np.prod(self.hilbert_space), scale=self.factor(t)
            )

        elif max_modes == 1:
            # All term are on only one mode.
            opers = [None] * len(self.hilbert_space)
            for pterm in self.prod_terms:
                oper = pterm.to_data(t)
                mode = pterm.modes[0]
                if opers[mode] is not None:
                    opers[mode] = opers[mode] @ oper
                else:
                    opers[mode] = oper

            for i in range(len(opers)):
                if opers[i] is not None:
                    continue
                opers[i] = _data.identity[dtype](self.hilbert_space[i])

            if not self.issuper:
                out = _data.mul(opers[0], self.factor(t))
                for oper in opers[1:]:
                    out = _data.kron(out, oper)
            else:
                N = len(self.hilbert_space) // 2
                pre = opers[0]
                post = _data.mul(opers[N], self.factor(t))
                for i in range(1, N):
                    pre = _data.kron(pre, opers[i])
                    post = _data.kron(post, opers[i + N])
                # When superrep == super: post.T & pre
                # The transpose already accounted for in pterm.transform
                out = _data.kron(post, pre)

        else:
            # TODO: To optimize
            # It's not always needed to expand to the full space before the
            #   product.
            out = self.prod_terms[0].expand_data(t)
            out = _data.mul(out, self.factor(t))
            for pterm in self.prod_terms[1:]:
                oper = pterm.expand_data(t)
                out = out @ pterm.expand_data(t)

        return out

    def _compress(self):
        """
        Reorder and merge prod_terms.
        Return a new _Term.
        """
        if len(self.prod_terms) == 0:
            return self

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
                layers[target_layer_idx][modes].append(term)
            else:
                target_layer_idx += 1
                if target_layer_idx == len(layers):
                    layers.append({})
                layers[target_layer_idx][modes] = [term]
                for m in modes:
                    frontier[m] = new_idx

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

                done_modes |= set(terms[0].modes)
                # TODO: Add real merge code
                # For now, we create an Operator with 1 term and just the local
                # hilbert space an pass it as one single matrix _ProdTerm
                oper = Operator(shape=[
                    np.prod(terms[-1].out_space),
                    np.prod(terms[0].in_space)
                ])
                dims = Dimensions([terms[0].dimension[0], terms[-1].dimension[1]])
                local_terms = [
                    _ProdTerm(
                        term.oper,
                        (0,),
                        Dimensions(term.oper.shape),
                        term.transform,
                    )
                    for term in terms
                ]
                oper.terms.append(_Term(
                    local_terms,
                    [local_terms[0].dimension[1].size],
                    False,
                    1.
                ))

                pterm.append(
                    _ProdTerm(
                        oper,
                        terms[0].modes,
                        dims,
                        Transform.DIRECT,
                    )
                )
        return _Term(
            tuple(pterm), self.hilbert_space, self.issuper, self.factor
        )


def _read_dims(shape, dimension):
    with CoreOptions(auto_tidyup_dims=False):
        if dimension is None and shape is None:
            return None
        if dimension is None:
            return Dimensions([[shape[0]], [shape[1]]])
        dimensions = Dimensions(dimension)
        if shape is not None and shape != dimensions.shape:
            raise ValueError("given dimensions and shape do not match")
        if dimensions[0].flat() != dimensions[1].flat():
            N = max(len(dimensions[0].flat()), len(dimensions[1].flat()))
            dimensions = Dimensions(
                [
                    dimensions[0].expand_scalar_dims(N),
                    dimensions[1].expand_scalar_dims(N),
                ]
            )

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
                _Term(
                    (
                        _ProdTerm(
                            arg,
                            modes,
                            self._dims,
                            Transform.DIRECT,
                        ),
                    ),
                    self._dims[1].flat(),
                    self._dims.issuper,
                    coefficient(1.0),
                )
            )

        else:
            raise TypeError(f"{type(arg)} not supported.")

    @classmethod
    def identity(self, hilbert_space):
        out = Operator(dimension=[hilbert_space] * 2)
        out.terms.append(
            _Term((), out._dims[1].flat(), out._dims.issuper, coefficient(1))
        )
        return out

    @classmethod
    def from_qobj(self, qobj):
        return Operator(
            qobj.data,
            dimension=qobj._dims,
            modes=tuple(range(len(qobj._dims[0].as_list()))),
        )

    @property
    def dims(self):
        return self._dims.as_list()

    @property
    def shape(self):
        return self._dims.shape

    @property
    def isconstant(self):
        # TODO: check for element in pterm operator
        return all(
            isinstance(term.factor, ConstantCoefficient) for term in self.terms
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

    def conj(self):
        new = Operator(dimension=self._dims)
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms:
                pterms.append(pterm._add_transform(conj_transform))
            new.terms.append(
                _Term(
                    pterms,
                    self._dims[1].flat(),
                    self._dims.issuper,
                    term.factor.conj(),
                )
            )
        return new

    def transpose(self):
        new = Operator(dimension=Dimensions(self._dims[0], self._dims[1]))
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                pterms.append(pterm._add_transform(trans_transform))
            new.terms.append(
                _Term(
                    pterms,
                    new._dims[1].flat(),
                    new._dims.issuper,
                    term.factor,
                )
            )
        return new

    def adjoint(self):
        new = Operator(dimension=Dimensions(self._dims[0], self._dims[1]))
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                pterms.append(pterm._add_transform(adjoint_transform))
            new.terms.append(
                _Term(
                    pterms,
                    new._dims[1].flat(),
                    new._dims.issuper,
                    term.factor.conj(),
                )
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
        # With this A - A would not result in `0`, but in 2 opposite terms.
        return self + -other

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            other = coefficient(other)
        if not isinstance(other, Coefficient):
            return NotImplemented
        new = Operator(dimension=self._dims)
        for term in self.terms:
            new.terms.append(
                _Term(
                    term.prod_terms,
                    term.hilbert_space,
                    term.issuper,
                    term.factor * other,
                )
            )
        return new

    def __truediv__(self, other):
        return self * (1 / other)

    def __matmul__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented

        out_dims = self._dims @ other._dims
        new = Operator(dimension=out_dims)

        for term_l, term_r in itertools.product(self.terms, other.terms):
            new.terms.append(
                _Term(
                    term_l.prod_terms + term_r.prod_terms,
                    out_dims[1].flat(),
                    out_dims.issuper,
                    term_l.factor * term_r.factor,
                )
            )

        new.merge_terms()
        return new

    def _tensor_super(left, right):
        N = len(left._dims[0]) // 2
        M = len(right._dims[0]) // 2

        space_out = [
            [left._dims[0].oper[0], right._dims[0].oper[0]],
            [left._dims[0].oper[1], right._dims[0].oper[1]],
        ]
        space_mid = [
            [left._dims[1].oper[0], right._dims[0].oper[0]],
            [left._dims[1].oper[1], right._dims[0].oper[1]],
        ]
        space_in = [
            [left._dims[1].oper[0], right._dims[1].oper[0]],
            [left._dims[1].oper[1], right._dims[1].oper[1]],
        ]

        left_ext = Operator(dimension=[space_out, space_mid])
        for term in left.terms:
            pterms = []
            for pterm in term.prod_terms:
                local_dim = Dimensions([
                    [
                        [pterm.dimension[0].oper[0], right._dims[0].oper[0]],
                        [pterm.dimension[0].oper[1], right._dims[0].oper[1]],
                    ], [
                        [pterm.dimension[1].oper[0], right._dims[0].oper[0]],
                        [pterm.dimension[1].oper[1], right._dims[0].oper[1]],
                    ]
                ])
                pterms.append(
                    _ProdTerm(
                        pterm.operator,
                        tuple(i if i < N else i + M for i in pterm.modes),
                        local_dim,
                        pterm.transform,
                    )
                )
            left_ext.terms.append(
                _Term(
                    tuple(pterms),
                    left_ext._dims[1].flat(),
                    left_ext._dims.issuper,
                    factor=term.factor,
                )
            )

        right_shifted = Operator(dimension=[space_mid, space_in])
        for term in right.terms:
            pterms = []
            for pterm in term.prod_terms:
                local_dim = Dimensions([
                    [
                        [left._dims[1].oper[0], pterm.dimension[0].oper[0]],
                        [left._dims[1].oper[1], pterm.dimension[0].oper[1]],
                    ], [
                        [left._dims[1].oper[0], pterm.dimension[1].oper[0]],
                        [left._dims[1].oper[1], pterm.dimension[1].oper[1]],
                    ]
                ])
                pterms.append(
                    _ProdTerm(
                        pterm.operator,
                        tuple(
                            i + N if i < M else i + 2 * N for i in pterm.modes
                        ),
                        local_dim,
                        pterm.transform,
                    )
                )
            right_shifted.terms.append(
                _Term(
                    tuple(pterms),
                    right_shifted._dims[1].flat(),
                    right_shifted._dims.issuper,
                    factor=term.factor,
                )
            )

        return left_ext @ right_shifted

    def __and__(left, right):
        """
        `kron` operation.
        """
        if left._dims.issuper and right._dims.issuper:
            return left._tensor_super(right)
        if left._dims.issuper or right._dims.issuper:
            raise TypeError("Can't compound normal and super space together.")
        N = len(left._dims[0].as_list())

        left_ext = Operator(
            dimension = [
                [left._dims[0], right._dims[0]],
                [left._dims[1], right._dims[0]],
            ]
        )
        for term in left.terms:
            pterms = []
            for pterm in term.prod_terms:
                local_dim = Dimensions([
                    [pterm.dimension[0], right._dims[0]],
                    [pterm.dimension[1], right._dims[0]],
                ])
                pterms.append(
                    _ProdTerm(
                        pterm.operator,
                        pterm.modes,
                        local_dim,
                        pterm.transform,
                    )
                )
            left_ext.terms.append(
                _Term(
                    tuple(pterms),
                    left_ext._dims[1].flat(),
                    left_ext._dims.issuper,
                    factor=term.factor,
                )
            )

        right_shifted = Operator(
            dimension=[
                [left._dims[1], right._dims[0]],
                [left._dims[1], right._dims[1]],
            ]
        )
        for term in right.terms:
            pterms = []
            for pterm in term.prod_terms:
                local_dim = Dimensions([
                    [left._dims[1], pterm.dimension[0]],
                    [left._dims[1], pterm.dimension[1]],
                ])
                pterms.append(
                    _ProdTerm(
                        pterm.operator,
                        tuple(i + N for i in pterm.modes),
                        local_dim,
                        pterm.transform,
                    )
                )
            right_shifted.terms.append(
                _Term(
                    tuple(pterms),
                    right_shifted._dims[1].flat(),
                    right_shifted._dims.issuper,
                    factor=term.factor,
                )
            )

        return left_ext @ right_shifted

    def spre(self):
        if self._dims.issuper:
            raise TypeError("Already a superoperator")
        if not self._dims.issquare:
            raise TypeError("spost only defined for square dimensions")

        super_op = Operator(dimension=[self._dims, self._dims])
        N = len(self._dims[1].flat())
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms:
                local_dim = Dimensions(
                    [
                        Dimensions([pterm.dimension[0], self._dims[1]]),
                        Dimensions([pterm.dimension[1], self._dims[0]]),
                    ]
                )
                pterms.append(
                    _ProdTerm(
                        pterm.operator,
                        pterm.modes,
                        local_dim,
                        pterm.transform,
                    )
                )
            super_op.terms.append(
                _Term(
                    pterms,
                    super_op._dims[1].flat(),
                    super_op._dims.issuper,
                    factor=term.factor,
                )
            )

        return super_op

    def spost(self):
        if self._dims.issuper:
            raise TypeError("Already a superoperator")
        if not self._dims.issquare:
            raise TypeError("spost only defined for square dimensions")

        super_op = Operator(dimension=[self._dims, self._dims])
        N = len(self._dims[1].flat())
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                local_dim = Dimensions(
                    [
                        Dimensions([self._dims[0], pterm.dimension[1]]),
                        Dimensions([self._dims[1], pterm.dimension[0]]),
                    ]
                )
                pterms.append(
                    _ProdTerm(
                        pterm.operator,
                        tuple(i + N for i in pterm.modes),
                        local_dim,
                        trans_transform[pterm.transform],
                    )
                )
            super_op.terms.append(
                _Term(
                    pterms,
                    super_op._dims[1].flat(),
                    super_op._dims.issuper,
                    factor=term.factor,
                )
            )

        return super_op

    def sprepost(self, post):
        if not self._dims.issquare:
            # TODO: implement
            raise NotImplementedError("sprepost only implemented for square dimensions")
        if self._dims != post._dims:
            # TODO: implement
            raise NotImplementedError("sprepost only implemented for operators of same dimensions")
        if self._dims.issuper or post._dims.issuper:
            raise TypeError("Already a superoperator")

        out_dims = Dimensions(
            [
                Dimensions([self._dims[0], post._dims[1]]),
                Dimensions([self._dims[1], post._dims[0]]),
            ]
        )

        pre_part = Operator(dimension=out_dims)
        post_part = Operator(dimension=out_dims)
        N = len(self._dims[0].flat())
        for term in self.terms:
            pterms = []
            for pterm in term.prod_terms:
                local_dim = Dimensions(
                    [
                        Dimensions([pterm.dimension[0], post._dims[1]]),
                        Dimensions([pterm.dimension[1], post._dims[0]]),
                    ]
                )
                pterms.append(
                    _ProdTerm(
                        pterm.operator,
                        pterm.modes,
                        local_dim,
                        pterm.transform,
                    )
                )
            pre_part.terms.append(
                _Term(
                    pterms,
                    out_dims[1].flat(),
                    out_dims.issuper,
                    factor=term.factor,
                )
            )
        for term in post.terms:
            pterms = []
            for pterm in term.prod_terms[::-1]:
                local_dim = Dimensions(
                    [
                        Dimensions([self._dims[0], pterm.dimension[1]]),
                        Dimensions([self._dims[1], pterm.dimension[0]]),
                    ]
                )
                pterms.append(
                    _ProdTerm(
                        pterm.operator,
                        tuple(i + N for i in pterm.modes),
                        local_dim,
                        trans_transform[pterm.transform],
                    )
                )
            post_part.terms.append(
                _Term(
                    pterms,
                    out_dims[1].flat(),
                    out_dims.issuper,
                    factor=term.factor,
                )
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
                and abs(factor(0)) < settings.core["atol"]
            ):
                continue
            new_terms.append(
                _Term(
                    term.prod_terms,
                    self._dims[1].flat(),
                    self._dims.issuper,
                    factor,
                )
            )

        self.terms = new_terms
