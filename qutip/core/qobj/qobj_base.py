from __future__ import annotations

import numpy as np
import numbers

import qutip
from .. import __version__
from ..settings import settings
from . import data as _data
from .dimensions import (
    enumerate_flat, collapse_dims_super, flatten, unflatten, Dimensions
)
from typing import Any, Literal
from qutip.typing import LayerType, DimensionLike
from numpy.typing import ArrayLike


class _QobjBuilder
    @classmethod
    def _initialize_data(arg, raw_dims, copy):
        flags = {}
        if isinstance(arg, _data.Data):
            data = arg.copy() if copy else arg
            dims = Dimensions(raw_dims or [[arg.shape[0]], [arg.shape[1]]])
        elif isinstance(arg, Qobj):
            data = arg.data.copy() if copy else arg.data
            dims = Dimensions(raw_dims or arg._dims)
            flags["isherm"] = arg._isherm
            flags["isunitary"] = arg._isunitary
        else:
            data = _data.create(arg, copy=copy)
            dims = Dimensions(
                raw_dims or [[self._data.shape[0]], [self._data.shape[1]]]
            )
        if dims.shape != data.shape:
            raise ValueError('Provided dimensions do not match the data: ' +
                             f"{self._dims.shape} vs {self._data.shape}")

        return data, dims, flags

    def __call__(
        cls,
        arg: ArrayLike | Any = None,
        dims: DimensionLike = None,
        copy: bool = True,
        superrep: str = None,
        isherm: bool = None,
        isunitary: bool = None
    ):
        data, dims, flags = self._initialize_data(arg, dims, copy)
        if isherm is not None:
            flags["isherm"] = isherm
        if isunitary is not None:
            flags["isunitary"] = isunitary
        if superrep is not None:
            dims = dims.replace_superrep(superrep)

        instance_class = {
            "ket": qutip.core.qobj_state.Ket,
            "bra": qutip.core.qobj_state.Bra,
            "operator-ket": qutip.core.qobj_operator.OperKet,
            "operator-bra": qutip.core.qobj_operator.OperBra,
            "scalar": qutip.core.qobj_state.Operator,
            "oper": qutip.core.qobj_state.Operator,
            "super": qutip.core.qobj_state.SuperOperator,
        }[dims.type]

        new_qobj = instance_class.__new__(instance_class)
        new_qobj.__init__(data, dims, **flags)
        return new_qobj


def _require_equal_type(method):
    """
    Decorate a binary Qobj method to ensure both operands are Qobj and of the
    same type and dimensions.  Promote numeric scalar to identity matrices of
    the same type and shape.
    """
    @functools.wraps(method)
    def out(self, other):
        if isinstance(other, Qobj):
            if self._dims != other._dims:
                msg = (
                    "incompatible dimensions "
                    + repr(self.dims) + " and " + repr(other.dims)
                )
                raise ValueError(msg)
            return method(self, other)
        if other == 0:
            return method(self, other)
        if self._dims.issquare and isinstance(other, numbers.Number):
            scale = complex(other)
            other = Qobj(_data.identity(self.shape[0], scale,
                                        dtype=type(self.data)),
                         dims=self._dims,
                         isherm=(scale.imag == 0),
                         isunitary=(abs(abs(scale)-1) < settings.core['atol']),
                         copy=False)
            return method(self, other)
        return NotImplemented

    return out


class Qobj(metaclass=_QobjBuilder):
    _dims: Dimensions
    dims: Dimensions
    _data: Data
    data: data

    def __init__(
        self,
        arg: ArrayLike | Any = None,
        dims: DimensionLike = None,
        copy: bool = True,
        superrep: str = None,
        isherm: bool = None,
        isunitary: bool = None
    ):
        if not (isinstance(arg, _data.Data) and isinstance(dims, Dimensions)):
            data, dims, flags = _QobjBuilder._initialize_data(arg, dims, copy)
            if isherm is None and flags["isherm"] is not None:
                isherm = flags["isherm"]
            if isunitary is None and flags["isunitary"] is not None:
                isunitary = flags["isunitary"]
            if superrep is not None:
                dims = dims.replace_superrep(superrep)

        self._data = data
        self._dims = dims
        self._isherm = isherm
        self._isunitary = isunitary

    @property
    def type(self) -> str:
        return self._dims.

    @property
    def data(self) -> _data.Data:
        return self._data

    @data.setter
    def data(self, data: _data.Data):
        if not isinstance(data, _data.Data):
            raise TypeError('Qobj data must be a data-layer format.')
        if self._dims.shape != data.shape:
            raise ValueError('Provided data do not match the dimensions: ' +
                             f"{self._dims.shape} vs {data.shape}")
        self._data = data

    @property
    def dtype(self):
        return type(self._data)

    @property
    def dims(self) -> list[list[int]] | list[list[list[int]]]:
        return self._dims.as_list()

    @dims.setter
    def dims(self, dims: list[list[int]] | list[list[list[int]]] | Dimensions):
        dims = Dimensions(dims, rep=self.superrep)
        if dims.shape != self._data.shape:
            raise ValueError('Provided dimensions do not match the data: ' +
                             f"{dims.shape} vs {self._data.shape}")
        self._dims = dims

    def copy(self) -> Qobj:
        """Create identical copy"""
        return Qobj(arg=self._data,
                    dims=self._dims,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=True)

    def to(self, data_type: LayerType, copy: bool=False) -> Qobj:
        """
        Convert the underlying data store of this `Qobj` into a different
        storage representation.

        The different storage representations available are the "data-layer
        types" which are known to :obj:`qutip.core.data.to`.  By default, these
        are :class:`~qutip.core.data.CSR`, :class:`~qutip.core.data.Dense` and
        :class:`~qutip.core.data.Dia`, which respectively construct a
        compressed sparse row matrix, diagonal matrix and a dense one.  Certain
        algorithms and operations may be faster or more accurate when using a
        more appropriate data store.

        Parameters
        ----------
        data_type : type, str
            The data-layer type or its string alias that the data of this
            :class:`Qobj` should be converted to.

        copy : Bool
            If the data store is already in the format requested, whether the
            function should return returns `self` or a copy.

        Returns
        -------
        Qobj
            A :class:`Qobj` with the data stored in the requested format.
        """
        data_type = _data.to.parse(data_type)
        if type(self._data) is data_type and copy:
            return self.copy()
        elif type(self._data) is data_type:
            return self
        return self.cls(
            _data.to(data_type, self._data),
            dims=self._dims,
            isherm=self._isherm,
            isunitary=self._isunitary,
            copy=False
        ):

    @_require_equal_type
    def __add__(self, other: Qobj | complex) -> Qobj:
        if other == 0:
            return self.copy()
        return Qobj(_data.add(self._data, other._data),
                    dims=self._dims,
                    isherm=(self._isherm and other._isherm) or None,
                    copy=False)

    def __radd__(self, other: Qobj | complex) -> Qobj:
        return self.__add__(other)

    @_require_equal_type
    def __sub__(self, other: Qobj | complex) -> Qobj:
        if other == 0:
            return self.copy()
        return Qobj(_data.sub(self._data, other._data),
                    dims=self._dims,
                    isherm=(self._isherm and other._isherm) or None,
                    copy=False)

    def __rsub__(self, other: Qobj | complex) -> Qobj:
        return self.__neg__().__add__(other)

    def __neg__(self) -> Qobj:
        return Qobj(_data.neg(self._data),
                    dims=self._dims,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def __mul__(self, other: complex) -> Qobj:
        """
        If other is a Qobj, we dispatch to __matmul__. If not, we
        check that other is a valid complex scalar, i.e., we can do
        complex(other). Otherwise, we return NotImplemented.
        """

        if isinstance(other, Qobj):
            return self.__matmul__(other)

        # We send other to mul instead of complex(other) to be more flexible.
        # The dispatcher can then decide how to handle other and return
        # TypeError if it does not know what to do with the type of other.
        try:
            out = _data.mul(self._data, other)
        except TypeError:
            return NotImplemented

        # Infer isherm and isunitary if possible
        try:
            multiplier = complex(other)
            isherm = (self._isherm and multiplier.imag == 0) or None
            isunitary = (abs(abs(multiplier) - 1) < settings.core['atol']
                         if self._isunitary else None)
        except TypeError:
            isherm = None
            isunitary = None

        return Qobj(out,
                    dims=self._dims,
                    isherm=isherm,
                    isunitary=isunitary,
                    copy=False)

    def __rmul__(self, other: complex) -> Qobj:
        # Shouldn't be here unless `other.__mul__` has already been tried, so
        # we _shouldn't_ check that `other` is `Qobj`.
        return self.__mul__(other)

    def __truediv__(self, other: complex) -> Qobj:
        return self.__mul__(1 / other)

    def __matmul__(self, other: Qobj) -> Qobj:
        if not isinstance(other, Qobj):
            return NotImplemented
        new_dims = self._dims @ other._dims
        if new_dims.type == 'scalar':
            return _data.inner(self._data, other._data)

        return Qobj(
            _data.matmul(self._data, other._data),
            dims=new_dims,
            isunitary=self._isunitary and other._isunitary,
            copy=False
        )

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Qobj) or self._dims != other._dims:
            return False
        # isequal uses both atol and rtol from settings.core
        return _data.isequal(self._data, other._data)

    def __and__(self, other: Qobj) -> Qobj:
        """
        Syntax shortcut for tensor:
        A & B ==> tensor(A, B)
        """
        return qutip.tensor(self, other)

    def dag(self) -> Qobj:
        """Get the Hermitian adjoint of the quantum object."""
        if self._isherm:
            return self.copy()
        return Qobj(_data.adjoint(self._data),
                    dims=Dimensions(self._dims[0], self._dims[1]),
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def conj(self) -> Qobj:
        """Get the element-wise conjugation of the quantum object."""
        return Qobj(_data.conj(self._data),
                    dims=self._dims,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def trans(self) -> Qobj:
        """Get the matrix transpose of the quantum operator.

        Returns
        -------
        oper : :class:`.Qobj`
            Transpose of input operator.
        """
        return Qobj(_data.transpose(self._data),
                    dims=Dimensions(self._dims[0], self._dims[1]),
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def _str_header(self):
        out = ", ".join([
            f"{self.__class__.__name__}: dims={self.dims}",
            f"shape={self._data.shape}",
            f"type={repr(self.type)}",
            f"dtype={self.dtype.__name__}",
        ])
        # TODO: Should this be here?
        # if self.type in ('oper', 'super'):
        #     out += ", isherm=" + str(self.isherm)
        # if self.issuper and self.superrep != 'super':
        #     out += ", superrep=" + repr(self.superrep)
        return out

    def __str__(self):
        if self.data.shape[0] * self.data.shape[0] > 100_000_000:
            # If the system is huge, don't attempt to convert to a dense matrix
            # and then to string, because it is pointless and is likely going
            # to produce memory errors. Instead print the sparse data string
            # representation.
            data = _data.to(_data.CSR, self.data).as_scipy()
        elif _data.iszero(_data.sub(self.data.conj(), self.data)):
            # TODO: that check could be slow...
            data = np.real(self.full())
        else:
            data = self.full()
        return "\n".join([self._str_header(), "Qobj data =", str(data)])

    def __repr__(self):
        # give complete information on Qobj without print statement in
        # command-line we cant realistically serialize a Qobj into a string,
        # so we simply return the informal __str__ representation instead.)
        return self.__str__()

    def _repr_latex_(self):
        """
        Generate a LaTeX representation of the Qobj instance. Can be used for
        formatted output in ipython notebook.
        """
        half_length = 5
        n_rows, n_cols = self.data.shape
        # Choose which rows and columns we're going to output, or None if that
        # element should be truncated.
        rows = list(range(min((half_length, n_rows))))
        if n_rows <= half_length * 2:
            rows += list(range(half_length, min((2*half_length, n_rows))))
        else:
            rows.append(None)
            rows += list(range(n_rows - half_length, n_rows))
        cols = list(range(min((half_length, n_cols))))
        if n_cols <= half_length * 2:
            cols += list(range(half_length, min((2*half_length, n_cols))))
        else:
            cols.append(None)
            cols += list(range(n_cols - half_length, n_cols))
        # Make the data array.
        data = r'$$\left(\begin{array}{cc}'
        data += r"\\".join(_latex_row(row, cols, self.data.to_array())
                           for row in rows)
        data += r'\end{array}\right)$$'
        return self._str_header() + data

    def __getstate__(self):
        # defines what happens when Qobj object gets pickled
        self.__dict__.update({'qutip_version': __version__[:5]})
        return self.__dict__

    def __setstate__(self, state):
        # defines what happens when loading a pickled Qobj
        # TODO: what happen with miss matched child class?
        if 'qutip_version' in state.keys():
            del state['qutip_version']
        (self.__dict__).update(state)

    def full(
        self,
        order: Literal['C', 'F'] = 'C',
        squeeze: bool = False
    ) -> np.ndarray:
        """Dense array from quantum object.

        Parameters
        ----------
        order : str {'C', 'F'}
            Return array in C (default) or Fortran ordering.
        squeeze : bool {False, True}
            Squeeze output array.

        Returns
        -------
        data : array
            Array of complex data from quantum objects `data` attribute.
        """
        out = np.asarray(self.data.to_array(), order=order)
        return out.squeeze() if squeeze else out

    def data_as(self, format: str = None, copy: bool = True) -> Any:
        """Matrix from quantum object.

        Parameters
        ----------
        format : str, default: None
            Type of the output, "ndarray" for ``Dense``, "csr_matrix" for
            ``CSR``. A ValueError will be raised if the format is not
            supported.

        copy : bool {False, True}
            Whether to return a copy

        Returns
        -------
        data : numpy.ndarray, scipy.sparse.matrix_csr, etc.
            Matrix in the type of the underlying libraries.
        """
        return _data.extract(self._data, format, copy)

    def contract(self, inplace: bool = False) -> Qobj:
        """
        Contract subspaces of the tensor structure which are 1D.  Not defined
        on superoperators.  If all dimensions are scalar, a Qobj of dimension
        [[1], [1]] is returned, i.e. _multiple_ scalar dimensions are
        contracted, but one is left.

        Parameters
        ----------
        inplace: bool, optional
            If ``True``, modify the dimensions in place.  If ``False``, return
            a copied object.

        Returns
        -------
        out: :class:`.Qobj`
            Quantum object with dimensions contracted.  Will be ``self`` if
            ``inplace`` is ``True``.
        """
        if self.isket:
            sub = [x for x in self.dims[0] if x > 1] or [1]
            dims = [sub, [1]*len(sub)]
        elif self.isbra:
            sub = [x for x in self.dims[1] if x > 1] or [1]
            dims = [[1]*len(sub), sub]
        elif self.isoper or self.isoperket or self.isoperbra:
            if self.isoper:
                oper_dims = self.dims
            elif self.isoperket:
                oper_dims = self.dims[0]
            else:
                oper_dims = self.dims[1]
            if len(oper_dims[0]) != len(oper_dims[1]):
                raise ValueError("cannot parse Qobj dimensions: "
                                 + repr(self.dims))
            dims_ = [
                (x, y) for x, y in zip(oper_dims[0], oper_dims[1])
                if x > 1 or y > 1
            ] or [(1, 1)]
            dims = [[x for x, _ in dims_], [y for _, y in dims_]]
            if self.isoperket:
                dims = [dims, [1]]
            elif self.isoperbra:
                dims = [[1], dims]
        else:
            raise TypeError("not defined for superoperators")
        if inplace:
            self.dims = dims
            return self
        return Qobj(self.data.copy(), dims=dims, copy=False)


    def permute(self, order: list) -> Qobj:
        """
        Permute the tensor structure of a quantum object.  For example,

            ``qutip.tensor(x, y).permute([1, 0])``

        will give the same result as

            ``qutip.tensor(y, x)``

        and

            ``qutip.tensor(a, b, c).permute([1, 2, 0])``

        will be the same as

            ``qutip.tensor(b, c, a)``

        For regular objects (bras, kets and operators) we expect ``order`` to
        be a flat list of integers, which specifies the new order of the tensor
        product.

        For superoperators, we expect ``order`` to be something like

            ``[[0, 2], [1, 3]]``

        which tells us to permute according to [0, 2, 1, 3], and then group
        indices according to the length of each sublist.  As another example,
        permuting a superoperator with dimensions of

            ``[[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]``

        by an ``order``

            ``[[0, 3], [1, 4], [2, 5]]``

        should give a new object with dimensions

            ``[[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]]``.

        Parameters
        ----------
        order : list
            List of indices specifying the new tensor order.

        Returns
        -------
        P : :class:`.Qobj`
            Permuted quantum object.
        """
        if self.type in ('bra', 'ket', 'oper'):
            structure = self.dims[1] if self.isbra else self.dims[0]
            new_structure = [structure[x] for x in order]
            if self.isbra:
                dims = [self.dims[0], new_structure]
            elif self.isket:
                dims = [new_structure, self.dims[1]]
            else:
                if self._dims[0] != self._dims[1]:
                    raise TypeError("undefined for non-square operators")
                dims = [new_structure, new_structure]
            data = _data.permute.dimensions(self.data, structure, order)
            return Qobj(data,
                        dims=dims,
                        isherm=self._isherm,
                        isunitary=self._isunitary,
                        copy=False)
        # If we've got here, we're some form of superoperator, so we work with
        # the flattened structure.
        flat_order = flatten(order)
        flat_structure = flatten(self.dims[1] if self.isoperbra
                                 else self.dims[0])
        new_structure = unflatten([flat_structure[x] for x in flat_order],
                                  enumerate_flat(order))
        if self.isoperbra:
            dims = [self.dims[0], new_structure]
        elif self.isoperket:
            dims = [new_structure, self.dims[1]]
        else:
            if self._dims[0] != self._dims[1]:
                raise TypeError("undefined for non-square operators")
            dims = [new_structure, new_structure]
        data = _data.permute.dimensions(self.data, flat_structure, flat_order)
        return Qobj(data,
                    dims=dims,
                    superrep=self.superrep,
                    copy=False)

    def tidyup(self, atol: float = None) -> Qobj:
        """
        Removes small elements from the quantum object.

        Parameters
        ----------
        atol : float
            Absolute tolerance used by tidyup. Default is set
            via qutip global settings parameters.

        Returns
        -------
        oper : :class:`.Qobj`
            Quantum object with small elements removed.
        """
        atol = atol or settings.core['auto_tidyup_atol']
        self.data = _data.tidyup(self.data, atol)
        return self


    def transform(
        self,
        inpt: list[Qobj] | ArrayLike,
        inverse: bool = False
    ) -> Qobj:
        """Basis transform defined by input array.

        Input array can be a ``matrix`` defining the transformation,
        or a ``list`` of kets that defines the new basis.

        Parameters
        ----------
        inpt : array_like
            A ``matrix`` or ``list`` of kets defining the transformation.
        inverse : bool
            Whether to return inverse transformation.

        Returns
        -------
        oper : :class:`.Qobj`
            Operator in new basis.

        Notes
        -----
        This function is still in development.
        """
        # TODO: What about superoper
        # TODO: split into each cases.
        if isinstance(inpt, list) or (isinstance(inpt, np.ndarray) and
                                      inpt.ndim == 1):
            if len(inpt) != max(self.shape):
                raise TypeError(
                    'Invalid size of ket list for basis transformation')
            base = np.hstack([psi.full() for psi in inpt])
            S = _data.adjoint(_data.create(base))
        elif isinstance(inpt, Qobj) and inpt.isoper:
            S = inpt.data
        elif isinstance(inpt, np.ndarray):
            S = _data.create(inpt).conj()
        else:
            raise TypeError('Invalid operand for basis transformation')

        # transform data
        if inverse:
            if self.isket:
                data = _data.matmul(S.adjoint(), self.data)
            elif self.isbra:
                data = _data.matmul(self.data, S)
            else:
                data = _data.matmul(_data.matmul(S.adjoint(), self.data), S)
        else:
            if self.isket:
                data = _data.matmul(S, self.data)
            elif self.isbra:
                data = _data.matmul(self.data, S.adjoint())
            else:
                data = _data.matmul(_data.matmul(S, self.data), S.adjoint())
        return Qobj(data,
                    dims=self.dims,
                    isherm=self._isherm,
                    superrep=self.superrep,
                    copy=False)

    # TODO: split
    def overlap(self, other: Qobj) -> complex:
        """
        Overlap between two state vectors or two operators.

        Gives the overlap (inner product) between the current bra or ket Qobj
        and and another bra or ket Qobj. It gives the Hilbert-Schmidt overlap
        when one of the Qobj is an operator/density matrix.

        Parameters
        ----------
        other : :class:`.Qobj`
            Quantum object for a state vector of type 'ket', 'bra' or density
            matrix.

        Returns
        -------
        overlap : complex
            Complex valued overlap.

        Raises
        ------
        TypeError
            Can only calculate overlap between a bra, ket and density matrix
            quantum objects.
        """
        if not isinstance(other, Qobj):
            raise TypeError("".join([
                "cannot calculate overlap with non-quantum object ",
                repr(other),
            ]))
        if (
            self.type not in ('ket', 'bra', 'oper')
            or other.type not in ('ket', 'bra', 'oper')
        ):
            msg = "only bras, kets and density matrices have defined overlaps"
            raise TypeError(msg)
        left, right = self._data.adjoint(), other.data
        if self.isoper or other.isoper:
            if not self.isoper:
                left = _data.project(left)
            if not other.isoper:
                right = _data.project(right)
            return _data.trace(_data.matmul(left, right))
        if other.isbra:
            right = right.adjoint()
        out = _data.inner(left, right, self.isket)
        if self.isket and other.isbra:
            # In this particular case, we've basically doing
            #   conj(other.overlap(self))
            # so we take care to conjugate the output.
            out = np.conj(out)
        return out

    @property
    def ishp(self) -> bool:
        # FIXME: this needs to be cached in the same ways as isherm.
        if self.type in ["super", "oper"]:
            try:
                J = qutip.to_choi(self)
                return J.isherm
            except:
                return False
        else:
            return False

    @property
    def iscp(self) -> bool:
        # FIXME: this needs to be cached in the same ways as isherm.
        if self.type not in ["super", "oper"]:
            return False
        # We can test with either Choi or chi, since the basis
        # transformation between them is unitary and hence preserves
        # the CP and TP conditions.
        J = self if self.superrep in ('choi', 'chi') else qutip.to_choi(self)
        # If J isn't hermitian, then that could indicate either that J is not
        # normal, or is normal, but has complex eigenvalues.  In either case,
        # it makes no sense to then demand that the eigenvalues be
        # non-negative.
        return J.isherm and np.all(J.eigenenergies() >= -settings.core['atol'])

    @property
    def istp(self) -> bool:
        if self.type not in ['super', 'oper']:
            return False
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
        return np.allclose(tr_oper.full(), np.eye(tr_oper.shape[0]),
                           atol=settings.core['atol'])

    @property
    def iscptp(self) -> bool:
        if not (self.issuper or self.isoper):
            return False
        reps = ('choi', 'chi')
        q_oper = qutip.to_choi(self) if self.superrep not in reps else self
        return q_oper.iscp and q_oper.istp

    @property
    def isherm(self) -> bool:
        if self._isherm is not None:
            return self._isherm
        self._isherm = _data.isherm(self._data)
        return self._isherm

    @isherm.setter
    def isherm(self, isherm: bool):
        self._isherm = isherm

    def _calculate_isunitary(self):
        """
        Checks whether qobj is a unitary matrix
        """
        if not self.isoper or self._data.shape[0] != self._data.shape[1]:
            return False
        cmp = _data.matmul(self._data, self._data.adjoint())
        iden = _data.identity_like(cmp)
        return _data.iszero(_data.sub(cmp, iden),
                            tol=settings.core['atol'])

    @property
    def isunitary(self) -> bool:
        if self._isunitary is not None:
            return self._isunitary
        self._isunitary = self._calculate_isunitary()
        return self._isunitary

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the Qobj data."""
        return self._data.shape

    @property
    def isoper(self) -> bool:
        """Indicates if the Qobj represents an operator."""
        return self._dims.type in ['oper', 'scalar']

    @property
    def isbra(self) -> bool:
        """Indicates if the Qobj represents a bra state."""
        return self._dims.type in ['bra', 'scalar']

    @property
    def isket(self) -> bool:
        """Indicates if the Qobj represents a ket state."""
        return self._dims.type in ['ket', 'scalar']

    @property
    def issuper(self) -> bool:
        """Indicates if the Qobj represents a superoperator."""
        return self._dims.type == 'super'

    @property
    def isoperket(self) -> bool:
        """Indicates if the Qobj represents a operator-ket state."""
        return self._dims.type == 'operator-ket'

    @property
    def isoperbra(self) -> bool:
        """Indicates if the Qobj represents a operator-bra state."""
        return self._dims.type == 'operator-bra'
