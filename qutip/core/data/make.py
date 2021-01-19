from .dispatch import Dispatcher as _Dispatcher
import numpy as np
from scipy.sparse import diags as scipy_diags
from .dense import Dense
from .csr import CSR
from . import csr, dense, CSR, Dense
from numbers import Number

__all__ = ['diag_csr', 'diag_dense', 'diag',
           'one_element_csr', 'one_element_dense', 'one_element']


def diag_csr(diagonals, offsets=0, shape=None):
    """
    Construct a matrix from diagonals.

    Parameters
    ----------

    diagonals : array_like of complex or list of array_like of complex
        One or multiple diagonals of the matrix.

    offsets : int or list of int, optional
        Positions of the diagonal. 0 for the main diagonals. Positive for upper
        diagonals.

    shape : (int, int), optional
        shape of the output matrix.
    """
    if not (isinstance(offsets, int) or len(offsets) == 1):
        # Many diagonals are given: we use scipy.
        return CSR(scipy_diags(diagonals, offsets, shape,
                               format='csr', dtype=np.complex128))
    diagonal = np.asarray(diagonals).ravel('K')
    offset = offsets if isinstance(offsets, int) else offsets[0]
    if shape is None:
        N = len(diagonal)
        d0 = len(diagonal) + abs(offset)
        shape = (d0, d0)
    else:
        d0, d1 = shape
        N = min(d0 + offset, d1 - offset, d0, d1)
        if len(diagonal) != N:
            raise ValueError("len of the diagonal does not match the shape.")

    ind = np.arange(max(0, offset), N + max(0, offset), dtype=np.int32)
    ptr = np.arange(min(0, offset), d0 + 1 + min(0, offset), dtype=np.int32)
    ptr[ptr<0] = 0
    ptr[ptr>N] = N

    return CSR((diagonal, ind, ptr), shape=shape)


def diag_dense(diagonals, offsets=0, shape=None):
    """
    Construct a matrix from diagonals.

    Parameters
    ----------

    diagonals : array_like of complex or list of array_like of complex
        One or multiple diagonals of the matrix.

    offsets : int or list of int, optional
        Positions of the diagonal. 0 for the main diagonals. Positive for upper
        diagonals.

    shape : (int, int), optional
        shape of the output matrix.
    """
    if isinstance(offsets, int):
        offsets = [offsets]
        if len(diagonals) != 1:
            diagonals = [diagonals]

    if len(diagonals) != len(offsets):
        raise ValueError("Number of diagonals and offsets does not match")

    if shape is None:
        Ns = []
        for diagonal in diagonals:
            try:
                Ns.append(len(diagonal))
            except TypeError:
                Ns.append(1)
        sizes = [N + abs(offset) for N, offset in zip(Ns, offsets)]
        shape = min(sizes), min(sizes)

    out = dense.zeros(*shape)
    nda = out.as_ndarray()

    for diagonal, offset in zip(diagonals, offsets):
        if isinstance(diagonal, Number):
            diagonal = np.ones(max(shape)) * diagonal
        i = max(0, -offset)
        j = max(0, offset)
        for k in range(0, max(shape)):
            if i >= shape[0] or j >= shape[1]:
                break
            nda[i, j] = diagonal[k]
            i += 1
            j += 1
    return out


diag = _Dispatcher(diag_dense, name='diag', inputs=(), out=True)
diag.add_specialisations([
    (CSR, diag_csr),
    (Dense, diag_dense),
], _defer=True)


def one_element_csr(shape, position, value=1):
    """
    Create a matrix with only one non-null elements.

    Parameters
    ----------
    shape : (int, int)
        shape of the output matrix.

    position : (int, int)
        position of the non zero in the matrix.

    value : complex, optional
        value of the non-null element.
    """
    if not (0 <= position[0] < shape[0] or 0 <= position[1] < shape[1]):
        raise ValueError("Position of the elements out of bound: " +
                         str(position) + " in " + str(shape))
    data = csr.empty(*shape, 1)
    sci = data.as_scipy(full=True)
    sci.data[0] = value
    sci.indices[0] = position[1]
    sci.indptr[:position[0]+1] = 0
    sci.indptr[position[0]+1:] = 1
    return data


def one_element_dense(shape, position, value=1):
    """
    Create a matrix with only one non-null elements.

    Parameters
    ----------
    shape : (int, int)
        shape of the output matrix.

    position : (int, int)
        position of the non zero in the matrix.

    value : complex, optional
        value of the non-null element.
    """
    if not (0 <= position[0] < shape[0] or 0 <= position[1] < shape[1]):
        raise ValueError("Position of the elements out of bound: " +
                         str(position) + " in " + str(shape))
    data = dense.zeros(*shape, 1)
    nda = data.as_ndarray()
    nda[position] = value
    return data


one_element = _Dispatcher(one_element_dense, name='one_element',
                          inputs=(), out=True)
one_element.add_specialisations([
    (CSR, one_element_csr),
    (Dense, one_element_dense),
], _defer=True)
