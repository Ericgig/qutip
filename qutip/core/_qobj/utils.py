from enum import Enum

__all__ = [
    "Transform",
    "conj_transform",
    "trans_transform",
    "adjoint_transform",
]


class Transform(Enum):
    DIRECT = 0
    CONJ = 1
    TRANSPOSE = 2
    ADJOINT = 3


conj_transform = {
    Transform.DIRECT : Transform.CONJ,
    Transform.CONJ : Transform.DIRECT,
    Transform.TRANSPOSE : Transform.ADJOINT,
    Transform.ADJOINT : Transform.TRANSPOSE,
}

trans_transform = {
    Transform.DIRECT : Transform.TRANSPOSE,
    Transform.CONJ : Transform.ADJOINT,
    Transform.TRANSPOSE : Transform.DIRECT,
    Transform.ADJOINT : Transform.CONJ,
}

adjoint_transform = {
    Transform.DIRECT : Transform.ADJOINT,
    Transform.CONJ : Transform.TRANSPOSE,
    Transform.TRANSPOSE : Transform.CONJ,
    Transform.ADJOINT : Transform.DIRECT,
}

def _apply_transform(matrix, transform):
    match transform:
        case Transform.DIRECT:
            out = matrix
        case Transform.CONJ:
            out = matrix.conj()
        case Transform.TRANSPOSE:
            out = matrix.transpose()
        case Transform.ADJOINT:
            out = matrix.adjoint()

    return out
