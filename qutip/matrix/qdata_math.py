
from qutip.matrix.cy.csr_math import zcsr_mult, zcsr_spmm_tr, zcsr_inner, zcsr_mult, zcsr_kron, zcsr_expect_ket

def mult(op1, op2, isherm):
    if op1.format == op2.format:
        if op1.format == "csr":
            return zcsr_mult(op1.cdata, op2.cdata, isherm)
    raise NotImplementedError("format mix not implemented")

def spmm_tr(op1, op2, isherm):
    if op1.format == op2.format:
        if op1.format == "csr":
            return zcsr_spmm_tr(op1.cdata, op2.cdata, isherm).to_qdata()
    raise NotImplementedError("format mix not implemented")

def inner(data_bra, data_ket):
    if data_bra.format == data_ket.format:
        if data_bra.format == "csr":
            return zcsr_inner(data_bra.cdata, data_ket.cdata).to_qdata()
    raise NotImplementedError("format mix not implemented")

def mat_elem(data_oper, data_bra, data_ket):
    if data_oper.format == data_bra.format and data_oper.format == data_ket.format:
        if data_oper.format == "csr":
            return zcsr_mult(data_oper.cdata, data_bra.cdata, data_ket.cdata).to_qdata()
    raise NotImplementedError("format mix not implemented")

def kron(op1, op2):
    if op1.format == op2.format:
        if op1.format == "csr":
            return zcsr_kron(op1.cdata, op2.cdata)
    raise NotImplementedError("format mix not implemented")

def expect_ket(oper, ket, isherm):
    if oper.format == ket.format:
        if oper.format == "csr":
            return zcsr_expect_ket(oper.cdata, ket.cdata, isherm).to_qdata()
    raise NotImplementedError("format mix not implemented")
