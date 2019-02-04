
from qutip.cy.csr_math import *
from qutip.cy.csr_matrix import cy_csr_matrix, CSR_from_scipy

def mult(op1, op2, isherm):
    cop1 = CSR_from_scipy(op1)
    cop2 = CSR_from_scipy(op2)
    return zcsr_mult(op1, op2, isherm).to_scipy()

def spmm_tr(op1, op2, isherm):
    cop1 = CSR_from_scipy(op1)
    cop2 = CSR_from_scipy(op2)
    return zcsr_spmm_tr(cop1, cop2, isherm)

def inner(data_bra, data_ket):
    bra = CSR_from_scipy(data_oper)
    ket = CSR_from_scipy(data_oper)
    return zcsr_inner(bra, ket)

def mat_elem(data_oper, data_bra, data_ket):
    oper = CSR_from_scipy(data_oper)
    bra = CSR_from_scipy(data_oper)
    ket = CSR_from_scipy(data_oper)
    return zcsr_mat_elem(oper, bra, ket)

def kron(op1, op2):
    cop1 = CSR_from_scipy(op1)
    cop2 = CSR_from_scipy(op2)
    return zcsr_kron(cop1, cop2).to_scipy()

def expect_ket(oper, ket, isherm):
    cop1 = CSR_from_scipy(oper)
    cop2 = CSR_from_scipy(ket)
    return zcsr_expect_ket(cop1, cop2, isherm)
