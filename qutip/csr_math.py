
from qutip.cy.csr_math import *
from qutip.cy.csr_matrix import cy_csr_matrix, CSR_from_scipy

def spmm_tr(op1, op2, isherm):
    cop1 = cy_csr_matrix(op1)
    cop2 = cy_csr_matrix(op2)
    return zcsr_spmm_tr(op1, op2, isherm)

def inner(data_bra, data_ket):
    bra = cy_csr_matrix(data_oper)
    ket = cy_csr_matrix(data_oper)
    return zcsr_inner(bra, ket)

def mat_elem(data_oper, data_bra, data_ket):
    oper = cy_csr_matrix(data_oper)
    bra = cy_csr_matrix(data_oper)
    ket = cy_csr_matrix(data_oper)
    return zcsr_mat_elem(oper, bra, ket)

def kron(op1, op2):
    cop1 = cy_csr_matrix(op1)
    cop2 = cy_csr_matrix(op2)
    return zcsr_kron(op1, op2).to_scipy()

def expect_ket(oper, ket, isherm):
    cop1 = cy_csr_matrix(oper)
    cop2 = cy_csr_matrix(ket)
    return zcsr_expect_ket(op1, op2, isherm)
