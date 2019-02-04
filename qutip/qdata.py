
from qutip.cy.csr_matrix import CSR_from_scipy




def cdata_from_scipy(qdata):
    return CSR_from_scipy(qdata)

##########################
def dense2D_to_data(dense_data):
    return dense2D_to_CSR(dense_data).to_scipy()
