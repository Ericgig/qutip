
from qutip.cy.csr_matrix import CSR_from_scipy




def cdata_from_scipy(qdata, target=""):
    return CSR_from_scipy(qdata)

##########################
def dense2D_to_data(dense_data, target=""):
    return dense2D_to_CSR(dense_data).to_scipy()



def coo_to_data(coo):
    ...
    return qdata



def data_as_coo(qdata):
    ...
    return coo
