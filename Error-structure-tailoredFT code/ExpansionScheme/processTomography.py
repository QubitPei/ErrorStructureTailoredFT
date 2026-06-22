import qutip as qt
import numpy as np
from scipy.linalg import block_diag

def initialDensityMatrixForProcessTomography(num_qubits):
    """
    Returns the initial density matrix for process tomography of a num_qubits system.
    """
    rho=[]
    rho.append(qt.basis(2,0)*qt.basis(2,0).dag())
    rho.append(qt.basis(2,1)*qt.basis(2,1).dag())
    rho.append((qt.basis(2,0)+qt.basis(2,1)).unit()*(qt.basis(2,0)+qt.basis(2,1)).unit().dag())
    rho.append((qt.basis(2,0)-1j*qt.basis(2,1)).unit()*(qt.basis(2,0)-1j*qt.basis(2,1)).unit().dag())
    rho_initial = []
    for num_rho in range(4**num_qubits):
        rho_initial.append(qt.tensor(*[rho[int(np.base_repr(num_rho,4).zfill(num_qubits)[num_q])] for num_q in range(num_qubits)]))
    return rho_initial

def initialStateVectorForProcessTomography(num_qubits):
    """
    Returns the initial density matrix for process tomography of a num_qubits system.
    """
    rho=[]
    rho.append(qt.basis(2,0))
    rho.append(qt.basis(2,1))
    rho.append((qt.basis(2,0)+qt.basis(2,1)).unit())
    rho.append((qt.basis(2,0)-1j*qt.basis(2,1)).unit())
    rho_initial = []
    for num_rho in range(4**num_qubits):
        rho_initial.append(qt.tensor(*[rho[int(np.base_repr(num_rho,4).zfill(num_qubits)[num_q])] for num_q in range(num_qubits)]))
    return rho_initial

def pauliForProcessTomography(num_qubits):
    """
    Returns the Pauli matrices for process tomography of a num_qubits system.
    """
    pauli=[]
    pauli.append(qt.qeye(2))
    pauli.append(qt.sigmax())
    pauli.append(-1j*qt.sigmay())
    pauli.append(qt.sigmaz())
    pauli_list = []
    for num_pauli in range(4**num_qubits):
        pauli_list.append(qt.tensor(*[pauli[int(np.base_repr(num_pauli,4).zfill(num_qubits)[num_q])] for num_q in range(num_qubits)]))
    return pauli_list

def cal_ChiMatrix(rho_list):
    """
    Returns the chi matrix for the given rho_list.
    """
    dim_rho = rho_list[0].shape[0]
    num_qubits = int(np.log2(dim_rho))
    rho_list_np = np.array([rho.full() for rho in rho_list])
    rho_list_dr = rho_list_np.reshape((4**num_qubits, 4**num_qubits),order = 'F').transpose()
    rho_ini_list = initialDensityMatrixForProcessTomography((num_qubits))
    rho_ini_list_np = np.array([rho.full() for rho in rho_ini_list])
    rho_ini_list_dr = rho_ini_list_np.reshape((4**num_qubits, 4**num_qubits),order = 'F').transpose()
    lambda_jk = np.linalg.solve(rho_ini_list_dr, rho_list_dr)
    pauli_ini_list = pauliForProcessTomography(num_qubits)
    beta=[]
    for n in range(4**num_qubits):
        for m in range(4**num_qubits):
            e_rho = []
            for j in range(4**num_qubits):
                e_rho.append(pauli_ini_list[m]*rho_ini_list[j]*pauli_ini_list[n].dag())
            e_rho_np_r = np.array([e.full() for e in e_rho]).reshape((4**num_qubits, 4**num_qubits),order = 'F').transpose()
            beta.append(np.linalg.solve(rho_ini_list_dr, e_rho_np_r))
    beta_r = np.array(beta).reshape(((4**num_qubits)**2, (4**num_qubits)**2),order = 'F').transpose()
    lambda_jk_r = lambda_jk.reshape(((4**num_qubits)**2, 1),order = 'F')
    chi = np.linalg.solve(beta_r ,lambda_jk_r)
    chi_matrix = chi.reshape((4**num_qubits, 4**num_qubits),order = 'F')
    return chi_matrix

def densityMatrix2PauliSet(rho):
    """
    Returns the Pauli set for the given density matrix.
    """
    dim_rho = rho.shape[0]
    num_qubits = int(np.log2(dim_rho))
    pauli=[]
    pauli.append(qt.qeye(2))
    pauli.append(qt.sigmax())
    pauli.append(qt.sigmay())
    pauli.append(qt.sigmaz())
    pauli_list = []
    for num_pauli in range(4**num_qubits):
        pauli_list.append(qt.tensor(*[pauli[int(np.base_repr(num_pauli,4).zfill(num_qubits)[num_q])] for num_q in range(num_qubits)]))
    pauli_set = []
    for pauli in pauli_list:
        pauli_set.append(qt.expect(pauli,rho))
    return pauli_set

def pauliSet2DensityMatrix(pauli_set):
    """
    Returns the density matrix for the given Pauli set.
    """
    dim_pauli_set = len(pauli_set)
    num_qubits = int(np.log2(dim_pauli_set)/2)
    pauli=[]
    pauli.append(qt.qeye(2))
    pauli.append(qt.sigmax())
    pauli.append(qt.sigmay())
    pauli.append(qt.sigmaz())
    rho = qt.tensor(*[qt.Qobj(np.zeros((2, 2)))]*num_qubits)
    for j in range(dim_pauli_set):
        rho += pauli_set[j]*qt.tensor(*[pauli[int(np.base_repr(j,4).zfill(num_qubits)[num_q])] for num_q in range(num_qubits)])
    return rho.unit()

def cal_transfer_R_matrix(pauli_out):
    """
    Returns the transfer R matrix for the given pauli_out.
    """
    dim_pauli_out = len(pauli_out)
    num_qubits = int(np.log2(dim_pauli_out)/2)
    p_out = np.reshape(pauli_out,(dim_pauli_out*dim_pauli_out, ))
    rho_ini_list = initialDensityMatrixForProcessTomography((num_qubits))
    p_in = [densityMatrix2PauliSet(rho) for rho in rho_ini_list]
    for j in range(dim_pauli_out):
        temp_row = block_diag(*[p_in[j]]*dim_pauli_out)
        if j==0:
            p_in_block = temp_row
        else:
            p_in_block = np.vstack((p_in_block, temp_row))

    transfer_R = np.linalg.solve(p_in_block, p_out)
    transfer_R_matrix = transfer_R.reshape((4**num_qubits, 4**num_qubits),order = 'F')
    return np.real(transfer_R_matrix)

def cal_transfer_R_matrix_from_rho(rho_list):
    """
    Returns the transfer R matrix for the given rho_list.
    """
    p_out = [densityMatrix2PauliSet(rho) for rho in rho_list]
    transfer_R_matrix = cal_transfer_R_matrix(p_out)
    return transfer_R_matrix

def transfer_R2chi(transfer_R_matrix):
    """
    Returns the chi matrix for the given transfer_R_matrix.
    """
    dim_transfer_R_matrix = transfer_R_matrix.shape[0]
    num_qubits = int(np.log2(dim_transfer_R_matrix)/2)
    rho_ini_list = initialDensityMatrixForProcessTomography((num_qubits))
    rho_out = []
    for k in range(dim_transfer_R_matrix):
        p_in = densityMatrix2PauliSet(rho_ini_list[k])
        p_out = np.dot(transfer_R_matrix,p_in)
        rho_out.append(pauliSet2DensityMatrix(p_out))
    chi_matrix = cal_ChiMatrix(rho_out)
    return chi_matrix

def chi_process(chi,rho_in):
    dim = rho_in.shape[0]
    num_qubits = int(np.log2(dim))
    pauli_d = pauliForProcessTomography(num_qubits)
    rho_out = qt.tensor(*[qt.Qobj(np.zeros((2, 2)))]*num_qubits)
    for i_ in range(4**num_qubits):
        for j_ in range(4**num_qubits):
            rho_out += chi[i_][j_] * pauli_d[i_] * rho_in * pauli_d[j_].dag()
    return rho_out

def chi2_transfer_R(chi):
    """
    Returns the transfer R matrix for the given chi.
    """
    dim_chi = chi.shape[0]
    num_qubits = int(np.log2(dim_chi)/2)
    rho_ini_list = initialDensityMatrixForProcessTomography((num_qubits))
    p_out = []
    for k in range(dim_chi):
        rho_out = chi_process(chi,rho_ini_list[k])
        p_out.append(densityMatrix2PauliSet(rho_out))
    transfer_R_matrix = cal_transfer_R_matrix(p_out)
    return transfer_R_matrix
