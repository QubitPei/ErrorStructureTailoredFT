import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import qutip_myw
from utils.processTomography import (
    initialDensityMatrixForProcessTomography,
    transfer_R2chi,
    cal_transfer_R_matrix_from_rho,
    cal_ChiMatrix
)

kappa2_c = 1e-3*2*np.pi
kappa1_c = 1.6e-6*2*np.pi
kappa2_t = 1e-3*2*np.pi
kappa1_t = 1.6e-6*2*np.pi

N = 30
c  = qt.tensor(qt.destroy(N), qt.qeye(N))
numN_c  = c.dag()*c
t  = qt.tensor(qt.qeye(N), qt.destroy(N))
numN_t  = t.dag()*t
nbar=5
alphacat=np.sqrt(nbar)
delta_omega = 0
kerr=0


c_ops = []
#
# Engineer disippation
c_ops.append(np.sqrt(kappa2_c) * (c**2-alphacat**2))
c_ops.append(np.sqrt(kappa1_c) * (c))
c_ops.append(np.sqrt(kappa1_t) * (t))

H0=delta_omega * numN_c + kerr/2 * c.dag() **2 * c **2 + delta_omega * numN_t + kerr/2 * t.dag() **2 * t **2

T_star = 0.282/nbar/np.sqrt(kappa1_c*kappa2_t)
H_drive = np.pi /4/alphacat/ T_star * (c.dag() + c - 2*alphacat) * (t.dag() * t - nbar)

basis_change = (qt.coherent(N,alphacat)+qt.coherent(N,-alphacat)).unit()*(qt.basis(2,0)+qt.basis(2,1)).unit().dag() + \
                (qt.coherent(N,alphacat)-qt.coherent(N,-alphacat)).unit()*(qt.basis(2,0)-qt.basis(2,1)).unit().dag()
basis_change_2 = qt.tensor(basis_change, basis_change)

rho_ini_list = initialDensityMatrixForProcessTomography(2)
rho_fin_list = []

for rho_ini in rho_ini_list:
    psi0 = basis_change_2 * rho_ini * basis_change_2.dag()
    tlist = np.linspace(0, T_star, 1000)
    result = qt.mesolve(H0+H_drive, psi0, tlist, c_ops=c_ops, options=qt.Options(nsteps=10000),progress_bar=True)
    c_ops_conv = [np.sqrt(kappa2_c) * (c**2-alphacat**2),np.sqrt(kappa2_t) * (t**2-alphacat**2)]
    result1 = qt.mesolve(H0, result.states[-1], tlist, c_ops=c_ops_conv, options=qt.Options(nsteps=10000),progress_bar=True)
    rho_fin_list.append(basis_change_2.dag()*result1.states[-1].unit()*basis_change_2)

U_post = qt.tensor(qt.sigmaz(), qt.qeye(2))
rho_fin_list1 = [U_post * rho * U_post.dag() for rho in rho_fin_list]
r_matrix = cal_transfer_R_matrix_from_rho(rho_fin_list1)
chi_matrix = cal_ChiMatrix(rho_fin_list1)
r2_matrix = np.dot(r_matrix,r_matrix)
chi2_matrix = transfer_R2chi(r2_matrix)

U_cnot = qt.cnot()
rho_ideal_list = [U_cnot * rho * U_cnot.dag() for rho in rho_ini_list]
chi_ideal_matrix = cal_ChiMatrix(rho_ideal_list)
cnot_fidelity = np.trace(np.dot(chi_ideal_matrix,chi_matrix))#/np.trace(np.dot(r_ideal_matrix,r_ideal_matrix))

# control_psit = result1.states[-1].ptrace(0).unit()
# target_psit = result1.states[-1].ptrace(1).unit()
# print(qt.fidelity(control_psit, qt.coherent(N,alphacat))**2)
# print(qt.fidelity(control_psit, qt.coherent(N,-alphacat))**2)
# print(qt.fidelity(target_psit, qt.coherent(N,alphacat))**2)
# print(qt.fidelity(target_psit, qt.coherent(N,-alphacat))**2)