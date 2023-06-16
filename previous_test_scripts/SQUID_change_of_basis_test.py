import numpy as np
import qutip
from qutip.wigner import wigner
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

from snail_solver.single_JJ_element import JJ
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit, tensor_out
from snail_solver.helper_functions import *

# Time unit is sec and frequency is Hz.
# Hamiltonian coefficients are expressed in Hz.


def get_interpolated_phi_rzpf():
    # Get phi zpf data as obtained from pyEPR distributed analysis and interpolate
    phi_rzpf = np.load('C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/phi_rzpf_high_f_squid.npz')["arr_0"]
    phase_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["phase"]
    cavity_phi_rzpf = interp1d(phase_list, phi_rzpf[:, 0])
    ancilla_phi_rzpf = interp1d(phase_list, phi_rzpf[:, 1])
    resonator_phi_rzpf = interp1d(phase_list, phi_rzpf[:, 2])
    return cavity_phi_rzpf, ancilla_phi_rzpf, resonator_phi_rzpf

def get_interpolated_ansys_fs():
    # Get HFSS frequency data and interpolate
    ansys_fs = np.load('C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ansys_fs.npz')["ansys_freqs"]
    phase_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["phase"]
    cavity_fs = interp1d(phase_list, ansys_fs[:,0].reshape(35, 3)[:,0])
    ancilla_fs = interp1d(phase_list, ansys_fs[:,0].reshape(35, 3)[:,1])
    resonator_fs = interp1d(phase_list, ansys_fs[:,0].reshape(35, 3)[:,2])
    return cavity_fs, ancilla_fs, resonator_fs

def get_interpolated_ljs():
    # Get HFSS frequency data and interpolate
    ljs_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["ljs"]
    phase_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["phase"]
    ljs = interp1d(phase_list, ljs_list)
    return ljs

ljs = get_interpolated_ljs()
cavity_phi_rzpf, ancilla_phi_rzpf, resonator_phi_rzpf = get_interpolated_phi_rzpf()
cavity_fs, ancilla_fs, resonator_fs = get_interpolated_ansys_fs()

def build_circuit(phase):
    # The phase input must be within the interpolation range given by the max and min values of phase
    # in the file ljs_and_phase.npz

    # Get circuit and SQUID parameters
    Lj = 1e-9*ljs(phase)
    Ej = 1 / (2 * np.pi * hbar * Lj) * (flux_quantum / 2 / np.pi) ** 2
    #freqs = 1e9*np.array([cavity_fs(phase), ancilla_fs(phase), resonator_fs(phase)])
    freqs = 1e9*np.array([cavity_fs(phase), ancilla_fs(phase)])
    #phi_rzpf = np.array([[cavity_phi_zpf(phase)], [ancilla_phi_zpf(phase)], [resonator_phi_zpf(phase)]])
    phi_rzpf = np.array([[cavity_phi_rzpf(phase)], [ancilla_phi_rzpf(phase)]])
    # Assemble circuit
    squid = JJ(Ej)
    ancilla = Ancilla(squid, freqs[np.argmax(phi_rzpf)], fock_trunc=fock_trunc)
    circuit = Circuit(ancilla, freqs, phi_rzpf)

    return circuit


fock_trunc = 7 
n_modes = 2

# Circuit whose eigenvectors will serve as basis
circuit_A = build_circuit(0.4)
print(circuit_A.get_H(basis = "eigen").diag())

# Another circuit
circuit_B = build_circuit(0.6)

# We can reassemble the hamiltonian from its list of eigenvals and eigenvecs
# Note that reassembling the hamiltonian changes the order of the eigenvectors
# and we lose information about which eigenvec corresponds to which mode. 
# This is a problem when building the number operator. So, since later on we want
# to change the hamiltonian to the interaction picture, better stick with the
# original hamiltonian.
eval, evec = circuit_A.get_circuit_spectrum()
H_A_fock = circuit_A.get_H()
H_A_assembled = sum([eval[i]*qutip.ket2dm(evec[i]) for i in range(len(eval))])
# The change-of-basis (COB) matrix can be constructed from the eigenvectors. It takes the matrix
# from the basis of Fock states of the dressed linear system to the basis of eigenvectors. 
# The COB matrix has to be choosen so to maintain the correct order of increasing excitations.
# Just concatenating the evecs without reordering will make the basis in increasing order of energy
#circuit_A_COB = np.concatenate([np.array(x) for x in evec], axis = 1).T
## Build num_ancilla by picking only those eigenstates in which the resonator is in ground state
ancilla_eigenstates = H_A_fock.ptrace(1).eigenstates()[1]
resonator_eigenstates = H_A_fock.ptrace(0).eigenstates()[1]
evec_excitation_label = []
for i, eigenvec in enumerate(evec):
    # Find the number of excitations in each eigenvector
    i_ancilla = np.argmax([qutip.metrics.fidelity(eigenvec.ptrace(1), qutip.ket2dm(x)) for x in ancilla_eigenstates])
    i_resonator = np.argmax([qutip.metrics.fidelity(eigenvec.ptrace(0), qutip.ket2dm(x)) for x in resonator_eigenstates])
    evec_excitation_label.append((eigenvec, eval[i], (i_resonator, i_ancilla)))

evec_reordered = sorted(evec_excitation_label, key=lambda x: x[2])
circuit_A_COB = np.concatenate([np.array(x[0]) for x in evec_reordered], axis = 1).T
H_A_assembled_diagonal = H_A_assembled.transform(circuit_A_COB)
H_A_diagonal = H_A_fock.transform(circuit_A_COB)
# For simulation purposes, we can set the first eigenvalue as zero. This can be done
# Before or after the change of basis, since the identity matrix is not affected
offset = eval[0]
H_A_assembled_diagonal_offset = H_A_assembled_diagonal - offset
H_A_diagonal_offset = H_A_diagonal - offset
# Next thing is to change to an interaction picture. The frame of choice is
# the rotating frame of the first transition between ancilla and resonator eigenstates.
# Notice this is convenient to be done after diagonalization so we can use the number operator.
num_ancilla = qutip.tensor(qutip.qeye(fock_trunc), qutip.num(fock_trunc))
num_resonator = qutip.tensor(qutip.num(fock_trunc), qutip.qeye(fock_trunc))
f01_ancilla = circuit_A.get_eigenstate({1:1})[0] - circuit_A.get_eigenstate({1:0})[0]
f01_resonator = circuit_A.get_eigenstate({0:1})[0] - circuit_A.get_eigenstate({0:0})[0]
H_A_assembled_diagonal_offset_rotated = H_A_assembled_diagonal_offset - f01_ancilla*num_ancilla - f01_resonator*num_resonator
H_A_diagonal_offset_rotated = H_A_diagonal_offset - f01_ancilla*num_ancilla - f01_resonator*num_resonator
# Can we write the matrix in the basis of the Fock states of the bare modes?
# In this case the cosine has only the SQUID mode and the capacitive coupling term reappears 
phi_rzpf_A = circuit_A.phi_rzpf
a_dressed = tensor_out(qutip.destroy(fock_trunc), 0, fock_trunc, n_modes) # resonator dressed destruction op
b_dressed = tensor_out(qutip.destroy(fock_trunc), 1, fock_trunc, n_modes) # ancilla dressed destruction op
# Get phi_rzpf of the ancilla
phi_rzpf = (phi_rzpf_A[0][0]**2 + phi_rzpf_A[1][0]**2)**0.5  # total reduced flux zpf
a_bare = phi_rzpf_A[0][0]/phi_rzpf*a_dressed + phi_rzpf_A[1][0]/phi_rzpf*b_dressed
b_bare = -phi_rzpf_A[1][0]/phi_rzpf*a_dressed + phi_rzpf_A[0][0]/phi_rzpf*b_dressed
dressing_angle = np.arctan(phi_rzpf_A[1][0]/phi_rzpf_A[0][0])
circuit_A_DCOB = (dressing_angle*(a_dressed.dag()*b_dressed - a_dressed*b_dressed.dag())).expm() # dressing change-of-basis matrix

# COB matrix can be used to change the basis of other hamiltonians.
# We want to bring all hamiltonians to the same basis to be able to compare them.
H_B_fock = circuit_B.get_H()
H_B_basisA = H_B_fock.transform(circuit_A_COB) 
print("\na transform", b_dressed.transform(circuit_A_DCOB))# circuit_A_COB*H_B_fock*circuit_A_COB.dag()
print("\na normal", b_dressed)

print(num_ancilla)
print(H_A_diagonal_offset.diag())
print(H_A_diagonal_offset_rotated.diag())
print(H_A_diagonal_offset)
# This doesn't work because the eigenvalues of H_A_assembled_diagonal_offset are ordered.

offset_A = circuit_A.get_eigenstate({})[0]
eval, evec = circuit_A.get_circuit_spectrum()
assembled_H = sum([eval[i]*qutip.ket2dm(evec[i]) for i in range(len(eval))])
matrix_change_of_basis = np.concatenate([np.array(x) for x in evec], axis = 1).T
print(matrix_change_of_basis.shape)
print(assembled_H.transform(matrix_change_of_basis).diag())
#print(assembled_H)
#print(assembled_H == H_A_fock)
#print(eval)
#H_A_fock = (H_A_fock - offset_A)/1e9
#print(H_A_fock)
#H_A_diag = H_A_fock.diag()
print(eval)


print(circuit_A_DCOB)
