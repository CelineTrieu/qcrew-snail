"""
The purpose of this script is to demonstrate how to choose a common basis with which to write 
SQUID hamiltonians at two or more different flux points.
"""

import numpy as np
import qutip
from qutip.wigner import wigner
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

from snail_solver.josephson_element.JJ_element import JJ
from snail_solver.ancillae import Ancilla
from snail_solver.circuit import Circuit
from snail_solver.helper_functions import *

# Time unit is sec and frequency is Hz.
# Hamiltonian coefficients are expressed in Hz.

fock_trunc = 10
def get_interpolated_phi_zpf():
    # Get phi zpf data as obtained from pyEPR distributed analysis and interpolate
    phi_zpf = np.load('C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/phi_rzpf_high_f_squid.npz')["arr_0"]
    phase_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["phase"]
    cavity_phi_zpf = interp1d(phase_list, phi_zpf[:, 0])
    ancilla_phi_zpf = interp1d(phase_list, phi_zpf[:, 1])
    resonator_phi_zpf = interp1d(phase_list, phi_zpf[:, 2])
    return cavity_phi_zpf, ancilla_phi_zpf, resonator_phi_zpf

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
cavity_phi_zpf, ancilla_phi_zpf, resonator_phi_zpf = get_interpolated_phi_zpf()
cavity_fs, ancilla_fs, resonator_fs = get_interpolated_ansys_fs()

def build_circuit(phase, reference_operators = None):
    # The phase input must be within the interpolation range given by the max and min values of phase
    # in the file ljs_and_phase.npz

    # Get circuit and SQUID parameters
    Lj = 1e-9*ljs(phase)
    Ej = 1 / (2 * np.pi * hbar * Lj) * (flux_quantum / 2 / np.pi) ** 2
    #freqs = 1e9*np.array([cavity_fs(phase), ancilla_fs(phase), resonator_fs(phase)])
    freqs = 1e9*np.array([cavity_fs(phase), ancilla_fs(phase)])
    #phi_rzpf = np.array([[cavity_phi_zpf(phase)], [ancilla_phi_zpf(phase)], [resonator_phi_zpf(phase)]])
    phi_rzpf = np.array([[cavity_phi_zpf(phase)], [ancilla_phi_zpf(phase)]])
    # Assemble circuit
    squid = JJ(Ej)
    ancilla = Ancilla(squid, freqs[np.argmax(phi_rzpf)], fock_trunc=fock_trunc)
    circuit = Circuit(ancilla, freqs, phi_rzpf, reference_operators = reference_operators)

    return circuit

reference_circuit = build_circuit(0.4)
ref_operators = reference_circuit.calc_ancilla_flux_charge_operators()
circuit_A = build_circuit(0.41, reference_operators = ref_operators)
print(reference_circuit.coupling_factor)
print(circuit_A.coupling_factor)

# circuit_B = build_circuit(0.41)
# # reference_circuit.get_H()
# circuit_A.get_H()
# print("CIRCUIT B")
# circuit_B.get_H()

# print(b0, flux_op, charge_op, circuit_ref.coupling_factor)
# circuit = build_circuit(0.5)
# circuit_operators = circuit.calc_mode_operators(b0, flux_op, charge_op)
# initial_state =  (initial_circuit.get_eigenstate({1: 0})[1] + initial_circuit.get_eigenstate({1: 1})[1]).unit()
# ref_operators = initial_circuit.calc_ancilla_flux_charge_operators()
# f01 = initial_circuit.get_eigenstate({1: 1})[0] - initial_circuit.get_eigenstate({1: 0})[0]
# offset = initial_circuit.get_eigenstate({1: 0})[0]
# ancilla_number_op = sum([np.sqrt(i)*qutip.ket2dm(initial_circuit.get_eigenstate({1: i})[1])
#                          for i in range(len(circuit.ancilla_spectrum[0]))])