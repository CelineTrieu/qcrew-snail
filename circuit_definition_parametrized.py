import numpy as np

from snail_solver.ancillae import JJ
from snail_solver.circuit import Circuit
from snail_solver.parameters import InterpolatedParameters

fock_trunc = 10

# Create SQUID
ancilla = JJ(fock_trunc=fock_trunc)

# Build interpolated circuit params: SQUID + 2 resonators
ljs_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["ljs"]*1e-9
ansys_fs = np.load('C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ansys_fs.npz')["ansys_freqs"][:,0].reshape(-1,3)*1e9
phi_rzpf = np.load('C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/phi_rzpf_high_f_squid.npz')["arr_0"]
phase_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["phase"]
params = InterpolatedParameters(ljs_list, ansys_fs, phi_rzpf, phase_list)

# Create circuit
circuit = Circuit(ancilla, params)
initial_phase = 0.2 # Initialize circuit with given SQUID phase
circuit.set_circuit_parameters(initial_phase)

# Calculate system hamiltonian for different phase points
# Previously calculated hamiltonians are saved in memory
circuit.hamiltonian
circuit.set_circuit_parameters(0.3)
circuit.hamiltonian
circuit.set_circuit_parameters(0.2)
circuit.hamiltonian