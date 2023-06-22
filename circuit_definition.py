import numpy as np

from snail_solver.ancillae import SNAIL, JJ 
from snail_solver.circuit import Circuit
from snail_solver.parameters import ContantParameters

fock_trunc = 10

# Create SNAIL
snail_parameters = {"n": 3, "alpha": 0.290, "phi_ext": 0.410 * 2 * np.pi}
ancilla = SNAIL(**snail_parameters, fock_trunc = fock_trunc)
# ancilla = JJ(fock_trunc = fock_trunc)

# Define constant circuit parameters: SNAIL + 1 resonator
Lj = 9.0e-9
freqs = np.array([5.727e9, 6.181e9])
phi_rzpf = np.array([[0.00588067], [0.35791346]])
params = ContantParameters(Lj, freqs, phi_rzpf)

# Create circuit
circuit = Circuit(ancilla, params)
circuit.set_circuit_parameters()

# calculate circuit hamiltonian
circuit.hamiltonian
circuit.circuit_spectrum
circuit.ancilla_spectrum
print(circuit.calc_bare_mode_freqs())