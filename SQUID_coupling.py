import os
import numpy as np
import matplotlib.pyplot as plt
from snail_solver.single_JJ_element import JJ
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit
from snail_solver.helper_functions import *
from snail_solver.epr_analysis import epr_analysis, get_epr_circuit_params

# Define circuit and SNAIL parameters
fock_trunc = 12
Lj = 7.5e-9
Ej = 1 / (2 * np.pi * hbar * Lj) * (flux_quantum / 2 / np.pi) ** 2
freqs = np.array([5711619202.44, 5805423210.32, 6995467289.219999])
phi_rzpf = np.array([[0.04554338], [0.35613048], [0.03060912]])

# Assemble circuit
squid = JJ(Ej)
ancilla = Ancilla(squid, freqs[np.argmax(phi_rzpf)], fock_trunc=fock_trunc)
circuit = Circuit(ancilla, freqs, phi_rzpf)

# Plotting
fig, axes = plt.subplots(2, 2)
# Ancilla spectrum when resonator is in ground state
ancilla_spec = [
    circuit.get_eigenstate({1: i})[0] for i in range(len(circuit.ancilla_spectrum[0]))
]
add_transition_energies_plot(axes[0, 0], ancilla_spec)
add_anharmonicity_plot(axes[1, 0], ancilla_spec, unit = 'MHz')

# Resonator spectrum when ancilla is in ground state
resonator_spec = [circuit.get_eigenstate({0: i})[0] for i in range(7)]
add_transition_energies_plot(axes[0, 1], resonator_spec)
add_anharmonicity_plot(axes[1, 1], resonator_spec, unit = 'kHz')
plt.show()

## Resonator fundamental frequency
resonator_f01_list = []
for i in range(6):
    resonator_f01_list.append(
        circuit.get_eigenstate({0: i+1})[0]
        - circuit.get_eigenstate({0: i+0})[0]
    )
print(resonator_f01_list)
resonator_f01_shifted_list = []
for i in range(6):
    resonator_f01_shifted_list.append(
        circuit.get_eigenstate({0: i+1, 1:1})[0]
        - circuit.get_eigenstate({0: i+0, 1:1})[0]
    )
## Dispersive shift
print(resonator_f01_shifted_list)
chi_list = [resonator_f01_shifted_list[i] - resonator_f01_list[i] for i in range(6)]
print(chi_list)
chi_prime_list = [chi_list[i+1] - chi_list[i] for i in range(len(chi_list)-1)]
print(chi_prime_list)
fig, axes = plt.subplots(1,2)
axes[0].scatter(range(len(chi_list)), np.array(chi_list)/1e6)
axes[0].set_ylabel(r"$\chi$ (MHz)")
axes[0].set_xlabel("Index")
axes[1].scatter(range(len(chi_prime_list)), np.array(chi_prime_list)/1e3)
axes[1].set_ylabel(r"$\chi$' (kHz)")
axes[1].set_xlabel("Index")
axes[0].grid()
axes[1].grid()
plt.show()