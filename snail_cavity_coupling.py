import numpy as np
import matplotlib.pyplot as plt
import qutip
from qutip.qobj import Qobj
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.snail_element import SNAIL
from snail_solver.resonator import Resonator
from snail_solver.circuit import Circuit
from functools import reduce

# Create SNAIL (alpha, phi) = (0.35, 0.35), (0.4, 0.35)
n = 3
alpha = 0.29
phi_ext = 0.41 * 2 * np.pi
Lj = 25e-9  # HFSS lumped element inductance
snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
# snail = SNAIL.from_Ej(Ej, n, alpha, phi_ext)

# Assemble circuit
fock_trunc = 18
freqs = np.array([5.104723e9, 7.465660e9])  # HFSS frequencies
PHI_zpf = np.array([[0.62331647], [0.01441409]])
ancilla = Ancilla(snail, freqs[np.argmax(PHI_zpf)], fock_trunc=fock_trunc)
circuit = Circuit(ancilla, freqs, PHI_zpf)

circuit.calc_ancilla_spectrum()
circuit.calc_spectrum()

# Plotting
fig, axes = plt.subplots(2, 2)
add_transition_energies_plot(
    axes[0, 0],
    [
        circuit.get_eigenstate({0: i})[0]
        for i in range(len(circuit.ancilla_spectrum[0]))
    ],
)
add_anharmonicity_plot(
    axes[1, 0],
    [
        circuit.get_eigenstate({0: i})[0]
        for i in range(len(circuit.ancilla_spectrum[0]))
    ],
)

add_transition_energies_plot(
    axes[0, 1],
    [circuit.get_eigenstate({1: i})[0] for i in range(fock_trunc - 3)],
)
add_anharmonicity_plot(
    axes[1, 1],
    [1e6 * circuit.get_eigenstate({1: i})[0] for i in range(fock_trunc - 3)],
)
plt.show()
