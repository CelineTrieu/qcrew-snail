import numpy as np
import matplotlib.pyplot as plt
from snail_solver.elements import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *

# Create SNAIL
Ej = 14.851e9 * 3  # in Hz
n = 3
alpha = 0.29
phi_ext = 0.41 * 2 * np.pi
snail = SNAIL(n, alpha, phi_ext, Ej=Ej)

# Create ancillas for given shunt capacitance
cap_list = [
    57.5e-15,
    77.5e-15,
    97.5e-15,
    117.5e-15,
    137.5e-15,
]  # shunt capacitance in F
ancilla_snail_list = [Ancilla(snail, cap) for cap in cap_list]


fig, axes = plt.subplots(3, 1, sharex=True)
for ancilla in ancilla_snail_list:
    # Ancilla analysis
    evals, evecs, H = ancilla.calculate_spectrum()
    evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states

    # Draw plots
    add_energy_diagram_plot(axes[0], evals)
    add_transition_energies_plot(axes[1], evals)
    add_anharmonicity_plot(axes[2], evals, label=str(ancilla.cap))

plt.legend()
plt.show()
