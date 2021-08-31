import numpy as np
import matplotlib.pyplot as plt
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *

# Create SNAIL
Ej = 14.851e9 * 3  # in Hz
n = 3
alpha_list = [0.23, 0.26, 0.29, 0.32]
phi_ext = 0.35 * 2 * np.pi
snail_list = [SNAIL.from_Ej(Ej, n, alpha, phi_ext) for alpha in alpha_list]

# Create ancillas for given shunt capacitance
freq = 5e9  # shunt capacitance in F
ancilla_snail_list = [Ancilla(snail, freq) for snail in snail_list]

fig, axes = plt.subplots(3, 1, sharex=True)
for ancilla in ancilla_snail_list:
    # Ancilla analysis
    evals, evecs, H, taylor_coef = ancilla.calculate_spectrum()
    evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states

    # Draw plots
    add_energy_diagram_plot(axes[0], evals)
    add_transition_energies_plot(axes[1], evals)
    add_anharmonicity_plot(axes[2], evals, label=ancilla.element.alpha)

plt.legend()
plt.show()
