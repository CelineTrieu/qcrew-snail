import numpy as np
import matplotlib.pyplot as plt
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *

# Create SNAIL
Ej = 33667189300.20908  # in Hz
n = 3
<<<<<<< HEAD
alpha = 0.29
phi_ext_list = [0.35 * 2 * np.pi, 0.37 * 2 * np.pi, 0.39 * 2 * np.pi, 0.41 * 2 * np.pi]
snail_list = [SNAIL(n, alpha, phi_ext, Ej=Ej) for phi_ext in phi_ext_list]
=======
alpha = 0.35
phi_ext_list = np.arange(0.4, 0.50, 0.01) * 2 * np.pi
snail_list = [SNAIL.from_Ej(Ej, n, alpha, phi_ext) for phi_ext in phi_ext_list]
>>>>>>> de1ca688b27c0b276730493130c011b4f6afc828

# Create ancillas for given shunt capacitance
freq = 5e9  # Hz
ancilla_snail_list = [Ancilla(snail, freq) for snail in snail_list]


fig, axes = plt.subplots(2, 1, sharex=True)
for ancilla in ancilla_snail_list:
    # Ancilla analysis
    evals, evecs, H, taylor_coef = ancilla.calculate_spectrum()
    evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states

    # Draw plots
    # add_energy_diagram_plot(axes[0], evals)
    add_transition_energies_plot(axes[0], evals)
    add_anharmonicity_plot(
        axes[1], evals, label=str(ancilla.element.phi_ext / 2 / np.pi)[:5] + "*2pi"
    )

plt.legend()
plt.show()
