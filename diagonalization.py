import numpy as np
import matplotlib.pyplot as plt
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.snail_element import SNAIL


# Create SNAIL
n = 3
alpha = 0.32
phi_ext = 0.34 * 2 * np.pi
Lj = 11.0e-9
freq = 5.19381e9
config = [n, alpha, phi_ext, Lj, freq]

# (alpha, phi) = (0.35, 0.35), (0.4, 0.35)

snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
ancilla = Ancilla(snail, freq)

# get qutip hamiltonian operator
evals, evecs, H, taylor_coef = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs, threshold=2.0)  # Remove weird states

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()
