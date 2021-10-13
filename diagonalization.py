import numpy as np
import matplotlib.pyplot as plt
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.snail_element import SNAIL

fock_trunc = 30

# Create SNAIL
n = 3
alpha = 0.320
phi_ext = 0.470 * 2 * np.pi
Lj = 11.0e-9
# Ej = 96151595754.02753
freq = 5.19381e9

# (alpha, phi) = (0.35, 0.35), (0.4, 0.35)

snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
ancilla = Ancilla(snail, freq, fock_trunc=fock_trunc)

# get qutip hamiltonian operator
evals, evecs = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()
