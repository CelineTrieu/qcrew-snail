import numpy as np
import matplotlib.pyplot as plt
from snail_solver.elements import SNAIL, SNAIL2
from snail_solver.ancilla import Ancilla, Ancilla2
from snail_solver.helper_functions import *

# Create SNAIL
n = 3
alpha = 0.45
phi_ext = 0.45 * 2 * np.pi
Lj = 17e-9
freq = 4.6e9

snail = SNAIL(n, alpha, phi_ext, Lj)
ancilla = Ancilla(snail, freq)

# get qutip hamiltonian operator
evals, evecs, H, a3, a4 = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states

# Draw plots
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# add_spectrum_plot(ax, evals, evecs, ancilla.fock_trunc)
# plt.show()

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()

# Create SNAIL2
Ej = 14.851e9 * 3  # in Hz
n = 3
alpha = 0.29
phi_ext = 0.39 * 2 * np.pi
snail = SNAIL2(Ej, n, alpha, phi_ext)

# Create ancillas for given shunt capacitance
cap = 97.5e-15  # shunt capacitance in F
ancilla = Ancilla2(snail, cap)

# get qutip hamiltonian operator
evals, evecs, H, a3, a4 = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states

# Draw plots
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
add_spectrum_plot(ax, evals, evecs, ancilla.fock_trunc)
plt.show()

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()
