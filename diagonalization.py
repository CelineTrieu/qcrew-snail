import numpy as np
import matplotlib.pyplot as plt
from snail_solver.elements import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *

# Create SNAIL
Ej = 14.851e9 * 3  # in Hz
n = 3
alpha = 0.29
phi_ext = 0.39 * 2 * np.pi
snail = SNAIL(n, alpha, phi_ext, Ej=Ej)

# Create ancillas for given shunt capacitance
freq = 5e9  # linear mode frequency in Hz
Lj = 16e-9  # junction inductance in H
ancilla = Ancilla(snail, freq, Lj)

# get qutip hamiltonian operator
evals, evecs, H, a3, a4 = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states

# plot potential
phi_list = np.arange(-4 * np.pi, 4 * np.pi, 0.01)
import matplotlib.pyplot as plt

f = ancilla.element.truncated_potential(norm=False)[0]
plt.plot(phi_list, ancilla.element.potential(phi_list))
plt.plot(phi_list, f(phi_list))
plt.show()

# Draw plots
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
add_spectrum_plot(ax, evals, evecs, ancilla.fock_trunc)
plt.show()

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()
