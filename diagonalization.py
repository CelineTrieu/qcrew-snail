import numpy as np
import matplotlib.pyplot as plt
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.snail_element import SNAIL

# Create SNAIL
n = 3
<<<<<<< HEAD
alpha = 0.29
phi_ext = 0.39 * 2 * np.pi
snail = SNAIL(n, alpha, phi_ext, Ej=Ej)

# Create ancillas for given shunt capacitance
freq = 5e9  # linear mode frequency in Hz
Lj = 16e-9  # junction inductance in H
ancilla = Ancilla(snail, freq, Lj)
=======
alpha = 0.35
phi_ext = 0.35 * 2 * np.pi
Lj = 11.0e-9
freq = 5.0e9

# (alpha, phi) = (0.35, 0.35), (0.4, 0.35)

snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
ancilla = Ancilla(snail, freq)
print(ancilla.Ej, ancilla.Lj)
>>>>>>> de1ca688b27c0b276730493130c011b4f6afc828

# get qutip hamiltonian operator
evals, evecs, H, taylor_coef = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states

<<<<<<< HEAD
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

=======
>>>>>>> de1ca688b27c0b276730493130c011b4f6afc828
fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()
