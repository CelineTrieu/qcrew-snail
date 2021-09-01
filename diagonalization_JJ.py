import numpy as np
import matplotlib.pyplot as plt
import qutip
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.single_JJ_element import JJ

# Create JJ
Ej = 14e9
freq = 5.19381e9

jj = JJ(Ej)
ancilla = Ancilla(
    jj,
    freq,
    taylor_degree=11,
    fock_trunc=90,
)
# get qutip hamiltonian operator
evals, evecs, H, taylor_coef = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states
print(H.data)
s0 = qutip.basis(ancilla.fock_trunc, 0)
s1 = qutip.basis(ancilla.fock_trunc, 1)
s2 = qutip.basis(ancilla.fock_trunc, 2)
s3 = qutip.basis(ancilla.fock_trunc, 3)
s4 = qutip.basis(ancilla.fock_trunc, 4)
f01 = s1.dag() * H * s1 - s0.dag() * H * s0
f12 = s2.dag() * H * s2 - s1.dag() * H * s1
f23 = s3.dag() * H * s3 - s2.dag() * H * s2
f34 = s4.dag() * H * s4 - s3.dag() * H * s3
print(f12 - f01, f23 - f12, f34 - f23)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
add_spectrum_plot(ax, evals, evecs, ancilla.fock_trunc)

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()
