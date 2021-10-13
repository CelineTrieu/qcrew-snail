import numpy as np
import matplotlib.pyplot as plt
import qutip
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.single_JJ_element import JJ

# Create JJ
Ej = 14e9
freq = 5.19381e9

ancilla = Ancilla(
    JJ(Ej),
    freq,
    taylor_degree=30,
    fock_trunc=30,
)
# get qutip hamiltonian operator
evals, evecs = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states
evecs = [qutip.Qobj(ev) for ev in evecs]

# plot
fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()
