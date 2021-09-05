import numpy as np
import matplotlib.pyplot as plt
import qutip
from qutip.qobj import Qobj
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.snail_element import SNAIL
from functools import reduce

HFSS_project_name = "SQUID"

# Create SNAIL (alpha, phi) = (0.35, 0.35), (0.4, 0.35)
n = 3
alpha = 0.32
phi_ext = 0.34 * 2 * np.pi
Lj = 11e-9  # HFSS lumped element inductance
freq = np.array([5.19381e9, 7.46624e9])  # HFSS frequencies
PHI_zpf = np.array([[0.41476267], [0.02612749]])
snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
ancilla = Ancilla(snail, freq[0])

# get eigenvectors of isolated SNAIL
evals_0, evecs_0, H_0, taylor_coef = ancilla.calculate_spectrum()
evals_0, evecs_0 = clean_spectrum(evals_0, evecs_0)

snail_mode = 0  # SNAIL mode number
n_modes = 2  # total number of modes


def eigenstate_on(d):
    """d={mode number: state number}"""
    eigenvectors_list = [
        qutip.basis(ancilla.fock_trunc, d.get(i, 0)) for i in range(n_modes)
    ]
    eigenvectors_list[snail_mode] = Qobj(evecs_0[d.get(snail_mode, 0)])

    return reduce(qutip.tensor, eigenvectors_list)


def fock_on(d):
    """d={mode number: state number}"""
    eigenvectors_list = [
        qutip.basis(ancilla.fock_trunc, d.get(i, 0)) for i in range(n_modes)
    ]

    return reduce(qutip.tensor, eigenvectors_list)


def tensor_out(op, loc):
    "Make operator <op> tensored with identities at locations other than <loc>"
    op_list = [qutip.qeye(ancilla.fock_trunc) for i in range(n_modes)]
    op_list[loc] = op
    return reduce(qutip.tensor, op_list)


def dot(ais, bis):
    """
    Dot product
    """
    return sum(ai * bi for ai, bi in zip(ais, bis))


# intantiate qutip operators
a = qutip.destroy(ancilla.fock_trunc)
ad = a.dag()
n = qutip.num(ancilla.fock_trunc)
mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
mode_ns = [tensor_out(n, i) for i in range(n_modes)]

# build hamiltonian
cos_interiors = PHI_zpf[0, 0] * mode_fields[0] + PHI_zpf[1, 0] * mode_fields[1]
Hl = dot(freq, mode_ns)
Hnl = ancilla.Ej * ancilla.nl_potential(cos_interiors)
H = Hl + Hnl

# diagonalize hamiltonian
evals, evecs = H.eigenstates()


def closest_state_to(s):
    def distance(s2):
        return (s.dag() * s2[1]).norm()

    return max(zip(evals, evecs), key=distance)


fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals_0)
add_anharmonicity_plot(axes[1], evals_0)
add_transition_energies_plot(
    axes[0], [closest_state_to(eigenstate_on({0: i}))[0] for i in range(len(evecs_0))]
)
add_anharmonicity_plot(
    axes[1], [closest_state_to(eigenstate_on({0: i}))[0] for i in range(len(evecs_0))]
)
add_transition_energies_plot(
    axes[0],
    [closest_state_to(eigenstate_on({0: i, 1: 1}))[0] for i in range(len(evecs_0))],
)
add_anharmonicity_plot(
    axes[1],
    [closest_state_to(eigenstate_on({0: i, 1: 1}))[0] for i in range(len(evecs_0))],
)
add_transition_energies_plot(
    axes[0],
    [closest_state_to(eigenstate_on({0: i, 1: 2}))[0] for i in range(len(evecs_0))],
)
add_anharmonicity_plot(
    axes[1],
    [closest_state_to(eigenstate_on({0: i, 1: 2}))[0] for i in range(len(evecs_0))],
)
add_transition_energies_plot(
    axes[0],
    [closest_state_to(eigenstate_on({0: i, 1: 3}))[0] for i in range(len(evecs_0))],
)
add_anharmonicity_plot(
    axes[1],
    [closest_state_to(eigenstate_on({0: i, 1: 3}))[0] for i in range(len(evecs_0))],
)
plt.show()

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(
    axes[0], [closest_state_to(eigenstate_on({1: i}))[0] for i in range(len(evecs_0))]
)
add_anharmonicity_plot(
    axes[1], [closest_state_to(eigenstate_on({1: i}))[0] for i in range(len(evecs_0))]
)
add_transition_energies_plot(
    axes[0],
    [closest_state_to(eigenstate_on({0: 1, 1: i}))[0] for i in range(len(evecs_0))],
)
add_anharmonicity_plot(
    axes[1],
    [closest_state_to(eigenstate_on({0: 1, 1: i}))[0] for i in range(len(evecs_0))],
)
plt.show()
