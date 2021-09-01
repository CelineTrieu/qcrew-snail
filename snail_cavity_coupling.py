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
alpha = 0.35
phi_ext = 0.35 * 2 * np.pi
Lj = 11e-9  # HFSS lumped element inductance
freq = np.array([5.19381e9, 7.46624])  # HFSS frequencies
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


def tensor_out(op, loc):
    "Make operator <op> tensored with identities at locations other than <loc>"
    op_list = [qutip.qeye(ancilla.fock_trunc) for i in range(n_modes)]
    op_list[loc] = op
    return reduce(qutip.tensor, op_list)


def closest_state_to(s):
    def distance(s2):
        return (s.dag() * s2[1]).norm()

    return max(zip(evals, evecs), key=distance)


def dot(ais, bis):
    """
    Dot product
    """
    return sum(ai * bi for ai, bi in zip(ais, bis))


# Hamiltonian building and diagonalization
a = qutip.destroy(ancilla.fock_trunc)
ad = a.dag()
n = qutip.num(ancilla.fock_trunc)
mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
mode_ns = [tensor_out(n, i) for i in range(n_modes)]

cos_interiors = [dot(phi, mode_fields) for phi in PHI_zpf]

Hl = dot(freq, mode_ns)
Hnl = dot([ancilla.Ej, ancilla.Ej], map(ancilla.nl_potential, cos_interiors))
H = Hl + Hnl
evals, evecs = H.eigenstates()

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals_0)
add_anharmonicity_plot(axes[1], evals_0)
add_transition_energies_plot(
    axes[0], [closest_state_to(eigenstate_on({0: i}))[0] for i in range(len(evecs_0))]
)
add_anharmonicity_plot(
    axes[1], [closest_state_to(eigenstate_on({0: i}))[0] for i in range(len(evecs_0))]
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


"""
    # scale the hamiltonian by Ej
    Hnl = self.Ej * self.nl_potential(self.phi_rzpf * (self.a + self.ad))
    Hl = self.n * self.freq
    return Hl, Hnl, self.taylor_coef


    
    cos_interiors = [dot(fzpf_row / fluxQ, mode_fields) for fzpf_row in fzpfs]
    nonlinear_part = dot(-fjs, map(cos, cos_interiors))
    if individual:
        return linear_part, nonlinear_part
    else:
        return linear_part + nonlinear_part
"""
