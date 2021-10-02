import numpy as np
import matplotlib.pyplot as plt
import qutip
from qutip.qobj import Qobj
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from functools import reduce


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


HFSS_project_name = "SQUID"
fock_trunc = 30

# Create SNAIL
n = 3
Lj = 25e-9  # HFSS lumped element inductance
freq = np.array([5.104723e9, 7.465660e9])  # HFSS frequencies
PHI_zpf = np.array([[0.62331647], [0.01441409]])

cavity_kerr_list = []
max_kerr_list = []
a3_list = []
a4_list = []

alpha_list = np.arange(0.2, 0.4, 0.01)
phi_ext_list = np.arange(0.2 * 2 * np.pi, 0.491 * 2 * np.pi, 0.01 * 2 * np.pi)
for alpha in alpha_list:
    for phi_ext in phi_ext_list:
        snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
        # snail = SNAIL.from_Ej(Ej, n, alpha, phi_ext)
        ancilla = Ancilla(snail, freq[0], fock_trunc=fock_trunc)

        # get eigenvectors of isolated SNAIL
        evals_0, evecs_0, H_0, taylor_coef = ancilla.calculate_spectrum()
        evals_0, evecs_0 = clean_spectrum(evals_0, evecs_0)

        snail_mode = 0  # SNAIL mode number
        n_modes = 2  # total number of modes

        # get a3 and a4
        a3_list.append(taylor_coef[3])
        a4_list.append(taylor_coef[4])

        # instantiate qutip operators
        a = qutip.destroy(ancilla.fock_trunc)
        ad = a.dag()
        num = qutip.num(ancilla.fock_trunc)
        mode_fields = [tensor_out(a + ad, i) for i in range(n_modes)]
        mode_ns = [tensor_out(num, i) for i in range(n_modes)]

        # build coupled hamiltonian
        cos_interiors = PHI_zpf[0, 0] * mode_fields[0] + PHI_zpf[1, 0] * mode_fields[1]
        Hl = dot(freq, mode_ns)
        Hnl = ancilla.Ej * ancilla.nl_potential(cos_interiors)
        H = Hl + Hnl

        # diagonalize coupled hamiltonian
        evals, evecs = H.eigenstates()

        def closest_state_to(s):
            def distance(s2):
                return (s.dag() * s2[1]).norm()

            return max(zip(evals, evecs), key=distance)

        cavity_evals = [
            closest_state_to(eigenstate_on({1: i}))[0] for i in range(len(evecs_0))
        ]

        relative_evals = np.real(cavity_evals - cavity_evals[0]) / 1e6
        transit_energies = relative_evals[1:] - relative_evals[:-1]
        anharm = transit_energies[1:] - transit_energies[:-1]
        avg_kerr = 1000 * np.average(anharm)
        max_kerr = 1000 * np.max(np.abs(anharm))
        cavity_kerr_list.append(avg_kerr)
        max_kerr_list.append(max_kerr)
        print(
            "%.3f %.3f %d %.3f %.3f"
            % (
                avg_kerr,
                max_kerr,
                len(anharm),
                alpha,
                phi_ext / 2 / np.pi,
            )
        )

phi_ext_list /= 2 * np.pi  # plot in units of 2pi
reshape_dim = (len(alpha_list), len(phi_ext_list))

# Reshape values for colormesh plotting
cavity_kerr_list = np.reshape(cavity_kerr_list, reshape_dim)
max_kerr_list = np.reshape(max_kerr_list, reshape_dim)

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)

im1 = axes[0].pcolormesh(
    phi_ext_list, alpha_list, cavity_kerr_list, shading="auto", cmap="bwr"
)
im1.set_clim(-0.6, 0.6)
fig.colorbar(im1, ax=axes[0])
axes[0].set_title("Average cavity Kerr")
axes[0].set_ylabel("Josephson energy proportion alpha")
axes[0].set_xlabel("External flux (per flux quanta)")

im2 = axes[1].pcolormesh(
    phi_ext_list, alpha_list, max_kerr_list, shading="auto", cmap="bwr"
)
im2.set_clim(0, 0.6)
fig.colorbar(im2, ax=axes[1])
axes[1].set_title("Maximum absolute cavity Kerr")
axes[1].set_xlabel("External flux (per flux quanta)")

plt.show()
