import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit
from snail_solver.helper_functions import *

# from snail_solver.epr_analysis import epr_analysis, get_epr_circuit_params
"""
# Executes EPR analysis
HFSS_project_path = os.getcwd()
HFSS_project_name = "SNAIL_test"
junction_info = [
    (
        "j1",  # assign junction name
        {
            "Lj_variable": "junction_LJ",
            "rect": "junction_left",
            "line": "junction_line",
            "length": "junction_l",
            "Cj_variable": "junction_CJ",
        },  # related HFSS variable names
    )
]
_ = epr_analysis(HFSS_project_path, HFSS_project_name, junction_info)
# Obtain circuit parameters from pyEPR
variation = 0
epr_Lj, epr_freqs, epr_phi_rzpf = get_epr_circuit_params(*_, variation)
"""

# Define fixed circuit parameters
fock_trunc = 10
n = 3
Lj = 14e-9
# freqs = [5.90273626e09, 6.31585224e09]  # , 7.12303432e+09]
freqs = [4.76634030e09, 6.12085314e09]
phi_rzpf = np.array([[0.43528079], [0.01620791]])

# Sweep SNAIL parameters alpha and phi_ext
cavity_kerr_list = []
max_kerr_list = []
alpha_list = np.arange(0.2, 0.5, 0.002)
phi_ext_list = np.arange(0.2 * 2 * np.pi, 0.47 * 2 * np.pi, 0.002 * 2 * np.pi)
for alpha in alpha_list:
    for phi_ext in phi_ext_list:

        # assemble circuit
        snail = SNAIL.from_Lj(Lj, {"n": 3, "alpha": alpha, "phi_ext": phi_ext})
        ancilla = Ancilla(snail, freqs[np.argmax(phi_rzpf)], fock_trunc=fock_trunc)
        circuit = Circuit(ancilla, freqs, phi_rzpf)

        # diagonalize coupled hamiltonian
        evals, evecs = circuit.circuit_spectrum

        cavity_evals = [
            circuit.get_eigenstate({1: i})[0] for i in range(fock_trunc - 3)
        ]

        # Calculate optimization variables
        relative_evals = np.real(cavity_evals - cavity_evals[0])
        transit_energies = relative_evals[1:] - relative_evals[:-1]
        anharm = transit_energies[1:] - transit_energies[:-1]
        avg_kerr = np.average(anharm)
        max_kerr = np.max(np.abs(anharm))

        # Save for plotting
        cavity_kerr_list.append(abs(avg_kerr))
        max_kerr_list.append(max_kerr)
        print(alpha, phi_ext, avg_kerr)

# Adjust and reshape values for colormesh plotting
phi_ext_list /= 2 * np.pi  # plot in units of 2pi
reshape_dim = (len(alpha_list), len(phi_ext_list))
cavity_kerr_list = np.reshape(cavity_kerr_list, reshape_dim)
max_kerr_list = np.reshape(max_kerr_list, reshape_dim)
# meshgrid
X, Y = np.meshgrid(phi_ext_list, alpha_list)

# Plot
fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)

im1 = axes[0].pcolormesh(
    phi_ext_list,
    alpha_list,
    cavity_kerr_list,
    norm=colors.LogNorm(vmin=cavity_kerr_list.min(), vmax=cavity_kerr_list.max()),
    shading="auto",
    cmap="Oranges",
)
fig.colorbar(im1, ax=axes[0])  # , extend="max"
axes[0].set_title("Average cavity Kerr")
axes[0].set_ylabel("Josephson energy proportion alpha")
axes[0].set_xlabel("External flux (per flux quanta)")
# Plot contour
CS = axes[0].contour(X, Y, cavity_kerr_list, [10, 100, 500, 1000])
axes[0].clabel(CS, inline=True, fontsize=9)


im2 = axes[1].pcolormesh(
    phi_ext_list,
    alpha_list,
    max_kerr_list,
    norm=colors.LogNorm(vmin=max_kerr_list.min(), vmax=max_kerr_list.max()),
    shading="auto",
    cmap="Oranges",
)
fig.colorbar(im2, ax=axes[1])  # , extend="max"
axes[1].set_title("Maximum absolute cavity Kerr")
axes[1].set_xlabel("External flux (per flux quanta)")
# Plot contour
CS = axes[1].contour(X, Y, max_kerr_list, [10, 100, 500, 1000])
axes[1].clabel(CS, inline=True, fontsize=9)

plt.show()
