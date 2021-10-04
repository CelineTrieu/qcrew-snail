import os
import numpy as np
import matplotlib.pyplot as plt
import qutip
from qutip.qobj import Qobj
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit
from snail_solver.helper_functions import *
from snail_solver.epr_analysis import epr_analysis

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
pinfo, eprh, epra = epr_analysis(HFSS_project_path, HFSS_project_name, junction_info)

# Quantum circuit analysis
variation = 0

# Create SNAIL (alpha, phi) = (0.35, 0.35), (0.4, 0.35)
n = 3
Lj = 1e-9 * float(
    eprh.get_ansys_variables()
    .loc[pinfo.junctions["j1"]["Lj_variable"]][variation]
    .replace("nH", "")
)
# snail = SNAIL.from_Ej(Ej, n, alpha, phi_ext)

fock_trunc = 18
freqs = 1e9 * np.array(
    eprh.get_ansys_frequencies_all().loc[str(variation)]["Freq. (GHz)"]
)  # HFSS frequencies
PHI_zpf = epra.get_epr_base_matrices(str(variation), _renorm_pj=True)

cavity_kerr_list = []
max_kerr_list = []

alpha_list = np.arange(0.2, 0.4, 0.01)
phi_ext_list = np.arange(0.2 * 2 * np.pi, 0.491 * 2 * np.pi, 0.01 * 2 * np.pi)
for alpha in alpha_list:
    for phi_ext in phi_ext_list:

        # Assemble circuit
        snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
        ancilla = Ancilla(snail, freqs[np.argmax(PHI_zpf)], fock_trunc=fock_trunc)
        circuit = Circuit(ancilla, freqs, PHI_zpf)

        # diagonalize coupled hamiltonian
        circuit.calc_ancilla_spectrum()
        evals, evecs = circuit.calc_spectrum()

        cavity_evals = [
            circuit.get_eigenstate({1: i})[0] for i in range(fock_trunc - 3)
        ]

        relative_evals = np.real(cavity_evals - cavity_evals[0]) / 1e6
        transit_energies = relative_evals[1:] - relative_evals[:-1]
        anharm = transit_energies[1:] - transit_energies[:-1]
        avg_kerr = 1000 * np.average(anharm)
        max_kerr = 1000 * np.max(np.abs(anharm))
        cavity_kerr_list.append(avg_kerr)
        max_kerr_list.append(max_kerr)
        print(
            "%.3f %.3f %.3f %.3f"
            % (
                avg_kerr,
                max_kerr,
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
