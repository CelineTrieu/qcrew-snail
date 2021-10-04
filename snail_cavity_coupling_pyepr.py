import os
import numpy as np
import matplotlib.pyplot as plt
import qutip

from qutip.qobj import Qobj
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.snail_element import SNAIL
from snail_solver.resonator import Resonator
from snail_solver.circuit import Circuit
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
alpha = 0.33
phi_ext = 0.410 * 2 * np.pi
Lj = 1e-9 * float(
    eprh.get_ansys_variables()
    .loc[pinfo.junctions["j1"]["Lj_variable"]][variation]
    .replace("nH", "")
)
snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
# snail = SNAIL.from_Ej(Ej, n, alpha, phi_ext)

# Assemble circuit
fock_trunc = 18
freqs = 1e9 * np.array(
    eprh.get_ansys_frequencies_all().loc[str(variation)]["Freq. (GHz)"]
)  # HFSS frequencies
PHI_zpf = epra.get_epr_base_matrices(str(variation), _renorm_pj=True)
ancilla = Ancilla(snail, freqs[np.argmax(PHI_zpf)], fock_trunc=fock_trunc)
circuit = Circuit(ancilla, freqs, PHI_zpf)

circuit.calc_ancilla_spectrum()
circuit.calc_spectrum()

# Plotting
fig, axes = plt.subplots(2, 2)
add_transition_energies_plot(
    axes[0, 0],
    [
        circuit.get_eigenstate({0: i})[0]
        for i in range(len(circuit.ancilla_spectrum[0]))
    ],
)
add_anharmonicity_plot(
    axes[1, 0],
    [
        circuit.get_eigenstate({0: i})[0]
        for i in range(len(circuit.ancilla_spectrum[0]))
    ],
)

add_transition_energies_plot(
    axes[0, 1],
    [circuit.get_eigenstate({1: i})[0] for i in range(fock_trunc - 3)],
)
add_anharmonicity_plot(
    axes[1, 1],
    [1e6 * circuit.get_eigenstate({1: i})[0] for i in range(fock_trunc - 3)],
)
plt.show()
