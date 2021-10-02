import os
import numpy as np
import matplotlib.pyplot as plt
import qutip
from functools import reduce

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
eprh, epra = epr_analysis(HFSS_project_path, HFSS_project_name, junction_info)


variation = 0  # for now working with a single variation

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
fock_trunc = 25
freq = np.array(
    eprh.get_ansys_frequencies_all().loc[str(variation)]["Freq. (GHz)"]
)  # HFSS frequencies
PHI_zpf = epra.get_epr_base_matrices(str(variation), _renorm_pj=True)
ancilla = Ancilla(snail, freq[0], fock_trunc=fock_trunc)
resonator = Resonator(freq[1], fock_trunc=fock_trunc)
circuit = Circuit(ancilla, [resonator], PHI_zpf)

# get eigenvectors of isolated SNAIL
evals_0, evecs_0, H_0, taylor_coef = ancilla.calculate_spectrum()
evals_0, evecs_0 = clean_spectrum(evals_0, evecs_0)
print(taylor_coef[3])
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


evals, evecs = circuit.eigenstates()


fig, axes = plt.subplots(2, 1, sharex=True)
# add_transition_energies_plot(axes[0], evals_0)
# add_anharmonicity_plot(axes[1], evals_0)
add_transition_energies_plot(
    axes[0],
    [
        closest_state_to(eigenstate_on({0: i}), evals, evecs)[0]
        for i in range(len(evecs_0))
    ],
)
add_anharmonicity_plot(
    axes[1],
    [
        closest_state_to(eigenstate_on({0: i}), evals, evecs)[0]
        for i in range(len(evecs_0))
    ],
)
plt.show()

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(
    axes[0],
    [
        closest_state_to(eigenstate_on({1: i}), evals, evecs)[0]
        for i in range(len(evecs_0))
    ],
)
add_anharmonicity_plot(
    axes[1],
    [
        1e6 * closest_state_to(eigenstate_on({1: i}), evals, evecs)[0]
        for i in range(len(evecs_0))
    ],
)
plt.show()
