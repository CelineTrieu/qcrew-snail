import os
import numpy as np
import matplotlib.pyplot as plt
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit
from snail_solver.helper_functions import *
from snail_solver.epr_analysis import epr_analysis, get_epr_circuit_params

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

# Define circuit and SNAIL parameters
fock_trunc = 18
n = 3
alpha = 0.33
phi_ext = 0.410 * 2 * np.pi
Lj = epr_Lj
freqs = epr_freqs
phi_rzpf = epr_phi_rzpf

# Assemble circuit
snail = SNAIL.from_Lj(Lj, n, alpha, phi_ext)
ancilla = Ancilla(snail, freqs[np.argmax(phi_rzpf)], fock_trunc=fock_trunc)
circuit = Circuit(ancilla, freqs, phi_rzpf)

# Plotting
fig, axes = plt.subplots(2, 2)
# Ancilla spectrum when resonator is in ground state
ancilla_spec = [
    circuit.get_eigenstate({0: i})[0] for i in range(len(circuit.ancilla_spectrum[0]))
]
add_transition_energies_plot(axes[0, 0], ancilla_spec)
add_anharmonicity_plot(axes[1, 0], ancilla_spec)

# Resonator spectrum when ancilla is in ground state
resonator_spec = [circuit.get_eigenstate({1: i})[0] for i in range(fock_trunc - 3)]
add_transition_energies_plot(axes[0, 1], resonator_spec)
add_anharmonicity_plot(axes[1, 1], resonator_spec)
plt.show()
