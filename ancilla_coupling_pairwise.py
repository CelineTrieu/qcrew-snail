import os
import numpy as np
import matplotlib.pyplot as plt
from snail_solver.snail_element import SNAIL
from snail_solver.single_JJ_element import JJ
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

# Define circuit and SNAIL parameters
fock_trunc = 12
snail_parameters = {"n": 3, "alpha": 0.280, "phi_ext": 0.450 * 2 * np.pi}
Lj = 14.0e-9
freqs = [4.76634030e09, 6.12085314e09, 6.94129285e09, 7.83171449e09]
phi_rzpf = np.array([[0.43528079], [0.01620791], [0.01854175], [0.01716157]])

# Assemble circuit with up to three elements, always including the ancilla
snail = SNAIL.from_Lj(Lj, snail_parameters)
ancilla_mode = np.argmax(phi_rzpf)
dressed_freqs_list = []
kerr_matrix_list = []
for i in range(len(freqs)):
    for j in range(i + 1, len(freqs)):
        if ancilla_mode in {i, j}:
            continue
        cut_modes = [ancilla_mode, i, j]
        cut_freqs = [freqs[index] for index in cut_modes]
        cut_phis = np.array([[phi_rzpf[index][0]] for index in [ancilla_mode, i, j]])
        # JJ(11672737030.61179)
        ancilla = Ancilla(snail, cut_freqs[np.argmax(cut_phis)], fock_trunc=fock_trunc)
        circuit = Circuit(ancilla, cut_freqs, cut_phis)
        dressed_freqs, kerr_matrix = report_H_params(circuit, len(freqs), cut_modes)
        dressed_freqs_list.append(dressed_freqs)
        kerr_matrix_list.append(kerr_matrix)


avg_freqs_list = np.nanmean(np.array(dressed_freqs_list), 0)
avg_kerr_matrix = np.nanmean(np.array(kerr_matrix_list), 0)
print(avg_freqs_list)
print(avg_kerr_matrix)
