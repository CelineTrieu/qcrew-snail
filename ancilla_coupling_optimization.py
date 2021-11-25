import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit
from snail_solver.helper_functions import *

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
fock_trunc = 14
n = 3
Lj = 10e-9
freqs = [5.04952809e09, 6.11913430e09, 6.93729448e09, 7.82853462e09]
phi_rzpf = np.array([[0.36789528], [0.02069451], [0.02306255], [0.02154341]])
n_modes = len(freqs)
ancilla_mode = np.argmax(phi_rzpf)

# Define cost function from kerr
def cost_function(x):

    # Unpack variables
    alpha, phi_ext = x

    # Assemble SNAIL
    snail = SNAIL.from_Lj(Lj, {"n": n, "alpha": alpha, "phi_ext": phi_ext})

    kerr_list = []
    # Calculate the Kerr of each resonator
    for indx in range(n_modes):
        if indx == ancilla_mode:
            continue

        # Select one resonator and bundle it with ancilla
        cut_modes = [ancilla_mode, indx]
        cut_freqs = [freqs[index] for index in cut_modes]
        cut_phis = np.array([[phi_rzpf[index][0]] for index in cut_modes])

        # Assemble circuit
        ancilla = Ancilla(snail, freqs[ancilla_mode], fock_trunc=fock_trunc)
        circuit = Circuit(ancilla, cut_freqs, cut_phis)

        evals = [circuit.get_eigenstate({1: i})[0] for i in range(fock_trunc - 3)]
        evals_data = np.real(evals - evals[0])

        transit_energies = evals_data[1:] - evals_data[:-1]
        anharm = transit_energies[1:] - transit_energies[:-1]
        kerr_list.append(np.average(anharm))
    cost = sum([x ** 2 for x in kerr_list])
    print(
        kerr_list,
        (alpha, phi_ext / 2 / np.pi),
        "%.4f" % ancilla.taylor_coef[3],
        np.sqrt(cost),
    )
    return cost


res = minimize(
    cost_function,
    (0.29, 0.41 * 2 * np.pi),
    bounds=((0.2, 0.4 * 2 * np.pi), (0.4, 0.5 * 2 * np.pi)),
)
print(res.x)
