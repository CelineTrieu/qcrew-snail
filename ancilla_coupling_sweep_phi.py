import os
import numpy as np
import matplotlib.pyplot as plt
from snail_solver.snail_element import SNAIL
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit
from snail_solver.helper_functions import *
from snail_solver.epr_analysis import epr_analysis, get_epr_circuit_params
import pyEPR as epr

# Open HFSS project
HFSS_project_path = "C:/Users/qcrew5/Desktop/Fernando"
HFSS_project_name = "SQUID_rectangular"
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

# Define circuit and SNAIL parameters
fock_trunc = 18
n = 3
alpha = 0.29
Ej = 52.71e9  # Hz
phi_ext_list = (
    list(np.array([0.405, 0.410, 0.415, 0.420, 0.425, 0.430]) * 2 * np.pi)
    # list(np.arange(0, 0.41, 0.05) * 2 * np.pi)
    # + [0.41 * 2 * np.pi]
    # + list(np.arange(0.45, 1.01, 0.05) * 2 * np.pi)
)
# Define Taylor expansion parameters for the SNAIL potential. Keep them consistent
# across the code. See SNAIL.truncated_potential for definition.
taylor_parameters = {"degree": 40, "scale": 9 * np.pi, "order": None}

ancilla_f01_list = []
anharm_list = []
resonator_f01_list = []
chi_list = []
# Solve HFSS simulation for different values of the external flux
for variation, phi_ext in enumerate(phi_ext_list):
    snail_parameters = {"n": n, "alpha": alpha, "phi_ext": phi_ext}
    snail = SNAIL.from_Ej(Ej, snail_parameters)

    # Now we have to get the Lj given Ej. We can get it from the nonlinear part of the
    # truncated snail potential expanded around the minimum
    Lj = snail.truncated_potential(taylor_parameters, shift=True, nonlinear=True)[3]
    # print(Lj)
    print(6.54e9 * np.sqrt(5e-9 / Lj), phi_ext / 2 / np.pi, Lj)
    # Do EPR analysis
    pinfo = epr.ProjectInfo(
        project_path=HFSS_project_path, project_name=HFSS_project_name
    )
    ## Change Lj in the HFSS design
    pinfo.design.set_variable(junction_info[0][1]["Lj_variable"], "%.2fnH" % (Lj * 1e9))
    for junction in junction_info:
        pinfo.junctions[junction[0]] = junction[1]
    pinfo.validate_junction_info()  # raise error if something is wrong

    ## run HFSS with given configurations
    pinfo.setup.analyze()

    ## Instantiate DistributedAnalysis object for EM field analysis
    eprh = epr.DistributedAnalysis(pinfo)
    eprh.do_EPR_analysis(variations=[str(variation)])  # Calculate participation ratios

    ## Instantiate QuantumAnalysis object to read EPR results. The actual epr quantum
    ## analysis will not take place.
    epra = epr.QuantumAnalysis(eprh.data_filename)

    # Obtain circuit parameters from pyEPR
    epr_Lj, epr_freqs, epr_phi_rzpf = get_epr_circuit_params(
        pinfo, eprh, epra, variation
    )
    freqs = epr_freqs
    phi_rzpf = epr_phi_rzpf
    print(phi_rzpf)
    print(freqs)

    # Assemble circuit
    ancilla = Ancilla(snail, freqs[np.argmax(phi_rzpf)], fock_trunc=fock_trunc)
    circuit = Circuit(ancilla, freqs, phi_rzpf)

    # Get hamiltonian parameters from ancilla and resonator spectra
    ancilla_index = circuit.ancilla_mode
    resonator_index = 0 if circuit.ancilla_mode == 1 else 1

    ## Ancilla fundamental frequency
    ancilla_f01 = (
        circuit.get_eigenstate({ancilla_index: 1})[0]
        - circuit.get_eigenstate({ancilla_index: 0})[0]
    )
    ancilla_f01_list.append(ancilla_f01)

    ## Ancilla anharmonicity
    ancilla_f12 = (
        circuit.get_eigenstate({ancilla_index: 2})[0]
        - circuit.get_eigenstate({ancilla_index: 1})[0]
    )
    anharmonicity = ancilla_f12 - ancilla_f01
    anharm_list.append(anharmonicity)

    ## Resonator fundamental frequency
    resonator_f01 = (
        circuit.get_eigenstate({resonator_index: 1})[0]
        - circuit.get_eigenstate({resonator_index: 0})[0]
    )
    resonator_f01_list.append(resonator_f01)

    ## Dispersive shift
    resonator_f01_shifted = (
        circuit.get_eigenstate({resonator_index: 1, ancilla_index: 1})[0]
        - circuit.get_eigenstate({resonator_index: 0, ancilla_index: 1})[0]
    )
    chi = resonator_f01_shifted - resonator_f01
    chi_list.append(chi)
    print(ancilla_f01, anharmonicity, resonator_f01, chi)

print(ancilla_f01_list)
print(anharm_list)
print(resonator_f01_list)
print(chi_list)
print("external flux list: ", phi_ext_list)
