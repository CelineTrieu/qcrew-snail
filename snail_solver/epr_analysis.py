"""
Script that executes the EPR analysis using the pyEPR package and returns the 
information needed for quantum analysis.
"""

import pyEPR as epr
import numpy as np


def epr_analysis(HFSS_project_path, HFSS_project_name, junction_info):
    pinfo = epr.ProjectInfo(
        project_path=HFSS_project_path, project_name=HFSS_project_name
    )
    for junction in junction_info:
        pinfo.junctions[junction[0]] = junction[1]
    pinfo.validate_junction_info()  # raise error if something is wrong
    pinfo.setup.analyze()  # run HFSS with given configurations

    # Instantiate DistributedAnalysis object for EM field analysis
    eprh = epr.DistributedAnalysis(pinfo)
    eprh.do_EPR_analysis()  # Calculate participation ratios

    # Instantiate QuantumAnalysis object to read EPR results. The actual quantum
    # analysis will not take place.
    epra = epr.QuantumAnalysis(eprh.data_filename)

    return pinfo, eprh, epra


def get_epr_circuit_params(pinfo, eprh, epra, variation):
    """[summary]

    Args:
        pinfo ([type]): [description]
        eprh ([type]): [description]
        epra ([type]): [description]
    """
    epr_Lj = 1e-9 * float(
        eprh.get_ansys_variables()
        .loc[pinfo.junctions["j1"]["Lj_variable"]][variation]
        .replace("nH", "")
    )

    epr_freqs = 1e9 * np.array(
        eprh.get_ansys_frequencies_all().loc[str(variation)]["Freq. (GHz)"]
    )

    epr_phi_rzpf = epra.get_epr_base_matrices(str(variation), _renorm_pj=True)[4]

    return epr_Lj, epr_freqs, epr_phi_rzpf
