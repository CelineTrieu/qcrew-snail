"""
Script that executes the EPR analysis using the pyEPR package and returns the 
information needed for quantum analysis.
"""

import pyEPR as epr


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

    # Instantiate QuantumAnalysis object to read EPR PHI_zpf results. The actual quantum
    # analysis will not take place.
    epra = epr.QuantumAnalysis(eprh.data_filename)

    return eprh, epra
