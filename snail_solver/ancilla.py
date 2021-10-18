"""The ancilla object is defined by a generalized nonlinear Josephson element shunted 
by a capacitance. It is designed to receive the linear mode frequency directly from 
HFSS to grant seamless connection to simulations.
"""

import qutip as qt
import numpy as np
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from snail_solver.helper_functions import *


class Ancilla:
    def __init__(self, element, freq, fock_trunc=20, taylor_parameters=None):
        """Define an ancilla from a nonlinear circuit element and a linear mode
        frequency.

        Args:

            element (JJ, SNAIL): JJ or SNAIL object. Defines nonlinear circuit
            element that composes the ancilla. This element is responsible for defining
            the inductance of the ancilla mode.

            freq (float): Linear mode frequency as obtained from HFSS (Hz).

            taylor_parameters (dict): Dictionary defining numerical parameters for
            fitting the element potential to a Taylor series. See SNAIL.solve_expansion
            method.

            fock_trunc (int, optional): Dimension of qutip matrices. Defaults to 20.
        """

        self.element = element

        # store numerical calc. parameters
        self.fock_trunc = fock_trunc
        if not taylor_parameters:
            taylor_parameters = {"degree": 40, "scale": 9 * np.pi, "order": None}
        self.taylor_parameters = taylor_parameters

        # Get nonlinear part of the truncated potential expanded around the minimum
        _ = self.element.truncated_potential(
            self.taylor_parameters,
            shift=True,
            nonlinear=True,
        )
        self.nl_potential, self.taylor_coef, self.Ej, self.Lj = _

        # circuit parameters
        self.freq = freq  # linear mode frequency in Hz

        # qutip mode operators
        self.destroy = qt.destroy(self.fock_trunc)
        self.create = self.destroy.dag()
        self.n = qt.num(self.fock_trunc)
        self._Hl, self._Hnl = None, None

    @property
    def Hl(self):
        """Linear part of ancilla hamiltonian when isolated from other modes, in Hz."""

        if self._Hl:
            return self._Hl
        else:
            self._Hl, self._Hnl = self._calculate_hamiltonian()
            return self._Hl

    @property
    def Hnl(self):
        """Nonlinear part of ancilla hamiltonian when isolated from other modes, in Hz."""

        if self._Hnl:
            return self._Hnl
        else:
            self._Hl, self._Hnl = self._calculate_hamiltonian()
            return self._Hnl

    def _calculate_hamiltonian(self):
        """Retrieve the hamiltonian of the ancilla assuming it is isolated from other
        modes.

        Since the ancilla is assumed to be isolated, the reduced flux zpf can be
        calculated analitically.

        Returns:
            tuple: (Hl, Hnl), where Hl and Hnl are Qobj operators of the linear and
            nonlinear parts of the ancilla hamiltonian in Hz.
        """

        # calculate the flux zpf for the isolated device
        cap = 1 / self.Lj / (2 * np.pi * self.freq) ** 2
        phi_zpf = np.sqrt(hbar / (2 * cap * 2 * np.pi * self.freq))
        phi_rzpf = 2 * np.pi * phi_zpf / flux_quantum  # reduced flux zpf

        # scale the hamiltonian by Ej
        Hnl = self.Ej * self.nl_potential(phi_rzpf * (self.destroy + self.create))
        Hl = self.n * self.freq
        return Hl, Hnl

    def calculate_spectrum(self):
        """Diagonalizes the circuit hamiltonian and retrieves its eigenvalues and
        eigenstates.

        Returns:
            tuple: (evals, evecs), where evals is an np.array of hamiltonian
            eigenvalues in Hz and evecs is the respective np.array of eigenvectors.
        """

        H = self.Hl + self.Hnl
        evals, evecs = H.eigenstates()

        return evals, evecs

    def analyze_anharmonicities(self):
        """
        Diagonalizes the circuit hamiltonian and retrieves its eigenvalues and
        eigenstates.
        """

        H = self.Hl + self.Hnl
        evals, evecs = H.eigenstates()
        # Clean the spectrum of weird eigenstates
        evals, evecs = clean_spectrum(evals, evecs)

        # Obtain anharmonicities
        evals_data = np.real(evals - evals[0])
        transit_energies = evals_data[1:] - evals_data[:-1]
        anharm = transit_energies[1:] - transit_energies[:-1]

        first_anharmonicity = anharm[0]
        # index of maximum anharmonicity.
        max_index = np.argmax(anharm)
        # minimum amount of quanta until the states become approx. linearly spaced.
        fock_cutoff = max_index + 2
        average_anharmonicity = np.average(np.abs(anharm[max_index:]))

        is_average_reliable = True
        if len(anharm[max_index:]) < 10:
            is_average_reliable = False

        return (
            first_anharmonicity / 1e6,
            fock_cutoff,
            average_anharmonicity / 1e6,
            is_average_reliable,
        )
