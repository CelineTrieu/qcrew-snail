"""
The ancilla is defined by a nonlinear Josephson element shunted by a capacitance cap.
"""

import qutip as qt
import numpy as np
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from snail_solver.helper_functions import *


class Ancilla:
    def __init__(
        self,
        element,
        freq,
<<<<<<< HEAD
        Lj,
=======
>>>>>>> de1ca688b27c0b276730493130c011b4f6afc828
        taylor_degree=40,
        taylor_scale=9 * np.pi,
        taylor_order=None,
        fock_trunc=70,
    ):

        self.element = element  # Josephson element

        # store numerical calc. parameters
        self.taylor_degree = taylor_degree
        self.taylor_scale = taylor_scale
        self.taylor_order = taylor_order if taylor_order else self.taylor_degree + 10
        self.fock_trunc = fock_trunc

<<<<<<< HEAD
        # circuit parameters
        self.element = element
        self.freq = freq  # linear mode frequency in Hz
        self.Lj = Lj  # element inductance
        self.element.Ej = self.calculate_Ej()  # update element Ej

        # pyEPR will substitute the three lines below
=======
        # Get nonlinear part of the truncated potential expanded around the minimum
        _ = self.element.truncated_potential(
            degree=self.taylor_degree,
            scale=self.taylor_scale,
            order=self.taylor_order,
            shift=False,
            nonlinear=True,
        )
        self.nl_potential, self.taylor_coef, self.Ej, self.Lj = _

        # circuit parameters
        self.freq = freq  # linear mode frequency in Hz
        # These should be replaced by pyEPR. They only make sense when the SNAIL is not
        # coupled to anything else.
>>>>>>> de1ca688b27c0b276730493130c011b4f6afc828
        self.cap = 1 / self.Lj / (2 * np.pi * self.freq) ** 2
        self.phi_zpf = np.sqrt(hbar / (2 * self.cap * 2 * np.pi * self.freq))
        self.phi_rzpf = 2 * np.pi * self.phi_zpf / flux_quantum  # reduced flux zpf
        print(self.phi_rzpf, self.cap, self.Lj, self.element.Ej)
        # qutip mode operators
        self.a = qt.destroy(self.fock_trunc)
        self.ad = self.a.dag()
        self.n = qt.num(self.fock_trunc)

    def calculate_Ej(self):
        """
        Returns the required Ej for a given SNAIL (external flux and alpha defined) to
        have a given inductance Lj.
        """

        taylor_expansion = self.element.solve_expansion(degree=self.taylor_degree)[1]

        # second-order term of the Taylor expansion of the SNAIL potential (normalized)
        a2 = taylor_expansion[2]

        Ej = 1 / 2 / (2 * np.pi * hbar * self.Lj * a2) * (flux_quantum / 2 / np.pi) ** 2

        return Ej

    def calculate_hamiltonian(self):
        """
        Retrieve a qutip hamiltonian operator from the nonlinear potential of the
        Josephson element.
        """

        # scale the hamiltonian by Ej
        Hnl = self.Ej * self.nl_potential(self.phi_rzpf * (self.a + self.ad))
        Hl = self.n * self.freq
<<<<<<< HEAD

        return Hl + Hnl, a3, a4
=======
        return Hl, Hnl, self.taylor_coef
>>>>>>> de1ca688b27c0b276730493130c011b4f6afc828

    def calculate_spectrum(self):
        """
        Diagonalizes the circuit hamiltonian and retrieves its eigenvalues and
        eigenstates.
        """

        Hl, Hnl, taylor_coef = self.calculate_hamiltonian()
        H = Hl + Hnl
        evals, evecs = H.eigenstates()

<<<<<<< HEAD
        return evals, evecs, H, a3, a4
=======
        return evals, evecs, H, taylor_coef
>>>>>>> de1ca688b27c0b276730493130c011b4f6afc828

    def analyze_anharmonicities(self):
        """
        Diagonalizes the circuit hamiltonian and retrieves its eigenvalues and
        eigenstates.
        """

        Hl, Hnl, taylor_coef = self.calculate_hamiltonian()
        H = Hl + Hnl
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

        a3 = taylor_coef[3]
        a4 = taylor_coef[4]

        return (
            first_anharmonicity / 1e6,
            fock_cutoff,
            average_anharmonicity / 1e6,
            is_average_reliable,
            a3,
            a4,
        )
