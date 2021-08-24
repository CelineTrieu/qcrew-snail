"""
The ancilla is defined by a nonlinear Josephson element shunted by a capacitance cap.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from helper_functions import *
import qutip as qt


class Ancilla:
    def __init__(self, element, cap, taylor_order=7, fock_trunc=20):

        # store numerical parameters
        self.fock_trunc = fock_trunc
        self.taylor_order = taylor_order

        # circuit parameters
        self.element = element
        self.cap = cap  # F
        print("calc_inductance")
        self.Lj = self.element.inductance(degree=self.taylor_order)
        self.freq = 1 / np.sqrt(self.cap * self.Lj) / 2 / np.pi  # linear mode frequency
        self.phi_zpf = np.sqrt(hbar / (2 * self.cap * 2 * np.pi * self.freq))
        self.phi_rzpf = 2 * np.pi * self.phi_zpf / flux_quantum  # reduced flux zpf

        # qutip mode operators
        self.a = qt.destroy(fock_trunc)
        self.ad = self.a.dag()
        self.n = qt.num(fock_trunc)

    def calculate_hamiltonian(self, term_scaling=None):
        """
        Retrieve a qutip hamiltonian operator from the nonlinear potential of the
        Josephson element. ::term_scaling:: accepts a list of floats in which the ith
        element scale the (i+3)th term of the taylor expansion of the potential.
        """

        # Get nonlinear part of the truncated potential expanded around the minimum
        nl_potential = self.element.truncated_potential(
            degree=self.taylor_order,
            norm=False,
            shift=False,
            nonlinear=True,
            term_scaling=term_scaling,
        )
        Hnl = nl_potential(self.phi_rzpf * (self.a + self.ad))
        Hl = self.n * self.freq

        return Hl, Hnl

    def calculate_spectrum(self, term_scaling=None):
        """
        Diagonalizes the circuit hamiltonian and retrieves its eigenvalues and
        eigenstates.
        """

        Hl, Hnl = self.calculate_hamiltonian(term_scaling=term_scaling)
        H = Hl + Hnl
        evals, evecs = H.eigenstates()

        return evals, evecs, H
