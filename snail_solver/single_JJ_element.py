import numpy as np
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from snail_solver.helper_functions import *
import math


class JJ:
    """ """

    def __init__(self, Ej):
        self.Ej = Ej
        self.Lj = 1 / (2 * np.pi * hbar * self.Ej) * (flux_quantum / 2 / np.pi) ** 2

    def potential(self, phi):
        """
        Normalized potential energy of the josephson junction as a function of the
        reduced flux ::phi:: over the element. Assumes Ej = 1.
        """

        return -np.cos(phi)

    def solve_expansion(self, degree=40, scale=9 * np.pi, order=None):
        """
        Returns the flux of minimum potential and the coefficients of the truncated
        Taylor expansion around the minimum.
        scale is the width of the interval used in the Taylor fit.
        """

        order = order if order else degree + 10

        # Initial guess for potential minimum
        phi_array = np.arange(-np.pi, np.pi, 0.01)
        phi_min_0 = min(self.potential(phi) for phi in phi_array)

        # find flux of minimum potential
        phi_min = minimize(self.potential, phi_min_0).x[0]

        # Fit Taylor expansion around minimum
        taylor_coef = approximate_taylor_polynomial(
            self.potential, phi_min, degree, scale, order=order
        )

        return phi_min, taylor_coef

    def truncated_potential_old(
        self,
        degree=40,
        scale=9 * np.pi,
        order=None,
        shift=True,
        nonlinear=False,
    ):
        """
        Calculates the potential Taylor expansion and reconstructs the potential
        function from the coefficients.
        degree is the order of the cosine expasion.
        nonlinear = True returns only the nonlinear part of the potential.
        element scale the (i+3)th term of the taylor expansion of the potential.
        """

        phi_0, taylor_coef = self.solve_expansion(
            degree=degree, scale=scale, order=order
        )
        phi_0 = phi_0 if shift else 0
        limit = 3 if nonlinear else 0

        def potential(x):
            return sum(
                taylor_coef[degree - k] * (x - phi_0) ** (degree - k)
                for k in range(len(taylor_coef) + 1 - limit)
            )

        return potential, taylor_coef, self.Ej, self.Lj

    def truncated_potential(
        self,
        degree=40,
        scale=9 * np.pi,
        order=None,
        shift=True,
        nonlinear=False,
    ):

        taylor_coef = [
            -np.real((1j) ** k / math.factorial(k)) if k % 2 == 0 else 0
            for k in range(degree)
        ]

        phi_0 = 0
        limit = 3 if nonlinear else 0

        def potential(x):
            return sum(
                taylor_coef[degree - k] * (x - phi_0) ** (degree - k)
                for k in range(1, len(taylor_coef) + 1 - limit)
            )

        return potential, taylor_coef, self.Ej, self.Lj
