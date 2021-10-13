import numpy as np
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from snail_solver.helper_functions import *
import math


class JJ:
    def __init__(self, Ej):
        """Define a Josephson junction to be used as a nonlinear circuit element.

        Args:
            Ej ([float]): Josephson energy of the junction in Hz.
        """
        self.Ej = Ej
        self.Lj = 1 / (2 * np.pi * hbar * self.Ej) * (flux_quantum / 2 / np.pi) ** 2

    def potential(self, phi):
        """Calculate normalized potential energy (Ej = 1) of the junction as a function of the reduced flux phi over the element.

        Args:
            phi ([float]): Reduced flux over the junction in radians.

        Returns:
            [float]: Value of potential energy of the junction in units of Ej.
        """

        return -np.cos(phi)

    def solve_expansion(self, degree=40, scale=9 * np.pi, order=None):
        """Calculate the flux of minimum potential and fit a truncated
        Taylor expansion around the minimum.

        Args:

            degree (int, optional): [description]. Defaults to 40.

            scale ([type], optional): Width of the fit interval of the Taylor expansion around the minimum. Defaults to 9*np.pi.

            order ([type], optional): [description]. Defaults to None.

        Returns:
            [tuple]: (phi_min, taylor_coef) where phi_min is the [float] flux of minimum potential and taylor_coef
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

    def truncated_potential(
        self,
        degree=40,
        scale=9 * np.pi,
        order=None,
        shift=True,
        nonlinear=False,
    ):
        """Return the Taylor expansion of the junction potential and, if needed, calculate Ej and Lj according to the expansion.

        Args:
            degree (int, optional): [description]. Defaults to 40.

            scale ([type], optional): Width of the fit interval of the Taylor expansion around the minimum. Defaults to 9*np.pi.

            order ([type], optional): [description]. Defaults to None.

            shift (bool, optional): If shift = False, redefine the point of minimum potential as zero flux. Defaults to True.

            nonlinear (bool, optional): If nonlinear = True, consider only the nonlinear terms. Defaults to False.

        Returns:
            [tuple]: [description]
        """

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
