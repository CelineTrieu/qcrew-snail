"""
Classes for defining a SNAIL circuit element.

The SNAIL is a two-terminal element composed of two parallel branches. One branch has a number n of Josephson junctions with characteristic energies Ej and the shunt branch has one junction with energy alpha*Ej, where alpha is a real number. The branches form a loop threaded by a reduced external flux phi_ext.

Reference: 3-Wave Mixing Josephson Dipole Element (doi: 10.1063/1.4984142)
"""

import numpy as np
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from snail_solver.helper_functions import *


class SNAIL:
    def __init__(self, n, alpha, phi_ext):
        """Base class for a SNAIL circuit element.

        The SNAIL element is used to define an ancilla object.

        Args:
            n (int): Number of series Josephson junctions. Equals to the total number
            of junctions minus one, corresponding to the shunting junction.

            alpha (float): Josephson energy scaling of the shunt Josephson junction.

            phi_ext (float): External reducing flux threading the circuit, in radians.
        """
        self.n = n
        self.alpha = alpha
        self.phi_ext = phi_ext

    @classmethod
    def from_Lj(self, Lj, snail_parameters):
        """The SNAIL is defined by a fixed Lj. Ej is a function of the parameters and
        the Taylor expansion.

        Returns:
            SNAIL_from_Lj: Child object that contains method to calculate Ej.
        """
        return SNAIL_from_Lj(Lj, snail_parameters)

    @classmethod
    def from_Ej(self, Ej, snail_parameters):
        """The SNAIL is defined by a fixed Ej. Lj is a function of the parameters and
        the Taylor expansion.

        Returns:
            SNAIL_from_Ej: Child object that contains method to calculate Lj.
        """
        return SNAIL_from_Ej(Ej, snail_parameters)

    def potential(self, phi):
        """Normalized (Ej=1) potential energy as a function of the reduced flux over
        the element.

        Args:
            phi (float): Reduced flux over the SNAIL element

        Returns:
            float: Value of the potential when the reduced flux over the junction is
            phi.
        """

        pot_branch1 = -self.alpha * np.cos(phi)
        pot_branch2 = -self.n * np.cos((self.phi_ext - phi) / self.n)

        return pot_branch1 + pot_branch2

    def solve_expansion(self, degree=40, scale=9 * np.pi, order=None):
        """Calculates the Taylor series of the potential around the minimum.

        Args:
            degree (int, optional): Numerical parameter. Defines degree of the
            Taylor series expansion of the potential. Defaults to 40.

            scale (float, optional): Range of fit for Taylor series expansion.
            Defaults to 9*np.pi.

            order (int, optional): The order of the polynomial to be used in the fitting. Defaults to None, in which case it is redefined as taylor_degree + 10.

        Returns:
            tuple: (phi_min, taylor_coef), where phi_min is the reduced flux of minimum
            potential and taylor_coef is a poly1d object defining the Taylor series
            around the minimum.
        """

        order = order if order else degree + 10

        # Initial guess for potential minimum
        phi_array = np.arange(-np.pi * self.n, np.pi * self.n, 0.01)
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
        taylor_parameters,
        shift=False,
        nonlinear=False,
    ):
        """Calculates the Taylor expansion of the potential and reconstructs the
        function from the coefficients.

        Args:

            taylor_parameters (dict): Dictionary defining numerical parameters for
            fitting the element potential to a Taylor series. See SNAIL.solve_expansion
            method.

            shift (bool, optional): If True, the returned potential is
            redefined in terms of the distance from its minimum. Defaults to False.

            nonlinear (bool, optional): If True, exclude the second-order term from the
            polynomial series. Defaults to False.

        Returns:
            tuple: (potential, taylor_coef, Ej, Lj)
        """

        phi_0, taylor_coef = self.solve_expansion(**taylor_parameters)
        phi_0 = phi_0 if not shift else 0
        excluded_terms = [2, 0] if nonlinear else [0]

        degree = taylor_parameters["degree"]

        def potential(x):
            return sum(
                taylor_coef[degree - k] * (x - phi_0) ** (degree - k)
                for k in range(len(taylor_coef) + 1)
                if degree - k not in excluded_terms
            )

        a2 = taylor_coef[2]
        Ej, Lj = self.get_Ej_Lj(a2)

        return potential, taylor_coef, Ej, Lj

    def potential_derivative(self, phi):
        """Derivative of the snail potential with respect to the reduced flux across
        the junction.

        Args:
            phi (float): Flux point where to calculate the derivative value.

        Returns:
            float: Value of the derivative of the potential.
        """

        pot_branch1_derivative = self.alpha * np.sin(phi)
        pot_branch2_derivative = -np.sin((self.phi_ext - phi) / self.n)

        return pot_branch1_derivative + pot_branch2_derivative

    @property
    def has_multiple_wells(self):
        """Analyzes if the potential has more than one local minimum per period.

        Returns:
            bool: Returns True if more than one local minimum is found in a period
            of the SNAIL potential.
        """

        return more_than_2_roots(self.potential_derivative, 0, self.n * 2 * np.pi)


class SNAIL_from_Lj(SNAIL):
    def __init__(self, Lj, snail_parameters):
        """Child SNAIL class defined from its equivalent inductance.

        The Josephon energy Ej of each series junction can be obtained from the
        inductance and the second-order coefficient of the Taylor series of the SNAIL
        potential.

        Args:
            Lj (float): Target equivalent inductance of the nonlinear element, in H.

            snail_parameters (dict): dictionary containing the parameters of the parent
            class n, alpha, and phi_ext.

        """
        self.Lj = Lj  # H
        super().__init__(**snail_parameters)  # Passes other parameters to parent

    def Ej(self, a2):
        """Calculate Ej from Lj and second-order term of Taylor series of the SNAIL
        potential.

        Args:
            a2 (float): Second-order term of the Taylor expansion of the potential.

        Returns:
            float: Equivalent inductance of the element.
        """
        Ej = 1 / 2 / (2 * np.pi * hbar * self.Lj * a2) * (flux_quantum / 2 / np.pi) ** 2
        return Ej

    def get_Ej_Lj(self, a2):
        """Bundle up and return both Josephson energy of shunt junctions and equivalent
        inductance.

        Args:
            a2 (float): Second-order term of the Taylor expansion of the potential.

        Returns:
            tuple: (Ej, Lj), where Ej is the Josephson energy of shunt junctions and
            Lj is the equivalent inductance.
        """
        return self.Ej(a2), self.Lj


class SNAIL_from_Ej(SNAIL):
    def __init__(self, Ej, snail_parameters):
        """Child SNAIL class defined from the Josephson energy Ej of the series
        junctions.

        The equivalent SNAIL inductance Lj can be obtained from the inductance and the
        second-order coefficient of the Taylor series of the SNAIL potential.

        Args:
            Ej (float): Equivalent Josephson energy of the series Josephson junctions,
            in Hz.

            snail_parameters (dict): dictionary containing the parameters of the parent
            class n, alpha, and phi_ext.
        """
        self.Ej = Ej  # Hz
        super().__init__(**snail_parameters)  # Passes other parameters to parent

    def Lj(self, a2):
        """Calculate equivalent inductance from Ej and second-order term of Taylor
        series of the SNAIL potential.

        Args:
            a2 (float): Second-order term of the Taylor expansion of the potential.

        Returns:
            float: Equivalent inductance of the element.
        """
        Lj = 1 / 2 / (2 * np.pi * hbar * self.Ej * a2) * (flux_quantum / 2 / np.pi) ** 2
        return Lj

    def get_Ej_Lj(self, a2):
        """Bundle up and return both Ej and Lj.

        Args:
            a2 (float): Second-order term of the Taylor expansion of the potential.

        Returns:
            tuple: (Ej, Lj), where Ej is the Josephson energy of shunt junctions and
            Lj is the equivalent inductance.
        """
        return self.Ej, self.Lj(a2)
