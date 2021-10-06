import numpy as np
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from snail_solver.helper_functions import *


class SNAIL:
    """
    Normalized SNAIL. The hamiltonian scaling is decided either by Lj or Ej, which is
    done in the child classes.

    The SNAIL element is defined as having two nodes connected by two parallel
    branches. One branch has ::n:: josephson junctions with characteristic energies
    :Ej: and the other branch has one junction with energy ::Ej:: scaled by ::alpha::.
    The branches form a loop threaded by a reduced external flux ::phi_ext::.

    Reference: 3-Wave Mixing Josephson Dipole Element (doi: 10.1063/1.4984142)
    """

    def __init__(self, n, alpha, phi_ext):
        self.n = n
        self.alpha = alpha
        self.phi_ext = phi_ext  # reduced flux

    @classmethod
    def from_Lj(self, Lj, n, alpha, phi_ext):
        return SNAIL_from_Lj(Lj, n, alpha, phi_ext)

    @classmethod
    def from_Ej(self, Ej, n, alpha, phi_ext):
        return SNAIL_from_Ej(Ej, n, alpha, phi_ext)

    def potential(self, phi):
        """
        Normalized potential energy of the josephson junction as a function of the
        reduced flux ::phi:: over the element. Assumes Ej = 1.
        """

        pot_branch1 = -self.alpha * np.cos(phi)
        pot_branch2 = -self.n * np.cos((self.phi_ext - phi) / self.n)

        return pot_branch1 + pot_branch2

    def solve_expansion(self, degree=40, scale=9 * np.pi, order=None):
        """
        Returns the flux of minimum potential and the coefficients of the truncated
        Taylor expansion around the minimum.
        scale is the width of the interval used in the Taylor fit.
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
        excluded_terms = [2, 0] if nonlinear else [0]

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
        """
        Derivative of the snail potential with respect to ::phi::. This function is
        used to analyze if the potential has multiple wells in self.has_multiple_wells
        () method.
        """

        pot_branch1_derivative = self.alpha * np.sin(phi)
        pot_branch2_derivative = -np.sin((self.phi_ext - phi) / self.n)

        return pot_branch1_derivative + pot_branch2_derivative

    @property
    def has_multiple_wells(self):
        """
        Analyzes if the potential has more than one local minimum per period.
        """

        return more_than_2_roots(self.potential_derivative, 0, self.n * 2 * np.pi)


class SNAIL_from_Lj(SNAIL):
    """ """

    def __init__(self, Lj, n, alpha, phi_ext):

        self.Lj = Lj  # H
        super().__init__(n, alpha, phi_ext)  # Passes other parameters to parent

    def Ej(self, a2):
        Ej = 1 / 2 / (2 * np.pi * hbar * self.Lj * a2) * (flux_quantum / 2 / np.pi) ** 2
        return Ej

    def get_Ej_Lj(self, a2):
        """
        a2 is the second-order coefficient of the Taylor expansion of the normalized
        potential.
        """
        return self.Ej(a2), self.Lj


class SNAIL_from_Ej(SNAIL):
    """ """

    def __init__(self, Ej, n, alpha, phi_ext):

        self.Ej = Ej  # Hz
        super().__init__(n, alpha, phi_ext)  # Passes other parameters to parent

    def Lj(self, a2):
        Lj = 1 / 2 / (2 * np.pi * hbar * self.Ej * a2) * (flux_quantum / 2 / np.pi) ** 2
        return Lj

    def get_Ej_Lj(self, a2):
        """
        a2 is the second-order coefficient of the Taylor expansion of the normalized
        potential.
        """
        return self.Ej, self.Lj(a2)
