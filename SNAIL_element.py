"""
The SNAIL element is defined as having two nodes connected by two parallel branches. 
One branch has ::n:: josephson junctions with characteristic energies :Ej: and the 
other branch has one junction with energy ::Ej:: scaled by ::alpha::. The branches form 
a loop threaded by a reduced external flux ::phi_ext::. 

Reference: 3-Wave Mixing Josephson Dipole Element (doi: 10.1063/1.4984142)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from helper_functions import *


class SNAIL:
    def __init__(self, Ej, n, alpha, phi_ext):

        self.Ej = Ej  # Hz
        self.n = n
        self.alpha = alpha
        self.phi_ext = phi_ext  # reduced flux

    def potential(self, phi, norm=True):
        """
        Potential energy of the josephson junction loop as a function of the reduced
        flux ::phi:: over the element. ::norm:: = True normalizes the cosines (Ej = 1).
        """

        pot_branch1 = -self.alpha * np.cos(phi)
        pot_branch2 = -self.n * np.cos((self.phi_ext - phi) / self.n)

        if norm:
            return pot_branch1 + pot_branch2

        return self.Ej * (pot_branch1 + pot_branch2)

    def solve_expansion(self, degree=7, norm=True):
        """
        Returns the flux of minimum potential and the coefficients of the truncated
        Taylor expansion around the minimum.
        """

        # Initial guess for potential minimum
        phi_array = np.arange(-np.pi * self.n, np.pi * self.n, 0.01)
        phi_min_0 = min(self.potential(phi) for phi in phi_array)

        # find flux of minimum potential
        phi_min = minimize(self.potential, phi_min_0).x[0]

        # Fit Taylor expansion around minimum
        scale = 1.0  # The width of the interval used to evaluate the polynomial.
        taylor_potential = approximate_taylor_polynomial(
            self.potential, phi_min, degree, scale, order=degree + 2
        )

        # return phi_min, taylor_potential
        Ej = 1 if norm else self.Ej

        return phi_min, Ej * taylor_potential

    def inductance(self, degree=7):
        """
        Obtain resonant frequency of the SNAIL shunted by a capacitance C. Also return SNAIL inductance Lj from 2nd order term.
        """

        taylor_expansion = self.solve_expansion(degree=degree, norm=False)[1]
        a2 = taylor_expansion[2]
        # Since self.Ej and consequently a2 are in units of Hz, the Lj expression needs
        # a 2*pi*hbar correction factor.
        Lj = 1 / 2 / (2 * np.pi * hbar * a2) * (flux_quantum / 2 / np.pi) ** 2

        return Lj

    def truncated_potential(
        self, degree=7, norm=True, shift=True, nonlinear=False, term_scaling=None
    ):
        """
        Calculates the potential Taylor expansion and reconstructs the potential
        function from the coefficients.
        degree is the order of the cosine expasion.
        norm = True returns potential in units of Ej.
        nonlinear = True returns only the nonlinear part of the potential.
        term_scaling accepts a list of floats in which the ith
        element scale the (i+3)th term of the taylor expansion of the potential.
        """

        phi_0, taylor_potential = self.solve_expansion(degree=degree, norm=norm)
        phi_0 = phi_0 if shift else 0
        limit = 2 if nonlinear else 0

        # padding term_scaling with unitary values
        if term_scaling:
            term_scaling = (
                [1, 1, 1] + term_scaling + [1] * (degree - len(term_scaling) - 2)
            )
        else:
            term_scaling = [1] * (degree + 1)

        def potential(x):
            return sum(
                taylor_potential[degree - k]
                * (x - phi_0) ** (degree - k)
                * term_scaling[degree - k]
                for k in range(len(taylor_potential) - limit)
            )

        return potential

    def potential_derivative(self, phi, norm=True):
        """
        Derivative of the snail potential with respect to ::phi::. ::norm:: = True
        normalizes the potential cosines (Ej = 1). This function is used to analyze if
        the potential has multiple wells in self.has_multiple_wells() method.
        """

        pot_branch1_derivative = self.alpha * np.sin(phi)
        pot_branch2_derivative = -np.sin((self.phi_ext - phi) / self.n)

        if norm:
            return pot_branch1_derivative + pot_branch2_derivative

        return self.Ej * (pot_branch1_derivative + pot_branch2_derivative)

    @property
    def has_multiple_wells(self):
        """
        Analyzes if the potential has more than one local minimum per period.
        """

        return more_than_2_roots(self.potential_derivative, 0, self.n * 2 * np.pi)

    def plot_potential(self, show=True):
        """
        Plots the potential in the interval (-n*pi, n*pi).
        """

        phi_array = np.arange(-np.pi * self.n, np.pi * self.n, 0.01)
        plt.plot(phi_array, [self.Ej * self.potential(phi) for phi in phi_array])
        plt.grid()

        if show:
            plt.show()

        return

    def plot_parameters_phi_ext(self, C=None):
        """
        Plots values of a2, a3, a4, a5 as a function of external potential.
        """

        phi_ext_array = np.arange(-1 * np.pi, 1 * np.pi, 0.01)

        a2_array = []
        a3_array = []
        a4_array = []
        a5_array = []
        for phi_ext in phi_ext_array:
            snail.phi_ext = phi_ext
            a2, a3, a4, a5 = snail.solve_expansion()[2:6]
            a2_array.append(a2)
            a3_array.append(a3)
            a4_array.append(a4)
            a5_array.append(a5)

        f, axes = plt.subplots(2, 2, sharex=True)

        # point of a4 minimum
        min_index = np.argmin(abs(np.array(a4_array)))

        if C:
            Lj = 1 / 2 / np.array(a2) * (flux_quantum / 2 / np.pi) ** 2
            freq_array = 1 / np.sqrt(Lj * C)
            freq_array = []
            axes[0, 0].plot(phi_ext_array, [x / 1e9 for x in freq_array])
            axes[0, 0].scatter(
                phi_ext_array[min_index],
                [x / 1e9 for x in freq_array][min_index],
                c="r",
            )
            axes[0, 0].set_ylabel("Mode frequency for C = %.2ffF" % (C * 1e15))
        else:
            axes[0, 0].plot(phi_ext_array, [x / Ej for x in a2_array])
            axes[0, 0].scatter(
                phi_ext_array[min_index], [x / Ej for x in a2_array][min_index], c="r"
            )
            axes[0, 0].set_ylabel("a2/Ej")

        axes[0, 1].plot(phi_ext_array, [x / Ej for x in a3_array])
        axes[0, 1].scatter(
            phi_ext_array[min_index], [x / Ej for x in a3_array][min_index], c="r"
        )
        axes[0, 1].set_ylabel("a3/Ej")
        axes[1, 0].plot(phi_ext_array, [x / Ej for x in a4_array])
        axes[1, 0].scatter(
            phi_ext_array[min_index], [x / Ej for x in a4_array][min_index], c="r"
        )
        axes[1, 0].set_ylabel("a4/Ej")
        axes[1, 1].plot(phi_ext_array, [x / Ej for x in a5_array])
        axes[1, 1].scatter(
            phi_ext_array[min_index], [x / Ej for x in a5_array][min_index], c="r"
        )
        axes[1, 1].set_ylabel("a5/Ej")

        plt.show()
        return

    def plot_parameters_alpha(self, C=None):
        """
        Plots values of a2, a3, a4, a5 as a function of external potential.
        """

        alpha_array = np.arange(0.1, 1, 0.01)

        a2_array = []
        a3_array = []
        a4_array = []
        a5_array = []
        for alpha in alpha_array:
            snail.alpha = alpha
            a2, a3, a4, a5 = snail.solve_expansion()[2:6]
            a2_array.append(a2)
            a3_array.append(a3)
            a4_array.append(a4)
            a5_array.append(a5)

        f, axes = plt.subplots(2, 2, sharex=True)

        # point of a4 minimum
        min_index = np.argmin(abs(np.array(a4_array)))

        if C:
            Lj = 1 / 2 / np.array(a2) * (flux_quantum / 2 / np.pi) ** 2
            freq_array = 1 / np.sqrt(Lj * C)
            freq_array = []
            axes[0, 0].plot(alpha_array, [x / 1e9 for x in freq_array])
            axes[0, 0].scatter(
                alpha_array[min_index], [x / 1e9 for x in freq_array][min_index], c="r"
            )
            axes[0, 0].set_ylabel("Mode frequency for C = %.2ffF" % (C * 1e15))
        else:
            axes[0, 0].plot(alpha_array, [x / Ej for x in a2_array])
            axes[0, 0].scatter(
                alpha_array[min_index], [x / Ej for x in a2_array][min_index], c="r"
            )
            axes[0, 0].set_ylabel("a2/Ej")

        axes[0, 1].plot(alpha_array, [x / Ej for x in a3_array])
        axes[0, 1].scatter(
            alpha_array[min_index], [x / Ej for x in a3_array][min_index], c="r"
        )
        axes[0, 1].set_ylabel("a3/Ej")
        axes[1, 0].plot(alpha_array, [x / Ej for x in a4_array])
        axes[1, 0].scatter(
            alpha_array[min_index], [x / Ej for x in a4_array][min_index], c="r"
        )
        axes[1, 0].set_ylabel("a4/Ej")
        axes[1, 1].plot(alpha_array, [x / Ej for x in a5_array])
        axes[1, 1].scatter(
            alpha_array[min_index], [x / Ej for x in a5_array][min_index], c="r"
        )
        axes[1, 1].set_ylabel("a5/Ej")

        plt.show()
        return

    def plot_a4(self, alpha_list, phi_ext_list):
        """ """

        x = alpha_list
        y = phi_ext_list
        z = []
        for phi_ext in y:
            for alpha in x:
                self.alpha = alpha
                self.phi_ext = phi_ext
                z.append(self.solve_expansion()[4])

        z = np.array(z).reshape(len(y), len(x))
        plot_imshow(x, y, z)
        return


if __name__ == "__main__":
    Ej = 2 * np.pi * hbar * 30e9
    n = 3
    alpha = 0.29
    phi_ext = 0.41 * 2 * np.pi
    snail = SNAIL(Ej, n, alpha, phi_ext)
    print([x / Ej for x in snail.solve_expansion()])

    a2 = snail.solve_expansion()[2]
    print(a2 / Ej)
    Lj = 1 / 2 / a2 * (flux_quantum / 2 / np.pi) ** 2
    C = 60e-15
    freq = 1 / np.sqrt(C * Lj) / 2 / np.pi
    print(Lj / 1e-9, "nH  ", freq / 1e9, "GHz")
