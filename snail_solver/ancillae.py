"""The ancilla object is defined by a generalized nonlinear Josephson element shunted 
by a capacitance. It is designed to receive the linear mode frequency directly from 
HFSS to grant seamless connection to simulations.
"""

import qutip as qt
import numpy as np
from abc import ABC, abstractmethod

from scipy.interpolate import approximate_taylor_polynomial
from scipy.optimize import minimize
from snail_solver.helper_functions import *


class Ancilla(ABC):
    def __init__(self, fock_trunc):
        """Define an ancilla from a nonlinear circuit element and a linear mode
        frequency.

        Args:

            element (JJ, SNAIL): JJ or SNAIL object. Defines nonlinear circuit
            element that composes the ancilla. This element is responsible for defining
            the inductance of the ancilla mode.

            freq (float): Linear mode frequency as obtained from HFSS (Hz).

            taylor_parameters (dict): Dictionary defining numerical parameters for
            fitting the element potential to a Taylor series. See solve_expansion
            method.

            fock_trunc (int, optional): Dimension of qutip matrices. Defaults to 20.
        """

        self.fock_trunc = fock_trunc
        self._freq = None

    @property
    @abstractmethod
    def Ej(self):
        if self._Ej == None:
            print("Value of Ej is not defined")
        return self._Ej
    
    @property
    @abstractmethod
    def Lj(self):
        if self._Lj == None:
            print("Value of Lj is not defined")
        return self._Lj

    @property
    def nl_potential(self):
        return self.calc_truncated_potential(shift=True, nonlinear=True)[0]
    
    @property
    def freq(self):
        if self._freq == None:
            print("Value of freq is not defined")
        return self._freq
    
    @freq.setter
    def freq(self, value):
        print("- Setting ancilla frequency")
        self._freq = value
    
    @abstractmethod
    def potential(self):
        pass

    @abstractmethod
    def calc_truncated_potential(self):
        pass 

    @abstractmethod
    def solve_expansion(self):
        pass 


    def calculate_ancilla_hamiltonian(self):
        """Retrieve the hamiltonian of the ancilla assuming it is isolated from other
        modes.

        Since the ancilla is assumed to be isolated, the reduced flux zpf can be
        calculated analitically.

        Returns:
            tuple: (Hl, Hnl), where Hl and Hnl are Qobj operators of the linear and
            nonlinear parts of the ancilla hamiltonian in Hz.
        """

        # calculate the flux zpf for the isolated device
        cap = 1 / self.Lj / (2 * np.pi * self._freq) ** 2
        phi_zpf = np.sqrt(hbar / (2 * cap * 2 * np.pi * self._freq))
        phi_rzpf = 2 * np.pi * phi_zpf / flux_quantum  # reduced flux zpf

        # scale the hamiltonian by Ej
        destroy = qt.destroy(self.fock_trunc)
        Hnl = self.Ej * self.nl_potential(phi_rzpf * (destroy + destroy.dag()))
        Hl = qt.num(self.fock_trunc) * self._freq
        return Hl, Hnl
    

class SNAIL(Ancilla):
    def __init__(self, n, alpha, phi_ext, fock_trunc=20, taylor_parameters=None):
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
        super().__init__(fock_trunc=fock_trunc)

        if not taylor_parameters:
            self.taylor_parameters = {"degree": 40, "scale": 9 * np.pi, "order": None}

        self._Ej = None
        self._Lj = None
        
        self.phi_min, self.taylor_coef = self.solve_expansion()


    @property
    def Ej(self):
        if self._Ej == None:
            print("Value of Ej is not defined")
        return self._Ej
    
    @property
    def Lj(self):
        if self._Lj == None:
            print("Value of Lj is not defined")
        return self._Lj
    
    @Ej.setter
    def Ej(self, value):
        print("- Setting Ej and Lj values")
        self._Ej = value
        a2 = self.taylor_coef[2]
        self._Lj = 1 / 2 / (2 * np.pi * hbar * self.Ej * a2) * (flux_quantum / 2 / np.pi) ** 2
    
    @Lj.setter
    def Lj(self, value):
        print("- Setting Ej and Lj values")
        self._Lj = value
        a2 = self.taylor_coef[2]
        self._Ej = Ej = 1 / 2 / (2 * np.pi * hbar * self.Lj * a2) * (flux_quantum / 2 / np.pi) ** 2
        
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

    def solve_expansion(self):
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

        degree = self.taylor_parameters["degree"]
        scale = self.taylor_parameters["scale"]
        order = self.taylor_parameters["order"]

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

    def calc_truncated_potential(
        self,
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

        phi_0, taylor_coef = self.phi_min, self.taylor_coef
        phi_0 = phi_0 if not shift else 0
        excluded_terms = [2, 0] if nonlinear else [0]

        degree = self.taylor_parameters["degree"]

        def potential(x):
            return sum(
                taylor_coef[degree - k] * (x - phi_0) ** (degree - k)
                for k in range(len(taylor_coef) + 1)
                if degree - k not in excluded_terms
            )


        return potential, taylor_coef

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
    
class JJ(Ancilla):
    def __init__(self, fock_trunc=20, taylor_parameters=None):
        """Define a Josephson junction to be used as a nonlinear circuit element.

        Args:
            Ej ([float]): Josephson energy of the junction in Hz.
        """

        super().__init__(fock_trunc=fock_trunc)

        if not taylor_parameters:
            self.taylor_parameters = {"degree": 40, "scale": 9 * np.pi, "order": None}
        
        self._Ej = None
        self._Lj = None
        
        self.phi_min, self.taylor_coef = self.solve_expansion()

    @property
    def Ej(self):
        return self._Ej
    
    @property
    def Lj(self):
        return self._Lj
    
    @Ej.setter
    def Ej(self, value):
        print("- Setting Ej and Lj values")
        self._Ej = value
        self._Lj = 1 / (2 * np.pi * hbar * value) * (flux_quantum / 2 / np.pi) ** 2
    
    @Lj.setter
    def Lj(self, value):
        print("- Setting Ej and Lj values")
        self._Lj = value
        self._Ej = 1 / (2 * np.pi * hbar * value) * (flux_quantum / 2 / np.pi) ** 2 
    

    def potential(self, phi):
        """Calculate normalized potential energy (Ej = 1) of the junction as a function of the reduced flux phi over the element.

        Args:
            phi ([float]): Reduced flux over the junction in radians.

        Returns:
            [float]: Value of potential energy of the junction in units of Ej.
        """

        return -np.cos(phi)

    def solve_expansion(self):
        """Calculate the flux of minimum potential and fit a truncated
        Taylor expansion around the minimum.

        Args:

            degree (int, optional): [description]. Defaults to 40.

            scale ([type], optional): Width of the fit interval of the Taylor expansion around the minimum. Defaults to 9*np.pi.

            order ([type], optional): [description]. Defaults to None.

        Returns:
            [tuple]: (phi_min, taylor_coef) where phi_min is the [float] flux of minimum potential and taylor_coef
        """

        degree = self.taylor_parameters["degree"]
        scale = self.taylor_parameters["scale"]
        order = self.taylor_parameters["order"]

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

    def calc_truncated_potential(
        self,
        shift=False,
        nonlinear=False,
    ):
        """Calculates the Taylor expansion of the potential and reconstructs the
        function from the coefficients.

        Args:

            taylor_parameters (dict): Dictionary defining numerical parameters for
            fitting the element potential to a Taylor series. In the case of the JJ, only the degree of the Taylor polynomial needs to be defined.

            shift (bool, optional): If True, the returned potential is
            redefined in terms of the distance from its minimum. Defaults to False.

            nonlinear (bool, optional): If True, exclude the second-order term from the
            polynomial series. Defaults to False.

        Returns:
            tuple: (potential, taylor_coef, Ej, Lj)
        """

        degree = self.taylor_parameters["degree"]

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

        return potential, taylor_coef
