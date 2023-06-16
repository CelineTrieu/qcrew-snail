"""
Class defining a circuit containing one ancilla coupled to one or more resonators.

It is designed to work seamlessly with pyEPR, having as inputs the list of reduced flux
zero-point fluctuations and a list of mode frequencies.
"""

import qutip as qt
import numpy as np
from scipy.interpolate import interp1d
from functools import reduce

from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *


def tensor_out(op, loc, fock_trunc, n):
    """Expands a single-mode operator into a multi-mode Hilbert space.

    The operator op tensored with identities at locations other than loc. fock_trunc is
    the dimension of each individual operator and n is the number of modes.

    Args:
        op (Qobj): Operator acting on a given mode.
        loc (int): Index of the mode.
        fock_trunc (int): Dimension of each operator.
        n (int): number of modes considered.

    Returns:
        Qobj: Operator op tensored up to a larger Hilbert space.
    """

    op_list = [qt.qeye(fock_trunc) for i in range(n)]
    op_list[loc] = op
    return reduce(qt.tensor, op_list)


def dot(ais, bis):
    """Dot product function between two input lists."""
    return sum(ai * bi for ai, bi in zip(ais, bis))


def clean_spectrum(evals, evecs, m=3, threshold=1e-3):
    """
    Remove from the list the eigenvectors that have too much population in the last
    m Fock states.

    TODO this is repeated from helper_functions. Find a way to import it from there.

    Args:
        evals (list): Total list of eigenvalues.

        evecs (list): Total list of Qobj eigenvectors.

        m (int, optional): The last m fock states are considered to be numerically
        unreliable. Defaults to 3.

        threshold ([type], optional): Maximum eigenstate population acceptable in the
        last m fock states. Defaults to 1e-3.

    Returns:
        tuple: (evals, evecs) with unreliable states removed.
    """

    clean_evals = []
    clean_evecs = []

    for i in range(len(evecs)):
        evecs_data = [float(np.absolute(x) ** 2) for x in evecs[i]]
        bad_state_pop = sum(evecs_data[-m:])
        if bad_state_pop < threshold:
            clean_evals.append(evals[i])
            clean_evecs.append(evecs[i])

    return clean_evals, clean_evecs


class Circuit:
    def __init__(
        self,
        ancilla,
        freqs,
        phi_rzpf,
        reference_operators = None
    ):
        """Define a circuit composed of multiple modes, one of which is an ancilla.

        The ancilla mode is identified as the mode with largest reduced flux zero-point
        fluctuation. The remaining are assumed to be resonator modes.

        Args:
            ancilla (Ancilla): ancilla object.

            freqs (list): List of linear frequencies of each mode in Hz.

            phi_rzpf (list): List of reduced flux zero-point fluctuations over the ancilla junction for each mode.

            reference_operators (tuple): This is a feature that only works with 2 modes (1 cavity + 1 ancilla).
            Tuple containing (cavity annihilation operator, ancilla flux operator, ancilla charge operator) as
            reference for building the operators of this circuit
        """

        self.fock_trunc = ancilla.fock_trunc
        self.n_modes = len(freqs)  # Assign one mode for each circ. element
        self.ancilla = ancilla
        self.phi_rzpf = phi_rzpf
        self.mode_freqs = freqs
        self.ancilla_mode = np.argmax(phi_rzpf)
        self.bogoliubov_factors = self.calc_bogoliubov_factors()
        self.coupling_factor, self.bare_mode_freqs = self.calc_bare_mode_freqs()
        self.reference_operators = reference_operators

        self._ancilla_spectrum = None
        self._en_circuit_spectrum, self._ex_circuit_spectrum = None, None
        self._COB_matrix = None

    def calc_bogoliubov_factors(self):
        # Only works for two modes for now
        ancilla_mode = self.ancilla_mode
        cavity_mode = (self.ancilla_mode - 1)%2

        # Find angle of Bogoliubov transformation
        phi_rzpf_ancilla = self.phi_rzpf[ancilla_mode, 0]
        phi_rzpf_cavity = self.phi_rzpf[cavity_mode, 0]
        phi_rzpf_total = np.sqrt(phi_rzpf_ancilla**2 + phi_rzpf_cavity**2)
        cos_lambda = phi_rzpf_ancilla/phi_rzpf_total
        sin_lambda = -phi_rzpf_cavity/phi_rzpf_total

        return cos_lambda, sin_lambda
    
    def calc_bare_mode_freqs(self):
        # Calculate pre-hybridization parameters
        # Only works for two modes for now
        ancilla_mode = self.ancilla_mode
        cavity_mode = (self.ancilla_mode-1)%2
        cos_lambda, sin_lambda = self.bogoliubov_factors

        bare_mode_freqs = [0,0]
        bare_mode_freqs[ancilla_mode] = cos_lambda**2*self.mode_freqs[ancilla_mode] + \
                                        sin_lambda**2*self.mode_freqs[cavity_mode]
        bare_mode_freqs[cavity_mode] = cos_lambda**2*self.mode_freqs[cavity_mode] + \
                                       sin_lambda**2*self.mode_freqs[ancilla_mode]
        coupling_factor = (self.mode_freqs[ancilla_mode] - self.mode_freqs[cavity_mode])*sin_lambda*cos_lambda

        return coupling_factor, bare_mode_freqs
    
    def calc_bare_mode_operators(self, mode_operators):
        # Only works for two modes for now
        ancilla_mode = self.ancilla_mode
        cavity_mode = (self.ancilla_mode-1)%2

        cos_lambda, sin_lambda = self.bogoliubov_factors

        bare_mode_operators = [0,0]
        bare_mode_operators[ancilla_mode] = cos_lambda*mode_operators[ancilla_mode] - \
                                            sin_lambda*mode_operators[cavity_mode]
        bare_mode_operators[cavity_mode] = cos_lambda*mode_operators[cavity_mode] + \
                                           sin_lambda*mode_operators[ancilla_mode]
        
        return bare_mode_operators
    
    def calc_dressed_mode_operators(self, bare_mode_operators):
        # Only works for two modes for now
        ancilla_mode = self.ancilla_mode
        cavity_mode = (self.ancilla_mode-1)%2
    
        cos_lambda, sin_lambda = self.bogoliubov_factors

        # get dressed mode operators 
        mode_operators = [0,0]
        mode_operators[ancilla_mode] = cos_lambda*bare_mode_operators[ancilla_mode] + \
                                       sin_lambda*bare_mode_operators[cavity_mode]
        mode_operators[cavity_mode] = cos_lambda*bare_mode_operators[cavity_mode] - \
                                      sin_lambda*bare_mode_operators[ancilla_mode]
        
        return mode_operators

    def calc_mode_operators(self, cavity_operator, flux_operator, charge_operator):
        # Only works for two modes for now
        ancilla_mode = self.ancilla_mode
        cavity_mode = (self.ancilla_mode-1)%2
        
        # get bare mode operators from circuit operators
        f0 = self.bare_mode_freqs[ancilla_mode]
        bare_mode_operators = [0,0]
        bare_mode_operators[ancilla_mode] = np.sqrt(1/(2*f0*self.ancilla.Lj))*flux_operator + \
                                            1j*np.sqrt(f0*self.ancilla.Lj/(2))*charge_operator
        bare_mode_operators[cavity_mode] = cavity_operator

        mode_operators = self.calc_dressed_mode_operators(bare_mode_operators)
        return mode_operators
    
    def calc_ancilla_flux_charge_operators(self):
        # Build bare mode operators and find ancilla circuit operators
        # Choose the hybridized Fock states as basis
        a = qt.destroy(self.fock_trunc)
        mode_operators = [
            tensor_out(a, i, self.fock_trunc, self.n_modes)
            for i in range(self.n_modes)
        ]
        bare_mode_operators = self.calc_bare_mode_operators(mode_operators)
        a0 = bare_mode_operators[self.ancilla_mode]
        b0 = bare_mode_operators[(self.ancilla_mode-1)%2]
        f0 = self.bare_mode_freqs[self.ancilla_mode]
        flux_operator = np.sqrt(f0*self.ancilla.Lj/2) * (a0.dag() + a0)
        charge_operator = 1j*np.sqrt(1/f0/self.ancilla.Lj/2) * (a0.dag() - a0)

        return b0, flux_operator, charge_operator
        
    
    def get_H(self, basis = "fock", basis_cob = None):
        """Returns the hamiltonian of the system. Allows the user to chose the basis.

        Args:
            basis (): if "fock", return the hamiltonian in the basis of excitations of the
            linear part of the circuit. If "eigen", return the hamiltonian diagonalized by
            transforming with its own COB matrix. If it is a matrix instead, try to transform
            using qutip method.
        """

        if self.reference_operators:
            mode_operators = self.calc_mode_operators(*self.reference_operators)
            mode_fields = [x + x.dag() for x in mode_operators]
            mode_ns = [x.dag()*x for x in mode_operators]

        else:
            # get auxiliary operators    
            a = qt.destroy(self.fock_trunc)
            ad = a.dag()
            n = qt.num(self.fock_trunc)
            # instantiate qutip operators for each hybridized mode
            mode_fields = [
                tensor_out(a + ad, i, self.fock_trunc, self.n_modes)
                for i in range(self.n_modes)
            ]
            mode_ns = [
                tensor_out(n, i, self.fock_trunc, self.n_modes) for i in range(self.n_modes)
            ]

        # build hamiltonian.
        cos_interiors = sum(
            [self.phi_rzpf[i, 0] * mode_fields[i] for i in range(self.n_modes)]
        )
        self._Hl = dot(self.mode_freqs, mode_ns)
        self._Hnl = self.ancilla.Ej * self.ancilla.nl_potential(cos_interiors)
        self._H = self._Hl + self._Hnl

        if basis_cob is not None:
            try:
                return self._H.transform(basis_cob)
            except:
                print("Could not identify basis. Returning hamiltonian in Fock basis.")
                return self._H

        if basis == "fock":
            return self._H
        if basis == "eigen":
            return self._H.transform(self.COB_matrix)
        

    @property
    def ancilla_spectrum(self):
        """Eigenvalues and eigenvectors of the hamiltonian of the isolated ancilla."""
        if self._ancilla_spectrum:
            return self._ancilla_spectrum

        else:
            self._ancilla_spectrum = self._calc_ancilla_spectrum()
            return self._ancilla_spectrum

    def get_circuit_spectrum(self, order = "energy"):
        """Eigenvalues and eigenvectors of the full hamiltonian of the circuit.
        
        Args:
            order (str): if "energy", return in lowest-to-highest order, as given by qutip.
            If "excitation", return spectrum ordered by the number of excitations of each mode,
            with the first mode being more significative.
        """

        if not (self._en_circuit_spectrum and self._ex_circuit_spectrum):
            self._en_circuit_spectrum = self.get_H().eigenstates()
            self._ex_circuit_spectrum = self._reorder_circuit_excitation_spectrum()
            
        if order == "energy":
            return self._en_circuit_spectrum

        elif order == "excitation":
            return self._ex_circuit_spectrum    

    @property
    def COB_matrix(self):
        """Matrix that transforms operators from Fock state basis to the
        basis of eigenvectors of the system.
        """
        if self._COB_matrix:
            return self._COB_matrix

        else:
            self._COB_matrix = self._calc_change_of_basis_matrix()
            return self._COB_matrix

    def _reorder_circuit_excitation_spectrum(self):
        """The purpose of this method is to reorder the eigenvectors according to
        the number of excitations in each mode. Assumes the eigenvectors of the 
        whole system are similar to the eigenvectors of the individual modes 
        (valid in the dispersive regime).
        """

        # Gets eigenvectors of each individual mode
        mode_eigenvec = [self.get_H().ptrace(i).eigenstates()[1] 
                         for i in range(len(self.mode_freqs))]

        # get evecs in the
        evals, evecs = self._en_circuit_spectrum

        evecs_labeled = []
        for i in range(len(evecs)):
            label = ()
            # Find the number of excitations in each eigenvector
            for mode in range(len(self.mode_freqs)):
                indx = np.argmax([qt.metrics.fidelity(evecs[i].ptrace(mode), qt.ket2dm(x)) 
                                  for x in mode_eigenvec[mode]])
                label += (indx,)
            evecs_labeled.append((evecs[i], evals[i], label))

        evecs_evals_reordered = sorted(evecs_labeled, key=lambda x: x[2])
        evecs_reordered = [x[0] for x in evecs_evals_reordered]
        evals_reordered = [x[1] for x in evecs_evals_reordered]

        return evals_reordered, evecs_reordered
        
    def _calc_ancilla_spectrum(self):
        """Calculate the spectrum of the isolated ancilla.

        This function is used to build a zeroth-order approximation of the circuit
        spectrum. The clean_spectrum function is used to remove states that are not
        numerically reliable.

        Returns:
            tuple: (evals, evecs), where evals is an np.array of hamiltonian
            eigenvalues in Hz and evecs is the respective np.array of eigenvectors.
        """

        evals_0, evecs_0 = self.ancilla.calculate_spectrum()
        return clean_spectrum(evals_0, evecs_0)

    def get_eigenstate(self, exc):
        """Return the eigenstate of the full coupled circuit corresponding to the given
        mode excitations.

        The function first finds the product state built from the eigenvectors of each
        individual mode hamiltonian. This first guess can be understood as an RWA
        eigenvector. Then returns the corresponding eigenstate of the full hamiltonian
        that has larger overlap.

        Args:
            exc (dict): dictionary in which the keys are mode indexes (same indexing as 
            self.mode_freqs) and the values are the number of excitations.

        Returns:
            tuple: (eval, evec), in which evec is the corresponding eigenstate of the 
            circuit hamiltonian and eval is its eigenvalue (Hz).
        """

        # Get eigestates of linear modes
        eigenvectors_list = [
            qt.basis(self.fock_trunc, exc.get(i, 0)) for i in range(self.n_modes)
        ]

        # get ancilla spectrum
        ancilla_eigenvecs = self.ancilla_spectrum[1]
        eigenvectors_list[self.ancilla_mode] = qt.Qobj(
            ancilla_eigenvecs[exc.get(self.ancilla_mode, 0)]
        )
        evec_guess = reduce(qt.tensor, eigenvectors_list)
        print(evec_guess[:10])
        # Return the eigenvector of the circuit spectrum that is closest to the guess.
        def distance(s2):
            return (evec_guess.dag() * s2[1]).norm()

        evals, evecs = self.get_circuit_spectrum()
        eval, evec = max(zip(evals, evecs), key=distance)
        print(evec[:10])
        return eval, evec

    def _calc_change_of_basis_matrix(self):
        """
        Calculates the matrix that transforms operators from Fock state basis to the
        basis of eigenvectors of the system. Keeps the structure of the hamiltonian
        by ordering basis per excitation.
        """

        _, evecs = self.get_circuit_spectrum(order = "excitation")
        COB_matrix = np.concatenate([np.array(x) for x in evecs], axis = 1).T
        return COB_matrix
    

class TunableCircuit:
    def __init__(
        self,
        junction_type,
        ljs_list,
        freqs_list,
        phi_rzpf_list,
        p_list,
        p_ref,
        fock_trunc,
    ):
        """Defines a circuit builder dependent on a tunable parameter p. For each value of phi, should
        receive a set of (Lj, freqs, phi_rzpf) which is used to build the circuit.
        """

        self.fock_trunc = fock_trunc
        self.junction_type = junction_type
        self.lj = self.interpolate(p_list, ljs_list)
        self.freqs = self.interpolate(p_list, freqs_list)
        self.phi_rzpf = self.interpolate(p_list, phi_rzpf_list)
        self.p_list = p_list
        self.p_ref = p_ref

        self.n_modes = len(freqs)  # Assign one mode for each circ. element
        self.ancilla = ancilla
        self.phi_rzpf = phi_rzpf
        self.mode_freqs = freqs
        self.ancilla_mode = np.argmax(phi_rzpf)


    def interpolate(self, p_list, y_list):
        return
        # return interp1d(p_list, ansys_fs[:,0].reshape(-1, 3)[:,2])

    def build_reference_circuit(self, p):

        # Get circuit and SQUID parameters
        Lj = self.ljs(p)
        Ej = 1 / (2 * np.pi * hbar * Lj) * (flux_quantum / 2 / np.pi) ** 2
        freqs = self.freqs(p)
        phi_rzpf = self.phi_rzpf(p)
        # Assemble circuit
        junction = self.junction_type(Ej)
        ancilla = Ancilla(junction, freqs[np.argmax(phi_rzpf)], fock_trunc=self.fock_trunc)
        self.ref_circuit = Circuit(ancilla, freqs, phi_rzpf)

        return self.ref_circuit
    
    def build_circuit(self, p):

        # Get circuit and SQUID parameters
        Lj = self.ljs(p)
        Ej = 1 / (2 * np.pi * hbar * Lj) * (flux_quantum / 2 / np.pi) ** 2
        freqs = self.freqs(p)
        phi_rzpf = self.phi_rzpf(p)
        # Assemble circuit
        junction = self.junction_type(Ej)
        ancilla = Ancilla(junction, freqs[np.argmax(phi_rzpf)], fock_trunc=self.fock_trunc)
        circuit = Circuit(ancilla, freqs, phi_rzpf)

        return circuit