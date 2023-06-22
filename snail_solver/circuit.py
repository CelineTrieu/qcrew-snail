"""
Class defining a circuit containing one ancilla coupled to one or more resonators.

It is designed to work seamlessly with pyEPR, having as inputs the list of reduced flux
zero-point fluctuations and a list of mode frequencies.
"""

import qutip as qt
import numpy as np
from functools import reduce

from snail_solver.ancillae import Ancilla
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
        circuit_parameters,
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
    
        self.circuit_parameters = circuit_parameters
        self.circuit_parameters_value = None
        self.ancilla = ancilla
        self.fock_trunc = ancilla.fock_trunc
        self.reference_operators = reference_operators

        self.phi_rzpf = None
        self.freqs = None
        self.ancilla_index = None

        self._en_circuit_spectrum, self._ex_circuit_spectrum = None, None

        self.circuit_H_memory = {}
        self.ancilla_spectrum_memory = {}
        self.circuit_spectrum_memory = {} # circuit spectrum ordered by eigenenergy


    
    @property
    def hamiltonian(self):
        """Hamiltonian of the circuit."""
        return self._calc_hamiltonian()

    @property
    def ancilla_spectrum(self):
        """Eigenvalues and eigenvectors of the hamiltonian of the isolated ancilla."""
        return self._calc_ancilla_spectrum()
    
    @property
    def circuit_spectrum(self):
        """Eigenvalues and eigenvectors of the hamiltonian of the circuit."""
        return self._calc_circuit_spectrum()
    
    def set_circuit_parameters(self, *args):
        print("\nSetting circuit parameters")
        Lj, freqs, phi_rzpf = self.circuit_parameters.get(*args)
        
        # update properties
        self.circuit_parameters_value = "constant" if args == () else args
        self.phi_rzpf = phi_rzpf
        self.freqs = freqs
        self.ancilla_index = np.argmax(phi_rzpf)
        self.ancilla.freq = self.freqs[self.ancilla_index]
        self.ancilla.Lj = Lj
        return

    def calc_bogoliubov_factors(self):
        # Only works for two modes for now
        ancilla_mode = self.ancilla_index
        cavity_mode = (self.ancilla_index - 1) % 2

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
        ancilla_mode = self.ancilla_index
        cavity_mode = (self.ancilla_index-1)%2
        cos_lambda, sin_lambda = self.calc_bogoliubov_factors()

        bare_mode_freqs = [0, 0]
        bare_mode_freqs[ancilla_mode] = cos_lambda**2*self.freqs[ancilla_mode] + \
                                        sin_lambda**2*self.freqs[cavity_mode]
        bare_mode_freqs[cavity_mode] = cos_lambda**2*self.freqs[cavity_mode] + \
                                       sin_lambda**2*self.freqs[ancilla_mode]
        coupling_factor = (self.freqs[ancilla_mode] - self.freqs[cavity_mode])*sin_lambda*cos_lambda

        return coupling_factor, bare_mode_freqs
    
    def calc_bare_mode_operators(self, mode_operators):
        # Only works for two modes for now
        ancilla_mode = self.ancilla_index
        cavity_mode = (self.ancilla_index-1) % 2

        cos_lambda, sin_lambda = self.calc_bogoliubov_factors()

        bare_mode_operators = [0,0]
        bare_mode_operators[ancilla_mode] = cos_lambda*mode_operators[ancilla_mode] - \
                                            sin_lambda*mode_operators[cavity_mode]
        bare_mode_operators[cavity_mode] = cos_lambda*mode_operators[cavity_mode] + \
                                           sin_lambda*mode_operators[ancilla_mode]
        
        return bare_mode_operators
    
    def calc_dressed_mode_operators(self, bare_mode_operators):
        # Only works for two modes for now
        ancilla_mode = self.ancilla_index
        cavity_mode = (self.ancilla_index-1)%2
    
        cos_lambda, sin_lambda = self.calc_bogoliubov_factors()

        # get dressed mode operators 
        mode_operators = [0,0]
        mode_operators[ancilla_mode] = cos_lambda*bare_mode_operators[ancilla_mode] + \
                                       sin_lambda*bare_mode_operators[cavity_mode]
        mode_operators[cavity_mode] = cos_lambda*bare_mode_operators[cavity_mode] - \
                                      sin_lambda*bare_mode_operators[ancilla_mode]
        
        return mode_operators

    def calc_mode_operators(self, cavity_operator, flux_operator, charge_operator):
        # Only works for two modes for now
        ancilla_mode = self.ancilla_index
        cavity_mode = (self.ancilla_index-1)%2
        
        # get bare mode operators from circuit operators
        f0 = self.calc_bare_mode_freqs()[ancilla_mode]
        bare_mode_operators = [0,0]
        bare_mode_operators[ancilla_mode] = np.sqrt(1/(2*f0*self.ancilla.Lj))*flux_operator + \
                                            1j*np.sqrt(f0*self.ancilla.Lj/(2))*charge_operator
        bare_mode_operators[cavity_mode] = cavity_operator

        mode_operators = self.calc_dressed_mode_operators(bare_mode_operators)
        return mode_operators
    
    def calc_ancilla_flux_charge_operators(self):
        # Build bare mode operators and find ancilla circuit operators
        # Choose the hybridized Fock states as basis

        n_modes = len(self.freqs)
        a = qt.destroy(self.fock_trunc)
        mode_operators = [
            tensor_out(a, i, self.fock_trunc, n_modes)
            for i in range(n_modes)
        ]
        bare_mode_operators = self.calc_bare_mode_operators(mode_operators)
        a0 = bare_mode_operators[self.ancilla_index]
        b0 = bare_mode_operators[(self.ancilla_index-1)%2]
        f0 = self.calc_bare_mode_freqs()[self.ancilla_index]
        flux_operator = np.sqrt(f0*self.ancilla.Lj/2) * (a0.dag() + a0)
        charge_operator = 1j*np.sqrt(1/f0/self.ancilla.Lj/2) * (a0.dag() - a0)

        return b0, flux_operator, charge_operator
        
    
    def _calc_hamiltonian(self, basis = "fock", basis_cob = None):
        """Returns the hamiltonian of the system. Allows the user to chose the basis.

        Args:
            basis (): if "fock", return the hamiltonian in the basis of excitations of the
            linear part of the circuit. If "eigen", return the hamiltonian diagonalized by
            transforming with its own COB matrix. If it is a matrix instead, try to transform
            using qutip method.
        """

        n_modes = len(self.freqs)

        H = self.circuit_H_memory.get(self.circuit_parameters_value)
        if not H:
            print("Calculating hamiltonian...", end = ' ')
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
                    tensor_out(a + ad, i, self.fock_trunc, n_modes)
                    for i in range(n_modes)
                ]
                mode_ns = [
                    tensor_out(n, i, self.fock_trunc, n_modes) for i in range(n_modes)
                ]

            # build hamiltonian
            cos_interiors = sum(
                [self.phi_rzpf[i, 0] * mode_fields[i] for i in range(n_modes)]
            )
            Hl = dot(self.freqs, mode_ns)
            Hnl = self.ancilla.Ej * self.ancilla.nl_potential(cos_interiors)
            H = Hl + Hnl
            self.circuit_H_memory[self.circuit_parameters_value] = H
            print("Done!")
        #else:
            #print("Hamiltonian found in memory. Proceeding...")

        if basis_cob is not None:
            try:
                return H.transform(basis_cob)
            except:
                print("Could not identify basis. Returning hamiltonian in Fock basis.")
                return H

        if basis == "fock":
            return H
        if basis == "eigen":
            COB_matrix = self._calc_change_of_basis_matrix()
            return H.transform(COB_matrix)

    def _calc_circuit_spectrum(self, order = "energy"):
        """Eigenvalues and eigenvectors of the full hamiltonian of the circuit.
        
        Args:
            order (str): if "energy", return in lowest-to-highest order, as given by qutip.
            If "excitation", return spectrum ordered by the number of excitations of each mode,
            with the first mode being more significative.
        """

        spectrum = self.circuit_spectrum_memory.get(self.circuit_parameters_value)
        if not spectrum:
            print("Calculating circuit spectrum...", end = ' ')
            spectrum = self.hamiltonian.eigenstates()
            print("Done!")
        # else:
        #     print("Circuit spectrum found in memory. Proceeding...")

        self.circuit_spectrum_memory[self.circuit_parameters_value] = spectrum
        if order == "excitation":
            spectrum = self._reorder_circuit_excitation_spectrum()

        return spectrum  
    
    def _reorder_circuit_excitation_spectrum(self):
        """The purpose of this method is to reorder the eigenvectors according to
        the number of excitations in each mode. Assumes the eigenvectors of the 
        whole system are similar to the eigenvectors of the individual modes 
        (valid in the dispersive regime).
        """

        # Gets eigenvectors of each individual mode
        mode_eigenvec = [self.hamiltonian.ptrace(i).eigenstates()[1] 
                         for i in range(len(self.freqs))]

        # get evecs in the
        evals, evecs = self.circuit_spectrum_memory.get(self.circuit_parameters_value)

        evecs_labeled = []
        for i in range(len(evecs)):
            label = ()
            # Find the number of excitations in each eigenvector
            for mode in range(len(self.freqs)):
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


        spectrum = self.ancilla_spectrum_memory.get(self.circuit_parameters_value)
        if not spectrum:
            print("Calculating ancilla spectrum...", end = ' ')
            Hl, Hnl = self.ancilla.calculate_ancilla_hamiltonian()
            H = Hl + Hnl
            evals_0, evecs_0 = H.eigenstates()
            spectrum = clean_spectrum(evals_0, evecs_0)
            print("Done!")

        #else:
            #print("Ancilla spectrum found in memory. Proceeding...")

        # save results in memory
        self.ancilla_spectrum_memory[self.circuit_parameters_value] = spectrum

        return spectrum


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
        n_modes = len(self.freqs)
        eigenvectors_list = [
            qt.basis(self.fock_trunc, exc.get(i, 0)) for i in range(n_modes)
        ]

        # get ancilla spectrum
        ancilla_eigenvecs = self.ancilla_spectrum[1]
        eigenvectors_list[self.ancilla_index] = qt.Qobj(
            ancilla_eigenvecs[exc.get(self.ancilla_index, 0)]
        )
        evec_guess = reduce(qt.tensor, eigenvectors_list)
        
        # Return the eigenvector of the circuit spectrum that is closest to the guess.
        def distance(s2):
            return (evec_guess.dag() * s2[1]).norm()

        evals, evecs = self.calc_circuit_spectrum()
        eval, evec = max(zip(evals, evecs), key=distance)
        return eval, evec

    def _calc_change_of_basis_matrix(self):
        """
        Calculates the matrix that transforms operators from Fock state basis to the
        basis of eigenvectors of the system. Keeps the structure of the hamiltonian
        by ordering basis per excitation.
        """

        _, evecs = self.calc_circuit_spectrum(order = "excitation")
        COB_matrix = np.concatenate([np.array(x) for x in evecs], axis = 1).T
        return COB_matrix
