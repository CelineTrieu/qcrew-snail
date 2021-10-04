"""
Class defining a circuit containing one ancilla coupled to one or more resonators.
"""

import qutip as qt
import numpy as np
from functools import reduce


def tensor_out(op, loc, fock_trunc, n):
    """
    Make operator <op> tensored with identities at locations other than <loc>.
    <fock_trunc> is the dimension of each individual operator and <n> is the number of
    operators.
    """
    op_list = [qt.qeye(fock_trunc) for i in range(n)]
    op_list[loc] = op
    return reduce(qt.tensor, op_list)


def dot(ais, bis):
    """
    Dot product
    """
    return sum(ai * bi for ai, bi in zip(ais, bis))


def clean_spectrum(evals, evecs, m=3, threshold=1e-3):
    """
    Remove from the list the eigenvectors that have too much population in the last
    ::m:: Fock states.

    TODO this is repeated from helper_functions. Find a way to import it from there.
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
        PHI_zpf,  # Comes from pyEPR. The higher PHI is assumed to be the ancilla's
    ):

        self.fock_trunc = ancilla.fock_trunc
        self.n_modes = len(freqs)  # Assign one mode for each circ. element
        self.ancilla = ancilla
        self.mode_freqs = freqs
        self.ancilla_mode = np.argmax(PHI_zpf)

        # get auxiliary operators
        a = qt.destroy(self.fock_trunc)
        ad = a.dag()
        n = qt.num(self.fock_trunc)
        # instantiate qutip operators for each hybridized mode
        self.mode_fields = [
            tensor_out(a + ad, i, self.fock_trunc, self.n_modes)
            for i in range(self.n_modes)
        ]
        self.mode_ns = [
            tensor_out(n, i, self.fock_trunc, self.n_modes) for i in range(self.n_modes)
        ]

        # build hamiltonian.
        cos_interiors = sum(
            [PHI_zpf[i, 0] * self.mode_fields[i] for i in range(self.n_modes)]
        )

        self.Hl = dot(self.mode_freqs, self.mode_ns)
        self.Hnl = ancilla.Ej * ancilla.nl_potential(cos_interiors)
        self.H = self.Hl + self.Hnl

        self.ancilla_spectrum = None  # To be defined in calc_ancilla_spectrum
        self.circuit_spectrum = None  # To be defined in calc_spectrum

    def calc_ancilla_spectrum(self):
        """
        Get eigenstates of the isolated ancilla. Used to build a 0-order approximation
        of the circuit spectrum. The clean_spectrum function is used to remove ill
        states.
        """

        evals_0, evecs_0, _, _ = self.ancilla.calculate_spectrum()
        self.ancilla_spectrum = clean_spectrum(evals_0, evecs_0)
        return self.ancilla_spectrum

    def calc_spectrum(self):
        """
        Full diagonalization of the circuit hamiltonian
        """

        self.circuit_spectrum = self.H.eigenstates()
        return self.circuit_spectrum

    def get_eigenstate(self, d):
        """
        d={mode number: state number}
        Gets a zeroth order approximation for the eigenstates assuming Fock states for
        the resonator modes and the isolated ancilla state. Then returns the eigenstate
        of the diagonalized circuit hamiltonian that is closest to this guess.
        """

        # Get eigestates of linear modes
        eigenvectors_list = [
            qt.basis(self.fock_trunc, d.get(i, 0)) for i in range(self.n_modes)
        ]
        # get ancilla spectrum
        ancilla_eigenvecs = self.ancilla_spectrum[1]
        eigenvectors_list[self.ancilla_mode] = qt.Qobj(
            ancilla_eigenvecs[d.get(self.ancilla_mode, 0)]
        )
        evec_guess = reduce(qt.tensor, eigenvectors_list)

        # Return the eigenvector (and resp. eigenval) of the circuit spectrum that is
        # closest to the guess.
        def distance(s2):
            return (evec_guess.dag() * s2[1]).norm()

        evals, evecs = self.circuit_spectrum
        eval, evec = max(zip(evals, evecs), key=distance)

        return eval, evec
