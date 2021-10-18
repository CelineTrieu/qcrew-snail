"""
Class defining a circuit containing one ancilla coupled to one or more resonators.

It is designed to work seamlessly with pyEPR, having as inputs the list of reduced flux
zero-point fluctuations and a list of mode frequencies.
"""

import qutip as qt
import numpy as np
from functools import reduce


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
    ):
        """Define a circuit composed of multiple modes, one of which is an ancilla.

        The ancilla mode is identified as the mode with largest reduced flux zero-point
        fluctuation. The remaining are assumed to be resonator modes.

        Args:
            ancilla (Ancilla): ancilla object.

            freqs (list): List of linear frequencies of each mode in Hz.

            phi_rzpf (list): List of reduced flux zero-point fluctuations over the ancilla junction for each mode.
        """

        self.fock_trunc = ancilla.fock_trunc
        self.n_modes = len(freqs)  # Assign one mode for each circ. element
        self.ancilla = ancilla
        self.mode_freqs = freqs
        self.ancilla_mode = np.argmax(phi_rzpf)

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
            [phi_rzpf[i, 0] * self.mode_fields[i] for i in range(self.n_modes)]
        )

        self.Hl = dot(self.mode_freqs, self.mode_ns)
        self.Hnl = ancilla.Ej * ancilla.nl_potential(cos_interiors)
        self.H = self.Hl + self.Hnl

        self._ancilla_spectrum, self._circuit_spectrum = None, None

    @property
    def ancilla_spectrum(self):
        """Eigenvalues and eigenvectors of the hamiltonian of the isolated ancilla."""
        if self._ancilla_spectrum:
            return self._ancilla_spectrum

        else:
            self._ancilla_spectrum = self._calc_ancilla_spectrum()
            return self._ancilla_spectrum

    @property
    def circuit_spectrum(self):
        """Eigenvalues and eigenvectors of the full hamiltonian of the circuit."""
        if self._circuit_spectrum:
            return self._circuit_spectrum

        else:
            self._circuit_spectrum = self.H.eigenstates()
            return self._circuit_spectrum

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
            exc (dict): dictionary in which the keys are mode indexes (same indexing as self.mode_freqs) and the values are the number of excitations.

        Returns:
            tuple: (eval, evec), in which evec is the corresponding eigenstate of the circuit hamiltonian and eval is its eigenvalue (Hz).
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

        # Return the eigenvector of the circuit spectrum that is closest to the guess.
        def distance(s2):
            return (evec_guess.dag() * s2[1]).norm()

        evals, evecs = self.circuit_spectrum
        eval, evec = max(zip(evals, evecs), key=distance)

        return eval, evec
