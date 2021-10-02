"""
Class defining a circuit containing one ancilla coupled to one or more resonators.
"""

import qutip as qt
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


class Circuit:
    def __init__(
        self,
        ancilla,
        resonator_list,
        PHI_zpf,  # Comes from pyEPR
    ):

        self.fock_trunc = ancilla.fock_trunc
        self.n_modes = len(resonator_list) + 1  # Assign one mode for each circ. element

        # get auxiliary operators
        a = qt.destroy(self.fock_trunc)
        ad = a.dag()
        n = qt.num(self.fock_trunc)
        # intantiate qutip operators for each hybridized mode
        self.mode_fields = [
            tensor_out(a + ad, i, self.fock_trunc, self.n_modes)
            for i in range(self.n_modes)
        ]
        self.mode_ns = [
            tensor_out(n, i, self.fock_trunc, self.n_modes) for i in range(self.n_modes)
        ]

        # build hamiltonian. First mode is assigned to the hybridized ancilla mode, while
        # the resonators follow in order. TODO this should be reviewed since in pyEPR the
        # first mode is not always the ancilla.
        self.mode_freqs = [ancilla.freq] + [res.freq for res in resonator_list]
        cos_interiors = sum(
            [PHI_zpf[i, 0] * self.mode_fields[i] for i in range(self.n_modes)]
        )

        self.Hl = dot(self.mode_freqs, self.mode_ns)
        self.Hnl = ancilla.Ej * ancilla.nl_potential(cos_interiors)
        self.H = self.Hl + self.Hnl

    def eigenstates(self):

        # diagonalize hamiltonian
        return self.H.eigenstates()
