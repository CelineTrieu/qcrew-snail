"""
Class for a cQED resonator modelled as a quantum harmonic oscillator.
Defines its own mode operators (creation and destruction ops.)
"""

import qutip as qt


class Resonator:
    def __init__(
        self,
        freq,
        fock_trunc=70,
    ):

        # numerical calc. parameters
        self.fock_trunc = fock_trunc

        # linear mode frequency in Hz
        self.freq = freq

        # qutip mode operators
        self.destroy = qt.destroy(self.fock_trunc)
        self.create = self.destroy.dag()
        self.n = qt.num(self.fock_trunc)
