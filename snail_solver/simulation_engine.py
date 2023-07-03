"""
The simulation engine will use the fock basis of the reference circuit
"""

import qutip
from snail_solver.helper_functions import *

class SimulationEngine():

    def __init__(self, circuit):

        self.circuit = circuit

        self.results = None
        self.initial_state = None
        self.pulses = []

        self.rotating_frame_fs = None # list of frequencies
        self.reference_params = None # parameters of the reference circuit

    def add_pulse(self, pulse):
        self.pulses.append(pulse)
        return
    
    @property
    def H(self):
        drift_hamiltonian = self.circuit.hamiltonian - self.rotating_frame_matrix
        print(self.circuit.hamiltonian)
        print(drift_hamiltonian)
        return [drift_hamiltonian] + [p.hamiltonian() for p in self.pulses]
    
    @property
    def rotating_frame_matrix(self):
        if self.rotating_frame_fs is None:
            print("Rotating frame not defined")
            return 0
        n_modes = len(self.rotating_frame_fs)
        fock_trunc = self.circuit.fock_trunc
        frame_matrix = 0
        for n in range(n_modes):
            freq = self.rotating_frame_fs[n]
            frame_matrix += freq*tensor_out(qutip.num(fock_trunc), n, fock_trunc, n_modes)
        return frame_matrix
    
    # @property
    # def reference_circuit(self):
    #     return 

    def simulate(self, t_sim, timeskip = 1):
        self.results = qutip.mesolve(self.H, self.initial_state, t_sim, options=qutip.Options(nsteps=2000))
        return self.results