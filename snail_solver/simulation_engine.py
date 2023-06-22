import qutip

class SimulationEngine():

    def __init__(self, circuit):

        self.circuit = circuit

        self.results = None
        self.initial_state = None
        self.pulses = []

    def add_pulse(self, pulse):
        self.pulses.append(pulse)
        return
    
    @property
    def H(self):
        return [self.circuit.hamiltonian] + [p.hamiltonian() for p in self.pulses]

    def simulate(self, t_sim, timeskip = 1):
        self.results = qutip.mesolve(self.H, self.initial_state, t_sim, options=qutip.Options(nsteps=2000))
        return self.results