import numpy as np
import qutip

from snail_solver.ancillae import SNAIL, JJ
from snail_solver.circuit import Circuit
from snail_solver.parameters import ContantParameters
from snail_solver.waveforms import SquareWaveform
from snail_solver.pulses import DrivePulse
from snail_solver.simulation_engine import SimulationEngine
from snail_solver.helper_functions import animate_wigner, tensor_out

fock_trunc = 10

# Create SNAIL
# snail_parameters = {"n": 3, "alpha": 0.290, "phi_ext": 0.410 * 2 * np.pi}
# ancilla = SNAIL(**snail_parameters, fock_trunc = fock_trunc)
ancilla = JJ(fock_trunc= fock_trunc)
# Define constant circuit parameters: SNAIL + 1 resonator
Lj = 9.0e-9
freqs = np.array([5.727e9, 6.181e9])
phi_rzpf = np.array([[0.00588067], [0.35791346]])
params = ContantParameters(Lj, freqs, phi_rzpf)

# Create circuit
circuit = Circuit(ancilla, params)
circuit.set_circuit_parameters()

# Start simulation engine
sim = SimulationEngine(circuit)
# sim.initial_state = qutip.tensor(qutip.fock(fock_trunc, n = 0), qutip.fock(fock_trunc, n = 1))
sim.initial_state = circuit.get_eigenstate({1:0})[1]
rot_frame_freqs =  [circuit.get_eigenstate({0:1})[0] - circuit.get_eigenstate({0:0})[0], circuit.get_eigenstate({1:1})[0] - circuit.get_eigenstate({1:0})[0]]
sim.rotating_frame_fs = rot_frame_freqs
# print(sim.rotating_frame_matrix)

# Add pulses
# a = tensor_out(qutip.destroy(fock_trunc), 1, fock_trunc, 2) # ancilla destruction op
# wf_1 = SquareWaveform(0*1.5e6, 0, 20e-9, 100e-9)
# pulse_1 = DrivePulse(a, wf_1)
#sim.add_pulse(pulse_1)
# Simulate
t_sim = np.arange(0e-9, 200e-9, 0.2e-9)
sim.simulate(t_sim, timeskip = 10)
result_states = sim.results.states

# Animate wigner function
displ_array = np.linspace(-6, 6, 101)
animate_wigner(sim.results, 2, "animation.gif", displ_array, skip = 20)