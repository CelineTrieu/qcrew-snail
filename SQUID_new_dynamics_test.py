import numpy as np
import qutip
from qutip.wigner import wigner
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

from snail_solver.single_JJ_element import JJ
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit
from snail_solver.helper_functions import *

# Time unit is sec and frequency is Hz.
# Hamiltonian coefficients are expressed in Hz.

fock_trunc = 10
def get_interpolated_phi_zpf():
    # Get phi zpf data as obtained from pyEPR distributed analysis and interpolate
    phi_zpf = np.load('C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/phi_rzpf_high_f_squid.npz')["arr_0"]
    phase_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["phase"]
    cavity_phi_zpf = interp1d(phase_list, phi_zpf[:, 0])
    ancilla_phi_zpf = interp1d(phase_list, phi_zpf[:, 1])
    resonator_phi_zpf = interp1d(phase_list, phi_zpf[:, 2])
    return cavity_phi_zpf, ancilla_phi_zpf, resonator_phi_zpf

def get_interpolated_ansys_fs():
    # Get HFSS frequency data and interpolate
    ansys_fs = np.load('C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ansys_fs.npz')["ansys_freqs"]
    phase_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["phase"]
    cavity_fs = interp1d(phase_list, ansys_fs[:,0].reshape(35, 3)[:,0])
    ancilla_fs = interp1d(phase_list, ansys_fs[:,0].reshape(35, 3)[:,1])
    resonator_fs = interp1d(phase_list, ansys_fs[:,0].reshape(35, 3)[:,2])
    return cavity_fs, ancilla_fs, resonator_fs

def get_interpolated_ljs():
    # Get HFSS frequency data and interpolate
    ljs_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["ljs"]
    phase_list = np.load("C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/ljs_and_phase.npz")["phase"]
    ljs = interp1d(phase_list, ljs_list)
    return ljs

ljs = get_interpolated_ljs()
cavity_phi_zpf, ancilla_phi_zpf, resonator_phi_zpf = get_interpolated_phi_zpf()
cavity_fs, ancilla_fs, resonator_fs = get_interpolated_ansys_fs()

def build_circuit(phase, reference_operators = None):
    # The phase input must be within the interpolation range given by the max and min values of phase
    # in the file ljs_and_phase.npz

    # Get circuit and SQUID parameters
    Lj = 1e-9*ljs(phase)
    Ej = 1 / (2 * np.pi * hbar * Lj) * (flux_quantum / 2 / np.pi) ** 2
    #freqs = 1e9*np.array([cavity_fs(phase), ancilla_fs(phase), resonator_fs(phase)])
    freqs = 1e9*np.array([cavity_fs(phase), ancilla_fs(phase)])
    #phi_rzpf = np.array([[cavity_phi_zpf(phase)], [ancilla_phi_zpf(phase)], [resonator_phi_zpf(phase)]])
    phi_rzpf = np.array([[cavity_phi_zpf(phase)], [ancilla_phi_zpf(phase)]])
    # Assemble circuit
    squid = JJ(Ej)
    ancilla = Ancilla(squid, freqs[np.argmax(phi_rzpf)], fock_trunc=fock_trunc)
    circuit = Circuit(ancilla, freqs, phi_rzpf, reference_operators = reference_operators)

    return circuit

# reference_circuit = build_circuit(0.4)
# ref_operators = reference_circuit.calc_ancilla_flux_charge_operators()
# circuit_A = build_circuit(0.41, reference_operators = ref_operators)
# circuit_B = build_circuit(0.41)
# # reference_circuit.get_H()
# circuit_A.get_H()
# print("CIRCUIT B")
# circuit_B.get_H()

def square_envelope(A, t0, length):
    def envelope(t, args):
        if t0<t<length+t0:
            return A
        else:
            return 0
    return envelope

def gaussian_envelope(A, t0, length, sigma):
    mu = t0 + length/2
    def envelope(t, args):
        if t0<t<length+t0:
            return A*np.exp(-(t-mu)**2/2/sigma**2)
        else:
            return 0
    return envelope

flux_offset = 0.4
flux_pulse_t0 = 0 # start of the pulse
flux_pulse_A = 0.65 - flux_offset # amplitude in reduced flux (rad)
flux_pulse_len = 40  
flux_pulse_sigma = 5
#flux_pulse = gaussian_envelope(flux_pulse_A, flux_pulse_t0, flux_pulse_len, flux_pulse_sigma)
flux_pulse = square_envelope(flux_pulse_A, flux_pulse_t0, flux_pulse_len)
hamiltonian_list = []
reshaped_t_list_simulation = []
hamiltonian_dictionary = {} # store pre-existing solutions

t_list_hamiltonian = np.arange(0, 40, 1)
t_list_simulation = np.arange(0, 40, 0.1)
t_list_simulation_reshaped = []
for i in range(len(t_list_hamiltonian)-1):
    t_current = t_list_hamiltonian[i]
    t_next = t_list_hamiltonian[i+1]
    a = t_current <= t_list_simulation
    b = t_next > t_list_simulation
    sliced_t_list_simulation = t_list_simulation[a & b]
    t_list_simulation_reshaped.append(sliced_t_list_simulation)
# Include last time window
t_current = t_list_hamiltonian[-1]
a = t_current <= t_list_simulation
t_list_simulation_reshaped.append(t_list_simulation[a])
initial_circuit = None
reference_circuit = build_circuit(flux_offset)
ref_operators = None #reference_circuit.calc_ancilla_flux_charge_operators()
for t_ham in t_list_hamiltonian:
    # get flux for that hamiltonian time
    flux = flux_pulse(t_ham, '') + flux_offset
    # calculate hamiltonian if its not in the dictionary. Else, retrieve
    if flux in hamiltonian_dictionary.keys():
        hamiltonian = hamiltonian_dictionary[flux]
    else:
        circuit = build_circuit(flux, reference_operators = ref_operators)
        # save initial circuit
        if not initial_circuit:
            initial_circuit = circuit
        hamiltonian = circuit.get_H()
        hamiltonian_dictionary[flux] = hamiltonian
    
    hamiltonian_list.append(hamiltonian)

# print(b0, flux_op, charge_op, circuit_ref.coupling_factor)
# circuit = build_circuit(0.5)
# circuit_operators = circuit.calc_mode_operators(b0, flux_op, charge_op)
print("states 0 and 1")
initial_state =  (initial_circuit.get_eigenstate({1: 0})[1] + initial_circuit.get_eigenstate({1: 1})[1]).unit()
print("^^^^^")
ref_operators = initial_circuit.calc_ancilla_flux_charge_operators()
f01 = initial_circuit.get_eigenstate({1: 1})[0] - initial_circuit.get_eigenstate({1: 0})[0]
offset = initial_circuit.get_eigenstate({1: 0})[0]
ancilla_number_op = sum([np.sqrt(i)*qutip.ket2dm(initial_circuit.get_eigenstate({1: i})[1])
                         for i in range(len(circuit.ancilla_spectrum[0]))])
                         
print(f01)
print(offset)
# Convert hamiltonian units to GHz
a = qutip.tensor(qutip.qeye(fock_trunc), qutip.destroy(fock_trunc))
hamiltonian_list = [(H-offset-f01*ancilla_number_op)/1e9 for H in hamiltonian_list] 

# simulate iteratively
state = initial_state*initial_state.dag()
state_list = [state,]
for i in range(len(t_list_simulation_reshaped)):
    print(i)
    H = hamiltonian_list[i]
    t_sim = t_list_simulation_reshaped[i]
    result = qutip.mesolve(H, state, t_sim, options=qutip.Options(nsteps=3000))
    # start next one where this one ended
    state = result.states[-1] 
    # concatenate list of states while excluding initial state
    state_list += result.states[1::]

print(state_list[0].ptrace(1).diag())
print(state_list[-1].ptrace(1).diag())
# plot wigner function
max_range = 2
displ_array = np.linspace(-max_range, max_range, 51)
wigner_list = [wigner(x.ptrace(1), displ_array, displ_array) for x in state_list[::5]]

# create first plot
fig, axes = plt.subplots()
axes.set_aspect('equal', 'box')
fig.set_size_inches(10, 8)
fig.tight_layout()
wigner_f = wigner(result.states[0], displ_array, displ_array)
cont0 = axes.pcolormesh(displ_array, displ_array, wigner_f, cmap = "bwr")
cb = fig.colorbar(cont0)

# refresh function
def animated_wigner(frame):
    wigner_f = wigner_list[frame]
    cont0 = axes.pcolormesh(displ_array, displ_array, wigner_f, cmap = "bwr")
    cont0.set_clim(-1/np.pi, 1/np.pi)

anim = FuncAnimation(fig, animated_wigner, frames=len(wigner_list), interval=100)
anim.save('animation.gif', writer='imagemagick')
