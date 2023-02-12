import numpy as np
import qutip
from qutip.wigner import wigner
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

from snail_solver.single_JJ_element import JJ
from snail_solver.ancilla import Ancilla
from snail_solver.circuit import Circuit, tensor_out
from snail_solver.helper_functions import *

# Time unit is sec and frequency is Hz.
# Hamiltonian coefficients are expressed in Hz.

fock_trunc = 12

def get_interpolated_phi_zpf():
    # Get phi zpf data as obtained from pyEPR distributed analysis and interpolate
    phi_zpf = np.load('C:/Users/Aftershock/Documents/qcrew-fastflux/squid_highf_data/phi_zpf_high_f_squid.npz')["arr_0"]
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

def build_circuit(phase):
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
    circuit = Circuit(ancilla, freqs, phi_rzpf)

    return circuit

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
flux_pulse_t0 = 10 # start of the pulse
flux_pulse_A = 0.635 - flux_offset # amplitude in reduced flux (rad)
flux_pulse_len = 3000
flux_pulse_sigma = 5
#flux_pulse = gaussian_envelope(flux_pulse_A, flux_pulse_t0, flux_pulse_len, flux_pulse_sigma)
flux_pulse = square_envelope(flux_pulse_A, flux_pulse_t0, flux_pulse_len)
hamiltonian_list = []
reshaped_t_list_simulation = []
hamiltonian_dictionary = {} # store pre-existing solutions

t_list_hamiltonian = np.arange(0, 3020, 1)
t_list_simulation = np.arange(0, 3020, 0.1)
t_list_simulation_reshaped = []

for i in range(len(t_list_hamiltonian)-1):
    t_current = t_list_hamiltonian[i]
    t_next = t_list_hamiltonian[i+1]
    a = t_current <= t_list_simulation
    b = t_next > t_list_simulation
    sliced_t_list_simulation = t_list_simulation[a & b]
    t_list_simulation_reshaped.append(sliced_t_list_simulation)

# Get the COB matrix, energy offset and the rotating frame of the initial circuit
initial_circuit = build_circuit(flux_pulse(t_list_hamiltonian[0], '') + flux_offset)
f01_res_ground = initial_circuit.get_eigenstate({0:1})[0] - initial_circuit.get_eigenstate({0:0})[0]
f01_res_excited = initial_circuit.get_eigenstate({0:1, 1:1})[0] - initial_circuit.get_eigenstate({0:0, 1:1})[0]
chi_0 = f01_res_excited - f01_res_ground
second_circuit = build_circuit(0.635)
f01_res_ground = second_circuit.get_eigenstate({0:1})[0] - second_circuit.get_eigenstate({0:0})[0]
f01_res_excited = second_circuit.get_eigenstate({0:1, 1:1})[0] - second_circuit.get_eigenstate({0:0, 1:1})[0]
chi_1 = f01_res_excited - f01_res_ground

COB_matrix = initial_circuit.COB_matrix
offset = initial_circuit.get_circuit_spectrum()[0][0]
rot_frame_matrix = 0
n_modes = len(initial_circuit.mode_freqs)
for mode in range(n_modes):
    f01_mode = initial_circuit.get_eigenstate({mode: 1})[0] - initial_circuit.get_eigenstate({mode: 0})[0]
    number_operator_mode = tensor_out(qutip.num(fock_trunc), mode, fock_trunc, n_modes)
    rot_frame_matrix += f01_mode*number_operator_mode

# Include last time window
t_current = t_list_hamiltonian[-1]
a = t_current <= t_list_simulation
t_list_simulation_reshaped.append(t_list_simulation[a])
for t_ham in t_list_hamiltonian:
    # get flux for that hamiltonian time
    flux = flux_pulse(t_ham, '') + flux_offset
    # calculate hamiltonian if its not in the dictionary. Else, retrieve
    if flux in hamiltonian_dictionary.keys():
        hamiltonian = hamiltonian_dictionary[flux]
    else:
        # Get hamiltonian on the basis of eigenvectors and rotating frame of the initial 
        # circuit and remove energy offset.

        circuit = build_circuit(flux)
        hamiltonian = circuit.get_H(basis_cob = COB_matrix)
        hamiltonian -= (rot_frame_matrix + offset)
        hamiltonian_dictionary[flux] = hamiltonian
    
    hamiltonian_list.append(hamiltonian)

# Ancilla starts in qubit manifold
initial_state_ancilla =  (qutip.basis(fock_trunc, 0) + qutip.basis(fock_trunc, 1)).unit()
# Resonator starts in odd cat state
initial_state_cavity = qutip.basis(fock_trunc,0)
initial_state = qutip.tensor(initial_state_cavity, initial_state_ancilla)
         
# Convert hamiltonian units to GHz
hamiltonian_list = [2*np.pi*H/1e9 for H in hamiltonian_list] 

# simulate step by step
state = initial_state
state_list = []
for i in range(len(t_list_simulation_reshaped)):
    H = hamiltonian_list[i]
    t_sim = t_list_simulation_reshaped[i]
    result = qutip.mesolve(H, state, t_sim, options=qutip.Options(nsteps=3000))
    # start next one where this one ended
    state = result.states[-1] 
    # concatenate list of states while excluding initial state
    state_list += result.states[0::]



# Ancilla starts in qubit manifold
initial_state_ancilla =  (qutip.basis(fock_trunc, 0) + qutip.basis(fock_trunc, 1)).unit()
# Resonator starts in odd cat state
initial_state_cavity = qutip.basis(fock_trunc,1)
initial_state = qutip.tensor(initial_state_cavity, initial_state_ancilla)
         
# Convert hamiltonian units to GHz
hamiltonian_list = [2*np.pi*H/1e9 for H in hamiltonian_list] 

# simulate step by step
state = initial_state
state_list2 = []
for i in range(len(t_list_simulation_reshaped)):
    H = hamiltonian_list[i]
    t_sim = t_list_simulation_reshaped[i]
    result = qutip.mesolve(H, state, t_sim, options=qutip.Options(nsteps=3000))
    # start next one where this one ended
    state = result.states[-1] 
    # concatenate list of states while excluding initial state
    state_list2 += result.states[0::]

print("AA")


plt.plot(t_list_simulation[::500], [qutip.metrics.fidelity(state_list[i].ptrace(1),state_list2[i].ptrace(1)) for i in range(len(state_list[::500]))])
plt.show()

"""
# plot wigner function
max_range = 4
displ_array = np.linspace(-max_range, max_range, 61)
wigner_list0 = [wigner(x.ptrace(0), displ_array, displ_array) for x in state_list[::100]]
wigner_list1 = [wigner(x.ptrace(1), displ_array, displ_array) for x in state_list[::100]]
# create first plot
fig, axes = plt.subplots(1,2)
axes[0].set_aspect('equal', 'box')
axes[1].set_aspect('equal', 'box')
fig.set_size_inches(20, 8)
fig.tight_layout()
wigner_f0 = wigner(state_list[0].ptrace(0), displ_array, displ_array)
wigner_f1 = wigner(state_list[0].ptrace(1), displ_array, displ_array)
cont0 = axes[0].pcolormesh(displ_array, displ_array, wigner_f0, cmap = "bwr")
cont1 = axes[1].pcolormesh(displ_array, displ_array, wigner_f1, cmap = "bwr")
cb = fig.colorbar(cont0)

# refresh function
def animated_wigner(frame):
    wigner_f0 = wigner_list0[frame]
    wigner_f1 = wigner_list1[frame]
    cont0 = axes[0].pcolormesh(displ_array, displ_array, wigner_f0, cmap = "bwr")
    cont1 = axes[1].pcolormesh(displ_array, displ_array, wigner_f1, cmap = "bwr")
    cont0.set_clim(-1/np.pi, 1/np.pi)
    cont1.set_clim(-1/np.pi, 1/np.pi)

anim = FuncAnimation(fig, animated_wigner, frames=len(wigner_list0), interval=100)
anim.save('animation_parity_1.gif', writer='imagemagick')
"""