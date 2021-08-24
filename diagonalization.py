import qutip
import numpy as np
import matplotlib.pyplot as plt
from SNAIL_element import SNAIL
from ancilla import Ancilla
from helper_functions import *

# Create SNAIL
Ej = 14.851e9 * 3  # in Hz
n = 3
alpha = 0.29
phi_ext = 0.41 * 2 * np.pi
snail = SNAIL(Ej, n, alpha, phi_ext)

# Create ancilla for given shunt capacitance
fock_trunc = 40  # Fock number truncation of qutip operators
degree = 9  # Taylor order for the expansion of the SNAIL potential
cap = 77.5e-15  # shunt capacitance in F
ancilla = Ancilla(snail, cap, taylor_order=degree, fock_trunc=fock_trunc)
Lj = ancilla.Lj
freq = ancilla.freq
phi_rzpf = ancilla.phi_rzpf

# get qutip hamiltonian operator
term_scaling = [1, 1, 1, 1, 1, 1, 1]  # Rescaling of nonlinear terms
evals, evecs, H = ancilla.calculate_spectrum(term_scaling=term_scaling)
plot_energy_spectrum(evals[:6], evecs[:6], 15)

print("f01 ", evals[1] - evals[0])
print("f12 ", evals[2] - evals[1])
print("f23 ", evals[3] - evals[2])
print("f34 ", evals[4] - evals[3])
