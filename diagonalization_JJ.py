import numpy as np
import matplotlib.pyplot as plt
import qutip
from snail_solver.ancilla import Ancilla
from snail_solver.helper_functions import *
from snail_solver.single_JJ_element import JJ

# Create JJ
Ej = 14e9
freq = 5.19381e9

jj = JJ(Ej)
ancilla = Ancilla(
    jj,
    freq,
    taylor_degree=30,
    fock_trunc=30,
)
# get qutip hamiltonian operator
evals, evecs, H, taylor_coef = ancilla.calculate_spectrum()
evals, evecs = clean_spectrum(evals, evecs)  # Remove weird states
evecs = [qutip.Qobj(ev) for ev in evecs]
# print(evals[:8] - evals[0])


def closest_state_to(s):
    def distance(s2):
        return (s.dag() * s2[1]).norm()

    return max(zip(evals, evecs), key=distance)


def fock_state(n):
    return qutip.basis(ancilla.fock_trunc, n)


"""
Obtain the RWA anharmonicities
"""
fock = [fock_state(n) for n in range(7)]
eigen_fock = [closest_state_to(s)[1] for s in fock]
# print(closest_state_to(fock[1])[:6])
transit_RWA = [
    fock[i + 1].dag() * H * fock[i + 1] - fock[i].dag() * H * fock[i]
    for i in range(len(fock) - 1)
]
anharm_RWA = [
    np.real((transit_RWA[i + 1] - transit_RWA[i])[0])
    for i in range(len(transit_RWA) - 1)
]

"""
Obtain the non RWA anharmonicities by handpicking the hamiltonian eigenvectors
"""
transit_nonRWA = [
    eigen_fock[i + 1].dag() * H * eigen_fock[i + 1]
    - eigen_fock[i].dag() * H * eigen_fock[i]
    for i in range(len(eigen_fock) - 1)
]
anharm_nonRWA = [
    np.real((transit_nonRWA[i + 1] - transit_nonRWA[i])[0])
    for i in range(len(transit_nonRWA) - 1)
]

"""
Obtain the non RWA anharmonicities by picking the hamiltonian eigenvectors from diagonalization
"""
evals_data = np.real(evals - evals[0]) / 1e6
transit_energies = evals_data[1:] - evals_data[:-1]
anharm = transit_energies[1:] - transit_energies[:-1]


print("anharmonicities (RWA)", anharm_RWA)
print("anharmonicities (non RWA, handpicked)", anharm_nonRWA)
print("anharmonicities (non RWA, automated)", anharm[:4])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
add_spectrum_plot(ax, evals, evecs, ancilla.fock_trunc)

fig, axes = plt.subplots(2, 1, sharex=True)
add_transition_energies_plot(axes[0], evals)
add_anharmonicity_plot(axes[1], evals)
plt.show()
