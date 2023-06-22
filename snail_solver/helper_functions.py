"""
Helper functions for SNAIL class.
roots finds all zeros of a function in a given interval.
more_than_2_roots returns True if more than two zeros are found.
Code taken from https://www.py4u.net/discuss/19528.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import qutip
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce

e_charge = 1.602e-19
hbar = 1.054e-34
flux_quantum = 2.067e-15  # magnetic flux quantum

unit_dict = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3, "Hz": 1}


def rootsearch(f, a, b, dx):
    x1 = a
    f1 = f(a)
    x2 = a + dx
    f2 = f(x2)
    while f1 * f2 > 0.0:
        if x1 >= b:
            return None, None
        x1 = x2
        f1 = f2
        x2 = x1 + dx
        f2 = f(x2)
    return x1, x2


def bisect(f, x1, x2, switch=0, epsilon=5.0e-4):
    f1 = f(x1)
    if f1 == 0.0:
        return x1
    f2 = f(x2)
    if f2 == 0.0:
        return x2
    if f1 * f2 > 0.0:
        return None
    n = int(math.ceil(math.log(abs(x2 - x1) / epsilon) / math.log(2.0)))
    for i in range(n):
        x3 = 0.5 * (x1 + x2)
        f3 = f(x3)
        if (switch == 1) and (abs(f3) > abs(f1)) and (abs(f3) > abs(f2)):
            return None
        if f3 == 0.0:
            return x3
        if f2 * f3 < 0.0:
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3
    return (x1 + x2) / 2.0


def roots(f, a, b, eps=1e-3):
    print("The roots on the interval [%f, %f] are:" % (a, b))
    while 1:
        x1, x2 = rootsearch(f, a, b, eps)
        if x1 != None:
            a = x2
            root = bisect(f, x1, x2, 1)
            if root != None:
                pass
                print(round(root, -int(math.log(eps, 10))))
        else:
            print("\nDone")
            break


def more_than_2_roots(f, a, b, eps=1e-3):
    n_roots = 0
    while 1:
        x1, x2 = rootsearch(f, a, b, eps)
        if x1 != None:
            a = x2
            root = bisect(f, x1, x2, 1)
            if root != None:
                n_roots += 1
                if n_roots > 2:
                    return True
                pass
        else:
            return False


def plot_imshow(x, y, z):

    fig, ax = plt.subplots()
    pos = ax.imshow(z, cmap="seismic", interpolation="bilinear")
    fig.colorbar(pos, ax=ax)
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)
    plt.show()

    return


def clean_spectrum(evals, evecs, m=3, threshold=1e-3):
    """
    Remove from the list the eigenvectors that have too much population in the last
    ::m:: Fock states.
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


def add_spectrum_plot(ax, evals, evecs, fock_trunc, unit="GHz"):
    """
    Adapted from https://stackoverflow.com/questions/35210337/can-i-plot-several-histograms-in-3d/35225919
    """

    # calculate probability of finding eigenvec in each Fock state
    evecs_data = [[float(np.absolute(x) ** 2) for x in evec] for evec in evecs]
    # get eigenfrequencies in GHz and remove offset
    evals_data = np.real(evals - evals[0]) / unit_dict[unit]

    for energy, eigenstate in zip(evals_data, evecs_data):
        fock_list = list(range(fock_trunc))
        eigenstate = eigenstate[:fock_trunc]
        ax.bar(fock_list, eigenstate, zs=energy, zdir="y", alpha=0.6)

    ax.set_xlabel("Fock state N")
    ax.set_ylabel("Eigenfrequency (%s)" % unit)
    ax.set_zlabel("P(N)")

    return


def add_energy_diagram_plot(ax, evals, label=None, unit="GHz"):

    evals_data = np.real(evals - evals[0]) / unit_dict[unit]
    ax.scatter(list(range(len(evals_data))), evals_data, label=label)

    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenfrequency (%s)" % unit)

    return


def add_transition_energies_plot(ax, evals, label=None, unit="GHz"):

    evals_data = np.real(evals - evals[0]) / unit_dict[unit]

    transit_energies = evals_data[1:] - evals_data[:-1]
    ax.scatter(list(range(len(transit_energies))), transit_energies, label=label)

    ax.set_xlabel("Index")
    ax.set_ylabel("Transition energies (%s)" % unit)

    return


def add_anharmonicity_plot(ax, evals, label=None, unit="MHz"):
    evals_data = np.real(evals - evals[0]) / unit_dict[unit]

    transit_energies = evals_data[1:] - evals_data[:-1]
    anharm = transit_energies[1:] - transit_energies[:-1]
    ax.scatter(list(range(len(anharm))), anharm, label=label)

    ax.set_xlabel("Index")
    ax.set_ylabel("Anharmonicities (%s)" % unit)

    return


def report_H_params(circuit, n_modes, cut_modes):
    cut_n_modes = len(circuit.phi_rzpf)
    circuit.ancilla_mode
    kerr_matrix = np.zeros([n_modes, n_modes])
    kerr_matrix[:] = np.nan
    freq_list = np.zeros(n_modes)
    freq_list[:] = np.nan
    for mode_index in range(cut_n_modes):
        matrix_indx1 = cut_modes[mode_index]

        # Get the first transition frequency of each mode after diagonalization
        freq_list[matrix_indx1] = (
            circuit.get_eigenstate({mode_index: 1})[0]
            - circuit.get_eigenstate({mode_index: 0})[0]
        )

        # Get the dispersive shift between each mode and the ancilla
        for second_mode_index in range(cut_n_modes):
            matrix_indx2 = cut_modes[second_mode_index]
            if second_mode_index == mode_index:
                # Calculate self-Kerr
                freq_2 = (
                    circuit.get_eigenstate({mode_index: 2})[0]
                    - circuit.get_eigenstate({mode_index: 1})[0]
                )
                freq_1 = (
                    circuit.get_eigenstate({mode_index: 1})[0]
                    - circuit.get_eigenstate({mode_index: 0})[0]
                )
                kerr = freq_2 - freq_1
            else:
                freq_2 = (
                    circuit.get_eigenstate({mode_index: 1, second_mode_index: 1})[0]
                    - circuit.get_eigenstate({mode_index: 0, second_mode_index: 1})[0]
                )
                freq_1 = (
                    circuit.get_eigenstate({mode_index: 1})[0]
                    - circuit.get_eigenstate({mode_index: 0})[0]
                )
                kerr = freq_2 - freq_1

            kerr_matrix[matrix_indx1, matrix_indx2] = kerr

    return freq_list, kerr_matrix


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

    op_list = [qutip.qeye(fock_trunc) for i in range(n)]
    op_list[loc] = op
    return reduce(qutip.tensor, op_list)

def animate_wigner(results, n_modes, filename, displ_array, skip = 1):
    """
    skip is the number of frames skipped during plotting
    """

    # create plot
    fig, axes = plt.subplots(1, n_modes)
    fig.set_size_inches(20, 8)
    fig.tight_layout()
    
    wigner_list = []
    cont_list = []
    
    # Iterate over modes
    for i in range(n_modes):
        # Get simulation results
        # Calculate wigner functions over time
        wigner_mode = [qutip.wigner(x.ptrace(i), displ_array, displ_array) for x in results.states[::skip]]
        # Plot the first wigner function
        axes[i].set_aspect('equal', 'box')
        cont_mode = axes[i].pcolormesh(displ_array, displ_array, wigner_mode[0], cmap = "bwr")
        wigner_list.append(wigner_mode)
        cont_list.append(cont_mode)

    # Set colorbar
    cb = fig.colorbar(cont_list[0])

    # refresh function
    def plot_frame(frame):
        for i in range(n_modes):
            wigner_frame = wigner_list[i][frame]
            cont = axes[i].pcolormesh(displ_array, displ_array, wigner_frame, cmap = "bwr")
            cont.set_clim(-1/np.pi, 1/np.pi)

    anim = FuncAnimation(fig, plot_frame, frames=len(wigner_list[0]), interval=100)
    anim.save(filename, writer='imagemagick')