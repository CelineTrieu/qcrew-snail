"""
Class defining a set of circuit parameters obtained from pyEPR.

Child classes have built-in functions that can infer circuit parameters from interpolation 
or extrapolation; alternatively, they can simply provide fixed values.
"""

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

class Parameters:
    def __init__(self):
        return
    
    def get(self):
        # defined in child class
        return


class ContantParameters(Parameters):
    def __init__(self, Lj, freqs, phi_rzpf):
        self._Lj = Lj
        self._freqs = freqs
        self._phi_rzpf = phi_rzpf
    
    def get(self, *args):
        return (self._Lj, self._freqs, self._phi_rzpf)
    
class InterpolatedParameters(Parameters):
    def __init__(self, Lj_array, freqs_array, phi_rzpf_array, var_array):

        # The arrays must be arranged with shape
        # (number of datapoints, number of modes)
        self._Lj_array = Lj_array
        self._freqs_array = freqs_array
        self._phi_rzpf_array = phi_rzpf_array
        self._n_modes = None # number of modes
        self._var_array = var_array # independent variable 
    
        self.calc_interpolation()

    def calc_interpolation(self):
        self._Lj_f = interp1d(self._var_array, self._Lj_array)

        self._n_modes = self._freqs_array.shape[1]
        
        list_of_freqs_functions = []
        for n in range(self._n_modes):
            freqs_mode_array = self._freqs_array[:, n]
            freqs_mode_f = interp1d(self._var_array, freqs_mode_array)
            list_of_freqs_functions.append(freqs_mode_f)
        def freqs_f(var):
            return [float(f(var)) for f in list_of_freqs_functions]
        self._freqs_f = freqs_f


        list_of_phi_functions = []
        for n in range(self._n_modes):
            phi_mode_array = self._phi_rzpf_array[:, n]
            phi_mode_f = interp1d(self._var_array, phi_mode_array)
            list_of_phi_functions.append(phi_mode_f)
        def phi_rzpf_f(var):
            return np.array([f(var) for f in list_of_phi_functions]).reshape(self._n_modes, -1)
        self._phi_rzpf_f = phi_rzpf_f

        return

    def get(self, *args):
        
        if len(args) > 1:
            print("Interpolated params only take one argument")
            return None
        
        var = args[0]
        Lj = self._Lj_f(var)
        freqs = self._freqs_f(var)
        phi_rzpf = self._phi_rzpf_f(var)

        return (Lj, freqs, phi_rzpf)
    
    def plot_interpolation(self):
        fig, axes = plt.subplots(1, 3)
        axes[0]

        var_list = np.linspace(min(self._var_array), max(self._var_array), int(1e4))

        # plot Ljs
        axes[0].plot(var_list, [self.get(x)[0] for x in var_list])

        # plot freqs
        for n in range(self._n_modes):
            axes[1].plot(var_list, [self.get(x)[1][n] for x in var_list])

        # plot phi_rzpf
        for n in range(self._n_modes):
            axes[2].plot(var_list, [self.get(x)[2][n] for x in var_list])

        plt.show()
        return