
"""
Pulses define time-dependent hamiltonians used as mesolve input.
"""
import numpy as np

class DrivePulse():
    def __init__(self, a, waveform, phase = 0):
        self.waveform = waveform
        self.a = a # destruction operator
        self.phase = phase # phase of the drive
        
    def hamiltonian(self):
        eps = np.exp(1j*self.phase)
        drive_H = 2*np.pi*1j*(eps*self.a.dag() - np.conj(eps)*self.a) 
        drive_env = self.waveform.waveform()
        return [drive_H, drive_env]