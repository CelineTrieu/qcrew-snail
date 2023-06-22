import matplotlib.pyplot as plt
import numpy as np

class SquareWaveform():

    def __init__(self, amp, w, t0, length):
        self.amp = amp
        self.w = w
        self.t0 = t0
        self.length = length

    def waveform(self):
        def wf(t, args):
            if self.t0<t<self.length+self.t0:
                return self.amp*np.cos(self.w*(t-self.t0))
            else:
                return 0
        return wf

    def plot_waveform(self):
        wf = self.waveform()
        t_list = np.linspace(0, self.length+2*self.t0, int(1e4))
        plt.plot(t_list, [wf(t, 0) for t in t_list])
        plt.show()

# class GaussianWaveform():
#     def __init__(self):
#         pass