import matplotlib.pyplot as plt
import numpy as np
from RBM import RBM

N = 100
M = 64

rbm = RBM(N, M)
rbm.load_model("persisted_data/test1.npz")
states = rbm.generate_rbm_states()

def plot_signal(signal):
    fs = 100
    time = np.arange(len(signal)) / fs

    plt.figure()
    plt.plot(time, signal, color='black')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.show()

for signal in states:
    plot_signal(signal)


