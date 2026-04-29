import pandas as pd
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from classes.ecg_viewer import ecg_viewer
from pathlib import Path
from RBM import RBM

PTBXL_ROOT = Path("~/ecg_classification/PTB-XL/").expanduser()
df_stemi = pd.read_csv("persisted_data/stemi_filenames.csv")
stemi_filenames = [str(PTBXL_ROOT/filename) for filename in df_stemi["filename_lr"]]

ev = ecg_viewer()

def load_ecg(filename):
    record = wfdb.rdsamp(filename)[0]
    return record

def process_ecg(signal):
    return signal[:,0]

data = []

index = 0
for filename in stemi_filenames:
    if index == 1000:
        break
    signal = load_ecg(filename)
    signal = process_ecg(signal)
    data.append(signal[:250])
    index += 1

data_set = np.array(data)
mean = data_set.mean()
std = data_set.std()

data_set = (data_set - mean) / std


N = data_set.shape[1]
M = 64

rbm = RBM(N, M)
rbm.train_model(eta=1e-4, k=1, data_set=data_set,n_epochs=50)

filename = "persisted_data/test1.npz"
#rbm.load_model(filename)
rbm.save_model(filename)
states = rbm.generate_rbm_states()
states = std*(states) + mean

data_set = data_set*std + mean
machine_average = np.mean(states, axis=0)
real_average = np.mean(data_set, axis=0)
print(f"machine average array: {machine_average}")
print(f"real average array: {real_average}")


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
