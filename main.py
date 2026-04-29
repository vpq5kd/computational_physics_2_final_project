import pandas as pd
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import argparse
from classes.ecg_viewer import ecg_viewer
from pathlib import Path
from RBM_vectorized import RBM

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
    if index == 5000:
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
M = 128

rbm = RBM(N, M)

filename = "persisted_data/test2.npz"
rbm.load_model(filename)

#rbm.train_model(eta=.0001, k=5, data_set=data_set,n_epochs=5000)
#rbm.save_model(filename)

states = rbm.generate_rbm_states()
states = states*std + mean

average_generated_signal = np.mean(states, axis=0)

rbm.display_epsilon_w("persisted_data/test2_epsilon_w.png")
rbm.display_inter_layer_couplings("persisted_data/test2_ic.png")
data_set_normalized = data_set*std + mean
average_data_set_signal = np.mean(data_set_normalized, axis=0)

plt.figure()
plt.plot(average_generated_signal, label='average generated signal')
plt.plot(average_data_set_signal ,label='average real sign')
plt.ylabel("voltage (mv)")
plt.legend()
plt.show()
plt.close()

v0 = data_set[0]
h = rbm.hidden_prob(v0)
v1 = rbm.visible_mean(h)

v0 = v0*std + mean
v1 = v1*std + mean

plt.figure()
plt.plot(v0, label='original')
plt.plot(v1, label='reconstruction')
plt.legend()
plt.show()


