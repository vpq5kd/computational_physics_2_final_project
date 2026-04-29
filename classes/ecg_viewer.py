import wfdb
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str)
args = parser.parse_args()

class ecg_viewer():
    def load_wfdb_record(self, record_path):
        signal, fields = wfdb.rdsamp(record_path)
        return signal, fields

    def view_ecg(self, record_path):
        signal, fields = wfdb.rdsamp(record_path)
        fs = fields["fs"]
        lead_names = fields["sig_name"]

        n_samples = signal.shape[0]
        time = np.arange(n_samples) / fs

        fig, axes = plt.subplots(12,1,figsize=(14,10),sharex=True)

        for i, ax in enumerate(axes):
            y = signal[:,i]
            ax.plot(time, signal[:,i], linewidth=0.8,color='black')
            ax.set_ylabel(lead_names[i], rotation=0, labelpad=25)
            ax.set_xlim(time[0], time[-1])

            ax.set_xticks(np.arange(0, time[-1], 0.2))
            ax.set_yticks(np.arange(np.floor(y.min()), np.ceil(y.max()), 0.5))

            ax.set_xticks(np.arange(0, time[-1], 0.04), minor=True)
            ax.set_yticks(np.arange(np.floor(y.min()), np.ceil(y.max()), 0.1), minor=True)

            ax.grid(which="major", linewidth=0.8)
            ax.grid(which="minor", linewidth=0.3)
            #ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()



        
