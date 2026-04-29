import pandas as pd
from classes.ecg_viewer import ecg_viewer
from pathlib import Path

PTBXL_ROOT = Path("~/ecg_classification/PTB-XL/").expanduser()
df_stemi = pd.read_csv("persisted_data/stemi_filenames.csv")
stemi_filenames = [str(PTBXL_ROOT/filename) for filename in df_stemi["filename_lr"]]

ev = ecg_viewer()

for ecg in stemi_filenames:
    ev.view_ecg(ecg)
    
