import ast
import wfdb
import torch
import pandas as pd
import numpy as np


PTBXL_ROOT = "~/ecg_classification/PTB-XL/"
METADATA_PATH = f"{PTBXL_ROOT}/ptbxl_database.csv"
SCP_PATH = f"{PTBXL_ROOT}/scp_statements.csv"

df_metadata = pd.read_csv(METADATA_PATH, index_col='ecg_id')
df_metadata.scp_codes = df_metadata.scp_codes.apply(lambda x: ast.literal_eval(x))

agg_df = pd.read_csv(SCP_PATH,index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
df_metadata['diagnostic_superclass'] = df_metadata.scp_codes.apply(aggregate_diagnostic)


df_stemi = df_metadata[df_metadata["diagnostic_superclass"].apply(lambda x: 'MI' in x)]

filenames = df_stemi[["filename_lr"]].to_csv("stemi_filenames.csv",index=False)
