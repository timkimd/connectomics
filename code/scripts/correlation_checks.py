#%% Imports
import statsmodels.api as sm
import numpy as np
import pandas as pd

import sys
import os
from os.path import join as pjoin
import platform
from matplotlib import pyplot as plt
import seaborn as sns
import glob
from hdmf_zarr import NWBZarrIO
from nwbwidgets import nwb2widget

platstring = platform.platform()
system = platform.system()
if system == "Darwin":
    # macOS
    data_root = "/Volumes/Brain2025/"

#%% Import metadata and example session
metadata = pd.read_csv(pjoin(data_root, 'v1dd_1196/V1DD_metadata.csv'))
mouse_id = '409828'
mouse_dir = glob.glob((os.path.join(data_root, mouse_id+'*')))[0]
session_name = metadata.query(f'subject_id == {mouse_id}')['name'].iloc[0]
session_dir = os.path.join(mouse_dir, session_name)
nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
nwb_path = os.path.join(session_dir, nwb_file)
with NWBZarrIO(str(nwb_path), 'r') as io:
    nwbfile = io.read()
    print('Loaded NWB file from:', nwb_path)

#%% Load in test metric infor for structural data
struc_data = pd.read_pickle(pjoin(data_root, 'v1dd_1196/soma_nuc_v666_soma_nuc_feats_ZNORM_basic_size_filtered.pkl'))
