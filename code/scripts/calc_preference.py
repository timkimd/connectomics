#%% Import blocks
import sys
import os
from os.path import join as pjoin
import platform
from caveclient import CAVEclient
import skeleton_plot as skelplot
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.sparse import csr_array
from typing import Union, Optional
import glob
from hdmf_zarr import NWBZarrIO
from nwbwidgets import nwb2widget

# Set the utils path
# utils_dir = pjoin(os.getcwd(), "code/utils")
# os.getcwd()

# Add utilities to path
# sys.path.insert(-1, utils_dir)

from utils import *
# from utils import filter_synapse_table, check_index, adjacencyplot

platstring = platform.platform()
system = platform.system()
if system == "Darwin":
    # macOS
    data_root = "/Volumes/Brain2025/"

#%% Import metadata and example session
metadata = pd.read_csv(pjoin(data_root, 'V1DD_metadata.csv'))
session_name = metadata['name'].iloc[0]
# data_dir = os.path.join(r'/data/v1', session_name)
mouse_id = '409828'
mouse_dir = glob.glob((os.path.join(data_root, mouse_id+'*')))[0]
nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
nwb_path = os.path.join(session_dir, nwb_file)

#%% Var exploration
