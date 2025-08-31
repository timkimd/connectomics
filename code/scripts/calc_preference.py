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
utils_dir = pjoin("..", "utils")

# Add utilities to path
sys.path.append(utils_dir)
from data_io import *
from utils import filter_synapse_table, check_index, adjacencyplot
from data_io import _get_data_dir

#%% Import metadata and example session
metadata = pd.read_csv('/data/metadata/V1DD_metadata.csv')
session_name = metadata['name'].iloc[0]
data_dir = os.path.join(r'/data/v1', session_name)

#%% Var exploration