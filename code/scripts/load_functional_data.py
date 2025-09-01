#%% imports
import sys
import os
from os.path import join as pjoin
import platform
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import itertools
import seaborn as sns
from scipy.sparse import csr_array
from scipy import stats, spatial 
from typing import Union, Optional
import glob
import collections
from hdmf_zarr import NWBZarrIO
from nwbwidgets import nwb2widget

# import allen-specific functions
from allen_v1dd.client import OPhysClient, OPhysSession
from allen_v1dd.stimulus_analysis.drifting_gratings import DriftingGratings
from allen_v1dd.stimulus_analysis.natural_images import NaturalImages
from allen_v1dd.stimulus_analysis.natural_movie import NaturalMovie

# Set the utils path
utils_dir = pjoin("..", "utils")

# Add utilities to path
sys.path.append(utils_dir)
from data_io import *
from utils import filter_synapse_table, check_index, adjacencyplot
from data_io import _get_data_dir

#%% get metadata table
data_dir = '/data/'
metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'V1DD_metadata.csv'))
metadata.head()

#%% get subject ids
subject_ids = np.sort(metadata['subject_id'].unique())
subject_id = subject_ids[0]
print('Selected subject_id is', subject_id)

#%% get one mouse metadata
this_mouse_metadata = metadata[metadata['subject_id']==subject_id].sort_values(by='session_date')
this_mouse_metadata.head()

#%% get this mouse directory
data_dir = '/data/'
mouse_id = '409828'
mouse_dir = glob.glob((os.path.join(data_dir, mouse_id+'*')))[0]
mouse_dir

#%% get session
session_name = this_mouse_metadata.name.values[-1]
print('Selected session is', session_name)

#%% get session dir
session_dir = os.path.join(mouse_dir, session_name)
print(session_dir)

#%% get nwb file path
nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
nwb_path = os.path.join(session_dir, nwb_file)
print(nwb_path)

#%% open nwb file
with NWBZarrIO(str(nwb_path), 'r') as io:
    nwbfile = io.read()
    print('Loaded NWB file from:', nwb_path)

#%% view nwb file
nwbfile

#%% get the stimulus data for images
ophys_table = nwbfile.processing["plane-0"].data_interfaces["dff"]
ophys_table

#%% preprocess data function

