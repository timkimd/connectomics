#%%
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

#%% Initialize CAVEclient
client = CAVEclient("v1dd_public", 
                    server_address="https://global.em.brain.allentech.org", 
                    auth_token=os.environ["API_SECRET"])

#%% get metadata table
metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'V1DD_metadata.csv'))
metadata.head()

#%% get subject ids
subject_ids = np.sort(metadata['subject_id'].unique())
subject_id = subject_ids[0]
print('Selected subject_id is', subject_id)

#%% get one mouse metadata
this_mouse_metadata = metadata[metadata['subject_id']==subject_id].sort_values(by='session_date')
this_mouse_metadata

#%% get this mouse directory
data_dir = '/data/'
mouse_id = '409828'
mouse_dir = glob.glob((os.path.join(data_dir, mouse_id+'*')))[0]
mouse_dir