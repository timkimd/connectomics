#%% imports
import sys
import os
from os.path import join as pjoin
import platform
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import itertools
import seaborn as sns
from scipy.sparse import csr_array
from scipy import stats, spatial, interpolate 
from typing import Union, Optional
import glob
import collections
from hdmf_zarr import NWBZarrIO
from nwbwidgets import nwb2widget
import tqdm as tqdm
import pickle

# Set the utils path
utils_dir = pjoin("..", "utils")

# Add utilities to path
sys.path.append(utils_dir)
from data_io import *
from utils import filter_synapse_table, check_index, adjacencyplot
from data_io import _get_data_dir
#%%
# get metadata
data_dir = '/data/'
mat_version = 1196
golden_mouse =  409828
metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'V1DD_metadata.csv'))
rf_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'rf_metrics_M409828.csv'))
window_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'drifting_gratings_windowed_M409828.csv'))
ssi_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'surround_supression_index_M409828.csv'))
position_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'window_positions.csv'))
position_metadata_gold = position_metadata[position_metadata["mouse"] == golden_mouse]
rf_metadata['volume'] = pd.to_numeric(rf_metadata['volume'], errors='coerce').astype('Int64')
coreg_df = pd.read_feather(f"{data_dir}/metadata/coregistration_{mat_version}.feather")
coreg_df_unq = coreg_df.drop_duplicates(subset="pt_root_id")
e_to_i = pd.read_feather(f"{scratch_dir}/E_to_I.feather")
i_to_e = pd.read_feather(f"{scratch_dir}/I_to_E.feather")
struc_df = pd.read_feather(f"{scratch_dir}/structural_data.feather")
#%%
## setup figure aesthetics
mpl.rcParams.update({
    #'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': 'black',
    'savefig.dpi': 600,
    'figure.dpi': 300,
    'figure.figsize': (12, 4),
    'figure.constrained_layout.use': False,
    #'text.latex.preamble': r'\renewcommand{\familydefault}{\sfdefault} \usepackage[helvet]{sfmath}'
})

from matplotlib import rcParams, font_manager
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase, HandlerTuple
font_kwargs = {'fontsize': rcParams['font.size'], 'family': rcParams['font.sans-serif'][0]}

#%% plotting RFs and Window locations
fig, ax = plt.subplots()
ax.scatter(x, y, c='k', label='all windows')
ax.scatter(x1, y1, c='r', s=5, label='golden windows')
ax.scatter(x2, y2, c='g', s=4, label='rf on')
ax.scatter(x3, y3, c='b', s=4, label='rf off')

ax.set_xlim(-75, 75)
ax.set_ylim(-50, 50)
ax.set_xlabel('azimuth')
ax.set_ylabel('altitude')
ax.legend()
plt.show()
