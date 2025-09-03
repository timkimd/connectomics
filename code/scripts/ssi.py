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

# get metadata
data_dir = '/data/'
metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'V1DD_metadata.csv'))

#%% load stim_int_df
file_path = '/root/capsule/scratch/stim_int_table.pkl'
with open(file_path, 'rb') as file:
    stim_int_table = pickle.load(file)
stim_int_table

#%%
session = '409828_2018-11-06_14-02-59_nwb_2025-08-08_16-27-52'
this_session = stim_int_table[stim_int_table['session_id']==session]
base_times = this_session.base_time
#%%
stim_int_df = pd.DataFrame({
    "stim_name":  this_session.stim_name,
    "start_time": this_session.start_time,
    "stop_time":  this_session.stop_time,
})
stim_int_df = stim_int_df.explode(["stim_name", "start_time", "stop_time"], ignore_index=True)

#%%
def get_stimulus_sections(stim_int_df, threshold=1.1):
    starts = stim_int_df["start_time"].values
    ends = stim_int_df["stop_time"].values
    
    # Compute gaps between consecutive end/start
    gaps = starts[1:] - ends[:-1]
    
    # Find where gaps exceed threshold → discontinuity
    discontinuity_indices = np.where(gaps >= threshold)[0] + 1
    
    # Add start and end anchors
    split_indices = np.concatenate(([0], discontinuity_indices, [len(stim_int_df)]))
    
    # Collect sections
    sections = [(starts[i], ends[j - 1]) for i, j in zip(split_indices[:-1], split_indices[1:])]
    return sections

#%%
def _to_1d_float_array(base_times):
    # pandas Series to numpy array
    if isinstance(base_times, (pd.Series, pd.Index)):
        if base_times.dtype == object:
            # flatten lists/arrays inside the Series
            base_times = np.concatenate([np.asarray(x).ravel() for x in base_times.tolist()])
        else:
            base_times = base_times.to_numpy()
    else:
        base_times = np.asarray(base_times)
    return base_times.astype(float, copy=False)

def get_stimulus_mask(sections, base_times):
    t = _to_1d_float_array(base_times)          
    mask = np.zeros(t.shape[0], dtype=bool)
    for start, end in sections:
        mask |= (t >= float(start)) & (t <= float(end))
    return mask

trial_sections = [(s, e) for s, e in zip(stim_int_df["start_time"], stim_int_df["stop_time"])]
stim = pd.unique(stim_int_df.stim_name)
#%%
def get_trial_sections(stim_int_df, stim_name):
    """Return (start, stop) sections for all trials of a given stim_name."""
    df = stim_int_df.loc[stim_int_df["stim_name"] == stim_name]
    return list(zip(df["start_time"], df["stop_time"]))

#%%
def get_trial_mask_from_t(t, trial):
    start, end = map(float, trial)
    return (t >= start) & (t <= end)

#%%
def get_trial_indices(base_times, trial):
    t = _to_1d_float_array(base_times)
    start, end = map(float, trial)
    mask = (t >= start) & (t <= end)
    return np.where(mask)[0]
#%% chop dff for every trial, then take an average of 
# the response during a trial type

t = _to_1d_float_array(base_times)  # compute once
dff_obj = this_session.dff.iloc[0] 
stim_name = 'drifting_gratings_windowed'
for stim_name in stim_int_df["stim_name"].unique():
    stim_sections = get_trial_sections(stim_int_df, stim_name)

#%%
trial_dff_traces = []
for trial in stim_sections:
    mask = get_trial_mask_from_t(t, trial)
    trial_times = t[mask]
    idx = get_trial_indices(base_times, trial)
    trial_dff_traces.append(dff_obj[:,idx])
#%% get the shortest trial and snip all trials to that length
min_trial = min(dff.shape[1] for dff in trial_dff_traces)
stacked = np.stack([dff[:,:min_trial] for dff in trial_dff_traces], axis =2) # get cells by time by trials
psth = stacked.mean(axis=2)

#%% plot psth for this session
fig, ax = plt.subplots(figsize=(5, 10))
n_bins = psth.shape[1]
fq = 6
trial_dur = n_bins / fq
time_axis = np.arange(n_bins) / fq 
ax.imshow(psth, aspect='auto', 
                origin='lower',
                interpolation='none', 
                extent=[time_axis[0], 
                time_axis[-1], 0, psth.shape[0]])
ax.set_xlabel('Time (s)',fontsize=20)
ax.set_ylabel('Neurons',fontsize=20)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
fig.colorbar(i, ax=ax, label='ΔF/F')
fig.tight_layout()

#%%
nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
nwb_path = os.path.join(session_dir, nwb_file)
with NWBZarrIO(str(nwb_path), 'r') as io:
    nwbfile = io.read()

















# #%%
# def get_combined_stimulus_map(stim_int_df, session):
#     stim_list = pd.unique(stim_int_df.stim_name)
#     stim_mask = np.zeros(len(session_data["base_time"]), dtype=int)
#     stim_map = {}
#     for i_stim, stim in enumerate(session):
#         stim_table = get_stimulus_table(stim)[0]
#         single_stim_mask = get_stimulus_mask(get_stimulus_sections(stim_table), session_data["base_time"])
#         stim_mask[single_stim_mask] = i_stim + 1
#         stim_map[i_stim + 1] = stim

#     return stim_map, stim_mask

