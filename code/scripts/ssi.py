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
scratch_dir = '/scratch/'
mat_version = 1196
metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'V1DD_metadata.csv'))
rf_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'rf_metrics_M409828.csv'))
window_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'drifting_gratings_windowed_M409828.csv'))
ssi_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'surround_supression_index_M409828.csv'))
position_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'window_positions.csv'))
golden_mouse =  409828
position_metadata_gold = position_metadata[position_metadata["mouse"] == golden_mouse]
rf_metadata['volume'] = pd.to_numeric(rf_metadata['volume'], errors='coerce').astype('Int64')
coreg_df = pd.read_feather(
    f"{data_dir}/metadata/coregistration_{mat_version}.feather"
)
coreg_df_unq = coreg_df.drop_duplicates(subset="pt_root_id")
#%% load coreg_stim_int
file_path = '/root/capsule/scratch/coreg_stim_int.pkl'
with open(file_path, 'rb') as file:
    coreg_stim_int = pickle.load(file)
coreg_stim_int
# remove empty sessions 
coreg_stim_int = coreg_stim_int[coreg_stim_int['pt_root_id'].apply(lambda x: len(x) > 0)]

#%% load 
file_path = '/root/capsule/scratch/coreg_dff_psth_table.pkl'
with open(file_path, 'rb') as file:
    coreg_dff_psth_table = pickle.load(file)
coreg_dff_psth_table

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

#%%
sessions = pd.unique(coreg_stim_int.session_id)
stim_traces = []
for session in sessions:
    this_session = coreg_stim_int[coreg_stim_int['session_id']==session]
    base_times = this_session.base_time
    t = _to_1d_float_array(base_times) 
    dff_obj = this_session.dff.iloc[0]  

    coreg_stim_df = pd.DataFrame({
        "stim_name":  this_session.stim_name,
        "start_time": this_session.start_time,
        "stop_time":  this_session.stop_time,
    })
    coreg_stim_df = coreg_stim_df.explode(["stim_name", "start_time", "stop_time"], ignore_index=True)

    for stim_name in ['drifting_gratings_windowed' , 'drifting_gratings_full']:
        stim_sections = get_trial_sections(coreg_stim_df, stim_name)

        trial_dff_traces = []
        for trial in stim_sections:
            mask = get_trial_mask_from_t(t, trial)
            trial_times = t[mask]
            idx = np.where(mask)[0]  
            #idx = get_trial_indices(base_times, trial)
            trial_dff_traces.append(dff_obj[:,idx])

        # make PSTH and get mean over trial
        min_trial = min(dff.shape[1] for dff in trial_dff_traces)
        stacked = np.stack([dff[:,:min_trial] for dff in trial_dff_traces], axis =2) # get cells by time by trials
        psth = stacked.mean(axis=2)
        psth_chop = psth[:,3:]
        psth_chop_mean = psth_chop.mean(axis=1)

        stim_traces.append({
            "session_id": session,
            "stimulus": stim_name,
            "ttraces": trial_dff_traces,
            "psth": psth,
            "psth_chop": psth_chop,
            "mean_chop": psth_chop_mean
        })

#%% to save the output of the for loop above
coreg_traces_df = pd.DataFrame(stim_traces)
out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "coreg_dff_psth_table.pkl")
coreg_traces_df.to_pickle(out_path, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved to {out_path}")

#%% merge the dff mean traces with the metadata
big_table = pd.merge(coreg_stim_int, coreg_dff_psth_table, how='outer', on='session_id')
cell_table = big_table[['session_id', 'pt_root_id', 'stimulus', 'mean_chop', 'plane', 'roi']]

for n in range(len(cell_table)):
    cell_table.at[n, 'stimulus'] = np.repeat(
        cell_table.at[n, 'stimulus'],
        len(cell_table.at[n, 'pt_root_id']))
cell_table = pd.DataFrame(cell_table)

#%% find the RFs amd windows from metadata, map root_ids to column, volume, roi, plane
sessions = pd.unique(coreg_stim_int.session_id)
cvpr_map = []
for session in sessions:
    cvpr = coreg_stim_int.loc[coreg_stim_int["session_id"] == session]
    column = (cvpr.column.values*(np.ones(cvpr.roi.values[0].shape))).astype(int)
    volume = (cvpr.volume.values*(np.ones(cvpr.roi.values[0].shape))).astype(int)
    plane = cvpr.plane.values[0]
    roi = cvpr.roi.values[0]
    cvpr_map.append({
        "session_id": session,
        "column": column,
        "volume": volume,
        #"plane": plane, 
        #"roi": roi
    }) 
cvpr_df = pd.DataFrame(cvpr_map)
#%%
cell_table = pd.merge(cell_table, cvpr_df, on='session_id')
#%%
for n in range(len(cell_table)):
    cell_table.at[n, 'session_id'] = np.repeat(
        cell_table.at[n, 'session_id'],
        len(cell_table.at[n, 'pt_root_id']))
cell_table = pd.DataFrame(cell_table)
#%%
coreg_sess_cells = cell_table.explode(
    ["session_id", "column", "volume", "pt_root_id", 
     "stimulus", "mean_chop", "plane", "roi"]
)

#%%
cell_rf = pd.merge(coreg_sess_cells, rf_metadata, on=['column', 'volume', 'plane', 'roi'])

#%% need to sub_select position_metadata for golden mouse, drop nans
cell_rf_windows = pd.merge(cell_rf, position_metadata_gold, on=['column', 'volume'], how='outer')
cell_rf_windows = cell_rf_windows.dropna(subset=['session_id'])
# save here for non ssi but all coreg, use these to compare to ssi cells

out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "cell_coreg.feather")
coreg_feather = cell_rf_windows.to_feather(out_path)
print(f"Saved to {out_path}")

#%% to read
cell_coreg_df = pd.read_feather(out_path)

#%%
cell_rf_windows = cell_rf_windows[cell_rf_windows["has_rf_on_or_off"]==True]
cell_rf_windows['window_pos'] = cell_rf_windows['azi'].astype(str) + '_' + cell_rf_windows['alt'].astype(str)
cell_rf_windows['rf_on_pos'] = cell_rf_windows['altitude_rf_on'].astype(str) + '_' + cell_rf_windows['azimuth_rf_on'].astype(str)
cell_rf_windows['rf_off_pos'] = cell_rf_windows['altitude_rf_off'].astype(str) + '_' + cell_rf_windows['azimuth_rf_off'].astype(str)
#%% check if RFs are in the windows
cell_rf_windows["rf_in_window_on"] = (
    (cell_rf_windows["altitude_rf_on"] - cell_rf_windows["alt"])**2
  + (cell_rf_windows["azimuth_rf_on"]  - cell_rf_windows["azi"])**2
) <= 15**2

cell_rf_windows["rf_in_window_off"] = (
    (cell_rf_windows["altitude_rf_off"] - cell_rf_windows["alt"])**2
  + (cell_rf_windows["azimuth_rf_off"]  - cell_rf_windows["azi"])**2
) <= 15**2

#%% get SSI
rows = []
for pt_root_id, g in cell_rf_windows.groupby('pt_root_id'):
    # pick the rows for each stimulus within this neuron's group
    w = g.loc[g['stimulus'] == 'drifting_gratings_windowed', 'mean_chop']
    f = g.loc[g['stimulus'] == 'drifting_gratings_full', 'mean_chop']
    if w.empty or f.empty:
        continue  # skip if one stimulus is missing
    window = w.iloc[0]
    full = f.iloc[0]
    denom = window + full
    ssi = (window - full) / denom if denom != 0 else np.nan
    rows.append({'pt_root_id': pt_root_id, 'ssi': ssi})

ssi_df = pd.DataFrame(rows)

#%%
base_df = cell_rf_windows.drop(columns=['stimulus']).drop_duplicates('pt_root_id')
cell_ssi = pd.merge(base_df, ssi_df, on='pt_root_id', how='inner')
#%% save feather
out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "cell_ssi.feather")
ssi_feather = cell_ssi.to_feather(out_path)
print(f"Saved to {out_path}")
cell_ssi_df = pd.read_feather(out_path)

#%% plotting RFs and Window locations
x = position_metadata.azi 
y = position_metadata.alt

x1 = cell_ssi_df.azi
y1 = cell_ssi_df.alt

x2 = cell_ssi_df.azimuth_rf_on
y2 = cell_ssi_df.altitude_rf_on

x3 = cell_ssi_df.azimuth_rf_off
y3 = cell_ssi_df.altitude_rf_off

plt.scatter(x,y,c='k', label='all windows')
plt.scatter(x1, y1, c='r', s=5, label='golden windows')
plt.scatter(x2, y2, c='g', s=4, label='rf on')
plt.scatter(x3, y3, c='b', s=4, label='rf off')
plt.xlim(-75, 75)
plt.ylim(-50, 50)
plt.xlabel('azimuth')
plt.ylabel('altitude')


#%% saving big table with stim and psth info 
stim_min = stim_int_table[['session_id', 'column', 'volume', 'roi', 'plane']]
stim_dff = pd.merge(dff_psth_table, stim_min, how='outer', on='session_id')
stim_dff_df = pd.DataFrame(stim_dff)
out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "stim_dff_table.pkl")
stim_dff_df.to_pickle(out_path, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved to {out_path}")

#%% example for loading single nwb when testing
nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
nwb_path = os.path.join(session_dir, nwb_file)
with NWBZarrIO(str(nwb_path), 'r') as io:
    nwbfile = io.read()

#%% plot psth for one session
fig, ax = plt.subplots(figsize=(5, 10))
n_bins = psth_chop_window.shape[1]
fq = 6
trial_dur = n_bins / fq
time_axis = np.arange(n_bins) / fq 
im = ax.imshow(psth_chop_window, aspect='auto', 
                origin='lower',
                interpolation='none', 
                extent=[time_axis[0], 
                time_axis[-1], 0, psth_chop_window.shape[0]])

ax.set_xlabel('Time (s)',fontsize=20)
ax.set_ylabel('Neurons',fontsize=20)
ax.set_title('window')
fig.colorbar(im, ax=ax, label='ΔF/F')
fig.tight_layout()

#%% plotting psth
index = (psth_chop_full - psth_chop_window) 
fig, ax = plt.subplots(figsize=(5, 10))
n_bins = index.shape[1]
fq = 6
trial_dur = n_bins / fq
time_axis = np.arange(n_bins) / fq 
im = ax.imshow(index, aspect='auto', 
                origin='lower',
                interpolation='none', 
                extent=[time_axis[0], 
                time_axis[-1], 0, index.shape[0]])

ax.set_xlabel('Time (s)',fontsize=20)
ax.set_ylabel('Neurons',fontsize=20)
ax.set_title('full')
fig.colorbar(im, ax=ax, label='ΔF/F')
fig.tight_layout()