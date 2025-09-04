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
file_path = '/root/capsule/scratch/dff_psth_table.pkl'
with open(file_path, 'rb') as file:
    dff_psth_table = pickle.load(file)
dff_psth_table

#%% load metadata for RFs
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
#%% find the RFs amd windows from metadata, map root_ids to column, volume, roi, plane
sessions = pd.unique(stim_int_table.session_id)
cvpr_map = []
for session in sessions:
    cvpr = stim_int_table.loc[stim_int_table["session_id"] == session]
    column = (cvpr.column.values*(np.ones(cvpr.roi.values[0].shape))).astype(int)
    volume = (cvpr.volume.values*(np.ones(cvpr.roi.values[0].shape))).astype(int)
    plane = cvpr.plane.values[0]
    roi = cvpr.roi.values[0]
    cvpr_map.append({
        "session_id": session,
        "column": column,
        "volume": volume,
        "plane": plane, 
        "roi": roi
    }) 
cvpr_df = pd.DataFrame(cvpr_map)

#%%
all_sess_root_ids = []
for session in sessions:
    cvpr = cvpr_df.loc[cvpr_df["session_id"] == session]
    sess_cvpr = cvpr.explode(["column", "volume", "plane", "roi"])
    root_id = pd.merge(coreg_df_unq, sess_cvpr)
    all_sess_root_ids.append(root_id)
#%%
rf_cvpr_root_id = pd.concat(all_sess_root_ids)

#%%
all_sess_rf = pd.merge(rf_cvpr_root_id, rf_metadata, on=['column', 'volume', 'plane', 'roi'])

#%% need to sub_select position_metadata for golden mouse
all_sess_rf_windows = pd.merge(all_sess_rf, position_metadata_gold, on=['column', 'volume'], how='outer')

#%%
all_sess_rf_windows = all_sess_rf_windows[all_sess_rf_windows["has_rf_on_or_off"]==True]
all_sess_rf_windows['window_pos'] = all_sess_rf_windows['azi'].astype(str) + '_' + all_sess_rf_windows['alt'].astype(str)
all_sess_rf_windows['rf_on_pos'] = all_sess_rf_windows['altitude_rf_on'].astype(str) + '_' + all_sess_rf_windows['azimuth_rf_on'].astype(str)
all_sess_rf_windows['rf_off_pos'] = all_sess_rf_windows['altitude_rf_off'].astype(str) + '_' + all_sess_rf_windows['azimuth_rf_off'].astype(str)
#%%
all_sess_rf_windows["rf_in_window_on"] = (
    (all_sess_rf_windows["altitude_rf_on"] - all_sess_rf_windows["alt"])**2
  + (all_sess_rf_windows["azimuth_rf_on"]  - all_sess_rf_windows["azi"])**2
) <= 15**2

all_sess_rf_windows["rf_in_window_off"] = (
    (all_sess_rf_windows["altitude_rf_off"] - all_sess_rf_windows["alt"])**2
  + (all_sess_rf_windows["azimuth_rf_off"]  - all_sess_rf_windows["azi"])**2
) <= 15**2
#%%
rf_in_windows = all_sess_rf_windows.query('rf_in_window_off==True or rf_in_window_on==True')
rf_in_windows["pt_root_id"] = rf_in_windows["pt_root_id"].astype(int)
#%%
x = position_metadata.azi 
y = position_metadata.alt

x1 = rf_in_windows.azi
y1 = rf_in_windows.alt

x2 = rf_in_windows.azimuth_rf_on
y2 = rf_in_windows.altitude_rf_on

x3 = rf_in_windows.azimuth_rf_off
y3 = rf_in_windows.altitude_rf_off

plt.scatter(x,y, label='all windows')
plt.scatter(x1, y1, c='r', s=2**(30), alpha=0.003, label='golden windows')
plt.scatter(x2, y2, c='g', s=4, label='rf on')
plt.scatter(x3, y3, c='b', s=4, label='rf off')
plt.xlabel('azimuth')
plt.ylabel('altitude')


#%%
stim_min = stim_int_table[['session_id', 'column', 'volume', 'roi', 'plane']]
stim_dff = pd.merge(dff_psth_table, stim_min, how='outer', on='session_id')
stim_dff_df = pd.DataFrame(stim_dff)
out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "stim_dff_table.pkl")
stim_dff_df.to_pickle(out_path, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved to {out_path}")
#%%
sessions = stim_min.session_id.unique()
all_sess_prs = []
for session in stim_min.session_id.unique():
    sess = stim_dff_df.loc[stim_dff_df["session_id"] == session]
    sess_pr = sess.explode(["plane", "roi"]).reset_index(drop=True)

    # per-session index: 0,1,2,3,... up to len(planes)-1
    sess_pr["cell_idx"] = range(len(sess_pr))  # same as range(len(sess_pr["plane"]))

    all_sess_prs.append(sess_pr[["session_id", "plane", "roi", "cell_idx"]])

all_sess_prs_idx = pd.concat(all_sess_prs, ignore_index=True)

#%%
per_session = (
    all_sess_prs_idx
    .groupby("session_id", as_index=False)
    .agg({
        "plane": list,
        "roi": list,
        "cell_idx": list
    })
)

#%%

stim_dff_df_idx = pd.merge(stim_dff_df, per_session, on='session_id')










#%% 
sessions = pd.unique(stim_int_table.session_id)
stim_traces = []
for session in sessions:
    this_session = stim_int_table[stim_int_table['session_id']==session]
    base_times = this_session.base_time
    t = _to_1d_float_array(base_times) 
    dff_obj = this_session.dff.iloc[0]  

    stim_int_df = pd.DataFrame({
        "stim_name":  this_session.stim_name,
        "start_time": this_session.start_time,
        "stop_time":  this_session.stop_time,
    })
    stim_int_df = stim_int_df.explode(["stim_name", "start_time", "stop_time"], ignore_index=True)

    for stim_name in ['drifting_gratings_windowed' , 'drifting_gratings_full']:
        stim_sections = get_trial_sections(stim_int_df, stim_name)

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

#%%
traces_df = pd.DataFrame(stim_traces)
out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "dff_psth_table.pkl")
traces_df.to_pickle(out_path, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved to {out_path}")


#%%
nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
nwb_path = os.path.join(session_dir, nwb_file)
with NWBZarrIO(str(nwb_path), 'r') as io:
    nwbfile = io.read()

#%% get the shortest trial and snip all trials to that length

min_trial = min(dff.shape[1] for dff in dff_session.ttraces[0])
stacked = np.stack([dff[:,:min_trial] for dff in dff_session.ttraces[0]], axis =2) # get cells by time by trials
psth = stacked.mean(axis=2)
psth_chop_window = psth[:,3:]
psth_chop_mean_window = psth_chop.mean(axis=1)
#%%
min_trial = min(dff.shape[1] for dff in dff_session.ttraces[1])
stacked = np.stack([dff[:,:min_trial] for dff in dff_session.ttraces[1]], axis =2) # get cells by time by trials
psth = stacked.mean(axis=2)
psth_chop_full = psth[:,3:]
psth_chop_mean_full = psth_chop.mean(axis=1)

#%% plot psth for this session
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
#ax.set_xticks([0, 0.5, 1, 1.5, 2])
fig.colorbar(im, ax=ax, label='ΔF/F')
fig.tight_layout()

#%%
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
#ax.set_xticks([0, 0.5, 1, 1.5, 2])
fig.colorbar(im, ax=ax, label='ΔF/F')
fig.tight_layout()

















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

