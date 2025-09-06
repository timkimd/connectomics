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
# sys.path.append(utils_dir)
# from data_io import *
# from utils import filter_synapse_table, check_index, adjacencyplot
# from data_io import _get_data_dir

# get metadata
data_dir = '/data/'
scratch_dir = '/scratch/'
mat_version = 1196
metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'V1DD_metadata.csv'))
rf_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'rf_metrics_M409828.csv'))
window_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'drifting_gratings_windowed_M409828.csv'))
full_metadata = pd.read_csv(os.path.join(data_dir, 'metadata', 'drifting_gratings_full_M409828.csv'))
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
file_path = '/root/capsule/scratch/coreg_dff_psth_dir_table.pkl'
with open(file_path, 'rb') as file:
    coreg_dff_psth_dir_table = pickle.load(file)
coreg_dff_psth_dir_table

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
def get_trial_sections(pairs, combined_name):
    """Return (start, stop) sections for all trials of a given stim_name."""
    df = pairs.loc[pairs["combined_name"] == combined_name]
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
# which drifting stimuli we care about
DG_STIMS = ['drifting_gratings_windowed', 'drifting_gratings_full']

sessions = pd.unique(coreg_stim_int.session_id)
stim_traces = []

for session in sessions:
    this_session = coreg_stim_int[coreg_stim_int['session_id'] == session]
    base_times = this_session.base_time
    t = _to_1d_float_array(base_times)
    dff_obj = this_session.dff.iloc[0]

    # explode trials per-row
    coreg_stim_df = pd.DataFrame({
        "stim_name":  this_session.stim_name,
        "start_time": this_session.start_time,
        "stop_time":  this_session.stop_time,
        "direction":  this_session.direction,
    }).explode(["stim_name", "start_time", "stop_time", "direction"], ignore_index=True)

    # only keep the two drifting-gratings families; then get unique (stim_name, direction) pairs
    pairs = (coreg_stim_df.loc[coreg_stim_df["stim_name"].isin(DG_STIMS)])
    pairs["combined_name"] = pairs["stim_name"].astype(str) + "_dir" + pairs["direction"].astype(str)

    for _, row in pairs.iterrows():
        stim_name      = row["stim_name"]       # original family
        direction      = row["direction"]       # direction value
        combined_name  = row["combined_name"]   # e.g. drifting_gratings_full_dir

        stim_sections = get_trial_sections(pairs, combined_name)

        trial_dff_traces = []
        for trial in stim_sections:
            mask = get_trial_mask_from_t(t, trial)
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            trial_dff_traces.append(dff_obj[:, idx])

        # PSTH over trials (truncate to shortest trial)
        min_trial = min(dff.shape[1] for dff in trial_dff_traces)
        stacked = np.stack([dff[:, :min_trial] for dff in trial_dff_traces], axis=2)  # cells x time x trials
        psth = stacked.mean(axis=2)
        psth_chop = psth[:, 3:]  # chop the first three windows where the cell is ramping up
        psth_chop_mean = psth_chop.mean(axis=1)

        # store fields to make “matching by direction” easy later:
        stim_traces.append({
                "session_id": session,
                "combined_name": combined_name,   # now unique stim+direction
                "stim_name": stim_name,      # keep the family name too if you want
                "direction": direction,
                "ttraces": trial_dff_traces,
                "psth": psth,
                "psth_chop": psth_chop,
                "mean_chop": psth_chop_mean,
            })

#%% to save the output of the for loop above
coreg_dff_psth_dir_table = pd.DataFrame(stim_traces)
out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "coreg_dff_psth_dir_table.pkl")
coreg_dff_psth_dir_table.to_pickle(out_path, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved to {out_path}")
coreg_dff_psth_dir_table

#%% merge the dff mean traces with the metadata
coreg_stim_int_no_stim = coreg_stim_int.drop(columns=['stim_name', 'direction'])
big_table = pd.merge(coreg_stim_int_no_stim, coreg_dff_psth_dir_table, on='session_id', how='outer')

#%%
cell_table = big_table[['session_id', 'pt_root_id', 'combined_name', 'mean_chop', 'plane', 'roi']]

for n in range(len(cell_table)):
    cell_table.at[n, 'combined_name'] = np.repeat(
        cell_table.at[n, 'combined_name'],
        len(cell_table.at[n, 'pt_root_id']))
cell_table = pd.DataFrame(cell_table)

#%% find the RFs amd windows from metadata, map root_ids to column, volume, roi, plane
sessions = pd.unique(coreg_stim_int_no_stim.session_id)
cvpr_map = []
for session in sessions:
    cvpr = coreg_stim_int_no_stim.loc[coreg_stim_int_no_stim["session_id"] == session]
    column = (cvpr.column.values*(np.ones(cvpr.roi.values[0].shape))).astype(int)
    volume = (cvpr.volume.values*(np.ones(cvpr.roi.values[0].shape))).astype(int)
    plane = cvpr.plane.values[0]
    roi = cvpr.roi.values[0]
    cvpr_map.append({
        "session_id": session,
        "column": column,
        "volume": volume,
    }) 
cvpr_df = pd.DataFrame(cvpr_map)
#%%
cell_table = pd.merge(cell_table, cvpr_df, on='session_id', how='outer')
#%%
for n in range(len(cell_table)):
    cell_table.at[n, 'session_id'] = np.repeat(
        cell_table.at[n, 'session_id'],
        len(cell_table.at[n, 'pt_root_id']))
cell_table = pd.DataFrame(cell_table)

#%%
coreg_sess_cells = cell_table.explode(
    ["session_id", "column", "volume", "pt_root_id", 
     "combined_name", "mean_chop", "plane", "roi"]
)
# this is 210816 rows: cells x 2 stims x 13 dirs x trials (average 14.77 trials per unique dir_sim combo)

#%%
rf_metadata = rf_metadata.drop(columns=['mouse', 'roi_unique_id'])
#%% its called cell_rf but actually its cells x stims x dirs x trials
cell_rf = pd.merge(coreg_sess_cells, rf_metadata, on=['column', 'volume', 'plane', 'roi'], how='inner')

#%% need to sub_select position_metadata for golden mouse, drop nans

cell_rf_windows = pd.merge(cell_rf, position_metadata_gold, on=['column', 'volume'], how='left')

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
# make strings of unique windows and rfs
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

#%%################################### 
window_metadata = window_metadata.drop(columns=['mouse', 'roi_unique_id'])
window_metadata["volume"] = pd.to_numeric(
    window_metadata["volume"], errors="coerce"
).astype("Int64")

pref_dir_cell_df = pd.merge(cell_rf_windows, window_metadata, on=['column', 'volume', 'plane', 'roi'], how='left')

#%%
## take into account pref direction
## add cross dir info
## add CMI and change ss calculation
## make table for pref dir and for cross dir

def wrap360(a):  # makes all values in the same coordinate space
    return np.mod(a, 360.0)

#%% extract dir and stim
#### fix bug
stim_dir_arr = np.empty(len(pref_dir_cell_df))
stim_arr = np.empty(len(pref_dir_cell_df), dtype='object')
for i, rows in pref_dir_cell_df.iterrows():
    my_str = rows['combined_name']
    expl_str = my_str.split('_')
    stim_dir = expl_str[-1][3:]
    stim = '_'.join(expl_str[:-1])
    stim_dir_arr[i] = stim_dir
    stim_arr[i] = stim

pref_dir_cell_df['stim_dir'] = stim_dir_arr
pref_dir_cell_df['stim'] = stim_arr

#%%

pref_dir_cell_df['cross_dir_neg'] = np.mod((pref_dir_cell_df['preferred_dir'] - 90), 360.0)
pref_dir_cell_df['cross_dir_pos'] = np.mod((pref_dir_cell_df['preferred_dir'] + 90), 360.0)

iso_df = pref_dir_cell_df[iso_df = pref_dir_cell_df[pref_dir_cell_df['stim_dir'] == pref_dir_cell_df['preferred_dir']] == pref_dir_cell_df['preferred_dir']]
cross_neg_df = pref_dir_cell_df[pref_dir_cell_df['cross_dir_neg'] == pref_dir_cell_df['stim_dir']]
cross_pos_df = pref_dir_cell_df[pref_dir_cell_df['cross_dir_pos'] == pref_dir_cell_df['stim_dir']]

#%% calculate surround suppression iso

iso = iso_df.groupby(['pt_root_id', 'stim'])['mean_chop'].mean()
iso_full = iso.xs('drifting_gratings_full', level='stim')
iso_window = iso.xs('drifting_gratings_windowed', level='stim')
iso_ssi_calc = (iso_window - iso_full) / (iso_window + iso_full)

#%% calculate surround suppression cross neg

cross_neg = cross_neg_df.groupby(['pt_root_id', 'stim'])['mean_chop'].mean()
cross_neg_full = cross_neg.xs('drifting_gratings_full', level='stim')
cross_neg_window = cross_neg.xs('drifting_gratings_windowed', level='stim')
cross_neg_calc = (cross_neg_window - cross_neg_full) / (cross_neg_window + cross_neg_full)

#%% calculate surround suppression cross pos

cross_pos = cross_pos_df.groupby(['pt_root_id', 'stim'])['mean_chop'].mean()
cross_pos_full = cross_pos.xs('drifting_gratings_full', level='stim')
cross_pos_window = cross_pos.xs('drifting_gratings_windowed', level='stim')
cross_pos_calc = (cross_pos_window - cross_pos_full) / (cross_pos_window + cross_pos_full)

#%% make final feather

iso_ssi = iso_ssi_calc.to_frame().rename(columns={'mean_chop': 'iso_ssi'})
cross_neg_ssi = cross_neg_calc.to_frame().rename(columns={'mean_chop': 'cross_neg_ssi'})
cross_pos_ssi = cross_pos_calc.to_frame().rename(columns={'mean_chop': 'cross_pos_ssi'})

iso_cross_neg = pd.merge(iso_ssi, cross_neg_ssi, on='pt_root_id', how='outer')
iso_cross_neg_pos = pd.merge(iso_cross_neg, cross_pos_ssi, on='pt_root_id', how='outer')

cell_df = pref_dir_cell_df[['pt_root_id', 'plane', 'roi', 'column', 'volume', 'dsi',
                            'frac_responsive_trials', 'is_responsive',
                            'lifetime_sparseness', 'osi', 'preferred_dir', 'preferred_sf',
                            'pref_dir_mean']]

cell_ssi = pd.merge(cell_df, iso_cross_neg_pos, on=['pt_root_id'], how='inner')















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