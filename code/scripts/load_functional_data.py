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

coreg_df = pd.read_feather(
    f"{data_dir}/metadata/coregistration_{mat_version}.feather"
)
coreg_df_unq = coreg_df.drop_duplicates(subset="pt_root_id")

#%% get the session list for one animal
def get_sessions(mouse_id, data_dir):
    mouse_dir = glob.glob((os.path.join(data_dir, mouse_id+'*')))[0]
    
    session_names = []
    _, session_names, _ = next(os.walk(mouse_dir))
    return sorted(session_names)

#%% preprocess data function
def interp_across_planes(nwbfile, column, volume, remove_known_bad_planes=True):
    planes = range(6)
    if remove_known_bad_planes:
        if column == 1 and volume == 5:
            planes = range(5)

    dff_traces = []
    planes_traces = []
    roi_traces = []
    column_traces = []
    volume_traces = []
    coreg_cvpr = {}
    base_times = nwbfile.processing["plane-0"].data_interfaces["dff"].timestamps
    for plane in planes:
        rois = nwbfile.processing[f"plane-{plane}"].data_interfaces["image_segmentation"]["roi_table"][:]
        good_rois = rois[rois.is_soma==True].roi.values
        print(f'good ROIs in {plane} = {len(good_rois)}')
        print(good_rois)
        traces_xarray = nwbfile.processing[f"plane-{plane}"].data_interfaces["dff"].data[:,good_rois]
        timestamps = nwbfile.processing[f"plane-{plane}"].data_interfaces["dff"].timestamps[:]
        f_interp = interpolate.interp1d(timestamps, traces_xarray, 
                            kind='linear', axis=0, bounds_error=False, fill_value="extrapolate")
        dff_traces.extend(f_interp(base_times).T)
        roi_traces.extend(good_rois)
        planes_traces.extend([plane]* len(good_rois))
        column_traces.extend([column]* len(good_rois))
        volume_traces.extend([volume]* len(good_rois))

    dff_traces = np.array(dff_traces)
    planes_traces = np.array(planes_traces)
    roi_traces = np.array(roi_traces)
    column_traces = np.array(column_traces)
    volume_traces = np.array(volume_traces)

    coreg_cvpr = {
        "column": column_traces,
        "volume": volume_traces,
        "plane": planes_traces,
        "roi": roi_traces,
    }

    return coreg_cvpr, {
        "dff": dff_traces,
        "plane": planes_traces,
        "roi": roi_traces,
        "base_time": base_times}

#%% pull stim information
def pull_stim_info(nwbfile):
    stim = nwbfile.epochs.to_dataframe().reset_index(drop=True)
    keep = ["stim_name", "start_time", "stop_time", "duration"]
    return stim[keep].copy()

#%% 
 
def pull_interval_info(nwbfile):
    stim_intervals = nwbfile.intervals["stimulus_table"].to_dataframe().reset_index(drop=True)
    keep_inters = ["stim_name", "start_time", "stop_time","temporal_frequency","spatial_frequency","direction","frame","image_order","image_index","stimulus_condition_id"]
    return stim_intervals[keep_inters].copy()

#%% loop over sessions
def pre_process(mouse_ids=None, sessions=None, data_dir = '/data/'):
    if mouse_ids is None:
        mouse_ids = ['409828']
    if sessions is not None and not isinstance(sessions, (list, tuple)):
            sessions = [sessions]

    cell_df = []
    coreg_root_ids = []
    for k, mouse in enumerate(tqdm.tqdm(mouse_ids, leave=False, desc='mouse_ids')):
        mouse_dir = glob.glob(os.path.join(data_dir, mouse + '*'))[0]
        use_sessions = get_sessions(mouse, data_dir) if sessions is None else list(sessions)
        print(f'use_sessions {use_sessions}')
        print(f'mouse: {mouse}')

        this_mouse_metadata = metadata[metadata['subject_id']==int(mouse)].sort_values(by='session_date')
        for i, session in enumerate(tqdm.tqdm(use_sessions, leave=False, desc='sessions')):
            column = this_mouse_metadata[metadata["name"]==session].column.values[0]
            volume = this_mouse_metadata[metadata["name"]==session].volume.values[0]
            print(f'column: {column}')
            print(f'volume: {volume}')
            session_dir = os.path.join(mouse_dir, session)

            nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
            nwb_path = os.path.join(session_dir, nwb_file)
            with NWBZarrIO(str(nwb_path), 'r') as io:
                nwbfile = io.read()

            stim_int_table = pull_interval_info(nwbfile)
            stim_int_cols = ["stim_name", "start_time", "stop_time","temporal_frequency","spatial_frequency","direction","frame","image_order","image_index","stimulus_condition_id"]
            stim_int_lists = {c: stim_int_table[c].tolist() for c in stim_int_cols}
            coreg_cvpr, interp = interp_across_planes(nwbfile, column, volume, remove_known_bad_planes=True)
            
            coreg_root_ids.append({
                **coreg_cvpr
                    })
            coreg_cvpr = pd.DataFrame(coreg_cvpr)
            coreg_cvpr_merge = pd.merge(coreg_cvpr, coreg_df)
            print(f'coreg merge: {coreg_cvpr_merge}')
            
            cell_df.append({
                    "mouse_id": mouse,
                    "session_id": session,
                    "column": column,
                    "volume": volume,
                    **interp,
                    **stim_int_lists
            })
            print(f'added {session} to cell_df')
    return pd.DataFrame(cell_df), coreg_cvpr_merge

#%%
df = pd.DataFrame(stim_interval_df) 
out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "stim_int_table.pkl")
df.to_pickle(out_path, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved to {out_path}")

#%% to load your pickle
file_path = '/root/capsule/scratch/stim_int_table.pkl'
with open(file_path, 'rb') as file:
    stim_int_table = pickle.load(file)
stim_int_table

#%% example for loading one session at a time for validation
session_dir = '/root/capsule/data/409828_V1DD_GoldenMouse/409828_2018-11-06_14-02-59_nwb_2025-08-08_16-27-52'
nwb_file = [file for file in os.listdir(session_dir) if 'nwb' in file][0]
nwb_path = os.path.join(session_dir, nwb_file)
with NWBZarrIO(str(nwb_path), 'r') as io:
    nwbfile = io.read()
