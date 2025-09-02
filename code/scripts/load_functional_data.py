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

#%% get subject ids
subject_ids = np.sort(metadata['subject_id'].unique())
subject_id = subject_ids[0]

#%% get one mouse metadata
this_mouse_metadata = metadata[metadata['subject_id']==subject_id].sort_values(by='session_date')

#%% get this mouse directory
data_dir = '/data/'
mouse_id = str(subject_id)
mouse_dir = glob.glob((os.path.join(data_dir, mouse_id+'*')))[0]
mouse_dir

#%% get session
session_name = this_mouse_metadata.name.values[-1]
session_name

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

    base_times = nwbfile.processing["plane-0"].data_interfaces["dff"].timestamps
    for plane in planes:
        rois = nwbfile.processing[f"plane-{plane}"].data_interfaces["image_segmentation"]["roi_table"][:]
        good_rois = rois[rois.is_soma==True].roi.values
        print(f'good ROIs in {plane} = {len(good_rois)}')
        traces_xarray = nwbfile.processing[f"plane-{plane}"].data_interfaces["dff"].data[:,good_rois]
        timestamps = nwbfile.processing[f"plane-{plane}"].data_interfaces["dff"].timestamps[:]
        f_interp = interpolate.interp1d(timestamps, traces_xarray, 
                            kind='linear', axis=0, bounds_error=False, fill_value="extrapolate")
        dff_traces.extend(f_interp(base_times).T)
        roi_traces.extend(good_rois)
        planes_traces.extend([plane]* len(good_rois))

    dff_traces = np.array(dff_traces)
    planes_traces = np.array(planes_traces)
    roi_traces = np.array(roi_traces)
    
    return {
        "dff": dff_traces,
        "plane": planes_traces,
        "roi": roi_traces,
        "base_time": base_times}
#%%
col_vol = []
        for col in this_mouse_metadata["column"].unique():
            for vol in this_mouse_metadata.loc[this_mouse_metadata["column"] == col, "volume"].unique():
                if pd.isna(vol): 
                    continue
                else:
                    col_vol.append([col, vol])
#%% loop over sessions

def pre_process(mouse_ids=None, sessions=None, data_dir = '/data/'):
    if mouse_ids is None:
        mouse_ids = ['409828']
    if sessions is not None and not isinstance(sessions, (list, tuple)):
            sessions = [sessions]

    cell_df = []
    
    for k, mouse in enumerate(tqdm.tqdm(mouse_ids, leave=False, desc='mouse_ids')):
        mouse_dir = glob.glob(os.path.join(data_dir, mouse + '*'))[0]
        use_sessions = get_sessions(mouse, data_dir) if sessions is None else list(sessions)
        print(f'use_sessions {use_sessions}')
        print(f'mouse: {mouse}')

        this_mouse_metadata = metadata[metadata['subject_id']==int(mouse)].sort_values(by='session_date')
        # for column, volume in tqdm.tqdm(col_vol, desc ="col/vol"):
        #     print(f'col_vol = {column}, {volume}')
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
            print(f'loaded and running interp on {nwb_path}')
            interp = interp_across_planes(nwbfile, column, volume, remove_known_bad_planes=True)
            cell_df.append({
                    "mouse_id": mouse,
                    "session_id": session,
                    "column": column,
                    "volume": volume,
                    **interp,
            })
            print(f'added {session} to cell_df')
    return cell_df

#%%
df = pd.DataFrame(cell_df)
df.to_pickle('cell_table.pkl') 
    