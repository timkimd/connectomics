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

#%%
def get_sessions(mouse_id, data_dir):
    mouse_dir = glob.glob((os.path.join(data_dir, mouse_id+'*')))[0]
    
    session_names = []
    _, session_names, _ = next(os.walk(mouse_dir))
    return sorted(session_names)

#%%
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

    base_times = nwbfile.processing["plane-0"].data_interfaces["dff"].timestamps
    for plane in planes:
        rois = nwbfile.processing[f"plane-{plane}"].data_interfaces["image_segmentation"]["roi_table"][:]
        good_rois = rois[rois.is_soma == True].roi.values

        traces = nwbfile.processing[f"plane-{plane}"].data_interfaces["dff"].data[:, good_rois]
        timestamps = nwbfile.processing[f"plane-{plane}"].data_interfaces["dff"].timestamps[:]
        f_interp = interpolate.interp1d(
            timestamps, traces, kind='linear', axis=0, bounds_error=False, fill_value="extrapolate"
        )
        # f_interp(base_times) -> (T, cells); we want (cells, T)
        dff_traces.extend(f_interp(base_times).T)

        roi_traces.extend(good_rois)
        planes_traces.extend([plane] * len(good_rois))
        column_traces.extend([column] * len(good_rois))
        volume_traces.extend([volume] * len(good_rois))

    dff_traces  = np.array(dff_traces)           # (n_cells, T)
    planes_arr  = np.array(planes_traces)
    roi_arr     = np.array(roi_traces)
    column_arr  = np.array(column_traces)
    volume_arr  = np.array(volume_traces)

    # one row per cell for joining
    coreg_index_df = pd.DataFrame({
        "column": column_arr,
        "volume": volume_arr,
        "plane":  planes_arr,
        "roi":    roi_arr,
    })
    coreg_index_df["row_idx"] = np.arange(len(coreg_index_df), dtype=int)

    return coreg_index_df, {
        "dff": dff_traces,          # (cells, T)
        "plane": planes_arr,        # (cells,)
        "roi": roi_arr,             # (cells,)
        "base_time": base_times
    }

#%%
def pull_interval_info(nwbfile):
    stim_intervals = nwbfile.intervals["stimulus_table"].to_dataframe().reset_index(drop=True)
    keep = ["stim_name","start_time","stop_time","temporal_frequency","spatial_frequency",
            "direction","frame","image_order","image_index","stimulus_condition_id"]
    return stim_intervals[keep].copy()

#%%

def pre_process(mouse_ids=None, sessions=None, coreg_df=coreg_df, data_dir='/data/'):
    if mouse_ids is None:
        mouse_ids = ['409828']
    if sessions is not None and not isinstance(sessions, (list, tuple)):
        sessions = [sessions]

    # load metadata/coreg once
    meta_path = os.path.join(data_dir, 'metadata', 'V1DD_metadata.csv')
    metadata = pd.read_csv(meta_path)

    # keep only needed columns + ensure dtypes line up
    key_cols = ['column','volume','plane','roi']
    root_col = 'pt_root_id'
    coreg_df = coreg_df[key_cols + [root_col]].copy()
    # (optional) enforce ints, in case feather loaded them as floats
    for c in key_cols:
        coreg_df[c] = coreg_df[c].astype(int)

    # outputs
    cell_df_records = []        # your original per-session records (unfiltered)
    coreg_cell_records = []     # per-session, filtered to matched cells and includes root_id

    for mouse in tqdm.tqdm(mouse_ids, leave=False, desc='mouse_ids'):
        mouse_dir = glob.glob(os.path.join(data_dir, mouse + '*'))[0]
        use_sessions = get_sessions(mouse, data_dir) if sessions is None else list(sessions)

        this_mouse_metadata = metadata[metadata['subject_id'] == int(mouse)].sort_values(by='session_date')

        for session in tqdm.tqdm(use_sessions, leave=False, desc='sessions'):
            # not sure about not indexing metadata (in lieu of this_mouse_metadata)
            row = this_mouse_metadata[this_mouse_metadata["name"] == session].iloc[0]
            column = int(row.column)
            volume = int(row.volume)

            session_dir = os.path.join(mouse_dir, session)
            nwb_file = [f for f in os.listdir(session_dir) if f.endswith('.nwb') or '.nwb' in f][0]
            nwb_path = os.path.join(session_dir, nwb_file)
            with NWBZarrIO(str(nwb_path), 'r') as io:
                nwbfile = io.read()

            stim_int_table = pull_interval_info(nwbfile)
            stim_int_cols = ["stim_name","start_time","stop_time","temporal_frequency","spatial_frequency",
                             "direction","frame","image_order","image_index","stimulus_condition_id"]
            stim_int_lists = {c: stim_int_table[c].tolist() for c in stim_int_cols}

            coreg_index_df, interp = interp_across_planes(nwbfile, column, volume, remove_known_bad_planes=True)

            # unfiltered 
            cell_df_records.append({
                "mouse_id": mouse,
                "session_id": session,
                "column": column,
                "volume": volume,
                **interp,            # dff, plane, roi, base_time
                **stim_int_lists
            })

            # coreg-only
            left = coreg_index_df.copy()
            for c in key_cols:
                left[c] = left[c].astype(int)

            merged = left.merge(coreg_df, on=key_cols, how='inner')  # keep only matched cells
            if merged.empty:
                # make an entry with empty arrays, (only 2 volumes coreg)
                coreg_cell_records.append({
                    "mouse_id": mouse,
                    "session_id": session,
                    "column": column,
                    "volume": volume,
                    "pt_root_id": np.array([], dtype=int),
                    "dff": interp["dff"][[], :],          # (0, T)
                    "plane": np.array([], dtype=int),
                    "roi": np.array([], dtype=int),
                    "base_time": interp["base_time"],
                    **{k: [] for k in stim_int_cols}
                })
                continue

            sel = merged["row_idx"].to_numpy()
            dff_coreg  = interp["dff"][sel, :]
            plane_coreg= interp["plane"][sel]
            roi_coreg  = interp["roi"][sel]
            root_ids   = merged[root_col].to_numpy()

            coreg_cell_records.append({
                "mouse_id": mouse,
                "session_id": session,
                "column": column,
                "volume": volume,
                "pt_root_id": root_ids,     # aligned 1:1 with rows in dff_coreg
                "dff": dff_coreg,
                "plane": plane_coreg,
                "roi": roi_coreg,
                "base_time": interp["base_time"],
                **stim_int_lists
            })

    # return two tidy dataframes (one row per session, arrays in object cols)
    cell_df        = pd.DataFrame(cell_df_records)
    coreg_cell_df  = pd.DataFrame(coreg_cell_records)
    return cell_df, coreg_cell_df

#%%
cell_df, coreg_cell_df = pre_process(
    mouse_ids=['409828'],          # or a list
    sessions=['409828_2018-11-06_14-02-59_nwb_2025-08-08_16-27-52'],                 # or '409828_2018-11-06_14-02-59_...' etc.
    coreg_df=coreg_df,
    data_dir='/data/'
)

#%%
df = pd.DataFrame(coreg_cell_df) 
out_dir = "/root/capsule/scratch"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "coreg_stim_int.pkl")
df.to_pickle(out_path, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved to {out_path}")

#%% to load your pickle
file_path = '/root/capsule/scratch/coreg_stim_int.pkl'
with open(file_path, 'rb') as file:
    coreg_stim_int = pickle.load(file)
coreg_stim_int