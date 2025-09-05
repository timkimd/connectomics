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
scratch_dir = '/scratch/'
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
e_to_e = pd.read_feather(f"{scratch_dir}/E_to_E_total_con.feather")
struc_df = pd.read_feather(f"{scratch_dir}/structural_data.feather")
cell_ssi_df = pd.read_feather(f"{scratch_dir}/cell_ssi.feather")
cell_coreg_df = pd.read_feather(f"{scratch_dir}/cell_coreg.feather")
i_to_e_chain = pd.read_feather(f"{scratch_dir}/new_IE_struct_cell_tbl_v1dd_1196.feather")
e_to_i_chain = pd.read_feather(f"{scratch_dir}/new_EI_struct_cell_tbl_v1dd_1196.feather")
e_to_e_chain = pd.read_feather(f"{scratch_dir}/E_to_E_pre_post_id.feather")
#%% plotting RFs and Window locations

x = position_metadata.azi 
y = position_metadata.alt

x1 = cell_ssi_df.azi
y1 = cell_ssi_df.alt

x2 = cell_ssi_df.azimuth_rf_on
y2 = cell_ssi_df.altitude_rf_on

x3 = cell_ssi_df.azimuth_rf_off
y3 = cell_ssi_df.altitude_rf_off

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

#%%
struc_func_df = pd.merge(struc_df, cell_ssi_df, on='pt_root_id', how='inner')
i_to_e.rename(columns={"post_pt_root_id": "pt_root_id"}, inplace=True)
i_to_ssi = pd.merge(i_to_e, cell_ssi_df, on='pt_root_id', how='inner')

#%% plot ssi by layer
plt.figure(figsize=(8,6))
sns.boxplot(
    data=struc_func_df, 
    x="cell_type",
    y="ssi",
    palette="Set2"
)

plt.title("SSI by Cell Type")
plt.xlabel("Cell Type")
plt.ylabel("SSI")
plt.show()

#%%

filtered = i_to_ssi[i_to_ssi["target_structure"] != "unknown"]
plt.figure(figsize=(8,6))
sns.boxplot(
    data=filtered, 
    x="target_structure",     # Shaft / Soma / Spine
    y="num_connections",      # # of synapses
    hue="cell_type_post",     # DTC / ITC / STC / PTC
    palette="Set2"            # color scheme
)

# Axis labels
plt.ylabel("# of synapses", fontsize=14)
plt.xlabel("")  # no label, to match your plot
plt.xticks(fontsize=12)

# Legend title
plt.legend(title="Cell type", fontsize=10, title_fontsize=12)

# Remove plot title (your target image doesn’t have one)
plt.title("SSI synapses by cell type")

plt.show()

#%% plot target structure based on unique connection types

df = i_to_ssi.copy()
# Make a readable pair label
df["pair"] = df["cell_type_pre"].astype(str) + " \u2192 " + df["cell_type_post"].astype(str)
# enforce a target_structure order
order_struct = ["shaft", "soma", "spine"]
df["target_structure"] = pd.Categorical(df["target_structure"], categories=order_struct, ordered=True)
# Stacked percentage bars (distribution of structures per pre→post pair)
ct = pd.crosstab(df["pair"], df["target_structure"])              # counts
prop = ct.div(ct.sum(axis=1), axis=0).fillna(0)                   # row-wise proportions

ax = prop[order_struct].plot(kind="bar", stacked=True, figsize=(11,6))
ax.set_ylabel("Share of synapses")
ax.set_xlabel("Pre to Post cell-type pair")
ax.legend(title="Target structure", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

#%% plot the unique connections between coreg types and inhib inputs

i_to_e.rename(columns={"post_pt_root_id": "pt_root_id"}, inplace=True)
i_to_coreg = pd.merge(i_to_e, cell_coreg_df, on='pt_root_id', how='inner')

#%% coreg # of synapses by target

filtered = i_to_coreg[i_to_coreg["target_structure"] != "unknown"]
plt.figure(figsize=(8,6))
sns.boxplot(
    data=filtered, 
    x="target_structure",     # Shaft / Soma / Spine
    y="num_connections",      # # of synapses
    hue="cell_type_post",     # DTC / ITC / STC / PTC
    palette="Set2"            # color scheme
)

# Axis labels
plt.ylabel("# of synapses", fontsize=14)
plt.xlabel("")  # no label, to match your plot
plt.xticks(fontsize=12)

# Legend title
plt.legend(title="Cell type", fontsize=10, title_fontsize=12)

# Remove plot title (your target image doesn’t have one)
plt.title("coreg synapses by cell type")

plt.show()

#%% plot target structure based on unique connection types

df = i_to_ssi.copy()
# Make a readable pair label
df["pair"] = df["cell_type_pre"].astype(str) + " \u2192 " + df["cell_type_post"].astype(str)
# enforce a target_structure order
order_struct = ["shaft", "soma", "spine"]
df["target_structure"] = pd.Categorical(df["target_structure"], categories=order_struct, ordered=True)
# Stacked percentage bars (distribution of structures per pre→post pair)
ct = pd.crosstab(df["pair"], df["target_structure"])              # counts
prop = ct.div(ct.sum(axis=1), axis=0).fillna(0)                   # row-wise proportions

ax = prop[order_struct].plot(kind="bar", stacked=True, figsize=(11,6))
ax.set_ylabel("Share of synapses")
ax.set_xlabel("Pre to Post cell-type pair")
ax.legend(title="Target structure", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

#%% E>I>E chain

i_to_e_chain
i_to_e_chain.rename(columns={"post_pt_root_id": "pt_root_id"}, inplace=True)
i_to_ssi = pd.merge(i_to_e, cell_ssi_df, on='pt_root_id', how='inner')
