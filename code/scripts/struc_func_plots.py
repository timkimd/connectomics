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
from scipy.stats import mannwhitneyu

# Set the utils path
utils_dir = pjoin("..", "utils")

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
cell_ssi = pd.read_feather(f"{scratch_dir}/cell_ssi.feather")
cell_coreg_df = pd.read_feather(f"{scratch_dir}/cell_coreg.feather")
i_to_e_chain = pd.read_feather(f"{scratch_dir}/new_IE_struct_cell_tbl_v1dd_1196.feather")
e_to_i_chain = pd.read_feather(f"{scratch_dir}/EI_new_v1dd_1196.feather")
e_to_e_chain = pd.read_feather(f"{scratch_dir}/E_to_E_pre_post_id.feather")

#%%

struc_df.rename(columns={"volume": "cell_volume"}, inplace=True)
struc_func_df = pd.merge(struc_df, cell_ssi, on='pt_root_id', how='inner')
#%%
i_to_e.rename(columns={"post_pt_root_id": "pt_root_id"}, inplace=True)
i_to_ssi = pd.merge(cell_ssi, i_to_e, on='pt_root_id', how='inner')
#%%
for col in ["dtc_num_connections", "itc_num_connections", "ptc_num_connections", "stc_num_connections"]:
    inhib_type = col.split("_")[0]
    struc_func_df[f"{inhib_type}_mean_strength_synapse"] = struc_func_df[f"{inhib_type}_sum_size"]/struc_func_df[col]
#%%

long_df = struc_func_df.melt(
    value_vars=['dtc_mean_strength_synapse',
                'itc_mean_strength_synapse',
                'stc_mean_strength_synapse',
                'ptc_mean_strength_synapse'],
    var_name='input_type',
    value_name='synaptic_strength',
)

# make short labels for x-ticks: DTC / ITC / PTC
long_df['input_short'] = long_df['input_type'].str.split('_').str[0].str.upper()

# order + colors to match your other plot
order = ['DTC','PTC','STC', 'ITC']  # change order if you prefer
palette = {'DTC': '#1f77b4',  # blue
           'PTC': '#ff7f0e',  # orange
           'STC': '#2ca02c',
           'ITC': '#d62728'}  # red

plt.figure(figsize=(8,6))
sns.boxplot(
    x='input_short', y='synaptic_strength',
    data=long_df, order=order, palette=palette
)
sns.despine()
plt.xlabel("")                     # optional: cleaner look
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Synaptic Strength", fontsize=16)
plt.title("Synaptic Strength", fontsize=20)
plt.tight_layout()
plt.savefig("/scratch/syn_strength_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()


#%%
e_to_e.rename(columns={"pre_pt_root_id": "pt_root_id"}, inplace=True)
e_to_ssi = pd.merge(cell_ssi, e_to_e, on='pt_root_id', how='inner')


# --- data prep ---
need_cols = {"cell_type", "iso_ssi", "cross_neg_ssi"}
missing = need_cols - set(struc_func_df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = struc_func_df.loc[:, ["cell_type", "iso_ssi", "cross_neg_ssi"]].copy()
df["iso_ssi"] = pd.to_numeric(df["iso_ssi"], errors="coerce")
df["cross_neg_ssi"] = pd.to_numeric(df["cross_neg_ssi"], errors="coerce")

# EXCLUDE these cell types entirely
exclude_ct = {"L4-IT", "L5-IT"}
df = df[~df["cell_type"].isin(exclude_ct)].copy()

# plotting dataframe: ONLY Iso values (drop NaNs in iso)
plot_df = (
    df.loc[:, ["cell_type", "iso_ssi"]]
      .rename(columns={"iso_ssi": "ssi"})
      .dropna(subset=["ssi"])
      .assign(condition="Iso")
)
plot_df["ssi"] = plot_df["ssi"].clip(-2, 2)
plot_df["group"] = plot_df["condition"].astype(str) + " " + plot_df["cell_type"].astype(str)

# group order: only Iso groups (after exclusion)
order = sorted(plot_df["group"].unique())

# summaries for overlays
means = plot_df.groupby("group")["ssi"].mean()

# --- paired stats (Wilcoxon) per cell_type ---
def p_to_stars(p):
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

pvals = {}
for ct, dct in df.groupby("cell_type"):
    # keep only rows with both values for the paired test
    paired = dct.dropna(subset=["iso_ssi", "cross_neg_ssi"])
    if len(paired) < 1:
        pvals[ct] = np.nan
        continue

    diffs = paired["iso_ssi"] - paired["cross_neg_ssi"]
    if np.allclose(diffs, 0, equal_nan=False):
        pvals[ct] = 1.0
        continue

    try:
        stat = mannwhitneyu(
            paired["iso_ssi"],
            paired["cross_neg_ssi"],
            alternative="two-sided"
        )
        pvals[ct] = stat.pvalue
    except ValueError:
        pvals[ct] = 1.0

# map cell_type p-values to group labels (e.g., "Iso L2/3")
pvals_group = {f"Iso {ct}": p for ct, p in pvals.items()}

# --- plot ---
plt.figure(figsize=(8, 6))
ax = sns.stripplot(
    data=plot_df,
    x="group", y="ssi",
    order=order,
    jitter=0.25, size=8,
    color="k", alpha=0.35,
)

ax.set_ylim(-2.1, 2.1)
ax.axhline(0, ls="--", lw=1, color="k", alpha=0.5)

# overlay thick mean bars (no yellow dots)
for i, g in enumerate(order):
    y = means[g]
    ax.plot([i-0.35, i+0.35], [y, y], color="k", lw=4, solid_capstyle="round", zorder=4)

# labels & cosmetics
ax.set_xlabel("")
ax.set_ylabel("SSI", fontsize=15)
sns.despine()

# two-line ticks "Iso\nL2/3"
ax.set_xticklabels([lab.replace(" ", "\n") for lab in order], fontsize=15)

# significance stars only (NO counts anywhere)
xt = ax.get_xticks()
for i, g in enumerate(order):
    p = pvals_group.get(g, np.nan)
    stars = p_to_stars(p) if pd.notna(p) else "NA"
    ax.text(xt[i], 2.02, stars, ha="center", va="bottom", fontsize=20)

plt.tight_layout()
plt.savefig("/scratch/SSI_by_cell_type.png", dpi=300, bbox_inches="tight")
plt.show()


#%%
filtered = i_to_ssi[i_to_ssi["target_structure"] != "unknown"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=filtered,
    x="target_structure",     # Shaft / Soma / Spine
    y="num_connections",      # # of synapses
    hue="cell_type_post",     # l3-IT etc
    palette="tab10",
    ax=ax
)
sns.despine(ax=ax)

ax.set_ylabel("# of synapses", fontsize=14)
ax.set_xlabel("")  # no label, to match your plot
ax.tick_params(axis='x', labelsize=12)
ax.legend(title="Cell type", fontsize=10, title_fontsize=12)
ax.set_title("SSI synapses")

fig.tight_layout()
fig.savefig("/scratch/SSI_synapses_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
#i_to_e.rename(columns={"post_pt_root_id": "pt_root_id"}, inplace=True)
i_to_coreg = pd.merge(cell_coreg_df, i_to_e, on='pt_root_id', how='inner')
filtered = i_to_coreg[i_to_coreg["target_structure"] != "unknown"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=filtered,
    x="target_structure",     # Shaft / Soma / Spine
    y="num_connections",      # # of synapses
    hue="cell_type_post",     # l3-IT etc
    palette="tab10",
    ax=ax
)
sns.despine(ax=ax)

ax.set_ylabel("# of synapses", fontsize=14)
ax.set_xlabel("")  # no label, to match your plot
ax.tick_params(axis='x', labelsize=12)
ax.legend(title="Cell type", fontsize=10, title_fontsize=12)
ax.set_title("Coregistered synapses")

fig.tight_layout()
fig.savefig("/scratch/coreg_synapses_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
i_to_coreg = pd.merge(cell_coreg_df, i_to_e, on='pt_root_id', how='inner')
filtered = i_to_coreg[i_to_coreg["target_structure"] != "unknown"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=filtered,
    x="cell_type_pre",     # Shaft / Soma / Spine
    y="num_connections",      # # of synapses     # l3-IT etc
    palette="tab10",
    ax=ax
)
sns.despine(ax=ax)

ax.set_ylabel("# of synapses", fontsize=16)
ax.set_xlabel("")  # no label, to match your plot
ax.tick_params(axis='x', labelsize=16)
#ax.legend(title="Cell type", fontsize=10, title_fontsize=12)
ax.set_title("Coregistered", fontsize=20)

fig.tight_layout()
fig.savefig("/scratch/coreg_synapses_from_inhib_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()
#%%
#i_to_ssi = pd.merge(cell_coreg_df, i_to_e, on='pt_root_id', how='inner')
filtered = i_to_ssi[i_to_ssi["target_structure"] != "unknown"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=filtered,
    x="cell_type_pre",     # Shaft / Soma / Spine
    y="num_connections",      # # of synapses     # l3-IT etc
    palette="tab10",
    ax=ax
)
sns.despine(ax=ax)

ax.set_ylabel("# of synapses", fontsize=16)
ax.set_xlabel("")  # no label, to match your plot
ax.tick_params(axis='x', labelsize=16)
#ax.legend(title="Cell type", fontsize=10, title_fontsize=12)
ax.set_title("SSI", fontsize=20)

fig.tight_layout()
fig.savefig("/scratch/ssi_synapses_from_inhib_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()



#%%
# 1) Build a single master order of categories across BOTH datasets
cats = pd.Index(
    pd.concat([
        i_to_coreg["cell_type_pre"],
        i_to_ssi["cell_type_pre"]
    ]).dropna().unique()
)

# 2) Make a label->color mapping once
palette = dict(zip(cats, sns.color_palette("tab10", n_colors=len(cats))))

# ---- Plot 1 ----
filtered_coreg = i_to_coreg[i_to_coreg["target_structure"] != "unknown"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=filtered_coreg,
    x="cell_type_pre",
    y="num_connections",
    order=cats,            # <-- enforce same x ordering
    palette=palette,       # <-- enforce same colors by label
    ax=ax
)
sns.despine(ax=ax)
ax.set_ylabel("# of synapses", fontsize=16)
ax.set_xlabel("")
ax.tick_params(axis='x', labelsize=16)
ax.set_title("Coregistered", fontsize=20)
fig.tight_layout()
fig.savefig("/scratch/coreg_synapses_from_inhib_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

# ---- Plot 2 ----
filtered = i_to_ssi[i_to_ssi["target_structure"] != "unknown"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=filtered,
    x="cell_type_pre",
    y="num_connections",
    order=cats,            # same order
    palette=palette,       # same colors
    ax=ax
)
sns.despine(ax=ax)
ax.set_ylabel("# of synapses", fontsize=16)
ax.set_xlabel("")
ax.tick_params(axis='x', labelsize=16)
ax.set_title("SSI", fontsize=20)
fig.tight_layout()
fig.savefig("/scratch/ssi_synapses_from_inhib_boxplot.png", dpi=300, bbox_inches="tight")
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
