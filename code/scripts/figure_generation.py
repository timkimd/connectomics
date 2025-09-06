#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams, font_manager
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase, HandlerTuple
import scipy.stats as stats

import os
from os.path import join as pjoin
import platform

from helpers import OLS_CV, fit_beta_model_with_cv, plot_beta_cv_results

platstring = platform.platform()
system = platform.system()
if system == "Darwin":
    # macOS
    data_root = "/Volumes/Brain2025/"

fig_path = pjoin(data_root, 'temp_results')
np.random.seed(42)

#%% MPL params
## setup figure aesthetics
# mpl.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'sans-serif',
#     'font.sans-serif': ['Helvetica'],
#     'pdf.fonttype': 42,
#     'ps.fonttype': 42,
#     'font.size': 8,
#     'axes.titlesize': 9,
#     'axes.labelsize': 8,
#     'xtick.labelsize': 7,
#     'ytick.labelsize': 7,
#     'legend.fontsize': 7,
#     'legend.title_fontsize': 8,
#     'axes.linewidth': 0.5,
#     'xtick.major.width': 0.5,
#     'ytick.major.width': 0.5,
#     'xtick.minor.width': 0.4,
#     'ytick.minor.width': 0.4,
#     'xtick.direction': 'out',
#     'ytick.direction': 'out',
#     'axes.spines.top': False,
#     'axes.spines.right': False,
#     'axes.edgecolor': 'black',
#     'savefig.dpi': 600,
#     'figure.dpi': 300,
#     'figure.figsize': (12, 4),
#     'figure.constrained_layout.use': False,
#     'text.latex.preamble': r'\renewcommand{\familydefault}{\sfdefault} \usepackage[helvet]{sfmath}'
# })
# font_kwargs = {'fontsize': rcParams['font.size'], 'family': rcParams['font.sans-serif'][0]}
#%% Load in test metric info for structural data and example func data
metadata = pd.read_csv(os.path.join(data_root, 'v1dd_1196/V1DD_metadata.csv'))
# struc_data = pd.read_feather(pjoin(data_root, 'v1dd_1196/joint_clustering_feat_df_v1dd.feather'))
struc_data = pd.read_feather(pjoin(data_root, 'v1dd_1196/structural_data.feather'))
# func_data = pd.read_csv(pjoin(data_root, 'v1dd_1196/surround_supression_index_M409828.csv'))
func_data = pd.read_feather(pjoin(data_root, 'v1dd_1196/cell_ssi.feather'))
coregistered_cells = pd.read_feather(pjoin(data_root, 'v1dd_1196/coregistration_1196.feather'))

# Cleaning funcitonal data
func_sub_table = func_data[['ssi', 'column', 'volume', 'plane', 'roi', 'pt_root_id']]
func_sub_table['volume'] = pd.to_numeric(func_sub_table['volume'], errors='coerce').astype('Int64')
# func_sub_table.dropna(inplace=True)

# Renaming column from root_id to pt_root_id
if "root_id" in struc_data.columns:
    struc_data.rename(columns={'root_id': 'pt_root_id'}, inplace=True)

#%% Merging time
# func_co_cells = pd.merge(func_sub_table, coregistered_cells, on=['column', 'volume', 'plane', 'roi'], how='inner')
func_co_cells = pd.merge(func_sub_table, coregistered_cells, on=["pt_root_id"], how='inner')
final_co_table = pd.merge(func_co_cells, struc_data, on=['pt_root_id'], how='inner')

final_co_table["ssi"] = np.tanh(final_co_table["ssi"])
for col in ["dtc_num_connections", "itc_num_connections", "ptc_num_connections",]:
    inhib_type = col.split("_")[0]
    final_co_table[col+"_vol_norm"] = final_co_table[col]/final_co_table["volume"]
    final_co_table[f"{inhib_type}_mean_strength_synapse"] = final_co_table[f"{inhib_type}_sum_size"]/final_co_table[col]

#%% Figure 1 -- Scatter of # inhib connections vs SSI
fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
# Calculate p-value
ssi_vals = final_co_table['ssi'].values
dtc_vals = final_co_table["dtc_num_connections_vol_norm"].values
r, p_val = stats.pearsonr(dtc_vals, ssi_vals)

sns.scatterplot(x="dtc_num_connections_vol_norm", y="ssi", data=final_co_table, alpha=0.7, s=10)
ax1.set_xlabel("Number of DTC Connections normalized by volume")
ax1.set_ylabel("SSI")
ax1.set_title(f"SSI vs Number of DTC Connections, p-value: {p_val:.3f}, r: {r:.3f}")
# ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'dtc_num_connections_vs_ssi.png'), dpi=300, bbox_inches='tight')
plt.show()



#%% Figure 3a -- Synapse target on excitatory cell -- Tim will do

#%% Figure 3b -- Histogram of strength of connection -- Tim as well?

#%% Figure 4a -- Histogram of euc distance between Soma_inhib and Soma_sursup
def categorize_ssi(ssi_value):
    if ssi_value < -0.25:
        return 'SSI < -0.25'
    elif ssi_value <= 0.25:
        return '-0.25 ≤ SSI ≤ 0.25'
    else:
        return 'SSI > 0.25'

final_co_table['ssi_group'] = final_co_table['ssi'].apply(categorize_ssi)
# Shuffle SSI values to calculate if significance
shuffle_count = 1000
shuffle_frame = final_co_table["avg_eucl_dist"].to_frame()
shuffled_dist_sig_SSI = []

for i in range(shuffle_count):
    shuffle_ssi = np.random.permutation(final_co_table['ssi'].values)
    shuffle_frame[f"ssi"] = shuffle_ssi
    shuffle_frame[f"shuffle_group"] = shuffle_frame["ssi"].apply(categorize_ssi)
    group_means = shuffle_frame.groupby([f"shuffle_group"]).mean()
    shuffled_dist_sig_SSI.append(group_means.avg_eucl_dist.iloc[2])

lower_b, upper_b = np.percentile(np.array(shuffled_dist_sig_SSI), [2.5, 97.5], axis=0)
mean_dist = final_co_table.groupby(["ssi_group"])["avg_eucl_dist"].mean().iloc[2]
p_value = sum(shuffled_dist_sig_SSI < mean_dist) / len(shuffled_dist_sig_SSI)
if p_value == 0:
    p_value = 1/shuffle_count
fig4a, ax4a = plt.subplots(1, 1, figsize=(5, 5))
sns.histplot(final_co_table["avg_eucl_dist"], bins=20, kde=True, ax=ax4a)
ax4a.set_xlabel("Euclidean Distance (um)")
ax4a.set_ylabel("Density")
ax4a.set_title(f"Distance between inhib and supressed soma")
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'euc_dist_soma_inhib_soma_supres.png'), dpi=300, bbox_inches='tight')
plt.show()

fig4b, ax4b = plt.subplots(1, 1, figsize=(10, 6))
sns.histplot(data=final_co_table, x='avg_eucl_dist', hue='ssi_group',
             bins=25, alpha=0.6, edgecolor='black', kde=True)
ax4b.set_xlabel('Euclidean Distance (um)')
ax4b.set_ylabel('Count')
ax4b.set_title(f'Distribution of Euclidean Distance by SSI Groups -- p-value < {p_value:.3f} for SSI > 0.25')
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'euc_dist_hist_.png'), dpi=300, bbox_inches='tight')
plt.show()


#%% Figure 4c -- Scatter of euc distance vs SSI
fig_4c, ax_4c = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x="euclidean_distance", y="ssi", data=final_co_table, alpha=0.7, s=10)
ax_4c.set_xlabel("Number of DTC Connections")
ax_4c.set_ylabel("SSI")
ax_4c.set_title("SSI vs Number of DTC Connections")
# ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'scatter_ssi_vs_euc_dist.png'), dpi=300, bbox_inches='tight')
plt.show()
#%% Figure 5 -- Plot of layer of cell vs SSI (does not need to be coregistered) - Sasha

#%% Figure Test
fig_test, ax_test = plt.subplots(1, 1, figsize=(3, 3))
sns.scatterplot(x="euclidean_distance", y="ssi", data=final_co_table, alpha=0.7, s=10)
ax_test.set_xlabel("Number of DTC Connections")
ax_test.set_ylabel("SSI")
ax_test.set_title("SSI vs Number of DTC Connections")
# ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# plt.savefig(pjoin(fig_path, 'dtc_num_connections_vs_ssi.png'), dpi=300, bbox_inches='tight')
plt.show()