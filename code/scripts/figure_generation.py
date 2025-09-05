#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams, font_manager
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase, HandlerTuple

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
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': 'black',
    'savefig.dpi': 600,
    'figure.dpi': 300,
    'figure.figsize': (12, 4),
    'figure.constrained_layout.use': False,
    'text.latex.preamble': r'\renewcommand{\familydefault}{\sfdefault} \usepackage[helvet]{sfmath}'
})
font_kwargs = {'fontsize': rcParams['font.size'], 'family': rcParams['font.sans-serif'][0]}
#%% Load in test metric info for structural data and example func data
metadata = pd.read_csv(os.path.join(data_root, 'v1dd_1196/V1DD_metadata.csv'))
# struc_data = pd.read_feather(pjoin(data_root, 'v1dd_1196/joint_clustering_feat_df_v1dd.feather'))
struc_data = pd.read_feather(pjoin(data_root, 'v1dd_1196/structurarl_data.feather'))
# func_data = pd.read_csv(pjoin(data_root, 'v1dd_1196/surround_supression_index_M409828.csv'))
func_data = pd.read_feather(pjoin(data_root, 'v1dd_1196/cell_ssi.feather'))
coregistered_cells = pd.read_feather(pjoin(data_root, 'v1dd_1196/coregistration_1196.feather'))

# Cleaning funcitonal data
func_sub_table = func_data[['ssi', 'column', 'volume', 'plane', 'roi', 'pt_root_id']]
func_sub_table['volume'] = pd.to_numeric(func_sub_table['volume'], errors='coerce').astype('Int64')
func_sub_table.dropna(inplace=True)

for col in ["dtc_num_connections", "tc_num_connections", "ptc_num_connections",]:
    struc_data[col+"_vol_norm"] = struc_data[col]/struc_data["volume"]

# Renaming column from root_id to pt_root_id
if "root_id" in struc_data.columns:
    struc_data.rename(columns={'root_id': 'pt_root_id'}, inplace=True)

#%% Merging time
# func_co_cells = pd.merge(func_sub_table, coregistered_cells, on=['column', 'volume', 'plane', 'roi'], how='inner')
func_co_cells = pd.merge(func_sub_table, coregistered_cells, on=["pt_root_id"], how='inner')
final_co_table = pd.merge(func_co_cells, struc_data, on=['pt_root_id'], how='inner')

#%% Figure 1 -- Scatter of # inhib connections vs SSI
fig1, ax1 = plt.subplots(1, 1, figsize=(3, 3))
sns.scatterplot(x="dtc_num_connections", y="ssi", data=final_co_table, alpha=0.7, s=10)
ax1.set_xlabel("Number of DTC Connections")
ax1.set_ylabel("SSI")
ax1.set_title("SSI vs Number of DTC Connections")
# ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'dtc_num_connections_vs_ssi.png'), dpi=300, bbox_inches='tight')
plt.show()


#%% Figure 3a -- Synapse target on excitatory cell -- Tim will do

#%% Figure 3b -- Histogram of strength of connection

#%% Figure 4 -- Histogram of euc distance between Soma_inhib and Soma_sursup
fig4, ax4 = plt.subplots(1, 1, figsize=(3, 3))
sns.histplot(final_co_table["euclidean_distance"], bins=20, kde=True, ax=ax4)
ax4.set_xlabel("Euclidean Distance (um)")
ax4.set_ylabel("Density")
ax4.set_title("Distance between inhib and supressed soma")
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'euc_dist_soma_inhib_soma_supres.png'), dpi=300, bbox_inches='tight')
plt.show()

#%% Figure 5 -- Plot of layer of cell vs SSI (does not need to be coregistered)

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