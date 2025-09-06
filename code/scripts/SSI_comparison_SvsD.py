#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
from os.path import join as pjoin
import platform

from helpers import OLS_CV, fit_beta_model_with_cv, plot_beta_cv_results
from scripts.correlation_checks_one_param_looping import func_sub_table

platstring = platform.platform()
system = platform.system()
if system == "Darwin":
    # macOS
    data_root = "/Volumes/Brain2025/"

fig_path = pjoin(data_root, 'temp_results')
np.random.seed(42)
#%% Load in test metric info for structural data and example func data
metadata = pd.read_csv(os.path.join(data_root, 'v1dd_1196/V1DD_metadata.csv'))
coregistered_cells = pd.read_feather(pjoin(data_root, 'v1dd_1196/coregistration_1196.feather'))
func_data_sasha = pd.read_feather(pjoin(data_root, 'v1dd_1196/cell_ssi.feather'))
func_data_david = pd.read_csv(pjoin(data_root, 'v1dd_1196/surround_supression_index_M409828.csv'))

func_data_david['volume'] = pd.to_numeric(func_data_david['volume'], errors='coerce').astype('Int64')
func_co_cells_david = pd.merge(func_data_david, coregistered_cells, on=["column", "volume", "plane", "roi"], how='inner')
func_co_cells_sasha = pd.merge(func_data_sasha, coregistered_cells, on=["pt_root_id"], how='inner')
func_co_cells_sasha.rename(columns={'ssi': 'ssi_sasha'}, inplace=True)
func_co_cells_david.rename(columns={'ssi': 'ssi_david'}, inplace=True)

func_co_cells_both = pd.merge(func_co_cells_david, func_co_cells_sasha, on=['pt_root_id'], how='inner')

func_sub_table = func_co_cells_both[['ssi_sasha', 'ssi_david']]
func_sub_table.dropna(inplace=True)

#%% Scatter of sasha ssi vs david ssi
plt.figure(figsize=(3, 3))
plt.scatter(func_sub_table['ssi_sasha'], func_sub_table['ssi_david'], alpha=0.7, s=10)
plt.xlabel("SSI (Sasha)")
plt.ylabel("SSI (David)")
plt.title("SSI Comparison")
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'ssi_comparison_sasha_vs_david.png'), dpi=300, bbox_inches='tight')
plt.show()

vals_sasha = func_sub_table['ssi_sasha'].values
vals_david = func_sub_table['ssi_david'].values
mean_sub_vals_sasha = np.clip(vals_sasha - np.mean(vals_sasha), -1, 1)
mean_sub_vals_david = np.clip(vals_david - np.mean(vals_david), -1, 1)
corr_value = np.corrcoef(mean_sub_vals_david, mean_sub_vals_sasha)[0, 1]
print(f"Correlation between SSI values is {corr_value}")
# mean_sub_vals_david = vals_david - np.mean(vals_david)
# corr_value = np.corrcoef(mean_sub_vals_david, mean_sub_vals_sasha)[0, 1]
# print(f"Correlation between SSI values is {corr_value}")