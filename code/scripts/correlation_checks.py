#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import KFold
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.families.links import log

import sys
import os
from os.path import join as pjoin
import platform
import glob
from hdmf_zarr import NWBZarrIO
from nwbwidgets import nwb2widget

from helpers import OLS_CV, poisson_glm_cv, fit_beta_model_with_cv, plot_beta_cv_results

platstring = platform.platform()
system = platform.system()
if system == "Darwin":
    # macOS
    data_root = "/Volumes/Brain2025/"

np.random.seed(42)
#%% Load in test metric info for structural data and example func data
metadata = pd.read_csv(os.path.join(data_root, 'v1dd_1196/V1DD_metadata.csv'))
struc_data = pd.read_feather(pjoin(data_root, 'v1dd_1196/joint_clustering_feat_df_v1dd.feather'))
func_data = pd.read_csv(pjoin(data_root, 'v1dd_1196/v1dd_metrics.csv'))
coregistered_cells = pd.read_feather(pjoin(data_root, 'v1dd_1196/coregistration_1196.feather'))

# Renaming column from root_id to pt_root_id
struc_data.rename(columns={'root_id': 'pt_root_id'}, inplace=True)

#%% Merging time
func_co_cells = pd.merge(func_data, coregistered_cells, on=['column', 'volume', 'plane', 'roi'], how='inner')
final_co_table = pd.merge(func_co_cells, struc_data, on=['pt_root_id'], how='inner')

#%% Correlation checking
# SSI options are 'ssi', 'ssi_prefered_both', or 'ssi_orth'
var_list = ["ssi_orth", "soma_area_to_volume_ratio", "median_density_spine", "soma_synapse_density_um", "median_density_shaft"]
sub_df = final_co_table[var_list].copy().reset_index(drop=True)
# # Check for NaNs in any values and drop those rows
sub_df.dropna(inplace=True)

# Set layout for pairwise plot - 3 X 3 plot grid
g = sns.PairGrid(sub_df,  vars=var_list, diag_sharey=False)

# Plot 2D density plot in the lower triangle
g.map_lower(sns.scatterplot, s=15, alpha=0.5, linewidth=0)

# Hide the upper triangle
def hide_current_axis(*args, **kwds):
    # function to hide upper triangle of the pairwise plots
    plt.gca().set_visible(False)
g.map_upper(hide_current_axis)

# Plot 1D density plot
g.map_diag(sns.kdeplot, hue=None, legend=False, bw_method = 'scott')

# Formatting
g.fig.set_size_inches(6,6)
g.fig.tight_layout()
g.fig.show()

corr_mat = pd.DataFrame(sub_df.corr())
plt.figure(figsize=(8, 6))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# Create a pairplot to visualize relationships
sns.pairplot(sub_df, diag_kind='hist')
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

#%% Split into X and Y
X = sub_df[var_list[1:]]
y = sub_df[var_list[0]]

#%% OLS model fitting ======================================================================
ols_cv_scores = OLS_CV(X, y, cv_folds=5)

summary_metrics = []
for result in ols_cv_scores:
    summary_metrics.append({
        'Fold': result['fold'],
        'R²': result['r_squared'],
        'RMSE': result['rmse'],
        'MAE': result['mae'],
        'Train_Size': result['train_size'],
        'Test_Size': result['test_size']
    })

summary_df = pd.DataFrame(summary_metrics)
print("\nDetailed Cross-Validation Results:")
print(summary_df)

print(f"\nOverall CV Statistics:")
print(f"Mean R²: {summary_df['R²'].mean():.4f} ± {summary_df['R²'].std():.4f}")
print(f"Mean RMSE: {summary_df['RMSE'].mean():.4f} ± {summary_df['RMSE'].std():.4f}")
print(f"Mean MAE: {summary_df['MAE'].mean():.4f} ± {summary_df['MAE'].std():.4f}")

#%% Visualize CV results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot R² across folds
axes[0].bar(summary_df['Fold'], summary_df['R²'], alpha=0.7)
axes[0].axhline(y=summary_df['R²'].mean(), color='red', linestyle='--',
                label=f'Mean: {summary_df["R²"].mean():.3f}')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('R²')
axes[0].set_title('R² by Fold')
axes[0].legend()

# Plot RMSE across folds
axes[1].bar(summary_df['Fold'], summary_df['RMSE'], alpha=0.7, color='orange')
axes[1].axhline(y=summary_df['RMSE'].mean(), color='red', linestyle='--',
                label=f'Mean: {summary_df["RMSE"].mean():.3f}')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('RMSE')
axes[1].set_title('RMSE by Fold')
axes[1].legend()

# Plot coefficient stability across folds
coeff_df = pd.DataFrame([result['model_summary']['params'] for result in ols_cv_scores])
coeff_df.index = range(1, len(coeff_df) + 1)
coeff_df.index.name = 'Fold'

axes[2].plot(coeff_df.index, coeff_df, marker='o', alpha=0.7)
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('Coefficient Value')
axes[2].set_title('Coefficient Stability')
axes[2].legend(coeff_df.columns, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

#%% GLM Time baby
# TODO: Need to change Poisson GLM to Beta or something else
# Perform Poisson GLM cross-validation
print("Poisson GLM Cross-Validation:")
print("=" * 50)

cv_results = fit_beta_model_with_cv(X, y, cv_folds=5)

# Calculate summary statistics
cv_pseudo_r2_mean = np.mean(cv_results['pseudo_r2'])
cv_pseudo_r2_std = np.std(cv_results['pseudo_r2'])
cv_deviance_mean = np.mean(cv_results['deviance'])
cv_deviance_std = np.std(cv_results['deviance'])

print(f"\nPoisson GLM Cross-Validation Summary:")
print(f"Mean Pseudo R²: {cv_pseudo_r2_mean:.4f} ± {cv_pseudo_r2_std:.4f}")
print(f"Mean Deviance: {cv_deviance_mean:.4f} ± {cv_deviance_std:.4f}")
print(f"Mean AIC: {np.mean(cv_results['aic']):.4f} ± {np.std(cv_results['aic']):.4f}")

# Average coefficients across folds
coeff_array = np.array(cv_results['coefficients'])
mean_coeffs = np.mean(coeff_array, axis=0)
std_coeffs = np.std(coeff_array, axis=0)

param_names = ['const'] + X.columns.tolist()
print(f"\nAverage Coefficients Across Folds:")
for i, name in enumerate(param_names):
    exp_coeff = np.exp(mean_coeffs[i]) if name != 'const' else np.exp(mean_coeffs[i])
    print(f"{name}: {mean_coeffs[i]:.4f} ± {std_coeffs[i]:.4f} (exp: {exp_coeff:.4f})")