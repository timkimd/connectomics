#%% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
func_sub_table.dropna(inplace=True)

# Renaming column from root_id to pt_root_id
if "root_id" in struc_data.columns:
    struc_data.rename(columns={'root_id': 'pt_root_id'}, inplace=True)

#%% Merging time
# func_co_cells = pd.merge(func_sub_table, coregistered_cells, on=['column', 'volume', 'plane', 'roi'], how='inner')
func_co_cells = pd.merge(func_sub_table, coregistered_cells, on=["pt_root_id"], how='inner')
final_co_table = pd.merge(func_co_cells, struc_data, on=['pt_root_id'], how='inner')

#%% Additional column calcs
for col in ["dtc_num_connections", "itc_num_connections", "ptc_num_connections",]:
    inhib_type = col.split("_")[0]
    final_co_table[col+"_vol_norm"] = final_co_table[col]/final_co_table["volume"]
    final_co_table[col+"_strength_norm"] = final_co_table[col]/final_co_table[f"{inhib_type}_sum_size"]

sub_df = final_co_table.drop(columns=final_co_table.columns[1:10])
sub_df = sub_df.drop(columns="cell_type").reset_index(drop=True)

#%% Correlation checking
# SSI options are 'ssi', 'ssi_pref_both', or 'ssi_orth'
# var_list = ["ssi", "soma_area_to_volume_ratio", "median_density_spine", "soma_synapse_density_um", "median_density_shaft", "soma_depth"]
var_list = sub_df.columns.tolist()
# sub_df = final_co_table[var_list].copy().reset_index(drop=True)
# Check for NaNs in any values and drop those rows
sub_df.dropna(inplace=True)

# The following code is adapted from Shawn Olsen's presentation on correlation comparisons
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
g.fig.set_size_inches(24,24)
g.fig.tight_layout()
plt.savefig(pjoin(fig_path, 'pairwise_density.png'), dpi=300, bbox_inches='tight')
g.fig.show()

corr_mat = pd.DataFrame(sub_df.corr())
plt.figure(figsize=(24, 24))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'corr_mat.png'), dpi=300, bbox_inches='tight')
plt.show()

# Create a pairplot to visualize relationships
# sns.pairplot(sub_df, diag_kind='hist')
# plt.suptitle('Pairwise Relationships', y=1.02)
# plt.savefig(pjoin(fig_path, 'pairplot.png'), dpi=300, bbox_inches='tight')
# plt.show()

#%% Split into X and Y
X = sub_df[np.array(var_list)[np.array([3, 12, 14, 16])]]
y = sub_df[var_list[0]]

#%% OLS model fitting
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

fig.savefig(pjoin(fig_path, 'ols_cv_results.png'), dpi=300, bbox_inches='tight')

#%% GLM Time baby
print("Poisson GLM Cross-Validation:")
print("=" * 50)
# "Normal" CV
# cv_results = fit_beta_model_with_cv(X, y, cv_folds=5)
# LOO CV
cv_results = fit_beta_model_with_cv(X, y, cv_folds=X.shape[0]//2)


# Calculate summary statistics
cv_pseudo_r2_mean = cv_results["cv_summary"]['mean_pseudo_r2']
cv_pseudo_r2_std = np.std(cv_results["cv_summary"]['std_pseudo_r2'])

print(f"\nPoisson GLM Cross-Validation Summary:")
print(f"Mean Pseudo R²: {cv_pseudo_r2_mean:.4f} ± {cv_pseudo_r2_std:.4f}")
fold_info = pd.DataFrame(cv_results["cv_results"]["fold_info"])
print(f"Mean AIC: {fold_info.model_aic.mean():.4f} ± {fold_info.model_aic.std():.4f}")

# Average coefficients across folds
coeff_array = np.array(cv_results['cv_results']['coefficients'])
mean_coeffs = np.mean(coeff_array, axis=0)
std_coeffs = np.std(coeff_array, axis=0)

param_names = ['const'] + X.columns.tolist()
print(f"\nAverage Coefficients Across Folds:")
for i, name in enumerate(param_names):
    exp_coeff = np.exp(mean_coeffs[i]) if name != 'const' else np.exp(mean_coeffs[i])
    print(f"{name}: {mean_coeffs[i]:.4f} ± {std_coeffs[i]:.4f} (exp: {exp_coeff:.4f})")

# box plot of coefficient values, not including precision
sns.boxplot(data=pd.DataFrame(coeff_array[:, :-1], columns=[param_names]), orient='h')
plt.title('Average Coefficient Values Across Folds')
plt.tight_layout()
plt.savefig(pjoin(fig_path, 'glm_boxplot.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot beta results
beta_fig, beta_axs = plot_beta_cv_results(cv_results)
beta_fig.savefig(pjoin(fig_path, 'glm_cv_results.png'), dpi=300, bbox_inches='tight')

#%% Shuffle data and rerun regression
print("Shuffling data and re-running regression:")
print("=" * 50)
times_to_shuffle = 100
shuffle_results = []

for i in range(times_to_shuffle):
    random_seed = np.random.randint(1, 10000)
    X_shuffled = X.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    y_shuffled = y.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    cv_results_shuffled = fit_beta_model_with_cv(X_shuffled, y_shuffled, cv_folds=5, verbose=False)
    shuffled_corr = cv_results['cv_results']['correlations']
    cv_params = cv_results_shuffled['cv_results']['coefficients']
    shuffle_results.append({
        'shuffled_corr': shuffled_corr,
        'params': cv_params,
    })

final_params = cv_results["full_model"]['model'].params.values[1:-1]
final_param_names = param_names[1:]

# Compare shuffled parameter values to actual model parameter values
shuffled_param_df = pd.DataFrame(shuffle_results)
shuffled_params = np.hstack(shuffled_param_df["params"].values).T
shuffled_corr = np.hstack(shuffled_param_df["shuffled_corr"].values).T

# Compute 95% CI for shuffled params
shuffled_param_ci = np.percentile(shuffled_params, [2.5, 97.5], axis=0)

# Check if each param is as extreme as the 95% CI intervals
for param_idx in range(len(final_params)):
    final_param = final_params[param_idx]
    final_param_name = final_param_names[param_idx]
    shuffled_ci = shuffled_param_ci[:, param_idx]
    if final_param < shuffled_ci[0] or final_param > shuffled_ci[1]:
        print(f"{final_param_name} is as extreme as the 95% CI for shuffled parameters!")
