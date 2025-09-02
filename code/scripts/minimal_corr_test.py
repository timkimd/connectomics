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

#%% Functions written
def OLS_CV(X, y, cv_folds=5, random_state=42):
    """
    Detailed cross-validation with comprehensive statistics
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Add constant for intercept
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        # Fit model
        model = sm.OLS(y_train, X_train_const).fit()

        # Predictions
        y_pred = model.predict(X_test_const)

        # Calculate various metrics
        residuals = y_test - y_pred
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'r_squared': r_squared,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'model_summary': {
                'params': model.params,
                'pvalues': model.pvalues,
                'conf_int': model.conf_int(),
                'rsquared_adj': model.rsquared_adj,
                'fvalue': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'aic': model.aic,
                'bic': model.bic,
            }
        }
        fold_results.append(fold_result)

    return fold_results

def poisson_glm_cv(X, y, cv_folds=5, random_state=42):
    """
    Perform cross-validation for Poisson GLM with log link
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_results = {
        'deviance': [],
        'aic': [],
        'bic': [],
        'pseudo_r2': [],
        'pearson_chi2': [],
        'coefficients': [],
        'pvalues': [],
        'predictions': [],
        'actual': []
    }

    # Prepare Poisson data (ensure integer counts and positive values)
    y_poisson = np.round(y).astype(int)
    y_poisson = np.maximum(y_poisson, 1)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_poisson[train_idx], y_poisson[test_idx]

        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)

        model = sm.GLM(y_train, X_train_const,
                       family=Poisson(link=log())).fit()

        # Make predictions on test data
        y_pred = model.predict(X_test_const)

        # Calculate metrics
        # For test set deviance, we need to calculate it manually
        test_deviance = 2 * np.sum(y_test * np.log(y_test / y_pred) - (y_test - y_pred))

        # Pseudo R-squared for test set
        null_deviance = 2 * np.sum(y_test * np.log(y_test / np.mean(y_test)) - (y_test - np.mean(y_test)))
        pseudo_r2 = 1 - (test_deviance / null_deviance)

        # Store results
        cv_results['deviance'].append(test_deviance)
        cv_results['aic'].append(model.aic)  # From training
        cv_results['bic'].append(model.bic)  # From training
        cv_results['pseudo_r2'].append(pseudo_r2)
        cv_results['pearson_chi2'].append(model.pearson_chi2)  # From training
        cv_results['coefficients'].append(model.params.values)
        cv_results['pvalues'].append(model.pvalues.values)
        cv_results['predictions'].extend(y_pred)
        cv_results['actual'].extend(y_test)

        print(f"Fold {fold + 1}: Pseudo R² = {pseudo_r2:.4f}, Deviance = {test_deviance:.4f}")

    return cv_results, y_poisson
#%% Sham data creation
np.random.seed(42)
n_cells = 100
firing_rates = np.random.normal(loc=10, scale=1, size=n_cells)

noise1 = np.random.normal(0, 1.2, n_cells)
noise2 = np.random.normal(0, 0.5, n_cells)
noise3 = np.random.normal(0, 1.0, n_cells)

membrane_potential = 0.5 * firing_rates + noise1 - 65
synaptic_input = 0.85 * firing_rates + noise2 + 5
cell_diameter = 0.4 * firing_rates + noise3 + 15

cell_df = pd.DataFrame({
    'firing_rates': firing_rates,
    'membrane_potential': membrane_potential,
    'synaptic_input': synaptic_input,
    'cell_diameter': cell_diameter
})

#%% Calculate correlations
cell_df.corr()

# Set layout for pairwise plot - 3 X 3 plot grid
g = sns.PairGrid(cell_df,  vars=['firing_rates', 'synaptic_input', 'cell_diameter'], diag_sharey=False)

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


#%% Correlation matrix
corr_mat = pd.DataFrame(cell_df.corr())
plt.figure(figsize=(8, 6))
sns.heatmap(corr_mat, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# Create a pairplot to visualize relationships
sns.pairplot(cell_df, diag_kind='hist')
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

#%% Split into X and Y
X = cell_df[['membrane_potential', 'synaptic_input', 'cell_diameter']]
y = cell_df['firing_rates']

#%% Advanced cross-validation with detailed statistics i.e. Roid results

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

#%% GLM Time
# 1. Poisson GLM (good for count data, but firing rates might be continuous)
print("=" * 60)
print("1. POISSON GLM")
print("=" * 60)

X_with_const = sm.add_constant(X)

# Convert to counts if needed (for Poisson)
y_poisson = np.round(y).astype(int)  # Convert to integer counts
y_poisson = np.maximum(y_poisson, 1)  # Ensure positive values

poisson_model = sm.GLM(y_poisson, X_with_const, family=Poisson()).fit()
print(poisson_model.summary())

print(f"Poisson GLM Performance:")
print(f"AIC: {poisson_model.aic:.4f}")
print(f"BIC: {poisson_model.bic:.4f}")
print(f"Deviance: {poisson_model.deviance:.4f}")
print(f"Pearson chi2: {poisson_model.pearson_chi2:.4f}")

#%% Poisson with CV

# Perform Poisson GLM cross-validation
print("Poisson GLM Cross-Validation:")
print("=" * 50)
X = cell_df[['membrane_potential', 'synaptic_input', 'cell_diameter']]
y = cell_df['firing_rates']

cv_results, y_poisson = poisson_glm_cv(X, y, cv_folds=5)

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

#%% Validation time
X_with_const = sm.add_constant(X)
poisson_full_model = sm.GLM(y_poisson, X_with_const,
                           family=Poisson(link=log())).fit()

print("Full Poisson GLM Model Summary:")
print("=" * 40)
print(poisson_full_model.summary())

# Make predictions
y_pred_poisson = poisson_full_model.predict(X_with_const)

pred_corr_poisson, _ = stats.pearsonr(y_poisson, y_pred_poisson)
rmse_poisson = np.sqrt(np.mean((y_poisson - y_pred_poisson)**2))
mae_poisson = np.mean(np.abs(y_poisson - y_pred_poisson))
pseudo_r2_full = 1 - (poisson_full_model.deviance / poisson_full_model.null_deviance)

print(f"\nPoisson GLM Performance Metrics:")
print(f"Correlation (actual vs predicted): {pred_corr_poisson:.4f}")
print(f"RMSE: {rmse_poisson:.4f}")
print(f"MAE: {mae_poisson:.4f}")
print(f"Pseudo R-squared: {pseudo_r2_full:.4f}")
print(f"AIC: {poisson_full_model.aic:.4f}")
print(f"BIC: {poisson_full_model.bic:.4f}")

# Exponentiated coefficients (multiplicative effects)
print(f"\nExponentiated Coefficients (Multiplicative Effects):")
exp_coefs = np.exp(poisson_full_model.params.drop('const'))
for param, exp_coef in exp_coefs.items():
    percent_change = (exp_coef - 1) * 100
    p_value = poisson_full_model.pvalues[param]
    print(f"{param}: {exp_coef:.4f} ({percent_change:+.1f}% per unit, p={p_value:.4f})")

#%% GLM-CV Visualization
fig = plt.figure(figsize=(16, 12))

# Create a 3x3 subplot grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Actual vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_poisson, y_pred_poisson, alpha=0.6, color='blue')
ax1.plot([y_poisson.min(), y_poisson.max()], [y_poisson.min(), y_poisson.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Firing Rates')
ax1.set_ylabel('Predicted Firing Rates')
ax1.set_title(f'Poisson GLM: Actual vs Predicted\n(r = {pred_corr_poisson:.3f})')

# Plot 2: Residuals vs Fitted
ax2 = fig.add_subplot(gs[0, 1])
residuals_poisson = y_poisson - y_pred_poisson
ax2.scatter(y_pred_poisson, residuals_poisson, alpha=0.6, color='green')
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Fitted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Fitted')

# Plot 3: Pearson Residuals vs Fitted
ax3 = fig.add_subplot(gs[0, 2])
pearson_resid = poisson_full_model.resid_pearson
ax3.scatter(y_pred_poisson, pearson_resid, alpha=0.6, color='orange')
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Fitted Values')
ax3.set_ylabel('Pearson Residuals')
ax3.set_title('Pearson Residuals vs Fitted')

# Plot 4: Q-Q plot of Pearson residuals
ax4 = fig.add_subplot(gs[1, 0])
from scipy import stats
stats.probplot(pearson_resid, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot: Pearson Residuals')

# Plot 5: Histogram of residuals
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(residuals_poisson, bins=20, alpha=0.7, edgecolor='black', color='lightblue')
ax5.set_xlabel('Residuals')
ax5.set_ylabel('Frequency')
ax5.set_title('Residual Distribution')

# Plot 6: Coefficient plot with confidence intervals
ax6 = fig.add_subplot(gs[1, 2])
coef_names = X.columns.tolist()
coeffs = poisson_full_model.params.drop('const')
conf_int = poisson_full_model.conf_int().drop('const')
y_pos = np.arange(len(coef_names))

ax6.errorbar(coeffs.values, y_pos,
            xerr=[coeffs.values - conf_int.iloc[:, 0],
                  conf_int.iloc[:, 1] - coeffs.values],
            fmt='o', capsize=5)
ax6.set_yticks(y_pos)
ax6.set_yticklabels([name.replace('_', ' ').title() for name in coef_names])
ax6.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax6.set_xlabel('Coefficient Value')
ax6.set_title('Coefficients with 95% CI')

# Plot 7: Cross-validation results
ax7 = fig.add_subplot(gs[2, 0])
cv_pseudo_r2 = cv_results['pseudo_r2']
folds = range(1, len(cv_pseudo_r2) + 1)
ax7.bar(folds, cv_pseudo_r2, alpha=0.7, color='purple')
ax7.axhline(y=np.mean(cv_pseudo_r2), color='red', linestyle='--',
           label=f'Mean: {np.mean(cv_pseudo_r2):.3f}')
ax7.set_xlabel('Fold')
ax7.set_ylabel('Pseudo R²')
ax7.set_title('Cross-Validation: Pseudo R²')
ax7.legend()

# Plot 8: Individual parameter relationships
ax8 = fig.add_subplot(gs[2, 1])
# Show the strongest predictor (should be synaptic_input)
strongest_param = exp_coefs.abs().idxmax()
ax8.scatter(cell_df[strongest_param], y_poisson, alpha=0.6, color='red')
corr_strongest = cell_df[strongest_param].corr(pd.Series(y_poisson))
ax8.set_xlabel(strongest_param.replace('_', ' ').title())
ax8.set_ylabel('Firing Rates (Poisson)')
ax8.set_title(f'Strongest Predictor: {strongest_param.replace("_", " ").title()}\n(r = {corr_strongest:.3f})')

# Plot 9: Predicted vs actual (CV results)
ax9 = fig.add_subplot(gs[2, 2])
cv_actual = np.array(cv_results['actual'])
cv_predicted = np.array(cv_results['predictions'])
cv_corr, _ = stats.pearsonr(cv_actual, cv_predicted)
ax9.scatter(cv_actual, cv_predicted, alpha=0.4, color='darkgreen')
ax9.plot([cv_actual.min(), cv_actual.max()], [cv_actual.min(), cv_actual.max()], 'r--', lw=2)
ax9.set_xlabel('Actual (CV)')
ax9.set_ylabel('Predicted (CV)')
ax9.set_title(f'Cross-Validation Results\n(r = {cv_corr:.3f})')

plt.suptitle('Poisson GLM Comprehensive Analysis', fontsize=16, y=0.98)
plt.show()

# %% Compare Poisson GLM vs OLS
print("\n" + "=" * 60)
print("POISSON GLM vs OLS COMPARISON")
print("=" * 60)

# Fit OLS model with all data
ols_model = sm.OLS(y, X_with_const).fit()
y_pred_ols = ols_model.predict(X_with_const)


pred_corr_ols, _ = stats.pearsonr(y, y_pred_ols)
rmse_ols = np.sqrt(np.mean((y - y_pred_ols) ** 2))
mae_ols = np.mean(np.abs(y - y_pred_ols))

ols_r2_cv = [fold['model_summary']['rsquared_adj'] for fold in ols_cv_scores]

comparison_data = {
    'Metric': ['Correlation (r)', 'RMSE', 'MAE', 'CV Score (mean)', 'CV Score (std)', 'AIC', 'Model Fit'],
    'Poisson GLM': [
        f'{pred_corr_poisson:.4f}',
        f'{rmse_poisson:.4f}',
        f'{mae_poisson:.4f}',
        f'{cv_pseudo_r2_mean:.4f}',
        f'{cv_pseudo_r2_std:.4f}',
        f'{poisson_full_model.aic:.2f}',
        f'Pseudo R² = {pseudo_r2_full:.4f}'
    ],
    'OLS': [
        f'{pred_corr_ols:.4f}',
        f'{rmse_ols:.4f}',
        f'{mae_ols:.4f}',
        f'{np.mean(ols_r2_cv):.4f}',
        f'{np.std(ols_r2_cv):.4f}',
        f'{ols_model.aic:.2f}',
        f'R² = {ols_model.rsquared:.4f}'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Actual vs Predicted comparison
axes[0].scatter(y_poisson, y_pred_poisson, alpha=0.6, label=f'Poisson GLM (r={pred_corr_poisson:.3f})', color='blue')
axes[0].scatter(y, y_pred_ols, alpha=0.6, label=f'OLS (r={pred_corr_ols:.3f})', color='red')
axes[0].plot([min(y.min(), y_poisson.min()), max(y.max(), y_poisson.max())],
             [min(y.min(), y_poisson.min()), max(y.max(), y_poisson.max())], 'k--', lw=2)
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title('Actual vs Predicted Comparison')
axes[0].legend()

# Cross-validation comparison
cv_data = pd.DataFrame({
    'Poisson GLM': cv_results['pseudo_r2'],
    'OLS': ols_r2_cv
})
cv_data.boxplot(ax=axes[1])
axes[1].set_ylabel('Cross-Validation Score')
axes[1].set_title('Cross-Validation Performance')

# Residual comparison
axes[2].hist(residuals_poisson, alpha=0.6, label='Poisson GLM', bins=15, color='blue')
axes[2].hist(y - y_pred_ols, alpha=0.6, label='OLS', bins=15, color='red')
axes[2].set_xlabel('Residuals')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Residual Distributions')
axes[2].legend()

plt.tight_layout()
plt.show()

print(
    f"\nCross-validation shows {'Poisson GLM' if cv_pseudo_r2_mean > np.mean(ols_r2_cv) else 'OLS'} performs better on average")
