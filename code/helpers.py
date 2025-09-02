"""
This is a helpers file for all of our helper functions.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
import scipy.stats as stats
from scipy.special import logit as scipy_logit, expit
from sklearn.model_selection import KFold
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.families.links import Log

import glob

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
                       family=Poisson(link=Log())).fit()

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

def beta_regression_manual(X, y, max_iter=100, tol=1e-6):
    """
    Manual implementation of Beta regression using iterative fitting
    """
    X_const = sm.add_constant(X)

    # Initialize with OLS on logit-transformed data
    y_logit = scipy_logit(y)
    ols_init = sm.OLS(y_logit, X_const).fit()
    beta = ols_init.params.values

    # Iterative fitting (simplified IRLS)
    for i in range(max_iter):
        eta = X_const @ beta
        mu = expit(eta)  # inverse logit

        # Beta regression weights (simplified)
        var_mu = mu * (1 - mu)
        weights = 1 / var_mu
        weights = np.clip(weights, 0.1, 100)  # Stabilize weights

        # Weighted least squares update
        W = np.diag(weights)
        try:
            beta_new = np.linalg.solve(X_const.T @ W @ X_const, X_const.T @ W @ y_logit)
            if np.linalg.norm(beta_new - beta) < tol:
                break
            beta = beta_new
        except np.linalg.LinAlgError:
            break

    # Final predictions
    final_eta = X_const @ beta
    predictions = expit(final_eta)

    return {
        'coefficients': beta,
        'predictions': predictions,
        'fitted_values': final_eta,
        'param_names': ['const'] + list(X.columns)
    }


def fit_beta_model_with_cv(X, y, formula=None, add_constant=True, precision_formula=None,
                           transform_to_unit=True, epsilon=1e-6, cv_folds=5, random_state=42):
    """
    Fit a Beta regression model using statsmodels BetaModel with cross-validation

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.array
        Predictor variables
    y : pandas.Series or numpy.array
        Response variable (should be in (0,1) or will be transformed)
    formula : str, optional
        Formula string for the model. If None, will auto-generate from column names
    add_constant : bool, default=True
        Whether to add intercept term
    precision_formula : str, optional
        Formula for precision parameter modeling
    transform_to_unit : bool, default=True
        Whether to transform y to (0,1) interval if it's not already
    epsilon : float, default=1e-6
        Small value to ensure y is strictly between 0 and 1
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    dict : Dictionary containing full model results, CV results, and predictions
    """

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if X.ndim == 1:
            X = pd.DataFrame({'x': X})
        else:
            X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])

    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='y')

    # Transform y to (0,1) if needed
    y_beta = y.copy()
    original_range = (y.min(), y.max())

    if transform_to_unit:
        if y.min() < 0 or y.max() > 1:
            print(f"Transforming y from [{y.min():.3f}, {y.max():.3f}] to (0,1)")
            y_beta = (y - y.min()) / (y.max() - y.min())

    # Ensure y is strictly between 0 and 1
    y_beta = np.clip(y_beta, epsilon, 1 - epsilon)

    # Auto-generate formula if not provided
    if formula is None:
        predictor_names = X.columns.tolist()
        formula = f"y ~ {' + '.join(predictor_names)}"
        print(f"Auto-generated formula: {formula}")

    # Perform Cross-Validation
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    print("=" * 50)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_results = {
        'correlations': [],
        'rmse_scores': [],
        'mae_scores': [],
        'pseudo_r2_scores': [],
        'log_likelihoods': [],
        'coefficients': [],
        'pvalues': [],
        'predictions': [],
        'actual': [],
        'fold_info': []
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{cv_folds}", end=" - ")

        try:
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_beta.iloc[train_idx], y_beta.iloc[test_idx]

            # Add constant if requested
            if add_constant:
                X_train_model = sm.add_constant(X_train)
                X_test_model = sm.add_constant(X_test)
            else:
                X_train_model = X_train.copy()
                X_test_model = X_test.copy()

            # Fit Beta model on training data
            beta_model = BetaModel(endog=y_train, exog=X_train_model)
            beta_results = beta_model.fit(disp=0)  # Suppress output

            # Make predictions on test data
            y_pred = beta_results.predict(X_test_model)

            # Transform predictions back to original scale if needed
            if transform_to_unit and (original_range[0] < 0 or original_range[1] > 1):
                y_pred_orig = y_pred * (original_range[1] - original_range[0]) + original_range[0]
                y_test_orig = y_test * (original_range[1] - original_range[0]) + original_range[0]
            else:
                y_pred_orig = y_pred
                y_test_orig = y_test

            # Calculate metrics
            from scipy.stats import pearsonr
            corr, _ = pearsonr(y_test_orig, y_pred_orig)
            rmse = np.sqrt(np.mean((y_test_orig - y_pred_orig) ** 2))
            mae = np.mean(np.abs(y_test_orig - y_pred_orig))

            # Pseudo R² for test set
            ss_res = np.sum((y_test_orig - y_pred_orig) ** 2)
            ss_tot = np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)
            pseudo_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Store results
            cv_results['correlations'].append(corr)
            cv_results['rmse_scores'].append(rmse)
            cv_results['mae_scores'].append(mae)
            cv_results['pseudo_r2_scores'].append(pseudo_r2)
            cv_results['log_likelihoods'].append(beta_results.llf)
            cv_results['coefficients'].append(beta_results.params.values)
            cv_results['pvalues'].append(beta_results.pvalues.values)
            cv_results['predictions'].extend(y_pred_orig)
            cv_results['actual'].extend(y_test_orig)
            cv_results['fold_info'].append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'model_aic': beta_results.aic,
                'model_bic': beta_results.bic
            })

            print(f"r = {corr:.3f}, RMSE = {rmse:.3f}, Pseudo R² = {pseudo_r2:.3f}")

        except Exception as e:
            print(f"FAILED - {str(e)}")
            continue

    # Calculate CV summary statistics
    if cv_results['correlations']:
        cv_summary = {
            'mean_correlation': np.mean(cv_results['correlations']),
            'std_correlation': np.std(cv_results['correlations']),
            'mean_rmse': np.mean(cv_results['rmse_scores']),
            'std_rmse': np.std(cv_results['rmse_scores']),
            'mean_mae': np.mean(cv_results['mae_scores']),
            'std_mae': np.std(cv_results['mae_scores']),
            'mean_pseudo_r2': np.mean(cv_results['pseudo_r2_scores']),
            'std_pseudo_r2': np.std(cv_results['pseudo_r2_scores']),
            'mean_log_likelihood': np.mean(cv_results['log_likelihoods']),
            'std_log_likelihood': np.std(cv_results['log_likelihoods'])
        }

        print(f"\nCross-Validation Summary:")
        print("=" * 30)
        print(f"Mean Correlation: {cv_summary['mean_correlation']:.4f} ± {cv_summary['std_correlation']:.4f}")
        print(f"Mean RMSE: {cv_summary['mean_rmse']:.4f} ± {cv_summary['std_rmse']:.4f}")
        print(f"Mean MAE: {cv_summary['mean_mae']:.4f} ± {cv_summary['std_mae']:.4f}")
        print(f"Mean Pseudo R²: {cv_summary['mean_pseudo_r2']:.4f} ± {cv_summary['std_pseudo_r2']:.4f}")

        # Average coefficients across folds
        if cv_results['coefficients']:
            coeff_array = np.array(cv_results['coefficients'])
            mean_coeffs = np.mean(coeff_array, axis=0)
            std_coeffs = np.std(coeff_array, axis=0)

            if add_constant:
                param_names = ['const'] + X.columns.tolist()
            else:
                param_names = X.columns.tolist()

            print(f"\nAverage Coefficients Across Folds:")
            for i, name in enumerate(param_names):
                if i < len(mean_coeffs):
                    print(f"{name:>15}: {mean_coeffs[i]:>8.4f} ± {std_coeffs[i]:.4f}")
    else:
        cv_summary = None
        print("Cross-validation failed for all folds")

    # Fit full model on all data
    print(f"\nFitting full model on all data...")
    print("=" * 40)

    try:
        # Add constant if requested
        if add_constant:
            X_full = sm.add_constant(X)
        else:
            X_full = X.copy()

        # Fit full Beta model
        beta_model_full = BetaModel(endog=y_beta, exog=X_full)
        beta_results_full = beta_model_full.fit()

        print("Beta Model fitted successfully!")
        print(beta_results_full.summary())

        # Make predictions on full data
        y_pred_full = beta_results_full.predict(X_full)

        # Transform predictions back to original scale if needed
        if transform_to_unit and (original_range[0] < 0 or original_range[1] > 1):
            y_pred_full_orig = y_pred_full * (original_range[1] - original_range[0]) + original_range[0]
            y_full_orig = y
        else:
            y_pred_full_orig = y_pred_full
            y_full_orig = y_beta

        # Calculate full model performance metrics
        from scipy.stats import pearsonr
        corr_full, p_val_full = pearsonr(y_full_orig, y_pred_full_orig)
        rmse_full = np.sqrt(np.mean((y_full_orig - y_pred_full_orig) ** 2))
        mae_full = np.mean(np.abs(y_full_orig - y_pred_full_orig))

        # Pseudo R-squared for full model
        ll_full = beta_results_full.llf
        ll_null = BetaModel(endog=y_beta, exog=np.ones((len(y_beta), 1))).fit(disp=0).llf
        pseudo_r2_full = 1 - (ll_full / ll_null)

        print(f"\nFull Model Performance:")
        print(f"Correlation: {corr_full:.4f} (p-value: {p_val_full:.4f})")
        print(f"RMSE: {rmse_full:.4f}")
        print(f"MAE: {mae_full:.4f}")
        print(f"Pseudo R²: {pseudo_r2_full:.4f}")
        print(f"AIC: {beta_results_full.aic:.4f}")
        print(f"BIC: {beta_results_full.bic:.4f}")

        full_model_results = {
            'model': beta_results_full,
            'predictions': y_pred_full,
            'predictions_original_scale': y_pred_full_orig,
            'actual_original_scale': y_full_orig,
            'correlation': corr_full,
            'correlation_pvalue': p_val_full,
            'rmse': rmse_full,
            'mae': mae_full,
            'pseudo_r2': pseudo_r2_full,
            'aic': beta_results_full.aic,
            'bic': beta_results_full.bic,
            'log_likelihood': ll_full
        }

    except Exception as e:
        print(f"Error fitting full Beta model: {e}")
        full_model_results = None

    # Compile final results
    final_results = {
        'full_model': full_model_results,
        'cv_results': cv_results,
        'cv_summary': cv_summary,
        'formula': formula,
        'transform_applied': transform_to_unit and (original_range[0] < 0 or original_range[1] > 1),
        'original_range': original_range,
        'n_successful_folds': len(cv_results['correlations']) if cv_results['correlations'] else 0,
        'total_folds': cv_folds
    }

    return final_results


def plot_beta_cv_results(results):
    """
    Plot comprehensive results from Beta regression with cross-validation
    """
    if results is None or results['full_model'] is None:
        print("No valid results to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    full_model = results['full_model']
    cv_results = results['cv_results']
    cv_summary = results['cv_summary']

    # Plot 1: Full model - Actual vs Predicted
    axes[0, 0].scatter(full_model['actual_original_scale'], full_model['predictions_original_scale'],
                       alpha=0.6, color='blue', label='Full Model')
    min_val = min(full_model['actual_original_scale'].min(), full_model['predictions_original_scale'].min())
    max_val = max(full_model['actual_original_scale'].max(), full_model['predictions_original_scale'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Full Model: Actual vs Predicted\nr = {full_model["correlation"]:.3f}')

    # Plot 2: CV - Actual vs Predicted
    if cv_results['actual'] and cv_results['predictions']:
        axes[0, 1].scatter(cv_results['actual'], cv_results['predictions'],
                           alpha=0.4, color='green', label='CV Predictions')
        cv_corr, _ = pearsonr(cv_results['actual'], cv_results['predictions'])
        min_val = min(min(cv_results['actual']), min(cv_results['predictions']))
        max_val = max(max(cv_results['actual']), max(cv_results['predictions']))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Values (CV)')
        axes[0, 1].set_ylabel('Predicted Values (CV)')
        axes[0, 1].set_title(f'Cross-Validation: Actual vs Predicted\nr = {cv_corr:.3f}')

    # Plot 3: Residuals vs Fitted (Full Model)
    residuals = full_model['actual_original_scale'] - full_model['predictions_original_scale']
    axes[0, 2].scatter(full_model['predictions_original_scale'], residuals, alpha=0.6, color='purple')
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Fitted Values')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Full Model: Residuals vs Fitted')

    # Plot 4: CV Correlation across folds
    if cv_results['correlations']:
        folds = range(1, len(cv_results['correlations']) + 1)
        axes[1, 0].bar(folds, cv_results['correlations'], alpha=0.7, color='orange')
        if cv_summary:
            axes[1, 0].axhline(y=cv_summary['mean_correlation'], color='red', linestyle='--',
                               label=f'Mean: {cv_summary["mean_correlation"]:.3f}')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Cross-Validation: Correlation by Fold')
        axes[1, 0].legend()

    # Plot 5: CV RMSE across folds
    if cv_results['rmse_scores']:
        folds = range(1, len(cv_results['rmse_scores']) + 1)
        axes[1, 1].bar(folds, cv_results['rmse_scores'], alpha=0.7, color='red')
        if cv_summary:
            axes[1, 1].axhline(y=cv_summary['mean_rmse'], color='black', linestyle='--',
                               label=f'Mean: {cv_summary["mean_rmse"]:.3f}')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Cross-Validation: RMSE by Fold')
        axes[1, 1].legend()

    # Plot 6: Coefficient stability across folds
    if cv_results['coefficients'] and len(cv_results['coefficients']) > 0:
        coeff_array = np.array(cv_results['coefficients'])
        if coeff_array.ndim == 2:
            param_names = ['const'] + list(results['full_model']['model'].exog_names[1:]) if \
            results['full_model']['model'].exog_names[0] == 'const' else list(results['full_model']['model'].exog_names)

            # Plot first few coefficients (excluding intercept if present)
            start_idx = 1 if param_names[0] == 'const' else 0
            n_params = min(4, coeff_array.shape[1] - start_idx)  # Show up to 4 parameters

            for i in range(n_params):
                param_idx = start_idx + i
                if param_idx < len(param_names) and param_idx < coeff_array.shape[1]:
                    axes[1, 2].plot(range(1, coeff_array.shape[0] + 1), coeff_array[:, param_idx],
                                    'o-', alpha=0.7, label=param_names[param_idx][:10])

            axes[1, 2].set_xlabel('Fold')
            axes[1, 2].set_ylabel('Coefficient Value')
            axes[1, 2].set_title('Coefficient Stability Across Folds')
            axes[1, 2].legend()

    plt.suptitle(f'Beta Regression with Cross-Validation\n{results["formula"]}', fontsize=14)
    plt.tight_layout()
    plt.show()

    