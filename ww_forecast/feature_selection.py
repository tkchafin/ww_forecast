import os 
import sys 

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.backends.backend_pdf import PdfPages


def feature_correlation(data: pd.DataFrame, method: str = 'pearson', prefix: str = 'correlation', 
                        plot: bool = True) -> pd.DataFrame:    
    """
    Calculate the correlation matrix and optionally plot it.
    
    Args:
    data (pd.DataFrame): Input data for correlation calculation.
    method (str): Correlation method, either 'pearson' or 'spearman'. Default is 'pearson'.
    prefix (str): Prefix for the output CSV and PDF files. Default is 'correlation'.
    plot (bool): Whether to plot the heatmap. Default is True.
    
    Returns:
    corr_matrix (pd.DataFrame): The correlation matrix.
    """
    
    # Calculate the correlation matrix
    corr_matrix = data.corr(method=method)
    
    # Save the correlation matrix to CSV
    corr_matrix.to_csv(f"{prefix}_{method}_correlation.csv")
    
    # Optionally plot the heatmap
    if plot:
        with PdfPages(f"{prefix}_{method}_correlation_heatmap.pdf") as pdf:
            plt.figure(figsize=(12, 10))
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f",
                        annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

            # Reducing title size for a cleaner look
            plt.title(f"{method.capitalize()} Correlation Heatmap", fontsize=12)
            
            # Reducing xtick and ytick label sizes for a cleaner look
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
    
    return corr_matrix


def feature_selection_RF(X: pd.DataFrame, y: pd.Series, n_trees: int = None, max_trees: int = 1000, 
                         max_depth: int = None, min_samples_split: int = 2, threads: int = 1,
                         min_samples_leaf: int = 1, max_features: str = 'sqrt', bootstrap: bool = True, 
                         max_samples: float = None, pdp_features: int = 3, prefix: str = "test", plot: bool = True) -> tuple:
    """
    Applies Random Forest Regressor for feature selection and generates relevant plots.

    Args: 
    X : pd.DataFrame
        Features matrix.
    y : pd.Series
        Target variable.
    n_trees : int, optional
        Number of trees to use without OOB optimization. If set, OOB optimization is skipped. Default is None.
    max_trees : int, optional
        Maximum number of trees to consider for OOB optimization. Default is 1000.
    max_depth : int, optional
        Maximum depth of the tree. Default is None.
    min_samples_split : int, optional
        Minimum number of samples required to split a node. Default is 2.
    min_samples_leaf : int, optional
        Minimum number of samples required to be at a leaf node. Default is 1.
    max_features : str, optional
        Number of features to consider when looking for the best split. Default is 'sqrt'.
    bootstrap : bool, optional
        Whether bootstrap samples are used when building trees. Default is True.
    max_samples : float or None, optional
        Number of samples to draw for bootstrapping. Default is None.
    pdp_features : int, optional
        Number of top features to consider for partial dependence plot. Default is 3.
    prefix : str, optional
        Prefix for saving output files. Default is "test".
    plot : bool, optional
        Whether to produce plots. Default is True.

    Returns:
    tuple:
        - pd.Series: Sorted feature importances.
        - pd.Series: Residuals after prediction.
    """
    
    if n_trees is None:
        # Define the RF regressor with warm_start=True to grow the number of trees incrementally
        rf = RandomForestRegressor(
            warm_start=True, 
            oob_score=True, 
            bootstrap=bootstrap, 
            max_depth=max_depth,
            n_jobs=threads,
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf,
            max_features=max_features, 
            max_samples=max_samples)

        oob_errors = []
        for i in range(1, max_trees + 1):
            rf.set_params(n_estimators=i)
            rf.fit(X, y)
            oob_error = 1 - rf.oob_score_
            oob_errors.append(oob_error)
            
        # Identify optimal number of trees using the Kneedle algorithm
        kneedle = KneeLocator(range(1, max_trees+1), oob_errors, curve='convex', direction='decreasing')
        optimal_trees = kneedle.elbow

        # Rerun RF with optimal number of trees (without warm_start to finalize the model)
        rf.set_params(n_estimators=optimal_trees, warm_start=False)
        rf.fit(X, y)
    else:
        optimal_trees = n_trees
        rf = RandomForestRegressor(
            n_estimators=optimal_trees, 
            max_depth=max_depth, 
            n_jobs=threads,
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            max_features=max_features, 
            bootstrap=bootstrap, 
            max_samples=max_samples)
        rf.fit(X, y)
        oob_errors = None  # Set to None since we don't compute them in this case

    # Predictions
    y_pred = rf.predict(X)
    residuals = y - y_pred

    # Get and sort the feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    ranks = importances.sort_values(ascending=False)

    # Save importances and residuals to CSV
    ranks.to_csv(f"{prefix}_RF_ranks.csv")
    residuals.to_csv(f"{prefix}_RF_residuals.csv")

    # Plot if required
    if plot:
        with PdfPages(f"{prefix}_RF_plots.pdf") as pdf:
            
            # Feature importances
            plt.figure(figsize=(10, 5))
            ranks.plot(kind="bar", title="Feature Importances", xlabel="Features", ylabel="Importance")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Predicted vs. Actual values
            plt.figure(figsize=(10, 5))
            plt.scatter(y, y_pred, alpha=0.5)
            plt.title("Predicted vs. Actual Values")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Residuals plot
            plt.figure(figsize=(10, 5))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.title("Residuals vs. Predicted Values")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # PDP for top features
            top_features = ranks.index[:pdp_features].tolist()
            PartialDependenceDisplay.from_estimator(
                rf, 
                X, 
                top_features, 
                kind='both')
            plt.suptitle(f"PDP for top {pdp_features} features", y=1.05)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # OOB error rate plot
            if oob_errors is not None:
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, max_trees+1), oob_errors, label="OOB Error Rate")
                plt.axvline(optimal_trees, color='red', linestyle='--', label=f"Optimal trees: {optimal_trees}")
                plt.xlabel("Number of Trees")
                plt.ylabel("OOB Error Rate")
                plt.title("OOB Error Rate vs. Number of Trees")
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

    return ranks, residuals

