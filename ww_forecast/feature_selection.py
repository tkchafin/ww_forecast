import os 
import sys 

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_pdf import PdfPages


def feature_correlation(data, method='pearson', prefix='correlation', plot=True):
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
            
            # Custom formatter function for annotations
            fmt = lambda x: f"{x:.2f}"
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt="",
                        annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

            # Reducing title size for a cleaner look
            plt.title(f"{method.capitalize()} Correlation Heatmap", fontsize=12)
            
            # Reducing xtick and ytick label sizes for a cleaner look
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
    
    return corr_matrix

def feature_selection_RF(X, y, n_estimators=100, max_depth=None, min_samples_split=2, 
                         min_samples_leaf=1, max_features='sqrt', bootstrap=True, max_samples=None, 
                         prefix="test", plot=True):
    """
    Applies Random Forest Regressor for feature selection.

    Args:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
    max_depth (int, optional): The maximum depth of the tree. If None, then nodes are expanded until 
                               all leaves are pure. Defaults to None.
    min_samples_split (int, optional): The minimum number of samples required to split an internal node.
                                       Defaults to 2.
    min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node. Defaults to 1.
    max_features (str, optional): The number of features to consider when looking for the best split.
                                  Can be int, float, string or None. Defaults to 'sqrt'.
    bootstrap (bool, optional): Whether bootstrap samples are used when building trees. Defaults to True.
    max_samples (float/int, optional): If bootstrap is True, the number of samples to draw from available samples 
                                       to train each tree. If None (default), then draw `X.shape[0]` samples.
    prefix (str, optional): Prefix for output files. Defaults to 'test'.
    plot (bool, optional): Whether to plot the feature importances and residuals. Defaults to True.

    Returns:
    importances (pd.Series): Sorted feature importances.
    residuals (pd.Series): Residuals after prediction.
    """
    
    # Instantiate and fit the random forest regressor
    rf = RandomForestRegressor(n_estimators=n_estimators, 
                               max_depth=max_depth, 
                               min_samples_split=min_samples_split, 
                               min_samples_leaf=min_samples_leaf, 
                               max_features=max_features, 
                               bootstrap=bootstrap, 
                               max_samples=max_samples)
    rf.fit(X, y)

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

    return ranks, residuals
