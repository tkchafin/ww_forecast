import sys 
import os 

import pandas as pd
import numpy as np 
from typing import Dict, List

from scipy.optimize import curve_fit
from typing import Generator, Tuple

from . import utils as utils
from . import lineage_mapper as lm


############ Genetic diversity metrics ##############


def get_genetic_diversity_features(
        df: pd.DataFrame, 
        lineage_mapper: lm.LineageMapper, 
        threshold: float = 0.001,                     
        abundance_col: str = "Weighted Abundance") -> pd.DataFrame:
    """
    Calculate the diversity metrics for each date.

    Args:
        df (pd.DataFrame): The dataframe to process. 
                           Contains dates and corresponding lineage abundances.
        lineage_mapper (LineageMapper): The LineageMapper object.
        threshold (float): Minimum abundance to consider a lineage for a given date.
        abundance_col (str): Name of the column representing abundance.

    Returns:
        pd.DataFrame: A dataframe with the calculated data. 
                      Each row corresponds to a unique date, 
                      and the columns correspond to the diversity metric for each gene and the whole genome. 
                      Column names are in the format: {gene_name}_Pi or Genome_Pi.
    """
    
    # Initialize an empty DataFrame to store the results
    diversity_df = pd.DataFrame()

    # Make a copy of the input DataFrame to avoid modifying it
    df_copy = df.copy()

    # Set 'Date' as the DataFrame index
    df_copy.set_index('Date', inplace=True)

    # Filter out lineages below threshold
    df_copy = df_copy[df_copy[abundance_col] > threshold]

    # For each unique date in df_copy
    for date in df_copy.index.unique():

        # Subset the df_copy for the given date
        date_df = df_copy.loc[date]

        # Calculate the diversity metrics using the LineageMapper method
        pi_dict = lineage_mapper.get_nucleotide_diversity(date_df, abundance_col)

        # Add the results to the diversity_df DataFrame
        for column, value in pi_dict.items():
            diversity_df.loc[date, column] = value

    return diversity_df.fillna(0.0)



######## Summary metrics for raw abundances #########

def get_abundance_features(df: pd.DataFrame, threshold: float = 0.001, 
                           abundance_col: str = "Weighted Abundance", 
                           aggregator_funcs: List[str] = ["all"]) -> pd.DataFrame:
    """
    Calculate various statistics about lineage abundance.

    Args:
        df (pd.DataFrame): The dataframe to process.
        threshold (float): Threshold for filtering lineages by abundance.
        abundance_col (str): Name of the column representing abundance.
        aggregator_funcs (List[str]): List of strings representing the name of functions to apply to each group.
            Options are: 
            - 'num': Counts the number of lineages. Equivalent to len in pandas.
            - 'mean': Computes the mean abundance.
            - 'evenness': Computes Simpson's Evenness of abundances. This is a custom function defined elsewhere.
            - 'diversity': Computes Simpson's Diversity. This is a custom function defined elsewhere.
            - 'std': Computes the standard deviation of abundance.
            - 'max': Finds the maximum abundance.
            - 'min': Finds the minimum abundance.
            - 'all': Applies all of the above functions.

    Returns:
        pd.DataFrame: A dataframe with the aggregated data. Each row corresponds to a unique date, and the columns
            correspond to the aggregated abundance statistics for each date. Column names are in the format:
            {statistic}_{abundance_col}.
    """
    
    func_dict = {
        "num": lambda x: len(x),
        "mean": lambda x: np.mean(x),
        "evenness": lambda x: utils.simpsons_evenness(x),
        "diversity": lambda x: utils.simpsons_diversity(x),
        "std": lambda x: np.std(x),
        "max": lambda x: np.max(x),
        "min": lambda x: np.min(x),
        "all": None
    }
    
    # Filter out lineages below threshold
    df = df[df[abundance_col] > threshold]

    # Create list of functions to apply
    if 'all' in aggregator_funcs:
        funcs_to_apply = [func for func, _ in func_dict.items() if func != 'all']
    else:
        funcs_to_apply = [func for func in aggregator_funcs if func_dict[func] is not None]
        
    agg_funcs = [func_dict[func] for func in funcs_to_apply]
    agg_df = df.groupby("Date")[abundance_col].agg(agg_funcs)

    # Create new column names
    new_col_names = [f'Abundance_{func}' for func in funcs_to_apply]
    agg_df.columns = new_col_names

    return agg_df



########### Lineage logistic growth rates ###########

def get_growth_rate_features(
        df: pd.DataFrame,
        window_width: int, 
        threshold: float, 
        abundance_col: str,
        serial_interval: float,
        aggregator_funcs: list) -> pd.DataFrame:
    """
    Calculate various statistics about lineage growth rates.
    
    Args:
        df (pd.DataFrame): The dataframe to process.
        window_width (int): Width of the window for computing growth rates.
        threshold (float): Threshold for filtering lineages by abundance.
        abundance_col (str): Name of the column representing abundance.
        serial_interval (float): The average time between successive cases in a chain transmission.
        aggregator_funcs (list): List of strings representing the name of functions to apply to each group.
            Options are: 
            - 'mean': Computes the mean growth rate, ignoring NaNs.
            - 'median': Computes the median growth rate, ignoring NaNs.
            - 'max': Finds the maximum growth rate.
            - 'min': Finds the minimum growth rate.
            - 'std': Computes the standard deviation of growth rate, ignoring NaNs.
            - 'fastest_lineage_abundance': Placeholder, will need to be replaced with a function.
            - 'slowest_lineage_abundance': Placeholder, will need to be replaced with a function.
            - 'weighted_mean': Placeholder, will need to be replaced with a function.
            - 'all': Applies all of the above functions.

    Returns:
        pd.DataFrame: A dataframe with the aggregated data. Each row corresponds to a unique date, and the columns
            correspond to the aggregated growth rate statistics for each date. Column names are in the format:
            {statistic}_{abundance_col}.
    """
    # Define mapping from function name strings to actual functions
    func_dict = {
        "mean": lambda x: np.nanmean(x),
        "median": lambda x: np.nanmedian(x),
        "max": np.max,
        "min": np.min,
        "std": lambda x: np.nanstd(x),
        "fastest_lineage_abundance": None,  
        "slowest_lineage_abundance": None, 
        "weighted_mean": None, 
        "all": None
    }

    # calc growth rates in a sliding window
    window_growth_df = calc_growth_sliding_window(
        df, window_width, threshold, abundance_col, serial_interval
    )

    agg_df = pd.DataFrame()

    # Handle weighted mean separately
    if 'weighted_mean' in aggregator_funcs or 'all' in aggregator_funcs:
        mask = ~np.isnan(window_growth_df['Estimated Advantage'])
        window_growth_df.loc[mask, 'Weighted Advantage'] = window_growth_df.loc[mask, 'Estimated Advantage'] * window_growth_df.loc[mask, abundance_col]
        weight_sum = window_growth_df.groupby('Date')[abundance_col].sum()
        weighted_sum = window_growth_df.groupby('Date')['Weighted Advantage'].sum()
        weighted_df = (weighted_sum / weight_sum).reset_index()
        weighted_df.columns = ['Date', 'Advantage_weighted_mean']
        agg_df = pd.concat([agg_df, weighted_df], axis=1)

    # Calculate abundances of max and min growth rate lineages
    if 'fastest_lineage_abundance' in aggregator_funcs or 'all' in aggregator_funcs:
        idxmax = window_growth_df.groupby('Date')['Estimated Advantage'].idxmax()
        max_df = window_growth_df.loc[idxmax, ['Date', abundance_col]].rename(columns={abundance_col: 'Fastest_lineage_abundance'})
        agg_df = pd.merge(agg_df, max_df, how='left', on='Date')
        
    if 'slowest_lineage_abundance' in aggregator_funcs or 'all' in aggregator_funcs:
        idxmin = window_growth_df.groupby('Date')['Estimated Advantage'].idxmin()
        min_df = window_growth_df.loc[idxmin, ['Date', abundance_col]].rename(columns={abundance_col: 'Slowest_lineage_abundance'})
        agg_df = pd.merge(agg_df, min_df, how='left', on='Date')

    # Group by date and apply aggregator functions (for remaining functions not handled separately)
    if 'all' in aggregator_funcs:
        funcs_to_apply = [func for func, _ in func_dict.items() if func if func_dict[func] is not None]
    else:
        funcs_to_apply = [func for func in aggregator_funcs if func_dict[func] is not None]

    for func in funcs_to_apply:
        agg_func = func_dict[func]
        agg_df_other = window_growth_df.groupby("Date")['Estimated Advantage'].agg(agg_func)
        agg_df_other.name = f'Advantage_{func}'
        agg_df = pd.merge(agg_df, agg_df_other.reset_index(), how='left', on='Date')

    return agg_df.set_index('Date', inplace=False)




def calc_growth_sliding_window(
        df: pd.DataFrame, 
        window_width: int = 14, 
        threshold: float = 0.001, 
        abundance_col: str = "Weighted Abundance", 
        serial_interval: float = 5.5,
        sliding_interval: int = 1
    ) -> pd.DataFrame:
    """
    Adapted from Freyja v1.3.1.

    Calculates relative growth rates for all lineages in sliding windows across the given dataframe.

    Parameters:
    df (pd.DataFrame): The data as a dataframe, must include 'Date' and 'Lineage' columns.
    window_width (int): The width of the sliding window in days.
    threshold (float): The abundance threshold to filter lineages.
    abundance_col (str): The column in the dataframe representing abundance.
    serial_interval (float): The serial interval of the disease.

    Returns:
    pd.DataFrame: A dataframe of relative growth rates for each lineage in each window.
    """
    # Initialize list to store DataFrame of each window's results
    results = []

    # Get the overall start and end dates
    start_date = df['Date'].min()
    end_date = df['Date'].max()

    # Loop over all windows
    for window_start, window_end in utils.get_window_dates(start_date, end_date, sliding_interval, window_width):
        # Subset the dataframe for the current window
        subset = df[(df['Date'] >= window_start) & (df['Date'] <= window_end)].copy()

        # Calculate mean abundance for each lineage over the window
        mean_abundances = subset.copy().groupby('Lineage')[abundance_col].mean().reset_index()
        total_mean_abundance = mean_abundances[abundance_col].sum()
        mean_abundances[abundance_col] = mean_abundances[abundance_col] / total_mean_abundance

        # Calculate growth rates for the current window
        window_growth_rates = calc_growth_rate(
            subset, 
            threshold=threshold, 
            serial_interval=serial_interval, 
            abundance_col=abundance_col,
            nboots=None)

        window_growth_rates["Date"] = window_end
        window_growth_rates['Estimated Advantage'] = window_growth_rates['Estimated Advantage'].astype(float)
        window_growth_rates = pd.merge(
            window_growth_rates,
            mean_abundances,on='Lineage', how='left')
        results.append(window_growth_rates)
    results_df = pd.concat(results, ignore_index=True)
    return results_df


def calc_growth_rate(
    df: pd.DataFrame, 
    threshold: float = 0.001, 
    serial_interval: float = 5.5, 
    abundance_col: str = "Weighted Abundance",
    nboots: int = None
) -> pd.DataFrame:
    """
    Calculates the growth rate of each lineage over the entire time window of the dataframe.
    
    Parameters:
    df (pd.DataFrame): The data as a dataframe, must include 'Date', 'Lineage' and abundance_col columns.
    threshold (float): The abundance threshold to filter lineages.
    serial_interval (float): The serial interval of the disease.
    abundance_col (str): The column in the dataframe representing abundance.
    nboots (int): Number of bootstrap samples to generate for calculating confidence intervals. If None, no bootstrapping is performed.
    
    Returns:
    pd.DataFrame: A dataframe with the lineage and its estimated advantage.
    """
    # Identify lineages with mean abundance > threshold
    lineage_mean_abundances = df.groupby('Lineage')[abundance_col].mean()
    valid_lineages = lineage_mean_abundances[lineage_mean_abundances > threshold].index
    
    # Initialize DataFrame to store results
    results = pd.DataFrame(columns=['Estimated Advantage'], index=valid_lineages)
    if nboots is not None:
        results['Lower Bound'] = np.nan
        results['Upper Bound'] = np.nan

    # Calculate growth rates for each valid lineage
    for lineage in valid_lineages:
        subset = df[df['Lineage'] == lineage]
        days = np.array([(date - subset['Date'].min()).days for date in subset['Date']])
        data = subset[abundance_col]
        
        # Skip if not enough data points
        if len(data) < 2:
            continue

        fit, covar = curve_fit(
            f=utils.logistic_growth,
            xdata=days,
            ydata=data,
            p0=[100, 0.1],
            bounds=([0, -10], [1000, 10])
        )
        rate0 = fit[1]
        trans_increase = serial_interval * rate0
        results.loc[lineage, 'Estimated Advantage'] = trans_increase

        # Bootstrapping if specified
        if nboots is not None:
            boot_rates = []
            for _ in range(nboots):
                boot_data = np.random.choice(data, size=len(data), replace=True)
                boot_fit, boot_covar = curve_fit(
                    f=logistic_growth,
                    xdata=days,
                    ydata=boot_data,
                    p0=[100, 0.1],
                    bounds=([0, -10], [1000, 10])
                )
                boot_rate0 = boot_fit[1]
                boot_trans_increase = serial_interval * boot_rate0
                boot_rates.append(boot_trans_increase)

            # Confidence intervals
            lower_bound, upper_bound = np.percentile(boot_rates, [2.5, 97.5])
            results.loc[lineage, 'Lower Bound'] = lower_bound
            results.loc[lineage, 'Upper Bound'] = upper_bound

    results.reset_index(drop=False, inplace=True)

    # Include lineages with low mean abundance
    low_abundance_lineages = set(df['Lineage']) - set(valid_lineages)
    low_abundance_df = pd.DataFrame({'Lineage': list(low_abundance_lineages), 'Estimated Advantage': np.nan})
    if nboots is not None:
        low_abundance_df['Lower Bound'] = np.nan
        low_abundance_df['Upper Bound'] = np.nan

    # Concatenate the results
    results = pd.concat([results, low_abundance_df], ignore_index=True)

    return results