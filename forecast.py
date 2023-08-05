import os
import sys

import argparse
import pandas as pd
from pathlib import Path

from ww_forecast.model_data import ModelData
from ww_forecast.lineage_mapper import LineageMapper
import ww_forecast.feature_selection as fs

def main():

    params = parseArgs()

    #############################################################
    # 1. Reading in and creating features 

    # load lineage metadata 
    if params.features_file is None:
        lineage_map = LineageMapper(
            lineages=params.lineages,
            usher_barcodes=params.usher_barcodes,
            curated_lineages=params.curated_lineages,
            regions=params.regions,
            mutations=params.mutations, 
            consensus_type=params.consensus
        )
        lineage_map.write(params.prefix)
    else:
        lineage_map = None

    # load input features 
    model_data = ModelData(
        lineages_file = params.lineages, 
        prevalence_file = params.prevalence, 
        threshold = params.threshold, 
        abundance_col = params.abundance_col, 
        peak_threshold = params.peak_threshold, 
        prevalence_col = params.prevalence_col, 
        serial_interval = params.serial_interval, 
        window_width = params.window_width, 
        features_file=params.features_file,
        lineage_map = lineage_map, 
        prefix=params.prefix)

    if params.extra_features is not None:
        model_data.add_features(params.extra_features, impute=True)

    # make some plots 
    model_data.plot_features_pairwise()
    model_data.plot_features_vs_prevalence()
    model_data.plot_features_vs_prevalence_timeseries()

    #############################################################
    # 2. Feature selection and transformation 

    # scale features and target 
    model_data.scale_data(scaling_method='standard', scale_target=True)

    # correlation 
    pearson = fs.feature_correlation(
        model_data.features,
        method="pearson",
        prefix=params.prefix,
        plot=True)
    
    spearman = fs.feature_correlation(
        model_data.features,
        method="spearman",
        prefix=params.prefix,
        plot=True)

    # drop features (hard-coded for now)
    model_data.drop_features(["S_Pi", 
                              "E_Pi", 
                              "M_Pi",
                              "N_Pi",
                              "Abundance_num",
                              "Abundance_max",
                              "Abundance_min", 
                              "Advantage_min",
                              "Advantage_max", 
                              "Abundance_std", 
                              "Abundance_diversity", 
                              "Advantage_median",
                              "Advantage_std", 
                              "Advantage_mean"])

    # correlation after feature removal 
    pearson = fs.feature_correlation(
        model_data.features,
        method="pearson",
        prefix=params.prefix+"_post",
        plot=True)
    
    spearman = fs.feature_correlation(
        model_data.features,
        method="spearman",
        prefix=params.prefix+"_post",
        plot=True)

    # variable importance via random forest regression  
    ranks, residuals = fs.feature_selection_RF(
        model_data.features,
        model_data.prevalence[params.prevalence_col],
        prefix=params.prefix,
        n_trees=1000,
        pdp_features=6,
        max_samples=0.5,
        bootstrap=True,
        max_depth=20,
        min_samples_leaf=2,
        threads=6,
        plot=True)
    
    #############################################################
    # 3. Model fitting and prediction 


def parseArgs():
    parser = argparse.ArgumentParser(description='Process command-line arguments.')

    parser.add_argument('-l', '--lineages',
                        default=None,
                        type=str,
                        help='Path to the lineages csv file')

    parser.add_argument('-p', '--prevalence',
                        default=None,
                        type=str,
                        help='Path to the prevalence csv file')

    parser.add_argument('-pc','--prevalence_col',
                        default='WWAvgMgc',
                        type=str,
                        help='Name of the column in the prevalence file')

    parser.add_argument('-ac','--abundance_col',
                        default='Weighted Abundance',
                        type=str,
                        help='Name of the abundance column in the lineages file')

    parser.add_argument('-t','--threshold',
                        default=0.001,
                        type=float,
                        help='Threshold lineage abundance to retain')

    parser.add_argument('-pt','--peak_threshold',
                        default=0.01,
                        type=float,
                        help='Threshold peak abundance to retain a lineage in the entire dataset')

    parser.add_argument('-im','--interpolation_method',
                        default='linear',
                        type=str,
                        help='Method for interpolation')

    parser.add_argument('-s', '--serial_interval',
                        default=5.5,
                        type=float,
                        help='The average time between successive cases in a chain transmission')

    parser.add_argument('-w','--window_width',
                        default=56,
                        type=int,
                        help='The width of the window for calculating growth rates')
    parser.add_argument('-ub', '--usher_barcodes',
                        default=None,
                        type=str,
                        help='Path to the barcodes file generated by UShER')

    parser.add_argument('-cl', '--curated_lineages',
                        default=None,
                        type=str,
                        help='Path to the curated_lineages file or directory')

    parser.add_argument('-r', '--regions',
                        default=None,
                        type=str,
                        help='Path to the BED file defining the regions')

    parser.add_argument('-m', '--mutations',
                        default=None,
                        type=str,
                        help='Path to the csv file listing specific mutations to consider')

    parser.add_argument('--consensus',
                        default="mode",
                        type=str,
                        help='mean, mode, or strict')

    parser.add_argument('-f','--features_file',
                        default=None,
                        type=str,
                        help='Path to the pre-computed features csv file')

    parser.add_argument('--extra_features', 
                        type=str, 
                        default = None, 
                        help='Path to the csv file with extra features')

    parser.add_argument('--prefix', 
                        type=str, 
                        default = "output", 
                        help='Prefix for output files')

    return parser.parse_args()



#Call main function
if __name__ == '__main__':
    main()
