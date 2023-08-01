import os
import sys

import argparse
import pandas as pd
from pathlib import Path

from ww_forecast.model_data import ModelData
from ww_forecast.lineage_mapper import LineageMapper

def main():

    params = parseArgs()

    # load lineage metadata 
    if params.features_file is not None:
        lineage_map = LineageMapper(
            lineages=params.lineages,
            usher_barcodes=params.usher_barcodes,
            curated_lineages=params.curated_lineages,
            regions=params.regions,
            mutations=params.mutations, 
            consensus_type=params.consensus
        )
        lineage_map.write("test")

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
        lineage_map = lineage_map)


    model_data.pretty_print('features', precision=5)

    # data = DataPreprocessor(summary_file, metadata_file, nationalavg_file)
    # #data.lineages = data.sort_normalise_filter(data.summarize_levels(data.lineages.copy(), 2), threshold=0.01)
    # print(data.lineages)
    # print(data.prevalence)
    # print(data.prevalence_validation)
    # #data.plot(0.8)

    # model = SARIMAXModel(*data.test_train_split(0.8))
    # model.plot()

    # true_values = model.test_prevalence['WWAvgMgc']
    # predicted_values = model.predictions['Predicted Value']

    # mse, rmse = model.evaluate(true_values, predicted_values)
    # print("Mean Squared Error (MSE):", mse)
    # print("Root Mean Squared Error (RMSE):", rmse)

    # sys.exit()

    # # read data tables
    # summary = sort_recent_spike(pd.read_csv(summary_file,
    #     header=0,
    #     sep="\t"))

    # # lineage = sort_lineage(pd.read_csv(params.lineage,
    # #     header=0,
    # #     sep="\t"))

    # # # summarized sub-lineages at different levels 
    # # level1=sort_normalise_filter(summarize_levels(lineage.copy(), 1), threshold=params.threshold)
    # # level2=sort_normalise_filter(summarize_levels(lineage.copy(), 2), threshold=params.threshold)
    # # lineages=sort_normalise_filter(lineage.copy(), threshold=params.threshold)

    # metadata = pd.read_csv(metadata_file,
    # header=0,
    # sep="\t")
    # metadata["Location"] = metadata.Site.astype("str")

    # wwavgmgc = pd.read_csv(nationalavg_file,
    #     sep=",",
    #     header=0)
    # wwavgmgc["Period"] = pd.to_datetime(wwavgmgc["Date7DayEnding"], format='%d/%m/%Y')
    # wwavgmgc["WWAvgMgc"] = wwavgmgc["WWAvgMgc"].astype("float")

    # # merge with metadata
    # merged = get_merged(summary, metadata)

    # # get population weighted prevalence
    # weighted = get_weighted_abundances(merged)

    # # smooth 
    # smoothed_weighted = sort_lineage(fill_smooth_normalise(weighted, "Weighted Abundance", 7, min_periods=1))

    # print(smoothed_weighted)

    # # Assuming the prevalence data is in a DataFrame called prevalence_df
    # first_date_abundance = smoothed_weighted['Period'].min()
    # prevalence_df = wwavgmgc[wwavgmgc['Period'] >= first_date_abundance]

    # # Pivot the lineage abundance data to create a wide format
    # lineage_abundance_wide = smoothed_weighted.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance')
    # lineage_abundance_wide.index = pd.to_datetime(lineage_abundance_wide.index)

    # # Resample the wide-format lineage abundance data to daily frequency
    # lineage_abundance_daily = lineage_abundance_wide.resample('D').asfreq()

    # # Use linear interpolation to fill in missing values
    # lineage_abundance_daily_interp = lineage_abundance_daily.interpolate(method='linear')

    # # Set the split date for training and testing sets
    # split_date = lineage_abundance_daily_interp.index[-1]  # The last date with lineage abundance data

    # # Split the lineage abundance data
    # train_lineage_abundance = lineage_abundance_daily_interp.loc[:split_date]
    # test_lineage_abundance = None  # There is no test lineage abundance data

    # # Split the total prevalence data
    # train_total_prevalence = prevalence_df.set_index('Period').loc[:split_date, 'WWAvgMgc']
    # test_total_prevalence = prevalence_df.set_index('Period').loc[split_date + pd.Timedelta(days=1):, 'WWAvgMgc']

    # print(train_total_prevalence)
    # print(train_lineage_abundance)
    # print(test_total_prevalence)

    # # Train the ARIMA model
    # arima_model = fit_arima(train_total_prevalence, train_lineage_abundance)

    # # Generate predictions for the test data
    # predictions = predict_arima(arima_model, test_total_prevalence, test_lineage_abundance, prediction_horizon=len(test_total_prevalence))

    # # Print the predictions
    # print(predictions)


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


    return parser.parse_args()



#Call main function
if __name__ == '__main__':
    main()
