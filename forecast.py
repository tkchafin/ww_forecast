import os
import sys

import argparse
import pandas as pd

from pathlib import Path
from ww_forecast.model_data import ModelData
#from ww_forecast.forecast import * 

def main():

    params = parseArgs()

    model_data = ModelData(
        lineages_file = params.lineages, 
        prevalence_file = params.prevalence, 
        threshold = params.threshold, 
        abundance_col = params.abundance_col, 
        peak_threshold = params.peak_threshold, 
        prevalence_col = params.prevalence_col, 
        serial_interval = params.serial_interval, 
        window_width = params.window_width, 
        treedb = params.treedb)


    model_data.pretty_print('lineages', precision=5)
    model_data.pretty_print('prevalence')
    model_data.write("test")

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

    parser.add_argument('--prevalence_col',
                        default='WWAvgMgc',
                        type=str,
                        help='Name of the column in the prevalence file')

    parser.add_argument('--abundance_col',
                        default='Weighted Abundance',
                        type=str,
                        help='Name of the abundance column in the lineages file')

    parser.add_argument('--threshold',
                        default=0.005,
                        type=float,
                        help='Threshold lineage abundance to retain')

    parser.add_argument('--peak_threshold',
                        default=0.01,
                        type=float,
                        help='Threshold peak abundance to retain a lineage in the entire dataset')

    parser.add_argument('--interpolation_method',
                        default='linear',
                        type=str,
                        help='Method for interpolation')

    parser.add_argument('--serial_interval',
                        default=5.5,
                        type=float,
                        help='The average time between successive cases in a chain transmission')

    parser.add_argument('--window_width',
                        default=56,
                        type=int,
                        help='The width of the window for calculating growth rates')

    parser.add_argument('--treedb',
                        default=None,
                        type=str,
                        help='Path to the database file for calculating phylogenetic diversity')

    return parser.parse_args()



#Call main function
if __name__ == '__main__':
    main()
