import os 
import sys 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX 
import matplotlib.gridspec as gridspec


class ForecastingModel:
    def __init__(self, train_lineage, test_lineage, train_prevalence, test_prevalence, validation=None):
        self.train_lineage = train_lineage
        self.test_lineage = test_lineage
        self.train_prevalence = train_prevalence
        self.test_prevalence = test_prevalence
        self.validation = validation
        self.predictions = None

    def fit(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def evaluate(self, true_values, predicted_values):
        # Implement evaluation metrics, e.g., mean squared error
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        return mse, rmse
    
    def plot(self):
        fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)

        # Reshape lineage abundance DataFrames
        train_lineage_pivot = self.train_lineage.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance', fill_value=0)
        test_lineage_pivot = self.test_lineage.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance', fill_value=0)

        # Prepare data for the stacked area plot
        abundance_data = pd.concat([self.train_lineage, self.test_lineage])
        abundance_data_wide = abundance_data.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance', fill_value=0)

        # Plot the abundance data as a stacked area plot
        ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
        abundance_data_wide.plot.area(ax=ax1, stacked=True, legend=False)
        ax1.set_ylabel("Lineage Abundance")
        ax1.set_ylim([0,1])
        ax1.set_xticklabels([])
        ax1.margins(0)
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.tick_params(axis='y', pad=10)

        # Add a smaller legend for the area plot
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper left', ncol=2, fontsize=8, frameon=True)

        # Main plot for prevalence
        ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, sharex=ax1)  # Add sharex=ax1
        ax2.plot(self.train_prevalence['Period'], self.train_prevalence['WWAvgMgc'], label='Training Data')
        ax2.plot(self.test_prevalence['Period'], self.test_prevalence['WWAvgMgc'], label='Test Data')
        ax2.plot(self.predictions.index, self.predictions['Predicted Value'], label='Predictions', linestyle='--')
        if self.validation is not None:
            ax2.plot(self.validation['Period'], self.validation['WWAvgMgc'], label='Validation Data', linestyle='-.')

        ax2.legend()
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Prevalence')

        plt.show()
    
    def validate_abundances(self):
        abundance_data = pd.concat([self.train_lineage, self.test_lineage])
        sums = abundance_data.groupby('Period')['Weighted Abundance'].sum()
        if not np.isclose(sums, 1, rtol=1e-6).all():
            problem_periods = sums[np.logical_not(np.isclose(sums, 1, rtol=1e-6))].index.tolist()
            print("The sum of Weighted Abundance values is not equal to 1 for the following periods: {}".format(problem_periods))
            sys.exit()



class SARIMAXModel(ForecastingModel):
    def __init__(self, train_lineage, test_lineage, train_prevalence, test_prevalence, validation=None, order=(1, 0, 0)):
        super().__init__(train_lineage, test_lineage, train_prevalence, test_prevalence, validation)
        self.order = order
        self.seasonal_order = (0, 0, 0, 0)  # You can adjust the seasonal_order if needed
        self.model = None

        self.fit()
        self.predict()

    def fit(self):
        # Preprocess the data
        train_data = self.train_prevalence[['WWAvgMgc']].set_index(pd.to_datetime(self.train_prevalence['Period']))
        exog_train = self.train_lineage.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance')
        exog_train.index = pd.to_datetime(exog_train.index)

        # Fit the model
        self.model = SARIMAX(train_data, exog=exog_train, order=self.order, seasonal_order=self.seasonal_order)
        self.model = self.model.fit()

    def predict(self):
        if self.model is None:
            raise RuntimeError("The model has not been fit yet. Call the fit method first.")

        # Preprocess the test data
        test_data = self.test_prevalence[['WWAvgMgc']].set_index(pd.to_datetime(self.test_prevalence['Period']))
        exog_test = self.test_lineage.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance')
        exog_test.index = pd.to_datetime(exog_test.index)

        # Calculate start and end indices
        n_train_periods = len(self.train_prevalence['Period'].unique())
        n_test_periods = len(self.test_prevalence['Period'].unique())
        start = n_train_periods
        end = start + n_test_periods - 1

        # Make predictions
        predictions = self.model.predict(start=start, end=end, exog=exog_test)
        self.predictions = predictions.to_frame(name='Predicted Value')
        self.predictions.index = self.test_prevalence['Period'].unique()
        return predictions


