import os 
import sys 

import pandas as pd

class ModelData:
    """
    Class to load, preprocess and interpolate lineage and prevalence data.
    
    Attributes:
    lineages (pd.DataFrame): Lineage data.
    prevalence (pd.DataFrame): Prevalence data.
    prevalence_col (str): Column name for prevalence values.
    abundance_col (str): Column name for abundance values.
    threshold (float): Threshold for abundance filtering.
    peak_threshold (float): Threshold for peak filtering.
    interpolation_method (str): Interpolation method for filling missing dates.
    """


    def __init__(
            self, 
            lineages_file: str, 
            prevalence_file: str, 
            prevalence_col: str, 
            abundance_col: str, 
            threshold: float = 0.005, 
            peak_threshold: float = 0.01, 
            interpolation_method: str = 'linear'
            ):
        """
        Initialize ModelData with files and column names. Performs preprocessing and interpolation on data.
        
        Args:
        lineages_file (str): Path to the lineage file.
        prevalence_file (str): Path to the prevalence file.
        prevalence_col (str): Column name for prevalence values in the prevalence file.
        abundance_col (str): Column name for abundance values in the lineage file.
        threshold (float, optional): Threshold for abundance filtering. Defaults to 0.005.
        peak_threshold (float, optional): Threshold for peak filtering. Defaults to 0.01.
        interpolation_method (str, optional): Interpolation method for filling missing dates. Defaults to 'linear'.
        """
        self.lineages = self._read_csv(lineages_file, ['Lineage', abundance_col])
        self.prevalence = self._read_csv(prevalence_file, [prevalence_col])
        self.prevalence_col = prevalence_col
        self.abundance_col = abundance_col
        self.threshold = threshold
        self.peak_threshold = peak_threshold
        self.interpolation_method = interpolation_method

        # pre-processing 
        if self.peak_threshold > 0.0:
            self.filter_lineages()
        self.trim() 

        # sort
        self.lineages.sort_values(by=['Lineage', 'Date'], inplace=True)
        self.prevalence.sort_values(by='Date', inplace=True)

        # # perform linear interpolation 
        self.lineages = self.interpolation(self.lineages, group="Lineage", var=self.abundance_col)
        self.prevalence = self.interpolation(self.prevalence, var=self.prevalence_col)


    def interpolation(self, df: DataFrame, var: str, group: Optional[str] = None) -> DataFrame:
        """
        Performs linear interpolation for missing dates in the data.
        
        Args:
        df (pd.DataFrame): Data to be interpolated.
        var (str): Variable column to be interpolated.
        group (str, optional): If given, interpolation will be done per group. Defaults to None.
        
        Returns:
        df (pd.DataFrame): Interpolated data.
        """

        if group is None:
            df.set_index('Date', inplace=True)
            date_index = pd.date_range(start=df.index.min(), end=df.index.max())
            df = df.reindex(date_index)
            df[var] = df[var].interpolate(method=self.interpolation_method)
            df.reset_index(inplace=True)
            if "index" in df.columns:
                df.rename(columns={"index": "Date"}, inplace=True)
        else:
            if group not in df.columns:
                raise ValueError(f"The group {group} is not a column in the dataframe.")

            df.set_index(['Date', group], inplace=True)
            date_index = pd.date_range(start=df.index.get_level_values('Date').min(), end=df.index.get_level_values('Date').max())
            lineage_index = df.index.get_level_values(group).unique()
            multi_index = pd.MultiIndex.from_product([date_index, lineage_index], names=['Date', group])
            df = df.reindex(multi_index)
            df[var] = df.groupby(level=group)[var].transform(lambda group: group.interpolate(method=self.interpolation_method))
            df.reset_index(['Date', group], inplace=True)  # reset the specified levels of the index

        return df
 

    def normalise_generic(self, df, group, var):
        """
        This function normalizes a column in a dataframe by the sum of its group.

        Parameters:
        df (pandas.DataFrame): The input dataframe to be processed.
        group (string or list): The column(s) to group by.
        var (string): The column to normalize.

        Returns:
        pandas.DataFrame: The input dataframe with the var column normalized by the sum of its group.
        """
        ret = df.copy()
        ret[var] = ret.groupby(group)[var].transform(lambda x: x/x.sum())
        return(ret)


    def filter_lineages(self):
        """
        This function filters lineages from the self.lineages dataframe based on a peak threshold. 
        It normalizes the abundance column for the selected lineages by date.

        The function modifies the self.lineages dataframe to include only selected lineages and normalized abundance.

        Returns:
        None
        """
        # Compute the max abundance for each lineage
        peaks = self.lineages.groupby('Lineage')[self.abundance_col].max()

        # Filter out lineages based on the peak threshold
        selects = pd.unique(peaks[peaks > self.peak_threshold].index).tolist()

        # Subset the lineages data
        subset = self.lineages[self.lineages['Lineage'].isin(selects)]
        
        # Normalize abundance
        subset = self.normalise_generic(subset, ['Date'], self.abundance_col)

        self.lineages = subset


    def trim(self):
        """
        This method trims the datasets to start from the earliest common date.

        Parameters:
        None

        Returns:
        None
        """
        # Get the first date from both datasets
        first_lineages_date = self.lineages['Date'].min()
        first_prevalence_date = self.prevalence['Date'].min()

        # Find the latest of these two dates
        start_date = max(first_lineages_date, first_prevalence_date)

        # Trim both datasets to start from the earliest common date
        self.lineages = self.lineages[self.lineages['Date'] >= start_date]
        self.prevalence = self.prevalence[self.prevalence['Date'] >= start_date]


    def _read_csv(self, file_path, keep_cols):
        """
        This method reads a CSV file and returns a dataframe with necessary columns and converted date.

        Parameters:
        file_path (str): The path to the CSV file.
        keep_cols (list): A list of columns to keep in the dataframe.

        Returns:
        pandas.DataFrame: The input CSV file as a dataframe with necessary columns and converted date.
        """
        df = pd.read_csv(file_path)
        if 'Date' in df.columns or 'Period' in df.columns:
            date_col = 'Date' if 'Date' in df.columns else 'Period'
            try:
                df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
            except ValueError:
                df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
            df.rename(columns={date_col: "Date"}, inplace=True)
            keep_cols.append("Date")
        # Keep only necessary columns
        df = df.filter(keep_cols)
        return df


    def pretty_print(self, data_type, max_rows=4, max_columns=None, width=1000, precision=3, colheader_justify='left'):
        """
        This method prints the specified dataframe in a pretty format with specified pandas display options.

        Parameters:
        data_type (str): The type of data to print. It can be either 'lineages' or 'prevalence'.
        max_rows (int, optional): The maximum number of rows to display. Defaults to 4.
        max_columns (int, optional): The maximum number of columns to display. Defaults to None.
        width (int, optional): The width of the display in characters. Defaults to 1000.
        precision (int, optional): The number of decimal places to display. Defaults to 3.
        colheader_justify (str, optional): The justification of the column headers. It can be either 'left' or 'right'. Defaults to 'left'.

        Returns:
        None
        """
        if data_type == 'lineages':
            df = self.lineages
        elif data_type == 'prevalence':
            df = self.prevalence
        else:
            print("Invalid data type. Please specify either 'lineages' or 'prevalence'.")
            return

        with pd.option_context('display.max_rows', max_rows,
                               'display.max_columns', max_columns,
                               'display.width', width,
                               'display.precision', precision,
                               'display.colheader_justify', colheader_justify):
            print(df)


    def describe(self, data_type, head_rows=10):
        """
        This method prints a summary of the specified dataframe.

        Parameters:
        data_type (str): The type of data to print a summary of. It can be either 'lineages' or 'prevalence'.
        head_rows (int, optional): The number of rows to display from the top of the dataframe. Defaults to 10.

        Returns:
        None
        """
        if data_type == 'lineages':
            df = self.lineages
        elif data_type == 'prevalence':
            df = self.prevalence
        else:
            print("Invalid data type. Please specify either 'lineages' or 'prevalence'.")
            return

        print(f"\nFirst {head_rows} rows:")
        print(df.head(head_rows))
        print("\nInfo:")
        print(df.info())
        print("\nDescriptive statistics:")
        print(df.describe())
    

    def write(self, prefix):
        """
        Writes the 'lineages' and 'prevalence' dataframes to CSV files.

        :param prefix: str, prefix for the filename
        """
        # sort
        self.lineages.sort_values(by=['Lineage', 'Date'], inplace=True)
        self.prevalence.sort_values(by='Date', inplace=True)

        # Write to CSV files
        self.lineages.to_csv(f"{prefix}_abundance.csv", index=True)
        self.prevalence.to_csv(f"{prefix}_prevalence.csv", index=True)


    # def plot(self, train_proportion=None):
    #     fig = plt.figure(figsize=(12, 6))
    #     gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)

    #     # Prepare data for the stacked area plot
    #     abundance_data_wide = self.lineages.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance', fill_value=0)

    #     # Plot the abundance data as a stacked area plot
    #     ax1 = plt.subplot(gs[0, :])
    #     periods = abundance_data_wide.index
    #     lineage_labels = abundance_data_wide.columns
    #     lineage_abundances = abundance_data_wide.T.values
    #     abundance_data_wide.plot.area(ax=ax1, stacked=True, legend=False, alpha=0.8)
    #     ax1.set_ylabel("Lineage Abundance")
    #     ax1.set_ylim(0, 1)
    #     ax1.set_xticks([])
    #     ax1.set_xlim(self.prevalence['Period'].min(), self.prevalence['Period'].max())


    #     # Add a smaller legend for the area plot
    #     handles, labels = ax1.get_legend_handles_labels()
    #     ax1.legend(handles, labels, loc='upper left', ncol=2, fontsize=8, frameon=True)

    #     # Plot the prevalence data as a line plot
    #     ax2 = plt.subplot(gs[1, :])
    #     self.prevalence.plot(x='Period', y='WWAvgMgc', ax=ax2, legend=False)
    #     ax2.set_ylabel("Prevalence")
    #     ax2.set_xlim(self.prevalence['Period'].min(), self.prevalence['Period'].max())

    #     # Add vertical line for training proportion
    #     if train_proportion is not None:
    #         unique_periods = self.lineages['Period'].unique()
    #         split_idx = int(len(unique_periods) * train_proportion)
    #         split_period = unique_periods[split_idx]

    #         ax1.axvline(x=split_period, color='r', linestyle='--', linewidth=1)
    #         ax2.axvline(x=split_period, color='r', linestyle='--', linewidth=1)

    #     plt.show()