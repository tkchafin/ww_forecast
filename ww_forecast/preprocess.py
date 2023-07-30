import sys 
import pandas as pd
import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, lineage_file, prevalence_file, lineage_interpolation='linear'):
        self.lineage_file = lineage_file
        self.prevalence_file = prevalence_file
        self.lineage_interpolation = lineage_interpolation

        # Read files
        lineage = pd.read_csv(self.lineage_file, sep="\t", header=0)
        metadata = pd.read_csv(self.metadata_file, sep="\t", header=0)
        metadata["Location"] = metadata.Site.astype("str")
        prevalence = pd.read_csv(self.prevalence_file, sep=",", header=0)
        prevalence["Period"] = pd.to_datetime(prevalence["Date7DayEnding"], format='%d/%m/%Y')
        prevalence["WWAvgMgc"] = prevalence["WWAvgMgc"].astype("float")

        # Process lineage and prevalence data separately
        self.lineages = self.process_lineage_data(lineage, metadata)
        self.prevalence = self.process_prevalence_data(lineage, prevalence)

        # interpolate lineage data 
        self.lineages = self.interpolate_lineage_data()
        self.check_abundances()


    def process_prevalence_data(self, lineages, prevalence):
        # Process prevalence data
        min_lineage_date = lineages['Period'].min()
        max_lineage_date = lineages['Period'].max()
        prevalence_validation = prevalence.loc[prevalence['Period'] > max_lineage_date]
        prevalence = prevalence.loc[prevalence['Period'] >= min_lineage_date]
        prevalence = prevalence.loc[prevalence['Period'] <= max_lineage_date]
        return (prevalence, prevalence_validation)

    def process_lineage_data(self, lineage, metadata):
        # Merge with metadata
        merged = self.get_merged(lineage, metadata)
        print(merged)
        # Get population weighted prevalence
        weighted = self.get_weighted_abundances(merged)

        # Smooth
        smoothed_weighted = self.sort_lineage(self.fill_smooth_normalise(weighted, "Weighted Abundance", 7, min_periods=1))

        return smoothed_weighted

    
    def get_merged(self, df, metadata):
        dfm = df.merge(metadata.drop_duplicates(['Location']), on="Location", how="left", indicator=True)
        bads = dfm[dfm["_merge"]!="both"] 
        if bads.shape[0] > 0: 
            print("Warning: Some sites weren't found in the metadata:")
            print(bads)
        return dfm[dfm["_merge"] == "both"]

    def get_weighted_abundances(self, df, pop_variable="Pop"):
        df["Prev_Pop"] = df.Abundance * df[pop_variable]
        
        df_sum = df.groupby(['Period']).agg({'Prev_Pop': 'sum'}).reset_index()
        df_sum.columns = ['Period', 'Total_Prev_Pop']

        df = df.merge(df_sum, on='Period', how='left')
        df['Weighted Abundance'] = df['Prev_Pop'] / df['Total_Prev_Pop']
        
        df2 = df[['Period', 'Lineage', 'Weighted Abundance']].sort_values(['Period', 'Lineage']).reset_index(drop=True)
        
        return df2

    def summarize_levels(self, df, level, var="Weighted Abundance"):
        df["Lineage"] = df['Lineage'].str.split('.',expand=False).str[0:(level+1)].str.join(".")
        df['Lineage'] = df['Lineage'] + ".X"
        df = df.groupby(['Lineage', 'Period'])[var].sum().reset_index()
        df['Abundance'] = df.groupby(['Period'])[var].transform(lambda x: x/x.sum())
        return(self.sort_lineage(df))

    def filter_lineages(self, df, var="Weighted Abundance", threshold=0.001):
        peaks = df.groupby(['Lineage'], sort=False)[var].max()
        selects = pd.unique(peaks[peaks > threshold].index).tolist()
        subset = df[df['Lineage'].isin(selects)]
        subset = self.normalise_generic(subset, ['Period'], var)
        return(subset)

    def sort_lineage(self, df):
        return df.sort_values(by="Lineage").reset_index(drop=True)

    def fill_smooth_normalise(self, df, var="Weighted Abundance", smooth=7, min_periods=1):
        df = self.fill_missing_by_group(df, "Period", "Lineage")
        df = self.smooth_abundance(df, var, interval=smooth, min_periods=min_periods)
        df = self.normalise_by_period(df, var)
        return df

    def sort_normalise_filter(self, df, var="Weighted Abundance", threshold=0.001):
        filtered = self.filter_lineages(df, threshold=threshold)
        normed = self.normalise_generic(filtered, ['Period'], var)
        return(self.sort_lineage(normed))

    def fill_missing_by_group(self, df, group, category):
        unique_periods = df[group].unique()
        unique_lineages = df[category].unique()
        mux = pd.MultiIndex.from_product([unique_periods, unique_lineages], names=[group, category])

        # Drop duplicate index combinations
        df = df.drop_duplicates(subset=[group, category])

        return df.set_index([group, category]).reindex(mux, fill_value=0).reset_index()

    def smooth_abundance(self, df, var, interval=7, min_periods=1):
        df = df.set_index('Period')
        df[var] = df.groupby('Lineage')[var].transform(lambda x: x.rolling(interval, min_periods).mean())
        df = df.reset_index()
        return df
    
    def get_scaled(self, df, wwavgmgc):
        df["Period"] = pd.to_datetime(df["Period"])
        sum_scaled = df.merge(wwavgmgc, on="Period")
        sum_scaled["Abundance (Mgc/ p)"] = sum_scaled["Abundance"] * sum_scaled["WWAvgMgc"]
        return(sum_scaled)

    def normalise_generic(self, df, group, var):
        ret = df.copy()
        ret[var] = ret.groupby(group)[var].transform(lambda x: x/x.sum())
        return(ret)

    def normalise_by_period(self, df, var):
        df[var] = df.groupby('Period')[var].transform(lambda x: x/x.sum())
        return(df)

    def interpolate_lineage_data(self):

        lineage_abundance_wide = self.lineages.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance')
        lineage_abundance_wide.index = pd.to_datetime(lineage_abundance_wide.index)

        # Create a new DataFrame with the specific periods from self.prevalence
        unique_periods = self.prevalence['Period'].unique()
        new_periods_df = pd.DataFrame(index=unique_periods, columns=lineage_abundance_wide.columns)

        # Reindex the original DataFrame to include the new periods and interpolate based on the specific periods
        interpolated_df = lineage_abundance_wide.reindex(unique_periods, method=None).interpolate(method=self.lineage_interpolation)

        # Keep only the periods from self.prevalence
        final_df = interpolated_df.loc[unique_periods]

        # Reset index and melt the wide DataFrame to long format
        final_df = final_df.reset_index().melt(id_vars='Period', var_name='Lineage', value_name='Weighted Abundance')

        # Normalize the abundances within each period
        final_df = self.normalise_by_period(final_df, 'Weighted Abundance')

        return final_df

    def check_abundances(self):
        sums = self.lineages.groupby('Period')['Weighted Abundance'].sum()
        if not np.isclose(sums, 1, rtol=1e-6).all():
            problem_periods = sums[np.logical_not(np.isclose(sums, 1, rtol=1e-6))]
            print("The sum of Weighted Abundance values is not equal to 1 for the following periods: {}".format(problem_periods))
            sys.exit()
    
    def check_abundances_wide(self, abundance_data_wide):
        sums = abundance_data_wide.sum(axis=1)
        if not np.isclose(sums, 1, rtol=1e-6).all():
            problem_periods = sums[np.logical_not(np.isclose(sums, 1, rtol=1e-6))]
            print("The sum of Weighted Abundance values in wide format is not equal to 1 for the following periods: {}".format(problem_periods))
            sys.exit()


    def test_train_split(self, train_proportion=0.80):
        # Calculate the split index based on periods
        unique_periods = self.lineages['Period'].unique()
        split_idx = int(len(unique_periods) * train_proportion)
        split_period = unique_periods[split_idx]

        # Split lineage data
        train_lineage = self.lineages[self.lineages['Period'] < split_period]
        test_lineage = self.lineages[self.lineages['Period'] >= split_period]

        # Split prevalence data
        train_prevalence = self.prevalence[self.prevalence['Period'] < split_period]
        test_prevalence = self.prevalence[self.prevalence['Period'] >= split_period]

        return train_lineage, test_lineage, train_prevalence, test_prevalence, self.prevalence_validation
    
    def plot(self, train_proportion=None):
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)

        # Prepare data for the stacked area plot
        abundance_data_wide = self.lineages.pivot_table(index='Period', columns='Lineage', values='Weighted Abundance', fill_value=0)

        # Plot the abundance data as a stacked area plot
        ax1 = plt.subplot(gs[0, :])
        periods = abundance_data_wide.index
        lineage_labels = abundance_data_wide.columns
        lineage_abundances = abundance_data_wide.T.values
        abundance_data_wide.plot.area(ax=ax1, stacked=True, legend=False, alpha=0.8)
        ax1.set_ylabel("Lineage Abundance")
        ax1.set_ylim(0, 1)
        ax1.set_xticks([])
        ax1.set_xlim(self.prevalence['Period'].min(), self.prevalence['Period'].max())


        # Add a smaller legend for the area plot
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper left', ncol=2, fontsize=8, frameon=True)

        # Plot the prevalence data as a line plot
        ax2 = plt.subplot(gs[1, :])
        self.prevalence.plot(x='Period', y='WWAvgMgc', ax=ax2, legend=False)
        ax2.set_ylabel("Prevalence")
        ax2.set_xlim(self.prevalence['Period'].min(), self.prevalence['Period'].max())

        # Add vertical line for training proportion
        if train_proportion is not None:
            unique_periods = self.lineages['Period'].unique()
            split_idx = int(len(unique_periods) * train_proportion)
            split_period = unique_periods[split_idx]

            ax1.axvline(x=split_period, color='r', linestyle='--', linewidth=1)
            ax2.axvline(x=split_period, color='r', linestyle='--', linewidth=1)

        plt.show()
