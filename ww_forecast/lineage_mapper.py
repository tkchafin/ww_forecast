import os
import sys

import pandas as pd
import numpy as np
import json
import glob 
from collections import defaultdict
from typing import Dict, Set, Tuple


class LineageMapper:
    """
    Class for mapping genomic lineages to their corresponding barcodes and clades.
    It also supports multiple consensus methods to summarize barcodes across lineages within the same clade.
    """

    def __init__(self, 
                 usher_barcodes: str, 
                 lineages: str = None, 
                 curated_lineages: str = None, 
                 regions: str = None, 
                 mutations: str = None,
                 consensus_type: str = 'mode'
                 ):
        """
        Initializes the LineageMapper with the provided data.
        
        Args:
        usher_barcodes (str): Path to the csv file containing the usher barcodes.
        lineages (str, optional): Path to the csv file containing the lineages. If None, no lookup will be created.
        curated_lineages (str, optional): Path to the csv file containing curated lineages. 
        regions (str, optional): Path to the csv file containing regions. If None, regions won't be considered.
        mutations (str, optional): Path to the csv file containing specific mutations. If None, mutations won't be considered.
        consensus_type (str, optional): Method for summarizing barcodes across lineages within the same clade. 
                                        Available options are: 'mean', 'mode', 'strict'. Default is 'mean'.
        
        """

        # read barcodes file 
        self.barcodes = pd.read_csv(usher_barcodes, index_col=0, header=0)

        # create clade_lookup
        if curated_lineages:
            self.clade_lookup = self.build_lineage_map(curated_lineages)
        elif lineages:
            self.clade_lookup = self.process_lineages(lineages)
        else:
            # if neither provided, then this LineageMapper will only work with fully resolved lineage files
            self.clade_lookup = None
            self.consensus_barcodes = None
        
        # create consensus_barcodes
        valid_consensus_types = ['mean', 'mode', 'strict']
        if consensus_type not in valid_consensus_types:
            raise ValueError(f"Invalid consensus_type. Available options are: {', '.join(valid_consensus_types)}.")

        if self.clade_lookup is not None:
            self.consensus_barcodes = self.get_consensus_barcodes(consensus_type)

        # parse BED file 
        if regions:
            self.regions = self.read_bed(regions)
        else:
            self.regions = None

        # parse mutation selects
        if mutations:
            #self.mutations = self.read_mutations(mutations)
            raise NotImplementedError("mutations not currently implemented")
        else:
            self.mutations = None
        

    def get_subset_barcodes(self, lineages: pd.Series) -> pd.DataFrame:
        """
        Get a subset of barcodes dataframe based on lineages.

        Args:
            lineages (pd.Series): The series of lineages to get barcodes for.

        Returns:
            pd.DataFrame: A dataframe with the subset of barcodes for the lineages.
        """
        
        df = self.consensus_barcodes if self.consensus_barcodes is not None else self.barcodes

        # get subset df 
        subset = lineages.unique()
        df_index_set = set(df.index)
        for lineage in subset:
            if lineage not in df_index_set:
                print(f"Warning: lineage {lineage} not found in DataFrame index")

        # Get the intersection 
        valid_subset = list(set(subset).intersection(df_index_set))

        # get subset barcodes df 
        subset_df = df.loc[valid_subset]

        # Check if resulting subset_df is empty
        if subset_df.empty:
            raise ValueError("Error: The subset barcodes dataframe is empty. Please check the input lineages and clade_map.")

        return subset_df


    def get_nucleotide_diversity(self, df: pd.DataFrame, abundance_col: str) -> Dict[str, float]:
        """
        Calculate the nucleotide diversity for the given dataframe of lineage abundances.

        Args:
            df (pd.DataFrame): The dataframe to process. Contains lineages and corresponding abundances.
            abundance_col (str): Name of the column representing abundance.

        Returns:
            Dict[str, float]: A dictionary with the nucleotide diversity for each gene and the whole genome.
        """
        # Get the barcodes dataframe for the lineages in df
        barcode_df = self.get_subset_barcodes(df['Lineage'])

        # drop monomorphic columns 
        barcode_df = barcode_df.loc[:, barcode_df.nunique() != 1]

        # Set Lineage as index for df
        df.set_index('Lineage', inplace=True)

        # Multiply df and barcode_df, and sum to get mutation freqs
        mutation_freqs = barcode_df.mul(df[abundance_col], axis=0).sum(axis=0)

        # get allele frequencies 
        allele_freqs_df = self.mutation_to_allele_freqs(mutation_freqs)
        
        # Calculate the nucleotide diversity (pi) for each gene and the whole genome
        pi_dict = {}
        if self.regions is not None:
            for gene_name, (start, end) in self.regions.items():
                gene_data = allele_freqs_df.loc[(allele_freqs_df.index >= start) & (allele_freqs_df.index <= end)]
                gene_data = gene_data.drop('Ref', axis=1, inplace=False)
                pi_value = (1 - (gene_data**2).sum(axis=1)).mean()
                # if np.isnan(pi_value):
                    # print(f"NaN calculated for {gene_name}. Allele frequencies:")
                    # print(gene_data)
                    # print("Corresponding mutation frequencies:")
                    # print(self.get_region(mutation_freqs, start, end))
                    # pv_value = 0.0
                pi_dict[f'{gene_name}_Pi'] = pi_value
        # Calculate for the whole genome
        genome_data = allele_freqs_df.copy()
        genome_data.drop('Ref', axis=1, inplace=True)
        pi_dict['Genome_Pi'] = (1 - (genome_data**2).sum(axis=1)).mean()

        return pi_dict


    def parse_mutation_name(self, mutation: str) -> Tuple[str, int, str]:
        """
        Parse a mutation name and return the reference allele, position, and alternate allele.

        Args:
            mutation (str): The mutation name, e.g. "A23012G"

        Returns:
            Tuple[str, int, str]: The reference allele, position, and alternate allele
        """
        ref = mutation[0]
        alt = mutation[-1]
        pos = int(mutation[1:-1])

        return ref, pos, alt


    def get_region(self, mutation_freqs: pd.Series, start: int, end: int) -> pd.Series:
        """
        Get the region from mutation frequencies series.

        Args:
            mutation_freqs (pd.Series): The mutation frequencies.
            start (int): Start of the region.
            end (int): End of the region.

        Returns:
            pd.Series: The subset of the mutation frequencies in the region.
        """
        region_positions = [self.parse_mutation_name(mutation)[1] for mutation in mutation_freqs.index]
        region_positions = pd.Series(region_positions, index=mutation_freqs.index) # Convert to Series for element-wise comparison
        region_freqs = mutation_freqs[(region_positions >= start) & (region_positions <= end)]
        return region_freqs


    def mutation_to_allele_freqs(self, df: pd.Series) -> pd.DataFrame:
        """
        Given a series of mutation frequencies, return a dataframe
        where each position has a frequency for each nucleotide

        Args:
            df (pd.Series): The mutation frequencies.

        Returns:
            pd.DataFrame: The nucleotide frequencies.
        """
        allele_freqs = []
        
        # first pass, get allele frequencies and ref for each mutation
        for mutation, freq in df.items():
            ref, pos, alt = self.parse_mutation_name(mutation)

            row = [pos, 0, 0, 0, 0, ref]

            if alt == 'A':
                row[1] = freq
            elif alt == 'C':
                row[2] = freq
            elif alt == 'G':
                row[3] = freq
            elif alt == 'T':
                row[4] = freq

            allele_freqs.append(row)

        allele_freqs_df = pd.DataFrame(allele_freqs, columns=["Pos", "A", "C", "G", "T", "Ref"])

        # adjust the reference allele frequencies
        for idx, row in allele_freqs_df.iterrows():
            ref = row["Ref"]
            other_freqs = [row[nt] for nt in ["A", "C", "G", "T"] if nt != ref]
            row[ref] = 1.0 - sum(other_freqs)
            allele_freqs_df.iloc[idx] = row

        # Normalize the frequencies so they sum to 1
        allele_freqs_df[["A", "C", "G", "T"]] = allele_freqs_df[["A", "C", "G", "T"]].div(
            allele_freqs_df[["A", "C", "G", "T"]].sum(axis=1), axis=0)

        # Ensure no negative frequencies (can happen due to floating point errors)
        allele_freqs_df[["A", "C", "G", "T"]] = allele_freqs_df[["A", "C", "G", "T"]].clip(lower=0)

        # return sorted, indexed by position
        allele_freqs_df.set_index('Pos', inplace=True)
        allele_freqs_df = allele_freqs_df.sort_index()
        return allele_freqs_df


    def get_consensus_barcodes(self, consensus_type: str) -> pd.DataFrame:
        """
        Computes the consensus barcodes based on the selected consensus type.

        Args:
        consensus_type (str): Method for summarizing barcodes across lineages within the same clade. 
                            Available options are: 'mean', 'mode', 'strict'.

        Returns:
        pd.DataFrame: A dataframe containing the consensus barcodes.

        """
        # Add clade information to the barcodes dataframe
        clade_df = self.barcodes.assign(clade = self.barcodes.index.map(self.clade_lookup))

        # Define consensus function based on the specified type
        if consensus_type == "mean":
            consensus_func = lambda x: x.mean() 
        elif consensus_type == "mode":
            consensus_func = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0.0
        elif consensus_type == "strict":
            consensus_func = lambda x: 1.0 if any(x > 0.0) else 0.0
        else:
            raise ValueError("Invalid consensus_type. Available options are: 'mean', 'mode', 'strict'.")

        # Initialize a new DataFrame to hold consensus barcodes
        consensus_barcodes = pd.DataFrame(columns = clade_df.columns[:-1])

        # Group by clade and compute consensus barcodes
        for clade, group in clade_df.groupby('clade'):
            consensus_barcodes.loc[clade, :] = group.iloc[:, :-1].apply(consensus_func, axis=0)

        return consensus_barcodes


    def process_lineages(self, lineages: str):
        """
        Processes the lineages. If the lineages are collapsed (i.e., end with '.X'), 
        it constructs the clade lookup by finding matches in the barcodes index.

        Args:
        lineages: str. The path to the CSV file containing the lineage information.

        Returns:
        None
        """
        # get input lineages 
        unique_lineages = set(pd.read_csv(lineages)['Lineage'])
        clade_lookup = {}
        if any(lineage.endswith('.X') for lineage in unique_lineages):
            for barcode in self.barcodes.index:
                lvl1 = '.'.join(barcode.split('.')[:1]) + '.X'
                lvl2 = '.'.join(barcode.split('.')[:2]) + '.X'
                # add to clade_lookup
                if lvl2 in unique_lineages:
                    clade_lookup[barcode] = lvl2
                elif lvl1 in unique_lineages:
                    clade_lookup[barcode] = lvl1
                else:
                    pass
                    #print(f"Warning [Building clade_lookup]: Barcode {barcode} could not be found under the aliases {lvl1} or {lvl2}.")
                    #clade_lookup[barcode] = barcode
            return(clade_lookup)
        else:
            return(None)



    def build_lineage_map(self, curated_lineages: str) -> dict:
        """
        Builds a lineage map using curated lineage data from outbreak.info.

        Args:
            curated_lineages (str): Directory path to the lineage data in JSON format.

        Returns:
            dict: A dictionary mapping lineages to clade names.

        Note: This function is adapted from Freyja v1.4.4.
        """
        map_dict = {}
        if os.path.isdir(curated_lineages):
            global_dat = defaultdict(set)
            for filename in glob.glob(os.path.join(curated_lineages, "*.json")):
                with open(filename, "r") as f0:
                    dat = json.load(f0)
                    dat = sorted(dat, key=lambda x: len(
                        x.get('pango_descendants', [])), reverse=True)
                    for record in dat:
                        if 'who_name' in record:
                            if record['who_name'] is not None:
                                global_dat[record['who_name']].update(
                                    record.get('pango_descendants', []))
            # Sort global_dat by number of descendants (largest first)
            sorted_dat = sorted(global_dat.items(),
                                key=lambda x: len(x[1]),
                                reverse=True)
            for clade, descendants in sorted_dat:
                for descendant in descendants:
                    map_dict[descendant] = clade
        else:
            with open(curated_lineages) as f0:
                dat = json.load(f0)
            # Sort records based on the length of 'pango_descendants'
            dat = sorted(dat, key=lambda x: len(
                x.get('pango_descendants', [])), reverse=True)
            for ind in range(len(dat)):
                if 'who_name' in dat[ind].keys():
                    for d0 in dat[ind]['pango_descendants']:
                        if dat[ind]['who_name'] is not None:
                            map_dict[d0] = dat[ind]['who_name']
        return map_dict


    def read_bed(self, bed_file: str) -> Dict[str, Tuple[int, int]]:
        """
        Reads a BED file and returns a dictionary where keys are the regions
        and values are tuples containing the start and end positions.
        
        Args:
        bed_file (str): Path to the BED file.

        Returns:
        A dictionary where keys are the regions and values are tuples containing the start and end positions.
        """
        regions = {}
        with open(bed_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                regions[line[3]] = (int(line[1]), int(line[2]))
        return regions


    def read_mutations(self, mutations_file: str):
        """
        Reads in the mutations file and returns a DataFrame.

        Args:
        mutations_file (str): Path to the csv file containing mutations.
        
        Returns:
        pandas DataFrame: DataFrame with mutation data.
        """
        mutations = pd.read_csv(mutations_file, index_col=None, header=0)
        return mutations.set_index('NUC')['AA'].to_dict()


    def write(self, prefix: str):
        """
        Writes the consensus barcodes and clade map to csv files with a given prefix.

        Args:
        prefix (str): The prefix for the output file names.
        """
        # write consensus barcodes
        self.consensus_barcodes.to_csv(f"{prefix}_consensus_barcodes.csv")

        # write clade map
        if self.clade_lookup is not None:
            # convert the clade lookup dictionary into a DataFrame
            clade_map = pd.DataFrame(list(self.clade_lookup.items()), columns=['Lineage', 'Clade'])

            # sort the DataFrame
            clade_map.sort_values(['Clade', 'Lineage'], inplace=True)

            # write the DataFrame to a csv file
            clade_map.to_csv(f"{prefix}_clade_map.csv", index=False)


    def extract_region_data(self, region: dict) -> pd.DataFrame:
        """
        Extracts data for a specific region from the DataFrame (either barcodes or consensus_barcodes).

        Args:
        region (dict): Dictionary containing the region information. Keys should be 'start' and 'stop'.

        Returns:
        pd.DataFrame: A DataFrame that only includes the columns (SNPs) that are within the specified region.
        """
        # Choose the data source: self.barcodes or self.consensus_barcodes
        data = self.consensus_barcodes if self.consensus_barcodes is not None else self.barcodes

        # Determine the columns that are within the specified region.
        # Here we assume that columns are named according to their positions (e.g., A10042C for position 10042).
        # We extract the numeric part of the column name, convert it to integer and compare with the region boundaries.
        region_columns = [col for col in data.columns if region['start'] <= int(''.join(filter(str.isdigit, col))) <= region['stop']]

        # Extract the data for the region.
        region_data = data[region_columns]

        return region_data

    

