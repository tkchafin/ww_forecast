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
        
        print(self.consensus_barcodes)

        # patse BED file 
        if regions:
            # Code to load regions
            pass

        # parse mutation selects
        if mutations:
            # Code to load specific mutations
            pass


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
                    clade_lookup[barcode] = barcode
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


    def read_mutations(self, mutations_file: str) -> Set[str]:
        """
        Reads a mutations file and returns a set of mutations.
        
        Args:
        mutations_file (str): Path to the mutations file.

        Returns:
        A set of mutations.
        """
        with open(mutations_file, 'r') as f:
            mutations = set(line.strip() for line in f)
        return mutations


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