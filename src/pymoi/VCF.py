import numpy as np
import pandas as pd
from scipy.stats import linregress

from .Variant import Variant


class VCF(object):
    """
    Represents a VCF file with methods for parsing and accessing data.
    """


    def __init__(self, vcf_file: str):
        """
        Initializes a VCF object.

        INPUT:
            vcf_file <str> : Path to the VCF file.
        """
        self.vcf_file = vcf_file
        self.info = []
        self.var_lines = []
        self.sample_ids = []
        self._parseVCF()


    def _parseVCF(self):
        """
        Parses the VCF file and populates the object's attributes.

        INPUT:
            None

        OUTPUT:
            None
        """
        with open(self.vcf_file, "r") as fin:
            for line in fin.readlines():
                if line.startswith("#CHROM"):
                    self.sample_ids = line.strip("\n").split("\t")[9:]
                    print(f"Detected {len(self.sample_ids)} samples: {self.sample_ids}")
                elif line.startswith("#"):
                    self.info.append(line)
                else:
                    variant = Variant()
                    variant.parseLine(line, self.sample_ids)
                    self.var_lines.append(variant)


    def getVars(self):
        """
        Returns the list of parsed Variant objects.

        INPUT:
            None

        OUTPUT:
            var_lines <list> : A list of Variant objects.
        """
        return self.var_lines
    

    def getSampleIds(self):
        """
        Returns the list of sample IDs from the VCF header.

        INPUT:
            None

        OUTPUT:
            sample_ids <list> : A list of sample IDs.
        """
        return self.sample_ids
    
    
    def getAlleleCounts(    self,
                            sample_id : str,
                            filter_homo : bool=False ) -> pd.DataFrame:
        """ takes a VCF object and outputs a dataframe of ref and alt depths for the given sample
        chrom    pos    alt_depth    ref_depth    alt_freq    ref_freq
        chr1     1      ##           ##           ##          ##

        where ## are read depths (int) or allele frequency (float) of that allele
        """

        allele_list = []

        for l in self.var_lines:

            ## use the dictionary containing allele information for the specified sample
            if sample_id not in l.samples:
                raise ValueError(f"{sample_id} not in dataset.")

            sample_dict = l.samples[sample_id]

            ## this block handles merged vcfs where there are no alternate alleles
            if sample_dict["AD"][0] == ".":
                continue
            
            else:
                r_depth = int(sample_dict["AD"][0])
                a_depths = sum([ int(i) for i in sample_dict["AD"][1:] ])

            total = r_depth + a_depths
            if total > 0:
                r_freq = r_depth/total
                a_freq = a_depths/total

                ## filter out homozygous reference alleles
                ## these should be flagged as 0/0 in a vcf
                if r_freq < 1.0:
                    if filter_homo:
                        ## filter out homozygous alt alleles
                        ## this may improve model fitting
                        if r_freq > 0.0:
                            allele_list.append({'chromosome': l.chrom, 'position': l.pos, 'ref_freq': r_freq, 'alt_freq': a_freq, 'ref_depth': r_depth, 'alt_depth': a_depths})
                    else:
                        allele_list.append({'chromosome': l.chrom, 'position': l.pos, 'ref_freq': r_freq, 'alt_freq': a_freq, 'ref_depth': r_depth, 'alt_depth': a_depths})

        return pd.DataFrame.from_records(allele_list)
    
        
    def getMAF(self, filter_homo: bool = False) -> np.array:
        """
        Generates B-allele frequency (BAF) and count matrices for a specific sample.

        INPUT:
            filter_homo <bool, optional> : Whether to filter out homozygous variants. Defaults to False.

        OUTPUT:
            maf_matrix <tuple> : A tuple containing two pandas DataFrames: the BAF matrix and the count matrix.
        """
        maf_matrix = {}
        for sample_id in self.sample_ids:
            sample_mafs = []
            for l in self.var_lines:

                ## use the dictionary containing allele information for the specified sample
                if sample_id not in l.samples:
                    raise ValueError(f"{sample_id} not in dataset.")
                
                sample_dict = l.samples[sample_id]

                ## this block handles merged vcfs where there are no alternate alleles
                if sample_dict["AD"][0] == ".":
                    sample_mafs.append(np.nan)
                    continue
                
                else:
                    r_depth = int(sample_dict["AD"][0])
                    a_depths = sum([ int(i) for i in sample_dict["AD"][1:] ])

                total = r_depth + a_depths
                if total > 0:
                    r_freq = r_depth/total
                    a_freq = a_depths/total

                    maf = min([r_freq, a_freq])
                    sample_mafs.append(maf)
                else:
                    sample_mafs.append(np.nan)

            maf_matrix[sample_id]=sample_mafs
        
        return pd.DataFrame(maf_matrix)
    

    def getPopMAF(self, filter_homo: bool = False) -> np.array:
        """
        Generates minor-allele frequency (MAF) across all samples.

        INPUT:
            filter_homo <bool, optional> : Whether to filter out homozygous variants. Defaults to False.

        OUTPUT:
            maf_matrix <pd>DataFrame> : the MAF matrix.
        """
        maf_matrix = []

        for l in self.var_lines:
            total = 0
            mac = 0
            for sample_id in self.sample_ids:
                sample_mafs = []

                ## use the dictionary containing allele information for the specified sample
                if sample_id not in l.samples:
                    raise ValueError(f"{sample_id} not in dataset.")
                
                sample_dict = l.samples[sample_id]

                ## this block handles merged vcfs where there are no alternate alleles
                if sample_dict["AD"][0] == ".":
                    sample_mafs.append(np.nan)
                    continue
                
                else:
                    r_depth = int(sample_dict["AD"][0])
                    a_depths = sum([ int(i) for i in sample_dict["AD"][1:] ])
                
                sample_total = r_depth + a_depths
                total += sample_total

                # print(sample_dict["AD"], r_depth, a_depths, total)

                sample_mac = min([r_depth, a_depths])
                mac += sample_mac

            # print(total, mac)
            maf = mac/total
            maf_matrix.append(maf)
        
        return np.array(maf_matrix)


    def filterMissingness( self, filter_t: int = 0.75 ) -> np.array:
        return None


    def fws( self, maf_matrix: np.array, n_maf_bins: int = 10 ) -> dict:

        print("NOTE: this method is in development and currently does not produce correct results.")

        # 1. Calculate Heterozygosity for Each SNP and Sample
        h_matrix = 1 - (maf_matrix**2 + (1 - maf_matrix)**2)  

        # 2. Calculate Population Heterozygosity (H_S) for Each SNP
        h_s = np.nanmean(h_matrix, axis=0)  # Mean across samples

        # 3. Define MAF Bins
        maf_bins = np.linspace(0, 0.5, n_maf_bins + 1)

        fws_values = {}
        for i, sample_id in enumerate(self.sample_ids):
            h_w_values = []  # Within-individual heterozygosity for each MAF bin
            h_s_values = []  # Within-population heterozygosity for each MAF bin

            # 4. Calculate Mean H_W and H_S for Each MAF Bin
            for j in range(n_maf_bins):
                maf_lower = maf_bins[j]
                maf_upper = maf_bins[j + 1]

                bin_indices = np.where((maf_matrix[i] >= maf_lower) & 
                                       (maf_matrix[i] < maf_upper))[0]

                if len(bin_indices) > 0:
                    h_w_values.append(np.nanmean(h_matrix[i, bin_indices]))
                    h_s_values.append(np.nanmean(h_s[bin_indices]))

            # 5. Linear Regression of H_W vs. H_S
            if len(h_w_values) > 1: 
                slope, _, _, _, _ = linregress(h_w_values, h_s_values)
                fws = 1 - slope 
            else:
                fws = np.nan  # Not enough data points for regression

            fws_values[sample_id] = fws

        return fws_values


        # he = self._getPopHeterozygosity(maf_matrix)

        # fws_box = []
        # for sample_maf in maf_matrix:
        #     hs = self._getHeterozygosity(sample_maf)
        #     print(he/hs)
        #     fws_box.append(1-(he/hs))
        
        # return zip(self.sample_ids, fws_box)
    
    
    def _getHeterozygosity(self, maf_array: np.array) -> float:

        return np.mean([2*(i*(1-i)) for i in maf_array if not np.isnan(i)])
    

    def _getPopHeterozygosity( self, maf_matrix: np.array ) -> float:
        
        ## calculate expected heterozygosity across the population
        n = maf_matrix.shape[0]  ## Number of samples

        he = (n/(n-1))*(1-np.sum(np.mean(maf_matrix, axis=0)))

        # mean_maf =  np.array(np.mean(maf_matrix, axis=0))  ## Calculate mean along axis 0 (samples)

        return he