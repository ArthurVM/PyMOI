import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple

from .mutation import Mutation

GMM = namedtuple("GMM", ["model", "cluster_assignments", "score", "k"])


def parse_bedcov(bedcov_file : str) -> np.array:
    """ Read a bedcov file and produce a numpy array of coverage
    """
    with open(bedcov_file, 'r') as fin:
        cov = np.array([int(line.strip("\n").split("\t")[-1]) for line in fin.readlines()])
        cov = cov[cov!=0]

    return cov


def parse_vcf(vcf_file : str) -> None:
    """ parses a vcf file
    """
    info = []
    lines = []

    with open(vcf_file, "r") as fin:
        for line in fin.readlines():
            if line.startswith("#CHROM"):
                sample_ids = line.strip("\n").split("\t")[9:]
                print(f"Detected {len(sample_ids)} samples : {sample_ids}")
            elif line.startswith("#"):
                info.append(line)
            else:
                muts = Mutation()
                muts.parse_line(line, sample_ids)
                lines.append(muts)

    return lines


def make_baf_matrix(    lines : list,
                        sample_id : str,
                        filter_homo : bool=True ) -> None:
    """ takes a VCF object and outputs B-allele frequency and count matrices of form
    chrom    pos    alt    ref
    chr1     1      ##     ##

    where ## are read depths (count matrix) or allele frequency (frequence matrix) of that allele
    """
    af_list = []
    ac_list = []
    for l in lines:

        ## use the dictionary containing allele information for the specified sample
        if sample_id not in l.samples:
            raise ValueError(f"{sample_id} not in dataset.")

        sample_dict = l.samples[sample_id]

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
                        af_list.append({'chromosome': l.chrom, 'position': l.pos, 'ref_freq': r_freq, 'alt_freq': a_freq})
                        ac_list.append({'chromosome': l.chrom, 'position': l.pos, 'ref_depth': r_depth, 'alt_depth': a_depths})
                else:
                    af_list.append({'chromosome': l.chrom, 'position': l.pos, 'ref_freq': r_freq, 'alt_freq': a_freq})
                    ac_list.append({'chromosome': l.chrom, 'position': l.pos, 'ref_depth': r_depth, 'alt_depth': a_depths})

    af_matrix = pd.DataFrame.from_records(af_list)
    ac_matrix = pd.DataFrame.from_records(ac_list)

    return af_matrix, ac_matrix
