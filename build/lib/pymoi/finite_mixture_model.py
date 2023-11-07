import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

class FiniteMixtureModel:
    def __init__(self, distribution='binomial', k=2, niter=1000):
        if distribution not in ['binomial', 'polynomial']:
            raise ValueError("Invalid distribution. Choose 'binomial' or 'polynomial'")

        if not isinstance(k, int) or k < 1 or k > 5:
            raise ValueError("Number of mixture components (k) must be an integer between 1 and 5")

        if not isinstance(niter, int) or niter < 1:
            raise ValueError("niter (number of iterations) must be a positive integer")

        self.distribution = distribution
        self.k = k
        self.niter = niter

    def fit(self, counts_matrix, sample_id, coverage_threshold=0):
        if not isinstance(counts_matrix, pd.DataFrame) or counts_matrix.shape[1] != 3:
            raise ValueError("Invalid alleleCounts object")

        if not isinstance(sample_id, str) or len(sample_id) != 1:
            raise ValueError("sample.id must be a single character")

        if sample_id not in counts_matrix.index:
            raise ValueError("sample.id not found in counts_matrix")

        if not isinstance(coverage_threshold, int) or coverage_threshold < 0:
            raise ValueError("coverage_threshold must be a non-negative integer")

        y = counts_matrix.loc[sample_id, ['alt', 'ref']].values.T
        ds = counts_matrix.loc[sample_id, 'dosage']

        ## Filter SNPs that are uninformative for MOI (i.e. non hets), low coverage, or missing
        keep_snps = ~np.isnan(y.sum(axis=0)) & (y.sum(axis=0) > coverage_threshold) & (~np.isnan(ds)) & (ds == 1)
        y_obs = y[:, keep_snps]

        if self.distribution == 'binomial':
            baf = y_obs[0] / y_obs.sum(axis=0)
            model = GaussianMixture(n_components=self.k, max_iter=self.niter)
        else:
            baf = None  ## For polynomial distribution, BAF is not computed
            model = GaussianMixture(n_components=self.k, max_iter=self.niter)

        model.fit(y_obs)

        return {'model': model, 'baf': baf, 'sample.id': sample_id}

# binomial_model = FiniteMixtureModel(distribution='binomial', k=3)
# result = binomial_model.fit(counts_matrix, sample_id, coverage_threshold=10)

# polynomial_model = FiniteMixtureModel(distribution='polynomial', k=2)
# result = polynomial_model.fit(counts_matrix, sample_id, coverage_threshold=10)
