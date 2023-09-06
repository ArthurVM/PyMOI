""" Simulate a distribution from a cov bed file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, poisson, uniform, kstest
from collections import namedtuple

Dist = namedtuple("Dist", ["X", "mean", "median", "s2", "sd", "N"])
GMM = namedtuple("GMM", ["model", "cluster_assignments", "score", "k"])


class SimDist(object):
    """docstring for SimDist."""

    def __init__(   self,
                    distribution : np.array ):

        super(SimDist, self).__init__()

        self.X = distribution
        self.D = self._inspect()


    def _inspect(self) -> Dist:
        """ Take a distribution as a numpy array and calculate the parameters.
        May also be transformed using natural log for the purpose of modelling the underlying normal of a lognormal dist
        """
        X = self.X

        N = len(X)
        mu = np.mean(X)
        median = np.median(X)
        s2_c = (1/(N-1))*np.sum((X-mu)**2)
        s2 = np.var(X, ddof=1)
        sd = np.std(X)

        D = Dist(X, mu, median, s2, sd, N)

        return D


    def simulate_distribution(self) -> np.array:
        """ simulate a distribution, either:
         - a lognormal distribution from a set of parameters taken from an underlying normal distribution
         - a Poisson distribution using the median from a modelled coverage distribution
        """
        # rng = np.random.default_rng()
        # return np.round(rng.lognormal(D.mean, D.sd, D.N), 0)
        D = self.D
        return np.random.poisson(D.median, D.N)


    def to_stnorm(self) -> np.array:
        """ transforms the data into N(0,1)
        """
        D = self.D
        # X = D.X/D.sd
        X = D.X-D.mean
        X = X/D.sd

        return X


    def ln_to_norm(self) -> np.array:
        ## adjust for lognormality
        return np.log(self.D)


    def plot_hist(self) -> plt:
        """ Simple plot function
        """
        D = self.D
        # D = transform(D)
        if max(D.X) <= 100:
            bins = int(max(D.X)*0.75)
        else:
            bins = 100

        n, bins, patches = plt.hist(D.X, bins = bins, density=True)
        ## Plot the PDF using the provided mean/median and standard deviation
        pdf_x = np.linspace(min(D.X), max(D.X), 100)
        pdf_y = (1 / (D.sd * np.sqrt(2 * np.pi))) * np.exp(-(pdf_x - D.median)**2 / (2 * D.sd**2))
        plt.plot(pdf_x, pdf_y, label="PDF")

        return plt


    def simulate_baf(   self,
                        simcov : np.array,
                        n : int ) -> pd.DataFrame:
        """ Takes a Dist struct containing randomly sampled data and produce a pandas dataframe of
        indices and randomly generated alt allele frequencies.
        """
        D = self.D
        sc_non0 = simcov[simcov>0.0]
        sample_indices = np.random.choice(len(sc_non0), size=n, replace=False)
        cov_ss = (sc_non0)[sample_indices]
        sim_baf = [ 1-np.random.randint(0, i)/i for i in cov_ss ]

        return pd.DataFrame({ "chromosome" : ["0"]*len(sample_indices), "position" : sample_indices, "alt_freq" : sim_baf }).sort_values(by=['position'])
