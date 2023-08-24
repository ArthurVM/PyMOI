import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple

from .mutation import Mutation

GMM = namedtuple("GMM", ["model", "cluster_assignments", "bic", "k"])

class MOI(object):
    """ MOI class
    """

    def __init__(self):
        super(MOI, self).__init__()

    def parse_vcf(self, vcf_file : str) -> None:
        """ parses a vcf file
        """
        self.vcf_path = vcf_file
        self.info = []
        self.lines = []

        with open(vcf_file, "r") as fin:
            for line in fin.readlines():
                if line.startswith("#CHROM"):
                    sample_ids = line.strip("\n").split("\t")[9:]
                    print(f"Detected {len(sample_ids)} samples : {sample_ids}")
                elif line.startswith("#"):
                    self.info.append(line)
                else:
                    muts = Mutation()
                    muts.parse_line(line, sample_ids)
                    self.lines.append(muts)


    def baf_matrix(self, sample_id : str, filter_homo : bool=False) -> None:
        """ takes a VCF object and outputs B-allele frequency and count matrices of form
        chrom    pos    alt    ref
        chr1     1      ##     ##

        where ## are read depths (count matrix) or allele frequency (frequence matrix) of that allele
        """
        af_list = []
        ac_list = []
        for l in self.lines:

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

        self.af_matrix = pd.DataFrame.from_records(af_list)
        self.ac_matrix = pd.DataFrame.from_records(ac_list)


    def plot_baf(self, gmm : GMM=None, plot_ref : bool=False) -> None:
        """ Plots the B-allele frequency with or without clustering data and responsibilities
        """

        if gmm == None:
            self._plot_non_clustered(plot_ref)

        else:
            self._plot_clustered(gmm, plot_ref)


    def _plot_non_clustered(self, plot_ref : bool=False) -> None:
        """ Make a SNV plot without cluster designations
        """

        df_sorted, chrom_positions = self._preprocess_for_plotting()

        scatter_kwargs = dict(alpha=0.5, linewidths=0.0, edgecolors=None)
        marginal_kwargs = dict(bins=25, fill=True, kde=True, weights=0.01)

        ## construct matrices which include reference data for plotting
        if plot_ref:
            x_axis_with_ref = []
            for i in df_sorted['x_axis']:
                x_axis_with_ref.append(i)
                x_axis_with_ref.append(i)

            dat = []
            for i in range(len(df_sorted['alt_freq'])):
                dat.append(df_sorted['alt_freq'][i])
                dat.append(df_sorted['ref_freq'][i])

            ## plot alt and ref data
            af_jp = sns.jointplot(np.array(x_axis_with_ref), np.array(dat), height=8, kind='scatter',
                joint_kws=scatter_kwargs,
                marginal_kws=marginal_kwargs)

        ## plot only alt allele freqs
        else:
            af_jp = sns.jointplot(x='x_axis', y='alt_freq', data=df_sorted, height=8, kind='scatter',
                joint_kws=scatter_kwargs,
                marginal_kws=marginal_kwargs)

        ## format plot
        af_jp.ax_joint.set_xticks(list(chrom_positions.values()), list(chrom_positions.keys()), rotation=90, ha='center')
        plt.subplots_adjust(bottom=0.15)
        plt.xlim(-0.5, len(chrom_positions) - 0.5)

        plt.xlabel('Chromosome')
        plt.ylabel('Allele Frequency')

        plt.tight_layout()
        plt.show()


    def _plot_clustered(self, gmm : GMM, plot_ref : bool=False) -> None:

        self.af_matrix['cluster'] = gmm.cluster_assignments
        self.af_matrix['responsibility'] = self._responsibilities(gmm)

        df_sorted, chrom_positions = self._preprocess_for_plotting()

        ## construct matrices which include reference data for plotting
        if plot_ref:
            x_axis_with_ref = []
            for i in df_sorted['x_axis']:
                x_axis_with_ref.append(i)
                x_axis_with_ref.append(i)

            dat = []
            for i in range(len(df_sorted['alt_freq'])):
                dat.append(df_sorted['alt_freq'][i])
                dat.append(df_sorted['ref_freq'][i])

            ## plot alt and ref data
            af_jp = sns.jointplot(np.array(x_axis_with_ref), np.array(dat), height=8, kind="scatter",
                joint_kws=scatter_kwargs,
                marginal_kws=marginal_kwargs)

        ## plot only alt allele freqs
        else:
            af_jp = sns.jointplot(x='x_axis', y='alt_freq', data=df_sorted, hue='cluster', palette='tab10', alpha=0.5, height=8)
            # af_jp.ax_joint.barh(df_sorted['alt_freq'], df_sorted['responsibility'], height=0.002, alpha=0.5)

        ## format plot
        af_jp.ax_joint.set_xticks(list(chrom_positions.values()), list(chrom_positions.keys()), rotation=90, ha='center')
        plt.subplots_adjust(bottom=0.15)
        plt.xlim(-0.5, len(chrom_positions) - 0.5)

        plt.xlabel('Chromosome')
        plt.ylabel('Allele Frequency')

        plt.tight_layout()
        plt.show()


    def _plot_clustered_responsibility(self, gmm : GMM, plot_ref : bool=False) -> None:

        import seaborn as sns
        import matplotlib.pyplot as plt

        # Assuming you have a DataFrame 'df' containing SNV data with columns: 'chromosome', 'position', 'allele_frequency', 'cluster_assignment', and 'responsibility'

        self.af_matrix['cluster'] = gmm.cluster_assignments
        self.af_matrix['responsibility'] = self._responsibilities(gmm)

        df_sorted, chrom_positions = self._preprocess_for_plotting()

        # Create the jointplot with scatter plot and histograms
        af_jp = sns.jointplot(x='x_axis', y='alt_freq', data=df_sorted, hue='cluster', palette='tab10', alpha=0.7, height=7)

        # Clear the existing scatter plot on the jointplot
        af_jp.ax_joint.cla()

        # Create scatter plot for each cluster
        for cluster, data in df_sorted.groupby('cluster'):
            # Plot scatter plot for each cluster
            af_jp.ax_joint.scatter(data['x_axis'], data['alt_freq'], label=f'Cluster {cluster}', alpha=0.7, color=f'C{cluster}')

            # Add point responsibilities on the left of the scatter plot
            af_jp.ax_joint.barh(data['alt_freq'], data['responsibility'], height=0.002, color=f'C{cluster}', alpha=0.5)

        # Set x-axis labels and title
        af_jp.set_axis_labels('SNV Position on Chromosome', 'Allele Frequency')
        af_jp.fig.suptitle('SNV Data Joint Plot with Cluster Assignments and Point Responsibilities')

        # Show the legend
        af_jp.ax_joint.legend(title='Cluster')

        # Show the updated jointplot
        plt.show()


    def fit_binomial_mixture_model(self, max_k : int=5, n_iter : int=5, random_state=None, plot : bool=False) -> GMM:
        from scipy.stats import binom
        from sklearn.metrics import silhouette_score
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

        """
        Fit a binomial mixture model on allele coverage data.

        Parameters:
            allele_counts (numpy array or list): Array of allele counts (number of successes).
            k (int): Maximum number of mixture components (clusters) to fit.
            n_iter (int): Number of iterations for the Gaussian Mixture Model algorithm.
            random_state (int or None): Random seed for reproducibility.

        Returns:
            model (sklearn.mixture.GaussianMixture): Fitted binomial mixture model.
            cluster_assignments (numpy array): Cluster assignments for each data point.
        """
        # Check if the input data is valid
        allele_counts = np.array(self.af_matrix['alt_freq'])

        if np.any(allele_counts < 0):
            raise ValueError("Allele counts must be non-negative integers.")

        ##### REDUNDANT BLOCK #####
        # ## Calculate the total number of trials (coverage) for each data point (assumed to be 1 for binomial distribution)
        # total_trials = np.ones_like(allele_counts)
        #
        # ## Estimate the success probability (p) using the method of moments
        # p_estimate = np.mean(allele_counts / total_trials)
        #
        # ## Create the binomial distribution object with the estimated p
        # binom_dist = binom(n=total_trials, p=p_estimate)
        ###########################

        ## find optimal k by minimising BIC

        ## store bic scores for plotting
        bic_scores = []
        sil_scores = []

        ## best score should be np.inf if BIC is used
        best_score = 0
        best_model = None
        best_k = None

        af_matrix = allele_counts[:, np.newaxis]

        ## iterate through cluster centers
        for k in range(1, max_k + 1):

            ## store cluster metrics for this k iteration
            tmp_sil = []
            tmp_bic = []

            ## carry out multiple iterations for each k
            for _ in range(n_iter):
                ## Initialize the Gaussian Mixture Model
                model = GaussianMixture(n_components=k, random_state=random_state, n_init=1)

                ## Fit the model using the Expectation-Maximization algorithm
                model.fit(af_matrix)

                cluster_assignments = model.predict(af_matrix)

                ## Calculate scores for the current model
                bic = model.bic(af_matrix)
                tmp_bic.append(bic)

                if k > 1:
                    sil = silhouette_score(af_matrix, cluster_assignments, metric='euclidean')
                    tmp_sil.append(sil)
                else:
                    tmp_sil.append(0)

            ## capture silhouette scores and calculate error
            sil_val = np.mean(self._sel_best(np.array(tmp_sil), int(n_iter/5)))
            sil_err = np.std(tmp_sil)
            sil_scores.append((k, sil_val, sil_err))

            ## capture bic scores and calculate error
            bic_val = np.mean(self._sel_best(np.array(tmp_bic), int(n_iter/5)))
            bic_err = np.std(tmp_bic)
            bic_scores.append((k, bic_val, bic_err))

            ## Update the best model if the current silhouette score is lower
            ## sil score seems to perform better on 1D clustering matrices than BIC
            if len(tmp_sil) == 0:
                max_sil = 0
            else:
                max_sil = max(tmp_sil)

            if max_sil > best_score:
                best_score = max_sil
                best_model = model
                best_k = k

        if plot:
            self._plot_cluster_measures(sil_scores[1:], bic_scores)

        # Get the cluster assignments for each data point
        cluster_assignments = best_model.predict(af_matrix)

        gmm = GMM(best_model, cluster_assignments, best_score, best_k)

        return gmm


    def _plot_cluster_measures(self, sil_scores : list, bic_scores : list) -> None:
        """ Plots the cluster metrics along with error bars
        """
        plt.figure(figsize=(12, 6))

        ## sil line plot
        plt.subplot(1, 2, 1)
        plt.errorbar([x for x, y, z in sil_scores], [y for x, y, z in sil_scores], yerr=[z for x, y, z in sil_scores])
        plt.title("Silhouette score", fontsize=20)
        plt.xticks([x for x, y, z in sil_scores])
        plt.xlabel("Clusters")
        plt.ylabel("Score")

        ## bic line plot
        plt.subplot(1, 2, 2)
        plt.errorbar([x for x, y, z in bic_scores], [y for x, y, z in bic_scores], yerr=[z for x, y, z in bic_scores])
        plt.title("BIC score", fontsize=20)
        plt.xticks([x for x, y, z in sil_scores])
        plt.xlabel("Clusters")
        plt.ylabel("Score")

        plt.tight_layout()
        plt.show()


    def _preprocess_for_plotting(self):
        """ Preprocesses the B-allele frequency matrix for constructing plots
        """

        ## sort the DataFrame by 'chromosome' and 'position'
        df_sorted = self.af_matrix.sort_values(by=['chromosome', 'position'])

        ## get unique chromosome names and set their positions such that relative size is conserved
        unique_chromosomes = df_sorted['chromosome'].unique()
        max_pos = max(self.af_matrix['position'])

        chrom_positions = {}
        summed_pos = 0
        for i, chrom in enumerate(unique_chromosomes):
            chrom_positions[chrom] = summed_pos
            summed_pos += max(self.af_matrix.loc[self.af_matrix['chromosome'] == chrom]['position'])/max_pos

        ## merge and position chromosomes along the x-axis
        df_sorted['x_axis'] = [chrom_positions[chrom] + int(pos) / max(self.af_matrix.loc[self.af_matrix['chromosome'] == chrom]['position'])
            for chrom, pos in zip(df_sorted['chromosome'], df_sorted['position'])]

        return df_sorted, chrom_positions


    def _sel_best(self, arr : list, X : int) -> list:
        '''
        returns the set of X configurations with shorter distance
        '''
        dx = np.argsort(arr)[:X]

        return arr[dx]


    def _responsibilities(self, gmm : GMM):
        """ Calculate the probability of correct cluster assignment for each data point
        """
        allele_counts = np.array(self.af_matrix['alt_freq'])
        responsibilities = gmm.model.predict_proba(allele_counts[:, np.newaxis])
        max_responsibility = np.max(responsibilities, axis=1)
        probability_correct_assignment = np.mean(max_responsibility)

        self.af_matrix['probs'] = max_responsibility

        return self


    def _make_base_matrix(self):
        """ construct a base matrix to store positional data of form

        chr    pos
        chr#   100
        chr#   200
        ...    ...
        """
        pos_box = []
        for l in self.lines:
            pos_box.append({'chromosome' : l.chrom, 'position' : l.pos})

        return pd.DataFrame.from_records(pos_box)


    def plot_af_deprecated(self, chr : str=False, ref : str=False):

        # Sort the DataFrame by 'chromosome' and 'position' for proper plotting order
        df_sorted = self.af_matrix.sort_values(by=['chromosome', 'position'])

        if chr != False:
            df_sorted = self.af_matrix.loc[self.af_matrix['chromosome'] == chr]

        # Get unique chromosome names and their corresponding positions
        unique_chromosomes = df_sorted['chromosome'].unique()
        chrom_positions = {chrom: i for i, chrom in enumerate(unique_chromosomes)}

        # Set the figure size
        plt.figure(figsize=(12, 6))

        # Create a scatter plot for ref and alt allele frequencies
        if ref:
            plt.scatter([chrom_positions[chrom] + int(pos) / max(self.af_matrix.loc[self.af_matrix['chromosome'] == chrom]['position']) for chrom, pos in zip(df_sorted['chromosome'], df_sorted['position'])],
                        df_sorted['ref_freq'], label='Ref Frequency', alpha=0.3)

        plt.scatter([chrom_positions[chrom] + int(pos) / max(self.af_matrix.loc[self.af_matrix['chromosome'] == chrom]['position']) for chrom, pos in zip(df_sorted['chromosome'], df_sorted['position'])],
                    df_sorted['alt_freq'], label='Alt Frequency', alpha=0.3)

        ## format and plot
        plt.xticks(list(chrom_positions.values()), list(chrom_positions.keys()), rotation=45, ha='right')

        plt.subplots_adjust(bottom=0)

        plt.xlabel('Chromosome')
        plt.ylabel('Allele Frequency')
        plt.title('Ref and Alt Allele Frequencies')

        plt.legend()

        plt.tight_layout()
        plt.show()


    def get_heterozygosity(self):
        """ compute allelic heterozygosity at each locus for sample and population level
        """
        ## dataframe to store heterozygosity values
        het_df = self._make_base_matrix()

        ref_counts_pop = []
        alt_counts_pop = []
        total_depth_pop = []

        for i, sample_id in enumerate(self.lines[0].samples):
            ## get alternate allele counts and total depth (AD and DP fields)
            ref_counts = np.array([i.samples[sample_id]['AD'][0] for i in self.lines])
            alt_counts = np.array([i.samples[sample_id]['AD'][1] for i in self.lines])
            total_depth = np.array([i.samples[sample_id]['DP'] for i in self.lines])

            if i == 0:
                ref_counts_pop = ref_counts
                alt_counts_pop = alt_counts
                total_depth_pop = total_depth
            else:
                ref_counts_pop += ref_counts
                alt_counts_pop += alt_counts
                total_depth_pop += total_depth

            ## compute heterozygosity for each variant
            ## ignore 0/n RunTime warnings for this block
            with np.errstate(divide='ignore',invalid='ignore'):
                raf = ref_counts / total_depth
                aaf = alt_counts / total_depth

            heterozygosity_values = 1 - (raf ** 2 + aaf ** 2)
            het_df[sample_id] = heterozygosity_values

        ## compute heterozygosity at each position across the entire population
        ## ignore 0/n RunTime warnings for this block
        with np.errstate(divide='ignore',invalid='ignore'):
            raf_pop = ref_counts_pop / total_depth_pop
            aaf_pop = alt_counts_pop / total_depth_pop

        het_df["ref_pop_heterozygosity"] = raf_pop
        het_df["alt_pop_heterozygosity"] = aaf_pop

        return het_df


    def old_get_heterozygosity(self):
        """ compute population-level heterozygosity (Hw)
        """
        # for sample_id in
        sample_mafs = self._get_MAF()
        return 1 - (mafs ** 2 + (1 - mafs) ** 2)


    def get_MAF(self, sample_id):
        """ compute Minor Allele Frequency (MAF)
        """
        # for sample_id in self.lines[0].samples:

        ## get alternate allele counts and total depth (AD and DP fields)
        alt_counts = np.array([min(i.samples[sample_id]['AD']) for i in self.lines])
        total_depth = np.array([i.samples[sample_id]['DP'] for i in self.lines])

        ## compute MAF for each variant
        mafs = np.array(alt_counts / total_depth)

        return mafs


    def get_Fws(self):
        """ compute within-host diversity statistic (Fws)
        """
        het_df = self.get_heterozygosity()

        Hw = np.mean(het_df["alt_pop_heterozygosity"])
        print(Hw, np.mean(het_df["ref_pop_heterozygosity"]))
        for sample_id in self.lines[0].samples:
            Hs = np.mean(het_df[sample_id])
            fws = 1-(Hs/Hw)

            print(sample_id, fws)

        # # Create MAF bins, 0..0.05, 0.05..0.1,...,0.45..0.5
        # maf_bins = np.digitize(get_MAF(vcf_file), np.linspace(0, 0.5, 11)) - 1
        #
        # mu_population_het = [np.mean(population_het[maf_bins == bin_id]) for bin_id in range(11)]
        # mu_sample_het = sample_het.groupby(maf_bins, axis=1).mean()
        #
        # def linear_regression(x, y):
        #     slope, intercept, _, _, _ = stats.linregress(x, y)
        #     return 1 - slope
        #
        # fws = mu_sample_het.apply(lambda x: linear_regression(mu_population_het, x))
        # return fws
