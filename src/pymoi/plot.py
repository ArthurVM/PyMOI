import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple

GMM = namedtuple("GMM", ["model", "cluster_assignments", "score", "k", "prob"])


def finite_mixture_model(   baf_matrix : pd.DataFrame,
                            min_k : int=1,
                            max_k : int=5,
                            n_iter : int=5,
                            responsibility_upper : float=0.1,
                            plot_cluster_fit : bool=False,
                            random_state=None   ) -> GMM:

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
    allele_counts = np.array(baf_matrix['alt_freq'])

    if np.any(allele_counts < 0):
        raise ValueError("Allele counts must be non-negative integers.")

    if responsibility_upper < 0.0 or responsibility_upper > 1.0:
        raise ValueError("responsibility_upper counts must be between 0-1.")

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
    for k in range(min_k, max_k + 1):

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
        sil_val = np.mean(_sel_best(np.array(tmp_sil), int(n_iter/5)))
        sil_err = np.std(tmp_sil)
        sil_scores.append((k, sil_val, sil_err))

        ## capture bic scores and calculate error
        bic_val = np.mean(_sel_best(np.array(tmp_bic), int(n_iter/5)))
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

    if plot_cluster_fit:
        _plot_cluster_fit(sil_scores[1:], bic_scores)

    # Get the cluster assignments for each data point
    cluster_assignments = best_model.predict(af_matrix)

    ## get responsibilities for each data point
    responsibilities = _responsibilities(baf_matrix, best_model)
    n = len(responsibilities)
    resp_ss = sorted(responsibilities)[:int(n*responsibility_upper)]

    gmm = GMM(best_model, cluster_assignments, best_score, best_k, np.mean(resp_ss))

    return gmm


def plot_baf(   baf_matrix : pd.DataFrame,
                sample_id : str,
                gmm : GMM=None,
                plot_ref : bool=False   ) -> None:
    """ Plots the B-allele frequency with or without clustering data and responsibilities
    """

    if gmm == None:
        _plot_non_clustered(baf_matrix, sample_id, plot_ref)
    else:
        _plot_clustered(baf_matrix, sample_id, gmm, plot_ref)


def _plot_non_clustered(    baf_matrix : pd.DataFrame,
                            sample_id : str,
                            plot_ref : bool=False   ) -> None:
    """ Make a BAF plot without cluster designations
    """

    df_sorted, chrom_positions = _preprocess_for_plotting(baf_matrix)

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
    af_jp.ax_marg_x.set_xlim(-0.1, len(chrom_positions)-0.5)
    af_jp.ax_joint.set_ylim([0, 1])

    af_jp.fig.set_figwidth(12)
    af_jp.fig.set_figheight(8)

    plt.xlabel('Chromosome')
    plt.ylabel('Allele Frequency')

    plt.tight_layout()


def _plot_clustered(    baf_matrix : pd.DataFrame,
                        sample_id : str,
                        gmm : GMM,
                        plot_ref : bool=False   ) -> None:
    """ Make a BAF plot with cluster designations
    """

    baf_matrix['cluster'] = gmm.cluster_assignments

    df_sorted, chrom_positions = _preprocess_for_plotting(baf_matrix)

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
        af_jp = sns.jointplot(np.array(x_axis_with_ref), np.array(dat), kind="scatter",
            joint_kws=scatter_kwargs,
            marginal_kws=marginal_kwargs)

    ## plot only alt allele freqs
    else:
        af_jp = sns.jointplot(x='x_axis', y='alt_freq', data=df_sorted, hue='cluster', palette='tab10', alpha=0.5)
        # af_jp.ax_joint.barh(df_sorted['alt_freq'], df_sorted['responsibility'], height=0.002, alpha=0.5)

    ## format plot
    af_jp.ax_joint.set_xticks(list(chrom_positions.values()), list(chrom_positions.keys()), rotation=90, ha='center')
    plt.subplots_adjust(bottom=0.15)
    af_jp.ax_marg_x.set_xlim(-0.1, len(chrom_positions)-0.5)
    af_jp.ax_joint.set_ylim([0, 1])

    af_jp.figure.set_figwidth(12)
    af_jp.figure.set_figheight(8)
    
    plt.title(sample_id)
    plt.xlabel('Chromosome')
    plt.ylabel('Allele Frequency')

    plt.tight_layout()

    plt.savefig(f"{sample_id}_allelefreq.png")


def _plot_cluster_fit( sil_scores : list,
                            bic_scores : list   ) -> None:
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


def _preprocess_for_plotting(baf_matrix : pd.DataFrame) -> [pd.DataFrame, dict]:
    """ Preprocesses the B-allele frequency matrix for constructing plots
    """

    ## sort the DataFrame by 'chromosome' and 'position'
    df_sorted = baf_matrix.sort_values(by=['chromosome', 'position'])

    ## get unique chromosome names and set their positions such that relative size is conserved
    unique_chromosomes = df_sorted['chromosome'].unique()
    max_pos = max(baf_matrix['position'])

    chrom_positions = {}
    summed_pos = 0
    for i, chrom in enumerate(unique_chromosomes):
        chrom_positions[chrom] = summed_pos
        summed_pos += max(baf_matrix.loc[baf_matrix['chromosome'] == chrom]['position'])/max_pos

    ## merge and position chromosomes along the x-axis
    df_sorted['x_axis'] = [chrom_positions[chrom] + int(pos) / max(baf_matrix.loc[baf_matrix['chromosome'] == chrom]['position'])
        for chrom, pos in zip(df_sorted['chromosome'], df_sorted['position'])]

    return df_sorted, chrom_positions


def _sel_best(  arr : list,
                X : int ) -> list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx = np.argsort(arr)[:X]

    return arr[dx]


def _responsibilities(  baf_matrix : pd.DataFrame,
                        model   ) -> np.array:
    """ Calculate the probability of correct cluster assignment for each data point
    """
    allele_counts = np.array(baf_matrix['alt_freq'])
    responsibilities = model.predict_proba(allele_counts[:, np.newaxis])
    responsibilities = np.max(responsibilities, axis=1)

    return responsibilities