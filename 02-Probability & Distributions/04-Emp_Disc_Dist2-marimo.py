import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import rv_discrete, poisson
    return mo, np, pd, plt, rv_discrete, poisson


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Empirical Discrete Distributions
    
    ## Learning goals
    - Construct an empirical probability mass function from counts.
    - Distinguish probability masses from cumulative probabilities.
    - Compare an empirical distribution with a known generating model.
    - Explain sample-size effects without assuming monotonic improvement.
    - Build a distribution from observations and explain what resampling can and cannot do.
    
    ## 1. Probability masses and cumulative probabilities
    An **empirical distribution** assigns equal weight to each recorded observation.
    For a discrete sample, its **empirical probability mass function (PMF)** assigns each distinct value its relative frequency:
    
    $$\widehat p(k)=\frac{\text{number of observations equal to }k}{n}.$$
    
    Its **empirical cumulative distribution function (ECDF)** gives the proportion at or below a threshold:
    
    $$F_n(t)=\frac{\text{number of observations at or below }t}{n}.$$
    
    For the sample [1, 1, 3, 5], the mass at 3 is 1/4, while the cumulative probability at 3 is 3/4.
    At 2, the mass is zero but the cumulative probability is 1/2.
    Masses add to one; cumulative probabilities never decrease and eventually reach one.
    The ECDF is a right-continuous step function: its value at a recorded point includes the observations equal to that point.
    For discrete distributions, use “probability mass,” not “probability density.”
    
    ### Try it yourself
    For [1, 1, 3, 5], calculate the mass at 1, the ECDF at 4, and the proportion strictly above 3.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({"Answers": mo.md("Mass at 1 = 2/4 = 0.50. ECDF at 4 = 3/4 = 0.75. Proportion strictly above 3 = 1/4 = 0.25.")})
    return


@app.cell
def _(np, pd, plt):
    def empirical_table(values, support):
        """Return counts, masses, and cumulative probabilities on a full grid."""
        values = np.asarray(values)
        support = np.asarray(support)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("Provide a nonempty one-dimensional sample.")
        if support.ndim != 1 or len(support) == 0 or not np.all(np.diff(support) > 0):
            raise ValueError("Support must be a nonempty increasing sequence.")
        if not np.isin(values, support).all():
            raise ValueError("The display support must contain every observation.")
        unique_values, counts = np.unique(values, return_counts=True)
        aligned_counts = pd.Series(counts, index=unique_values).reindex(support, fill_value=0).to_numpy()
        masses = aligned_counts / len(values)
        return pd.DataFrame({"Value": support, "Count": aligned_counts,
                             "Empirical_PMF": masses, "ECDF": np.cumsum(masses)})


    def compare_distributions(table, reference_pmf, reference_cdf, title, reference_label):
        """Plot masses and right-continuous cumulative distributions separately."""
        x = table["Value"].to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(x, table["Empirical_PMF"], alpha=0.55, color="tab:orange", label="Observed relative frequency")
        axes[0].plot(x, reference_pmf, "o", color="tab:green", label=reference_label)
        axes[0].set(xlabel="Value", ylabel="Probability mass", title="PMF comparison")
        # Padding shows zero below the support and the final cumulative level.
        padded_x = np.r_[x[0] - 1, x, x[-1] + 1]
        axes[1].step(padded_x, np.r_[0, table["ECDF"], table["ECDF"].iloc[-1]], where="post", color="tab:orange", label="Observed ECDF")
        axes[1].step(padded_x, np.r_[0, reference_cdf, reference_cdf[-1]], where="post", color="tab:green", linestyle="--", label=reference_label)
        axes[1].set(xlabel="Threshold t", ylabel="Probability at or below t", title="CDF comparison", ylim=(-0.03, 1.03))
        for ax in axes:
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        plt.close(fig)
        return fig
    return compare_distributions, empirical_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A known model: a weighted die
    P(5) = 0.5; every other face has probability 0.1. All probabilities are nonnegative and sum to one.
    `rv_discrete` constructs a model from an explicit list of values and their probabilities.
    
    We generate one long seeded sequence and display prefixes. Increasing the displayed size extends the same experiment.
    Independent, identically distributed trials support the law of large numbers: relative frequencies converge to their generating probabilities as sample size grows without bound.
    Finite samples need not match exactly, and agreement need not improve at every increase.
    """)
    return


@app.cell
def _(mo):
    die_seed = mo.ui.slider(start=1, stop=100, step=1, value=42, label="Die experiment seed")
    die_size = mo.ui.slider(steps=[10, 30, 100, 1000, 10000], value=10, label="Die rolls displayed")
    mo.hstack([die_seed, die_size])
    return die_seed, die_size


@app.cell
def _(die_seed, np, rv_discrete):
    die_support = np.arange(1, 7)
    die_probabilities = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.1])
    if not np.all(die_probabilities >= 0) or not np.isclose(die_probabilities.sum(), 1):
        raise ValueError("Probabilities must be nonnegative and sum to one.")
    die_model = rv_discrete(values=(die_support, die_probabilities))
    die_sequence = die_model.rvs(size=10000, random_state=np.random.default_rng(die_seed.value))
    return die_model, die_sequence, die_support


@app.cell
def _(die_model, die_sequence, die_size, die_support, empirical_table, np):
    die_table = empirical_table(die_sequence[:die_size.value], die_support)
    die_table["Theoretical_PMF"] = die_model.pmf(die_support)
    die_table["Theoretical_CDF"] = die_model.cdf(die_support)
    print("Empirical masses sum to:", die_table["Empirical_PMF"].sum())
    print("Largest absolute mass difference:", np.abs(die_table["Empirical_PMF"] - die_table["Theoretical_PMF"]).max())
    die_table.round(4)
    return (die_table,)


@app.cell
def _(compare_distributions, die_table):
    compare_distributions(die_table, die_table["Theoretical_PMF"].to_numpy(), die_table["Theoretical_CDF"].to_numpy(), "Weighted die: observations versus known model", "Theoretical model")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Find a face with count zero, if there is one. Does this make its theoretical probability zero?
    2. Compare the mass at 5 with the cumulative probability at 5. Which faces contribute to the cumulative value?
    3. Inspect 10, 100, 1,000, and 10,000 rolls. Is the largest absolute mass difference smaller at every increase?
    4. Change the seed while keeping the displayed size fixed. Explain why the results change but the model does not.
    
    **Discussion:** An unobserved face has zero empirical mass in this sample, not necessarily zero population probability. P(X ≤ 5) includes faces 1 through 5. Seeded experiments remain random-model simulations; the seed makes their realization reproducible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A distribution estimated from observations
    Generate 500 uniform integer outcomes from 1 through 6 and 500 Poisson outcomes with μ = 1. Pool the samples and build their empirical distribution.
    The combined distribution is a **mixture**, not generally uniform or Poisson.
    Equal component sizes give equal weights. With unequal sizes, the weights are the components' fractions of the pooled observations.
    
    Because we know the generating models in this teaching example, the pooled target PMF is:
    
    $$p_{\text{mixture}}(k)=w\,p_{\text{uniform}}(k)+(1-w)\,p_{\text{Poisson}}(k).$$
    
    Here w = 0.5. The original sample estimates this target with sampling error.
    Collecting fixed counts from each component differs from independently choosing a component for every observation; both have the same target marginal mixture proportions at these weights.
    """)
    return


@app.cell
def _(mo):
    source_seed = mo.ui.slider(start=1, stop=100, step=1, value=17, label="Original data seed")
    source_seed
    return (source_seed,)


@app.cell
def _(np, poisson, source_seed, empirical_table, rv_discrete):
    uniform_sample = np.random.default_rng(source_seed.value).integers(1, 7, size=500)
    poisson_sample = poisson.rvs(mu=1, size=500, random_state=np.random.default_rng(source_seed.value + 1000))
    original_values = np.concatenate([uniform_sample, poisson_sample])
    # Include observed values and enough of the known Poisson tail for comparison.
    mixture_max = max(int(original_values.max()), 6, int(poisson.ppf(0.9999, mu=1)))
    mixture_support = np.arange(mixture_max + 1)
    source_table = empirical_table(original_values, mixture_support)
    mixture_weight = len(uniform_sample) / len(original_values)
    uniform_mass = np.where((mixture_support >= 1) & (mixture_support <= 6), 1 / 6, 0)
    uniform_cdf = np.clip(mixture_support / 6, 0, 1)
    mixture_pmf = mixture_weight * uniform_mass + (1 - mixture_weight) * poisson.pmf(mixture_support, mu=1)
    mixture_cdf = mixture_weight * uniform_cdf + (1 - mixture_weight) * poisson.cdf(mixture_support, mu=1)
    observed_support, observed_counts = np.unique(original_values, return_counts=True)
    empirical_model = rv_discrete(values=(observed_support, observed_counts / len(original_values)))
    print("Original observations:", len(original_values))
    print("Uniform-component weight:", mixture_weight)
    print("Known mixture probability above display:", (1 - mixture_weight) * poisson.sf(mixture_max, mu=1))
    source_table.round(4)
    return empirical_model, mixture_cdf, mixture_pmf, mixture_support, original_values, source_table


@app.cell
def _(compare_distributions, mixture_cdf, mixture_pmf, source_table):
    compare_distributions(source_table, mixture_pmf, mixture_cdf, "Original sample versus known mixture", "Known generating mixture")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Explain why zero can occur in the pooled sample although the uniform component cannot produce it.
    2. Why is the probability of 5 in the mixture not simply 1/6?
    3. Change the original-data seed. Which changes: the known mixture or its empirical estimate?
    4. If you used 750 uniform observations and 250 Poisson observations, what would w be?
    
    **Discussion:** Zero comes from the Poisson component. Mixture probabilities weight both sources. Changing the original-data seed changes the estimate, not the component models or weights. With 750 and 250 observations, w = 0.75.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Draw new observations from the empirical model
    The empirical model assigns each recorded value its observed relative frequency.
    Drawing independently from it is equivalent in distribution to selecting observations uniformly **with replacement** from the original sample.
    Duplicates may appear, and some original observations may not be selected. This is the sampling operation used in the ordinary nonparametric bootstrap; we are not estimating confidence intervals here.
    
    There are now three distinct quantities:
    1. The known generating mixture.
    2. The original sample's empirical distribution, which estimates that mixture.
    3. The new sample's empirical distribution, which approximates the fixed empirical model.
    
    A larger new sample improves the approximation to item 2 in the usual probabilistic sense. It does not increase the amount of original information or remove item 2's estimation error.
    Values missing from the original sample have probability zero in this empirical model, even if the generating population can produce them.
    """)
    return


@app.cell
def _(mo):
    new_seed = mo.ui.slider(start=1, stop=100, step=1, value=73, label="Resampling seed")
    new_size = mo.ui.slider(steps=[10, 100, 1000, 10000, 100000], value=100, label="New sample size displayed")
    mo.hstack([new_seed, new_size])
    return new_seed, new_size


@app.cell
def _(empirical_model, new_seed, np):
    resampled_sequence = empirical_model.rvs(size=100000, random_state=np.random.default_rng(new_seed.value))
    return (resampled_sequence,)


@app.cell
def _(empirical_model, empirical_table, mixture_support, new_size, resampled_sequence):
    resample_table = empirical_table(resampled_sequence[:new_size.value], mixture_support)
    resample_table["Original_empirical_PMF"] = empirical_model.pmf(mixture_support)
    resample_table["Original_empirical_CDF"] = empirical_model.cdf(mixture_support)
    resample_table.round(4)
    return (resample_table,)


@app.cell
def _(compare_distributions, resample_table):
    compare_distributions(resample_table, resample_table["Original_empirical_PMF"].to_numpy(), resample_table["Original_empirical_CDF"].to_numpy(), "New observations versus original empirical model", "Original empirical model")
    return


@app.cell
def _(mixture_pmf, np, resample_table, source_table):
    print("Largest absolute PMF difference:")
    print("Original sample versus generating mixture:", np.abs(source_table["Empirical_PMF"] - mixture_pmf).max())
    print("New sample versus original empirical model:", np.abs(resample_table["Empirical_PMF"] - resample_table["Original_empirical_PMF"]).max())
    print("New sample versus generating mixture:", np.abs(resample_table["Empirical_PMF"] - mixture_pmf).max())
    print("These are descriptive discrepancies, not hypothesis tests.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Increase the new sample size while keeping the original-data seed fixed. Which distribution is the new sample approaching?
    2. Does drawing 100,000 new observations give us information equivalent to 100,000 newly collected population observations?
    3. Change only the resampling seed, then only the original-data seed. Explain the different effects.
    4. Can this empirical model generate a value absent from the original observations?
    
    **Discussion:** The new sample targets the original empirical model. It adds simulated observations, not independent population information. It cannot generate a value absent from the original sample.
    
    ## Conclusions
    - The empirical PMF describes equality to a value; the ECDF describes being at or below a threshold.
    - Known but unobserved outcomes should remain visible with zero empirical mass.
    - Larger independent samples generally approximate their generating distribution more closely; finite-sample improvement is not necessarily monotonic.
    - Pooling observations produces component-size weights, which must be included when interpreting the mixture.
    - An empirical model inherits the original observations' limitations, including unobserved values.
    - Resampling approximates the fitted empirical distribution; it does not repair the original sample's estimation error.
    - Seeds reproduce experiments, while separate seeds let us distinguish original-sample variability from resampling variability.
    
    ## Check your understanding
    1. In [0, 0, 2, 3], what are the empirical mass at 2 and ECDF at 2?
    2. An outcome is missing from a sample. Must it be impossible in the population?
    3. Must doubling sample size reduce every PMF discrepancy?
    4. Does a much larger resample remove uncertainty in the original empirical model?
    5. In a pooled sample with 200 uniform and 800 Poisson observations, what is the uniform weight?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({"Answers": mo.md("1. Mass = 1/4; ECDF = 3/4. 2. No. 3. No. 4. No. 5. 200/1000 = 0.20.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference
    Unpingco, J. (2019). *Python for Probability, Statistics, and Machine Learning*. Springer, Chapter 2.
    No external dataset is required.
    """)
    return


if __name__ == "__main__":
    app.run()
