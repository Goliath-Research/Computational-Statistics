import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import norm, gaussian_kde
    return mo, np, pd, plt, norm , gaussian_kde


@app.cell
def _(np):
    class ECDF:
        """Right-continuous empirical CDF, evaluated by binary search."""
        def __init__(self, values, side="right"):
            values = np.asarray(values, dtype=float)
            if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
                raise ValueError("Provide a nonempty finite one-dimensional sample.")
            if side != "right":
                raise ValueError("This lesson uses right-continuous ECDFs.")
            self.sorted_values = np.sort(values)

        def __call__(self, thresholds):
            return np.searchsorted(self.sorted_values, thresholds, side="right") / len(self.sorted_values)
    return (ECDF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Empirical Distributions of Continuous Measurements
    
    ## Learning goals
    - Construct and interpret a right-continuous empirical cumulative distribution function (ECDF).
    - Distinguish a histogram, a kernel density estimate, and an ECDF.
    - Calculate empirical probabilities with the correct endpoint conventions.
    - Resample observations efficiently and explain why resampling does not generate new population information.
    - Compare a known mixture, its empirical estimate, and a resampled estimate.
    
    ## 1. Continuous measurements, discrete empirical distribution
    A population model may be continuous. Nevertheless, the empirical distribution of a finite sample places mass on its observed values and is therefore **discrete**.
    Each observation has weight 1/n; repeated values combine those weights.
    The ECDF is:
    
    $$F_n(t)=\frac{1}{n}\sum_{i=1}^n\mathbf{1}(x_i\le t).$$
    
    The indicator is 1 if the observation is at or below t, otherwise 0.
    The ECDF works for any sample; the data do not need to fail a named model first.
    It is defined for **every real threshold**, not just between sample extremes.
    It equals zero below the minimum and one at and above the maximum. Those facts describe the empirical distribution, not the unknown population tails.
    
    For [1.2, 1.2, 2.8, 4.1], Fₙ(1.2) = 2/4, while the proportion strictly below 1.2 is zero.
    Unlike a continuous population model, the empirical distribution has positive exact-point masses.
    
    ### Try it yourself
    For this small sample, calculate Fₙ(2), the mass at 1.2, and the proportion in (1.2, 4.1].
    """)
    return


@app.cell
def _(mo):
    mo.accordion({"Answers": mo.md("Fₙ(2) = 0.50. Mass at 1.2 = 0.50. Proportion in (1.2, 4.1] = 0.50. The left endpoint is excluded and the right endpoint included.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A two-component generating mixture
    Generate 400 normal observations with mean 30 and standard deviation 5, and 700 with mean 50 and standard deviation 5.
    Use conventional notation N(30,25) and N(50,25): the second parameter is variance. SciPy's scale is standard deviation.
    The pooled target mixture has weights 400/1100 and 700/1100, not equal weights.
    These separated components give this example a two-peaked density. Combining two normal distributions does not always produce two visible peaks.
    We fix component counts; independently choosing a random component per observation would be a different sampling design with the same target marginal mixture at these weights.
    All data here are simulated; no external files are required.
    """)
    return


@app.cell
def _(mo):
    original_seed = mo.ui.slider(start=1, stop=100, step=1, value=42, label="Original data seed")
    bin_control = mo.ui.slider(start=10, stop=60, step=5, value=30, label="Histogram bins")
    bandwidth_control = mo.ui.slider(start=0.5, stop=2, step=0.25, value=1, label="KDE bandwidth multiplier")
    mo.vstack([original_seed, bin_control, bandwidth_control])
    return bandwidth_control, bin_control, original_seed


@app.cell
def _(ECDF, norm, np, original_seed):
    sample1 = norm.rvs(loc=30, scale=5, size=400, random_state=np.random.default_rng(original_seed.value))
    sample2 = norm.rvs(loc=50, scale=5, size=700, random_state=np.random.default_rng(original_seed.value + 1000))
    sample = np.concatenate([sample1, sample2])
    original_ecdf = ECDF(sample, side="right")
    mixture_weight = len(sample1) / len(sample)
    plot_grid = np.linspace(min(sample.min(), 10), max(sample.max(), 70), 800)
    mixture_pdf = mixture_weight * norm.pdf(plot_grid, loc=30, scale=5) + (1 - mixture_weight) * norm.pdf(plot_grid, loc=50, scale=5)
    mixture_cdf = mixture_weight * norm.cdf(plot_grid, loc=30, scale=5) + (1 - mixture_weight) * norm.cdf(plot_grid, loc=50, scale=5)
    print(f"Observations: {len(sample)}; weights: {mixture_weight:.4f}, {1 - mixture_weight:.4f}")
    print(f"Observed range: [{sample.min():.3f}, {sample.max():.3f}]")
    print("ECDF domain: all real thresholds. Both normal components have unbounded population support.")
    return mixture_cdf, mixture_pdf, mixture_weight, original_ecdf, plot_grid, sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Histogram and estimated density
    A density-normalized histogram has total area one; a bar's area is its observed proportion.
    A **kernel density estimate (KDE)** smooths the observations to estimate a population density. It is not the empirical distribution itself or a known true PDF.
    Bandwidth controls smoothing: small bandwidths reveal more local variation; large bandwidths can hide peaks.
    Neither histogram binning nor KDE bandwidth changes the ECDF.
    Density heights are not probabilities. A KDE can place density beyond the observed range; those tails come from smoothing, not newly observed information.
    """)
    return


@app.cell
def _(bandwidth_control, bin_control, gaussian_kde, mixture_pdf, np, plot_grid, plt, sample):
    original_kde = gaussian_kde(sample)
    original_kde.set_bandwidth(original_kde.scotts_factor() * bandwidth_control.value)
    _figure, _axes = plt.subplots(figsize=(9, 4))
    _axes.hist(sample, bins=np.linspace(plot_grid[0], plot_grid[-1], bin_control.value + 1), density=True, alpha=0.35, label="Original histogram")
    _axes.plot(plot_grid, original_kde(plot_grid), label="KDE estimate")
    _axes.plot(plot_grid, mixture_pdf, "--", label="Known mixture PDF")
    _axes.set(xlabel="Measurement", ylabel="Density", title="Density descriptions: observed and theoretical")
    _axes.legend()
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(mixture_cdf, np, original_ecdf, plot_grid, plt, sample):
    _unique, _counts = np.unique(sample, return_counts=True)
    _cumulative = np.cumsum(_counts) / len(sample)
    _step_x = np.r_[plot_grid[0], _unique, plot_grid[-1]]
    _step_y = np.r_[0, _cumulative, 1]
    _figure, _axes = plt.subplots(figsize=(9, 4))
    _axes.step(_step_x, _step_y, where="post", label="Original ECDF")
    _axes.plot(plot_grid, mixture_cdf, "--", label="Known mixture CDF")
    _axes.set(xlabel="Threshold t", ylabel="Proportion / probability at or below t", ylim=(-0.03, 1.03), title="ECDF versus generating CDF")
    _axes.legend()
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The original statsmodels implementation stores an extra x value of −∞ to represent the initial zero level. It is a plotting sentinel, not an observation or the lower bound of the ECDF's domain.
    This revision calculates the ECDF directly with sorted observations and binary searches. It needs no statsmodels dependency or sentinel. Our step plots use finite observations and explicit zero/one padding.
    
    ### Try it yourself
    1. Change histogram bins and KDE bandwidth separately. Which plot changes, and which remains the same?
    2. Compare the KDE with the known mixture PDF. Is the KDE exactly the generating density?
    3. Change the original-data seed. Does the theoretical mixture change?
    
    **Discussion:** The seed changes the observed data and empirical estimates, not the known mixture. Smooth density estimates and exact empirical cumulative proportions describe different objects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Probabilities from counts and the ECDF
    Use endpoint conventions precisely:
    - Fₙ(t) gives the proportion **at or below** t.
    - 1 − Fₙ(t) gives the proportion **strictly above** t.
    - Fₙ(b) − Fₙ(a) gives the proportion in **(a,b]**.
    - For strictly below t, count `sample < t` directly.
    At thresholds equal to recorded observations, these distinctions matter.
    """)
    return


@app.cell
def _(mo):
    query_control = mo.ui.slider(start=10, stop=70, step=1, value=40, label="Threshold t")
    query_control
    return (query_control,)


@app.cell
def _(np, original_ecdf, pd, query_control, sample):
    query_threshold = query_control.value
    probability_table = pd.DataFrame({"Event": ["X < t", "X ≤ t", "X = t", "X > t", "30 < X ≤ 50"],
        "Empirical proportion": [np.mean(sample < query_threshold), original_ecdf(query_threshold), np.mean(sample == query_threshold), 1 - original_ecdf(query_threshold), original_ecdf(50) - original_ecdf(30)]})
    print(f"Threshold t = {query_threshold}")
    print("Interval direct-count check:", np.mean((sample > 30) & (sample <= 50)))
    probability_table.round(4)
    return


@app.cell
def _(np, original_ecdf, sample):
    recorded_threshold = float(sample[0])
    print(f"Using a recorded observation t = {recorded_threshold:.6f}")
    print("Strictly below:", np.mean(sample < recorded_threshold))
    print("At or below:", original_ecdf(recorded_threshold))
    print("Exact-point empirical mass:", np.mean(sample == recorded_threshold))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Verify the interval result using direct counts.
    2. At a recorded observation, explain the difference between < and ≤.
    3. If the ECDF at 10 is zero, can the normal mixture population still produce values below 10?
    
    **Discussion:** Yes. No observed values below a threshold does not establish a zero population tail probability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Efficient resampling from the empirical distribution
    Selecting original observations uniformly **with replacement** samples from the empirical distribution.
    It can repeat recorded values but cannot create new measurement values or extrapolate beyond the observed extremes.
    This operation is used in the ordinary nonparametric bootstrap. Resampling does not create independent population information, and a huge resample does not remove the original estimation error.
    
    For day-to-day code, `generator.choice(sample, replace=True)` is simple and efficient.
    We also demonstrate inverse-transform sampling using the sorted observations and their finite cumulative probabilities.
    `np.searchsorted` finds indices without scanning the entire ECDF for each draw. Using only finite observations avoids selecting −∞ when u = 0.
    """)
    return


@app.cell
def _(np):
    def inverse_empirical_sample(values, uniforms):
        values = np.asarray(values, dtype=float)
        uniforms = np.asarray(uniforms, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Provide a nonempty one-dimensional finite sample.")
        if uniforms.ndim != 1 or not np.isfinite(uniforms).all() or np.any((uniforms < 0) | (uniforms > 1)):
            raise ValueError("Uniform inputs must lie in [0,1].")
        sorted_values = np.sort(values)
        probabilities = np.arange(1, len(values) + 1) / len(values)
        indices = np.searchsorted(probabilities, uniforms, side="left")
        return sorted_values[indices]
    return (inverse_empirical_sample,)


@app.cell
def _(inverse_empirical_sample, np, pd, sample):
    demonstration_u = np.array([0, 0.1, 0.5, 0.9, 1])
    pd.DataFrame({"Uniform input u": demonstration_u, "Inverse empirical value": inverse_empirical_sample(sample, demonstration_u)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The u = 0 boundary is assigned the sample minimum; u = 1 returns the maximum. These endpoint conventions have no effect on the ideal continuous-uniform sampling distribution.
    Sorting plus binary searches avoids the original per-draw full-array scan. Direct `choice` avoids sorting altogether.
    The two methods produce the same sampling distribution, not necessarily identical realized arrays with the same seed.
    """)
    return


@app.cell
def _(mo):
    resample_seed = mo.ui.slider(start=1, stop=100, step=1, value=73, label="Resampling seed")
    resample_size = mo.ui.slider(steps=[10, 100, 1000, 10000, 100000], value=1000, label="Resample size displayed")
    mo.hstack([resample_seed, resample_size])
    return resample_seed, resample_size


@app.cell
def _(np, resample_seed, sample):
    resample_sequence = np.random.default_rng(resample_seed.value).choice(sample, size=100000, replace=True)
    return (resample_sequence,)


@app.cell
def _(ECDF, np, resample_sequence, resample_size):
    new_sample = resample_sequence[:resample_size.value]
    new_ecdf = ECDF(new_sample, side="right")
    print("New sample size:", len(new_sample))
    print("Distinct values in resample:", len(np.unique(new_sample)))
    return new_ecdf, new_sample


@app.cell
def _(bin_control, new_sample, np, plot_grid, plt, sample):
    shared_edges = np.linspace(plot_grid[0], plot_grid[-1], bin_control.value + 1)
    _figure, _axes = plt.subplots(figsize=(9, 4))
    _axes.hist(sample, bins=shared_edges, density=True, alpha=0.4, label="Original")
    _axes.hist(new_sample, bins=shared_edges, density=True, histtype="step", linewidth=1.5, label="Resample")
    _axes.set(xlabel="Measurement", ylabel="Density", title="Shared bins and density normalization")
    _axes.legend()
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(mixture_cdf, new_ecdf, new_sample, np, original_ecdf, plot_grid, plt, sample):
    _figure, _axes = plt.subplots(figsize=(9, 4))
    for _values, _label in [(sample, "Original ECDF"), (new_sample, "Resample ECDF")]:
        _unique_values, _value_counts = np.unique(_values, return_counts=True)
        _axes.step(np.r_[plot_grid[0], _unique_values, plot_grid[-1]], np.r_[0, np.cumsum(_value_counts) / len(_values), 1], where="post", label=_label)
    _axes.plot(plot_grid, mixture_cdf, "--", label="Known mixture CDF")
    _axes.set(xlabel="Threshold t", ylabel="Cumulative probability / proportion", ylim=(-0.03, 1.03), title="Generating model, original estimate, and resample")
    _axes.legend()
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(new_ecdf, np, original_ecdf, sample):
    comparison_thresholds = np.unique(sample)
    print("Maximum ECDF difference, resample versus original:", np.max(np.abs(new_ecdf(comparison_thresholds) - original_ecdf(comparison_thresholds))))
    print("This is a descriptive discrepancy, not an independence-based two-sample test.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Compare 10, 1,000, and 100,000 resampled observations. Must every larger prefix be closer to the original ECDF?
    2. Change only the resampling seed. Then change only the original-data seed. Explain the different effects.
    3. Can a resample contain a value greater than the original maximum?
    4. Does a resample of 100,000 have the same information as 100,000 newly collected observations?
    
    **Discussion:** Larger resamples generally approximate the fixed empirical distribution more closely, but finite-sample agreement need not improve monotonically. Resampling inherits the original sample's limitations. It cannot generate an unobserved value.
    
    ## Conclusions
    - An ECDF is a step function defined for all real thresholds and built directly from observed counts.
    - Even for continuous measurements, the finite empirical distribution is discrete.
    - KDE is a smoothed estimate of density; it is not the ECDF or a known true PDF.
    - Use correct endpoint conventions when calculating empirical probabilities.
    - Mixture weights depend on the component sample proportions, and mixtures need not always be bimodal.
    - Resampling with replacement targets the original empirical distribution, not an independently known population distribution.
    - Seeds reproduce experiments; separate original-data and resampling seeds expose two sources of variability.
    
    ## Check your understanding
    1. Is the ECDF defined beyond the sample maximum?
    2. Does Fₙ(t) include observations equal to t?
    3. Can an ordinary empirical resample produce a previously unobserved value?
    4. Does increasing KDE bandwidth change the ECDF?
    5. Does Fₙ(b) − Fₙ(a) describe [a,b] or (a,b]?
    6. Are the two normal-component weights in this lesson equal?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({"Answers": mo.md("1. Yes; it equals 1 there. 2. Yes. 3. No. 4. No. 5. (a,b]. 6. No: 400/1100 and 700/1100.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference
    Unpingco, J. (2019). *Python for Probability, Statistics, and Machine Learning*. Springer, Chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
