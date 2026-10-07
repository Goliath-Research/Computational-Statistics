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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Empirical Distributions of Continuous Measurements
    
    ## Learning goals
    - Construct and interpret a right-continuous empirical cumulative distribution function (ECDF).
    - Distinguish a histogram, a kernel density estimate, and an ECDF.
    - Calculate empirical cumulative and survival probabilities with the correct endpoint conventions.
    - Resample observations efficiently and explain why resampling does not generate new population information.
    - Compare a known mixture, its empirical estimate, and a resampled estimate.
    
    ## 1. Continuous measurements, discrete empirical distribution
    A population model may be continuous. Nevertheless, the empirical distribution of a finite sample places mass on its observed values and is therefore **discrete**.
    Each observation has weight 1/n; repeated values combine those weights.
    The ECDF is:
    
    $$F_n(t)=\frac{1}{n}\sum_{i=1}^n\mathbf{1}(x_i\le t).$$
    
    The indicator is 1 if the observation is at or below t, otherwise 0.
    The **empirical survival function (SF)** gives the proportion strictly above t:

    $$\widehat{SF}_n(t)=1-F_n(t)=\frac{1}{n}\sum_{i=1}^n\mathbf{1}(x_i>t).$$

    Because the empirical distribution is discrete, “strictly above” and “at or above” can give different results at a recorded value.

    The ECDF works for any sample; the data do not need to fail a named model first.
    It is defined for **every real threshold**, not just between sample extremes.
    It equals zero below the minimum and one at and above the maximum. Those facts describe the empirical distribution, not the unknown population tails.
    
    Unlike a continuous population model, the empirical distribution has positive exact-point masses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Combining two normal samples

    We will generate two groups of measurements:

    - **Group 1:** 400 observations with mean **30** and standard deviation **5**.
    - **Group 2:** 700 observations with mean **50** and standard deviation **5**.

    Then we will combine them into one sample of **1,100 observations**.

    Because the groups have different means, we expect two peaks: one near **30** and another near **50**. The second group contains more observations, so it contributes more to the combined distribution.

    We will use this combined sample to build an **empirical cumulative distribution function (ECDF)** and calculate probabilities.

    All observations are simulated; no external dataset is needed.
    """)
    return


@app.cell
def _(mo):
    # Seed: Controls the random sample; the same seed reproduces the same observations.
    original_seed = mo.ui.slider(start=1, stop=100, step=1, value=42, show_value=True, label="Original data seed")
    # Bins: Controls the histogram's bin count; the larger the bins, the smoother the histogram.
    bin_control = mo.ui.slider(start=10, stop=60, step=5, value=30, show_value=True, label="Histogram bins")
    # Bandwidth: Controls the KDE's smoothing; the larger the bandwidth, the smoother the KDE. Smaller values show more details.
    bandwidth_control = mo.ui.slider(start=0.5, stop=2, step=0.25, value=1, show_value=True, label="KDE bandwidth multiplier")
    mo.vstack([original_seed, bin_control, bandwidth_control])
    return bandwidth_control, bin_control, original_seed


@app.cell
def _(np):
    class ECDF:
        """Right-continuous empirical CDF, evaluated by binary search."""
        def __init__(self, values):
            """Sort the observations for a right-continuous empirical CDF."""
            values = np.asarray(values, dtype=float)
            self.sorted_values = np.sort(values)

        def __call__(self, thresholds):
            """Return the proportion at or below each scalar or array threshold."""
            return np.searchsorted(self.sorted_values, thresholds, side="right") / len(self.sorted_values)
    return (ECDF,)


@app.cell
def _(ECDF, norm, np, original_seed):
    sample1 = norm.rvs(loc=30, scale=5, size=400, random_state=np.random.default_rng(original_seed.value))
    sample2 = norm.rvs(loc=50, scale=5, size=700, random_state=np.random.default_rng(original_seed.value + 1000))
    sample = np.concatenate([sample1, sample2])
    original_ecdf = ECDF(sample)
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

    A **histogram** groups observations into intervals called **bins**. Here, the bars are scaled so their total area equals one. Each bar’s **area** represents the proportion of observations in its interval.

    A **kernel density estimate (KDE)** draws a smooth curve that helps us see the shape of the data. It is an estimate of the population density.

    The **bandwidth** controls how smooth the curve is:

    - **Smaller bandwidth:** More detail, with more bumps.
    - **Larger bandwidth:** A smoother curve that may merge nearby peaks.

    Changing the bins or bandwidth changes the plot, but leaves the observations and their **ECDF** unchanged.

    Probabilities correspond to **areas**, not heights. The KDE may also extend beyond the smallest and largest observations because of smoothing.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This plot compares the observed data’s **histogram** and **smooth KDE estimate** with the **theoretical density of the two combined normal distributions**.
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

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This plot compares the **observed proportion at or below each threshold (ECDF)** with the **theoretical probability from the combined normal distributions (CDF)**.
    """)
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
    The ECDF equals zero below the smallest observation and one at or above the largest.

    Our code calculates it from the sorted observations. The plot includes these zero and one levels so students can see how the ECDF behaves outside the observed range.
        
    ### Try it yourself
    1. Change histogram bins and KDE bandwidth separately. Which plot changes, and which remains the same?
    2. Compare the KDE with the known mixture PDF. Is the KDE exactly the generating density?
    3. Change the original-data seed. Does the theoretical mixture change?
    
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Bins and bandwidth:** Bins change the histogram; bandwidth changes the KDE. Neither changes the observations, ECDF, or known mixture curves.
2. **KDE versus generating density:** No. The KDE is a smoothed estimate from the finite sample and need not equal the known mixture PDF.
3. **Original-data seed:** The data and their estimates can change; the component parameters and mixture weights remain fixed.

**Discussion:** The seed changes the observed data and empirical estimates, not the known mixture. Smooth density estimates and exact empirical cumulative proportions describe different objects.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Probabilities from counts and the ECDF
    Use endpoint conventions precisely:
    - Fₙ(t) gives the proportion **at or below** t.
    - The empirical SF, 1 − Fₙ(t), gives the proportion **strictly above** t.
    - Fₙ(b) − Fₙ(a) gives the proportion in **(a,b]**.
    - For strictly below t, count **sample < t** directly.
    At thresholds equal to recorded observations, these distinctions matter.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    This table shows the **observed proportions below, at, and above the selected threshold**, plus the proportion **greater than 30 and at or below 50**, checked by direct counting.
    """)
    return

@app.cell
def _(mo):
    query_control = mo.ui.slider(start=10, stop=70, step=1, value=40, show_value=True, label="Threshold t")
    query_control
    return (query_control,)


@app.cell
def _(np, original_ecdf, pd, query_control, sample, mo):
    query_threshold = query_control.value
    probability_table = pd.DataFrame({"Event": ["X < t", "X ≤ t", "X = t", "X > t", "30 < X ≤ 50"],
        "Empirical proportion": [np.mean(sample < query_threshold), original_ecdf(query_threshold), np.mean(sample == query_threshold), 1 - original_ecdf(query_threshold), original_ecdf(50) - original_ecdf(30)]})
    print(f"Threshold t = {query_threshold}")
    print("Interval direct-count check:", np.mean((sample > 30) & (sample <= 50)))
    mo.Html(
        probability_table.round(4).to_html(border=0, col_space=110, index=False)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(np, original_ecdf, sample):
    recorded_threshold = float(sample[0])
    print(f"Using a recorded observation t = {recorded_threshold:.4f}")
    print("Strictly below:", np.mean(sample < recorded_threshold))
    print("At or below:", original_ecdf(recorded_threshold).round(4))
    print("Exact-point empirical mass:", np.mean(sample == recorded_threshold).round(4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Calculate the empirical proportion in (30,50] using direct counts, then verify it using the ECDF difference.
    2. Calculate the empirical probability strictly above 50 using the survival function.
    3. At a recorded observation, explain the difference between < and ≤.
    4. If the ECDF at 10 is zero, can the normal mixture population still produce values below 10?

    In the next cell, replace **None** with your calculations. Use **np.mean()** with the appropriate comparisons for direct counts, and **original_ecdf()** for cumulative probabilities. The empirical SF is **1 − original_ecdf(t)**.
    Run your cell, then expand **Show answers** to compare with the explanations, complete solution code, and numerical answers.
    The numerical answers can change with the original-data seed.
    
    """)
    return


@app.cell
def _():
    # Proportion in (30,50] from direct counts
    interval_count_answer = None
    # The same proportion from the ECDF
    interval_ecdf_answer = None
    # Probability strictly above 50
    survival_answer = None
    print("Interval, direct counts:", interval_count_answer)
    print("Interval, ECDF:", interval_ecdf_answer)
    print("Strictly above 50:", survival_answer)
    return


@app.cell(hide_code=True)
def _(mo, np, original_ecdf, sample):
    _answers = f"""
1. **Interval (30,50]:** The direct-count proportion is {np.mean((sample > 30) & (sample <= 50)):.4f}; the ECDF difference is {original_ecdf(50) - original_ecdf(30):.4f}. Both exclude 30 and include 50.
2. **Strictly above 50:** The empirical SF is 1 − Fₙ(50) = {1 - original_ecdf(50):.4f}.
3. **Recorded threshold:** Fₙ(t) includes observations equal to t; the proportion strictly below t does not. Their difference is the empirical mass at t.
4. **Population tail:** Yes. Both normal components assign positive probability below 10, even when none of the recorded observations are there.

```python
interval_count_answer = np.mean((sample > 30) & (sample <= 50))
interval_ecdf_answer = original_ecdf(50) - original_ecdf(30)
survival_answer = 1 - original_ecdf(50)
```

**Discussion:** No observed values below a threshold does not establish a zero population tail probability.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Efficient resampling from the empirical distribution
    Selecting original observations uniformly **with replacement** samples from the empirical distribution.
    It can repeat recorded values but cannot create new measurement values or extrapolate beyond the observed extremes.
    This operation is used in the ordinary nonparametric bootstrap. Resampling does not create independent population information, and a huge resample does not remove the original estimation error.
    
    For day-to-day code, **generator.choice(sample, replace=True)** is simple and efficient.
    We also demonstrate inverse-transform sampling using the sorted observations and their finite cumulative probabilities.
    **np.searchsorted** finds indices without scanning the entire ECDF for each draw. Using only finite observations avoids selecting −∞ when u = 0.
    """)
    return


@app.cell
def _(np):
    def inverse_empirical_sample(values, uniforms):
        """Map uniform inputs in [0,1] to sorted observations via the empirical CDF.

        Use the first cumulative probability at or above each input.
        Assign input 0 to the sample minimum and input 1 to the maximum.
        """
        values = np.asarray(values, dtype=float)
        uniforms = np.asarray(uniforms, dtype=float)
        sorted_values = np.sort(values)
        probabilities = np.arange(1, len(values) + 1) / len(values)
        indices = np.searchsorted(probabilities, uniforms, side="left")
        return sorted_values[indices]
    return (inverse_empirical_sample,)


@app.cell
def _(inverse_empirical_sample, np, pd, sample, mo):
    demonstration_u = np.array([0, 0.1, 0.5, 0.9, 1])
    mo.Html(
        pd.DataFrame({"Uniform input u": demonstration_u, "Inverse empirical value": inverse_empirical_sample(sample, demonstration_u)})
        .round(4).to_html(border=0, col_space=110, index=False)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The u = 0 boundary is assigned the sample minimum; u = 1 returns the maximum. These endpoint conventions have no effect on the ideal continuous-uniform sampling distribution.
    Sorting plus binary searches avoids the original per-draw full-array scan. Direct **choice** avoids sorting altogether.
    The two methods produce the same sampling distribution, not necessarily identical realized arrays with the same seed.
    """)
    return


@app.cell
def _(mo):
    resample_seed = mo.ui.slider(start=1, stop=100, step=1, value=73, show_value=True, label="Resampling seed")
    resample_size = mo.ui.slider(steps=[10, 100, 1000, 10000, 100000], value=1000, show_value=True, label="Resample size displayed")
    mo.hstack([resample_seed, resample_size])
    return resample_seed, resample_size


@app.cell
def _(np, resample_seed, sample):
    resample_sequence = np.random.default_rng(resample_seed.value).choice(sample, size=100000, replace=True)
    return (resample_sequence,)


@app.cell
def _(ECDF, np, resample_sequence, resample_size):
    new_sample = resample_sequence[:resample_size.value]
    new_ecdf = ECDF(new_sample)
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
    
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Larger prefixes:** No. Larger resamples tend to approximate the original empirical distribution more closely, but finite-sample agreement need not improve at every increase.
2. **Seeds:** Changing only the resampling seed can change the resample while leaving the original empirical model fixed. Changing the original-data seed can change that model and therefore the resample, even with a fixed resampling seed.
3. **Beyond the original maximum:** No. Sampling with replacement selects only recorded values.
4. **Population information:** No. A larger resample reuses the original information; it does not supply newly collected population observations.

**Discussion:** Larger resamples generally approximate the fixed empirical distribution more closely, but finite-sample agreement need not improve monotonically. Resampling inherits the original sample's limitations. It cannot generate an unobserved value.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - An ECDF is a step function defined for all real thresholds and built directly from observed counts.
    - Even for continuous measurements, the finite empirical distribution is discrete.
    - KDE is a smoothed estimate of density; it is not the ECDF or a known true PDF.
    - Use correct endpoint conventions when calculating empirical probabilities; the empirical SF is 1 − Fₙ(t) and counts values strictly above t.
    - Mixture weights depend on the component sample proportions. This lesson's two normal components produce a bimodal mixture because their means, 30 and 50, are well separated relative to their common standard deviation of 5.
    - A mixture need not be bimodal: if component means are close relative to their spreads, their peaks can merge into one.
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


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Beyond the maximum:** Yes. The ECDF is defined for every real threshold and equals one at and above the sample maximum.
2. **Equality:** Yes. Fₙ(t) counts observations equal to t as well as those below it.
3. **Unobserved value:** No. Ordinary empirical resampling selects recorded observations.
4. **Bandwidth:** No. It changes the KDE, not the ECDF.
5. **Interval:** Fₙ(b) − Fₙ(a) describes (a,b], excluding a and including b.
6. **Mixture weights:** No. They are 400/1100 and 700/1100.
""")})
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
