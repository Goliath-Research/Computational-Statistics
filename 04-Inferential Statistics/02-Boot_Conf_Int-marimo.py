# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.25.1"

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as st
    from pathlib import Path

    sns.set_style("whitegrid")
    return Path, mo, np, pd, plt, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootstrap Confidence Intervals

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Explain why a bootstrap sample can repeat some observations and omit others.
    - Distinguish the original observations from bootstrap values of a statistic.
    - Interpret a percentile bootstrap confidence interval.
    - Calculate bootstrap intervals for center, spread, and shape using Python.
    - Explain why bootstrapping cannot correct an unrepresentative sample.
    - Apply the method to student grades and compare a bootstrap mean interval with a $t$ interval.

    ## 1. Resampling the sample you have
    Suppose we collect one sample and calculate its mean. A different sample would usually give a different mean, but collecting many new samples may be expensive or impractical.

    The **bootstrap** uses the observations we already have to approximate how a statistic varies from sample to sample.

    A **bootstrap sample** is drawn from the original observations **with replacement**. After selecting an observation, we leave it available to be selected again. This means some observations may appear more than once and others may not appear at all.

    Each bootstrap sample has the same number of observations as the original sample. We calculate the statistic from each resample. For example, 4,000 bootstrap samples give 4,000 means, called **bootstrap replicates** of the mean.

    ### From bootstrap replicates to an interval
    A **percentile bootstrap interval** uses two percentiles of those calculated statistics. For a 95% interval, the endpoints are the 2.5th and 97.5th percentiles of the bootstrap replicates. The interval therefore includes the middle 95% of their values.

    These percentiles come from the **bootstrap statistics**, not directly from the original observations. An interval for the mean estimates the population mean; it is not a range intended to contain 95% of individual observations.

    As with other confidence intervals, 95% confidence describes the method across repeated original samples. Bootstrap coverage is approximate, and the actual percentage can differ from 95%.

    ### What does the method assume?
    We do not have to choose a normal or uniform population model to perform this bootstrap. However, the ordinary method used here assumes independent observations from the same population and a sample that represents that population reasonably well.

    A very small sample may miss important population features. Resampling it cannot create those missing features. More resamples reduce randomness in the computed endpoints; they do not add new information about the population.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The observed sample is [3, 5, 8].
    1. How many observations are in each bootstrap sample?
    2. Can 5 appear more than once in one bootstrap sample?
    3. Can 4 appear in a bootstrap sample?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Size:** 3, the same size as the observed sample.
2. **Repeats:** Yes. Sampling with replacement can draw the same observation more than once.
3. **New values:** No. A bootstrap sample uses only values that were observed.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Simulated ages
    We will generate 1,000 example ages between 18 and 85 using a uniform distribution. This is a simple model for teaching, rather than a description of actual voter ages.

    The variable `ages` stores the observations. The seed `2026` reproduces the same sample whenever the cell is rerun. We display only the first 10 ages; calculations use all 1,000 unrounded values.

    The uniform distribution is used only to generate the example. The bootstrap calculations will resample the observed ages without using the population formula.

    In the histogram, each bar counts ages in a range. The two lines mark the sample mean and median.
    """)
    return


@app.cell
def _(np):
    ages = np.random.default_rng(2026).uniform(18, 85, size=1_000)
    print('First 10 ages:', ages[:10].round(0))
    return (ages,)


@app.cell
def _(ages, np, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.hist(ages, bins=15, color="#4C78A8", edgecolor="white")
    _ax.axvline(ages.mean(), color="#54A24B", linewidth=2, label=f"Mean {ages.mean():.1f}")
    _ax.axvline(np.median(ages), color="#F58518", linewidth=2, label=f"Median {np.median(ages):.1f}")
    _ax.set(title="Simulated ages", xlabel="Age", ylabel="Count")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculating an interval with SciPy
    `st.bootstrap` does the resampling and interval calculation for us. The important arguments are:

    - `data=(ages,)`: the sample in a one-item tuple. The comma is required.
    - `statistic=np.mean`: the statistic to calculate. Use `np.median` for a median.
    - `confidence_level=0.95`: the requested confidence level.
    - `method="percentile"`: the method taught in this lesson. SciPy otherwise defaults to a different method called BCa.
    - `n_resamples=4_000`: the number of bootstrap samples, not the number of observations in each sample.
    - `batch=100`: process up to 100 resamples at a time to limit memory use.
    - `rng=np.random.default_rng(2026)`: the seeded random generator.

    The returned result contains `confidence_interval.low` and `.high`, plus `bootstrap_distribution`, the values of the statistic from all resamples. Its `standard_error` summarizes the spread of those bootstrap statistics.

    The functions in this lesson accept `axis`, which lets SciPy calculate many resamples efficiently with `vectorized=True`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mean and median
    The **mean** is the arithmetic average. The **median** is the middle value after sorting the observations. Both describe center, but they respond differently to the data.

    The next calculation creates 4,000 bootstrap means and 4,000 bootstrap medians. The graph shows how these statistics vary across resamples. Its horizontal axis shows mean or median age, rather than individual ages.

    A wider bootstrap distribution indicates greater variability of that estimate. For this uniform population, the median generally varies more than the mean. This is not a rule for every population.

    The table reports 90%, 95%, and 99% percentile intervals for each statistic. For the same bootstrap distribution, higher confidence gives a wider interval.
    """)
    return


@app.cell
def _(ages, np, st):
    mean_bootstrap = st.bootstrap(
        (ages,), np.mean, confidence_level=0.95, method="percentile",
        n_resamples=4_000, batch=100, vectorized=True,
        rng=np.random.default_rng(2026)
    )
    median_bootstrap = st.bootstrap(
        (ages,), np.median, confidence_level=0.95, method="percentile",
        n_resamples=4_000, batch=100, vectorized=True,
        rng=np.random.default_rng(2027)
    )
    mean_replicates = mean_bootstrap.bootstrap_distribution
    median_replicates = median_bootstrap.bootstrap_distribution
    return mean_bootstrap, mean_replicates, median_bootstrap, median_replicates


@app.cell
def _(mean_replicates, median_replicates, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.hist(mean_replicates, bins=40, color="#F58518", alpha=0.55, label="Bootstrap means")
    _ax.hist(median_replicates, bins=40, color="#54A24B", alpha=0.45, label="Bootstrap medians")
    _ax.set(title="Bootstrap distributions", xlabel="Statistic", ylabel="Count")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(ages, mean_bootstrap, median_bootstrap, mo, np, pd, st):
    _rows = []
    for _name, _statistic, _previous in (
        ("Mean", np.mean, mean_bootstrap), ("Median", np.median, median_bootstrap)
    ):
        for _level in (0.90, 0.95, 0.99):
            # Reuse the existing replicates; no additional resamples are drawn.
            _result = st.bootstrap(
                (ages,), _statistic, confidence_level=_level, method="percentile",
                n_resamples=0, bootstrap_result=_previous,
                rng=np.random.default_rng(2028)
            )
            _rows.append({"Statistic": _name, "Confidence": f"{_level:.0%}",
                          "Low": _result.confidence_interval.low,
                          "High": _result.confidence_interval.high})
    center_intervals = pd.DataFrame(_rows)
    mo.Html(center_intervals.round(2).to_html(index=False, border=0, col_space=110))
    return (center_intervals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the mean intervals on a graph
    Each panel below shows the **same bootstrap means**. The red lines mark the endpoints for the confidence level shown above that panel.

    Compare the distance between the red lines: the 99% interval is wider than the 95% and 90% intervals. Higher confidence includes a larger part of the same bootstrap distribution.
    """)
    return


@app.cell
def _(center_intervals, mean_replicates, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 3))
    for _ax, _level in zip(_axes, ("90%", "95%", "99%")):
        _row = center_intervals.loc[
            (center_intervals["Statistic"] == "Mean") & (center_intervals["Confidence"] == _level)
        ].iloc[0]
        _ax.hist(mean_replicates, bins=30, color="#4C78A8", edgecolor="white")
        _ax.axvline(_row.Low, color="#E45756", linewidth=1.5)
        _ax.axvline(_row.High, color="#E45756", linewidth=1.5)
        _ax.set(title=f"{_level} mean interval", xlabel="Mean age (years)", ylabel="Count")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### New data: delivery times
    An online store records delivery times for a random sample of 80 orders. Times are measured in days. Deliveries can occasionally take much longer than usual, so we simulate positive data with a longer right tail.

    The next cell stores the times in `delivery_times`, using seed `2028`, and prints only the first 10. Assume orders are independent and the sample represents the store's deliveries.
    """)
    return


@app.cell
def _(np):
    delivery_times = np.random.default_rng(2028).lognormal(mean=1.2, sigma=0.5, size=80)
    print("First 10 delivery times (days):", delivery_times[:10].round(2))
    return (delivery_times,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate 95% percentile bootstrap confidence intervals for the population mean and median delivery time. What does each interval estimate?
    """)
    return


@app.cell
def _(delivery_times, np, st):
    student_delivery_mean_interval = None
    student_delivery_median_interval = None
    print("Mean interval (days):", student_delivery_mean_interval)
    print("Median interval (days):", student_delivery_median_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_delivery_mean_interval = st.bootstrap(
    (delivery_times,), np.mean, confidence_level=0.95,
    method="percentile", n_resamples=4_000, batch=100,
    rng=np.random.default_rng(2029)
).confidence_interval
student_delivery_median_interval = st.bootstrap(
    (delivery_times,), np.median, confidence_level=0.95,
    method="percentile", n_resamples=4_000, batch=100,
    rng=np.random.default_rng(2030)
).confidence_interval
print("Mean interval (days):", student_delivery_mean_interval)
print("Median interval (days):", student_delivery_median_interval)
```

The first interval estimates the mean time across the population of deliveries. The second estimates the population median: the time separating the shorter half of deliveries from the longer half. Neither interval is a range for the times of 95% of individual deliveries.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Spread and shape
    We now return to the **ages** sample to estimate other population properties:

    - **Variance**, $s^2$: spread measured in years squared. We use `ddof=1`, so the sample variance divides by $n-1$.
    - **Standard deviation**, $s$: spread measured in years. It is the square root of variance.
    - **Interquartile range (IQR)**: the 75th percentile minus the 25th percentile, measured in years.
    - **Skewness**: asymmetry. Positive values indicate a longer right tail, and negative values indicate a longer left tail.
    - **Excess kurtosis**: a measure related to tail weight. The normal distribution has population excess kurtosis 0; negative values generally indicate lighter tails than normal, and positive values heavier tails.

    For each statistic, SciPy resamples the ages, calculates the statistic, and finds the percentile interval. We use `bias=False` for the skewness and excess-kurtosis sample calculations, and `fisher=True` to request excess kurtosis.

    The short functions in the next cell tell SciPy which statistic to calculate and which options to use. `axis` tells the calculation which direction contains the observations when SciPy handles a batch of resamples.

    The table reports 90%, 95%, and 99% intervals. The graphs display the 95% intervals: the red lines mark the endpoints. These endpoints describe uncertainty about the population statistic, not the spread of individual ages.
    """)
    return


@app.cell
def _(ages, np, st):
    def sample_variance(sample, axis=-1):
        return np.var(sample, ddof=1, axis=axis)

    def sample_sd(sample, axis=-1):
        return np.std(sample, ddof=1, axis=axis)

    def age_skewness(sample, axis=-1):
        return st.skew(sample, bias=False, axis=axis)

    def age_excess_kurtosis(sample, axis=-1):
        return st.kurtosis(sample, fisher=True, bias=False, axis=axis)

    spread_statistics = {
        "Variance": sample_variance, "Standard deviation": sample_sd,
        "Interquartile range": st.iqr, "Skewness": age_skewness,
        "Excess kurtosis": age_excess_kurtosis,
    }
    spread_results = {}
    for _name, _statistic in spread_statistics.items():
        # A common seed uses the same resamples for comparisons between statistics.
        spread_results[_name] = st.bootstrap(
            (ages,), _statistic, confidence_level=0.95, method="percentile",
            n_resamples=4_000, batch=100, vectorized=True,
            rng=np.random.default_rng(2031)
        )
    return sample_sd, sample_variance, spread_results, spread_statistics


@app.cell
def _(ages, mo, np, pd, spread_results, spread_statistics, st):
    _rows = []
    for _name, _statistic in spread_statistics.items():
        for _level in (0.90, 0.95, 0.99):
            _result = st.bootstrap(
                (ages,), _statistic, confidence_level=_level, method="percentile",
                n_resamples=0, bootstrap_result=spread_results[_name],
                rng=np.random.default_rng(2032)
            )
            _rows.append({"Statistic": _name, "Confidence": f"{_level:.0%}",
                          "Low": _result.confidence_interval.low,
                          "High": _result.confidence_interval.high})
    spread_intervals = pd.DataFrame(_rows)
    mo.Html(spread_intervals.round(3).to_html(index=False, border=0, col_space=140))
    return (spread_intervals,)


@app.cell
def _(plt, spread_results):
    _fig, _axes = plt.subplots(2, 3, figsize=(9, 5.2))
    for _ax, (_name, _result) in zip(_axes.ravel(), spread_results.items()):
        _low, _high = _result.confidence_interval
        _ax.hist(_result.bootstrap_distribution, bins=30, color="#4C78A8", edgecolor="white")
        _ax.axvline(_low, color="#E45756", linewidth=1.5)
        _ax.axvline(_high, color="#E45756", linewidth=1.5)
        _ax.set_title(f"{_name}\n95% ({_low:.2f}, {_high:.2f})", fontsize=9)
    _axes.ravel()[-1].axis("off")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which two percentiles form a 90% percentile interval?
    2. Does a bootstrap interval for the median use a $t$ critical value?
    3. The age population in this simulation is uniform. Did the bootstrap calculation use that fact?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Percentiles:** The 5th and the 95th.
2. **Median:** No. The interval reads percentiles of the bootstrap medians.
3. **Uniform formula:** No. The calculation resamples the observed ages and takes percentiles of the resulting statistics. The uniform model was used only to create the sample.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Using `delivery_times`, calculate 95% percentile bootstrap confidence intervals for the population variance and standard deviation. Report the units of each interval.
    """)
    return


@app.cell
def _(sample_sd, sample_variance, delivery_times, np, st):
    student_delivery_variance_interval = None
    student_delivery_sd_interval = None
    print("Variance interval (days squared):", student_delivery_variance_interval)
    print("Standard deviation interval (days):", student_delivery_sd_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_delivery_variance_interval = st.bootstrap(
    (delivery_times,), sample_variance, confidence_level=0.95,
    method="percentile", n_resamples=4_000, batch=100,
    rng=np.random.default_rng(2033)
).confidence_interval
student_delivery_sd_interval = st.bootstrap(
    (delivery_times,), sample_sd, confidence_level=0.95,
    method="percentile", n_resamples=4_000, batch=100,
    rng=np.random.default_rng(2033)
).confidence_interval
print("Variance interval (days squared):", student_delivery_variance_interval)
print("Standard deviation interval (days):", student_delivery_sd_interval)
```

`sample_variance` and `sample_sd` calculate variance and standard deviation with `ddof=1`. SciPy applies each calculation to every resample and returns the confidence interval. Variance is measured in days squared; standard deviation is measured in days.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Grades in the student file
    The student performance data provide another application of bootstrap intervals. The file `student-mat.csv` must be available in the notebook folder or its `../data` folder. It uses semicolons to separate columns.

    Each row describes a student. The three grade columns are:

    - **G1:** first-period grade.
    - **G2:** second-period grade.
    - **G3:** final grade.

    All three grades are on a 0–20 scale. For each column, we calculate its sample mean and a 95% percentile bootstrap interval for the population mean. We also calculate the ordinary $t$ mean interval for comparison.

    Both intervals estimate the same population mean, but they use different methods and need not have identical endpoints. With a sufficiently large sample and suitable data, they may be close.

    The calculation treats students as independent observations. A confidence interval alone does not make this dataset representative of all students. The smoothed grade plots help compare distributions, but grades themselves are discrete values.
    """)
    return


@app.cell
def _(Path, mo, np, pd, st):
    _candidates = [Path("../data/student-mat.csv"), Path("student-mat.csv")]
    _path = next((_candidate for _candidate in _candidates if _candidate.is_file()), None)
    mo.stop(_path is None, mo.md("**Student data needed:** Place `student-mat.csv` in the notebook folder or its `../data` folder to run this section."))
    student_file = pd.read_csv(_path, sep=";")
    grades = student_file[["G1", "G2", "G3"]].copy()
    _rows = []
    for _name in ["G1", "G2", "G3"]:
        _values = grades[_name].to_numpy()
        _boot = st.bootstrap(
            (_values,), np.mean, confidence_level=0.95, method="percentile",
            n_resamples=4_000, batch=100, rng=np.random.default_rng(2034)
        ).confidence_interval
        _t = st.t.interval(0.95, len(_values) - 1, loc=_values.mean(), scale=st.sem(_values))
        _rows.append({
            "Grade": _name,
            "Mean": _values.mean(),
            "Bootstrap_low": _boot[0],
            "Bootstrap_high": _boot[1],
            "t_low": _t[0],
            "t_high": _t[1],
        })
    grade_intervals = pd.DataFrame(_rows)
    print(f"Loaded {len(grades)} records.")
    mo.Html(
        grade_intervals.round(3).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return grade_intervals, grades


@app.cell
def _(grades, plt, sns):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    for _name, _color in zip(["G1", "G2", "G3"], ["#E45756", "#4C78A8", "#54A24B"]):
        sns.kdeplot(grades[_name], ax=_ax, fill=True, alpha=0.25, color=_color, label=_name)
    _ax.set(title="Grade distributions", xlabel="Grade")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which grade has the lowest sample mean in the table?
    2. Are the bootstrap and $t$ intervals required to have the same endpoints?
    """)
    return


@app.cell(hide_code=True)
def _(grade_intervals, mo):
    _lowest = grade_intervals.loc[grade_intervals["Mean"].idxmin(), "Grade"]
    _answers = rf"""
1. **Lowest mean:** {_lowest}.
2. **Endpoints:** No. One interval uses bootstrap percentiles. The other uses a $t$ critical value and the standard error.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A bootstrap sample draws from the original observations **with replacement** and has the same sample size.
    - We recalculate the statistic for every resample. The resulting bootstrap statistics approximate its variation across samples.
    - A 95% percentile interval uses the 2.5th and 97.5th percentiles of the **bootstrap statistics**, not the original observations.
    - The method can estimate uncertainty for means, medians, and many measures of spread and shape without specifying a normal population model.
    - The ordinary bootstrap still depends on independent observations and a reasonably representative sample. It cannot recover population features missing from the sample.
    - SciPy's `bootstrap` calculates the intervals. We select the percentile method explicitly and seed every random calculation.
    - More resamples make the endpoints more stable. More original observations provide more information about the population; these are different improvements.

    ## Check your understanding
    1. A sample has 40 rows. How many rows does one bootstrap sample have?
    2. What does “with replacement” mean?
    3. Which percentiles form a 99% percentile bootstrap interval?
    4. Does a 95% bootstrap interval for the mean contain 95% of individual observations?
    5. Can increasing the number of resamples correct a sample that excludes an important part of the population?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Rows:** 40, the same number as in the original sample.
2. **Replacement:** After selecting a row, it remains available to be selected again. A resample may repeat rows and omit others.
3. **99% endpoints:** The 0.5th and 99.5th percentiles of the bootstrap statistics.
4. **Individual observations:** No. The interval estimates the population mean; it is not an interval for individual observations.
5. **Representativeness:** No. Resampling cannot supply information missing from the original sample.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer, Chapter 18.
    - [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), for the grade file.
    - Kim, A. *Comprehensive Confidence Intervals for Python Developers*. https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers
    """)
    return


if __name__ == "__main__":
    app.run()
