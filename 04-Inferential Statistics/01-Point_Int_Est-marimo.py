# /// script
# dependencies = [
#     "marimo",
#     "seaborn",
#     "scipy",
# ]
# ///

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as st

    sns.set_style("whitegrid")
    return mo, np, pd, plt, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point and Interval Estimation

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Distinguish a point estimate from a confidence interval.
    - Separate a known population variance from the two divisors used on a sample.
    - Interpret a confidence level as a long-run coverage rate.
    - Build a \(t\) interval for a mean and for a paired mean difference.
    - Build a chi-square interval for the variance of a normal population.
    - Explain why one computed interval either contains the parameter or does not.

    ## 1. Two kinds of estimate
    A **point estimate** is one number computed from a sample, such as a mean or a proportion.
    An **interval estimate** is a range built around that number.
    The range is an attempt to show how far the point estimate might sit from the population parameter.

    The sample in the next section is drawn from a normal population with mean 45 and standard deviation 8.
    We know those population values because this lesson chooses them for the simulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A sample of ages is [40, 42, 45, 48, 55].
    1. What is the point estimate of the mean?
    2. Is that number guaranteed to equal the population mean?
    3. What would an interval estimate add?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Point estimate:** (40 + 42 + 45 + 48 + 55) / 5 = 46.
2. **Equality:** No. A sample mean estimates the population mean. It does not have to equal it.
3. **Interval:** A range that, at a stated confidence level, covers the population mean in a known fraction of repeated samples.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A point estimate of mean age
    The next cell draws 1,000 ages from a normal population whose mean we set to 45 and whose standard deviation we set to 8.
    We know that the population mean is 45 only because this is a simulation.
    The sample mean estimates that value. It does not have to equal 45.
    We use a seed of 2026 to ensure reproducibility.
    """)
    return


@app.cell
def _(np):
    ages = np.random.default_rng(2026).normal(45, 8, size=1_000)
    print('First 10 ages:', ages[:10].round(0))
    return (ages,)


@app.cell
def _(ages, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.hist(ages, bins=20, color="#4C78A8", edgecolor="white")
    _ax.axvline(ages.mean(), color="#E45756", linewidth=2, label=f"Sample mean {ages.mean():.2f}")
    _ax.set(title="Simulated voter ages", xlabel="Age", ylabel="Count")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(ages, mo, pd):
    age_summary = pd.DataFrame({
        "Quantity": [
            "Sample size",
            "Sample mean",
            "Variance with divisor n",
            "Variance with divisor n - 1",
            "Standard deviation with divisor n",
            "Standard deviation with divisor n - 1",
        ],
        "Value": [
            len(ages),
            ages.mean(),
            ages.var(ddof=0),
            ages.var(ddof=1),
            ages.std(ddof=0),
            ages.std(ddof=1),
        ],
    })
    mo.Html(
        age_summary.round(2).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (age_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The known population variance is \(8^2 = 64\).
    Neither sample variance is that number.
    Divisor \(n\) describes the sample.
    Divisor \(n-1\) is the usual unbiased estimator of a normal population variance.
    `scipy.stats.sem` uses divisor \(n-1\).

    The original notebook labeled `a.var()` as the population variance and `st.sem(a)` as the sampling error.
    The first is a sample calculation. The second is the **standard error of the mean**, an estimate of the typical distance between the sample mean and the population mean.
    The sampling error of this sample is \(\bar x - 45\), and it is known here only because the simulation set the population mean.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a code task, replace each `None` in the next cell and run it in the marimo editor.

    ### Try it yourself
    Using `ages`:
    1. Store the sample mean in `age_mean`.
    2. Store the standard deviation with divisor \(n-1\) in `age_sd`.
    3. Store the proportion of ages below 50 in `proportion_under_50`.

    The original notes said "less than 55" while the code used 50. This lesson uses 50.
    """)
    return


@app.cell
def _():
    age_mean = None
    age_sd = None
    proportion_under_50 = None
    print("Mean:", age_mean)
    print("SD (n-1):", age_sd)
    print("Proportion under 50:", proportion_under_50)
    return age_mean, age_sd, proportion_under_50


@app.cell(hide_code=True)
def _(ages, mo):
    _mean = ages.mean()
    _sd = ages.std(ddof=1)
    _prop = float((ages < 50).mean())
    _answers = f"""
1. **Mean:** {_mean:.3f}.
2. **Standard deviation:** {_sd:.3f}, using `ddof=1`.
3. **Proportion:** {_prop:.3f}. For a normal population with mean 45 and standard deviation 8, the probability below 50 is about 0.73. The sample proportion estimates that probability.

```python
age_mean = ages.mean()
age_sd = ages.std(ddof=1)
proportion_under_50 = (ages < 50).mean()
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Confidence interval for a mean
    A **confidence level** is the proportion of intervals that would contain the parameter if the whole sampling procedure were repeated.
    A 95% procedure is built so that about 95% of those intervals contain the parameter.
    After one interval is computed, the parameter is either inside it or outside it. The 95% is not a probability attached to that one interval.

    For a normal sample with unknown population standard deviation, the interval for the mean is

    $$\bar x \pm t_{n-1,\,1-\alpha/2}\,\frac{s}{\sqrt n},$$

    where \(s\) uses divisor \(n-1\). That is the standard error `scipy.stats.sem` returns, and it is the scale passed to `scipy.stats.t.interval`.
    A higher confidence level uses a larger critical value, so the interval is wider.
    The \(t\) model is appropriate for a normal sample. A large normal sample satisfies that assumption closely.

    `Contains_45` checks whether the finished interval covers the known population mean.
    It is not part of calculating the interval. The interval is centered on the sample mean.
    In a real survey the population mean would be unknown, and this column could not be computed.
    """)
    return


@app.cell
def _(ages, mo, pd, st):
    _mean = ages.mean()
    _sem = st.sem(ages)
    _rows = []
    for _level in (0.90, 0.95, 0.99):
        _low, _high = st.t.interval(_level, len(ages) - 1, loc=_mean, scale=_sem)
        _rows.append({
            "Confidence": f"{_level:.0%}",
            "Low": _low,
            "High": _high,
            "Contains_45": _low <= 45 <= _high,
        })
    mean_intervals = pd.DataFrame(_rows)
    print(f"Sample mean = {_mean:.3f}; standard error = {_sem:.3f}.")
    mo.Html(
        mean_intervals.round(3).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (mean_intervals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which of the three intervals is widest?
    2. What does `Contains_45` check, and why can this lesson compute it?
    3. Does "95% confidence" mean that the population mean has a 95% chance of sitting in this particular interval?
    """)
    return


@app.cell(hide_code=True)
def _(mean_intervals, mo):
    _widths = mean_intervals["High"] - mean_intervals["Low"]
    _widest = mean_intervals.loc[_widths.idxmax(), "Confidence"]
    _answers = rf"""
1. **Widest interval:** {_widest}. Raising the confidence level raises the critical value.
2. **Known mean:** `Contains_45` asks whether the interval covers 45. The lesson can compute it because the simulation set the population mean to 45. The interval is centered on the sample mean.
3. **One interval:** No. The percentage describes the procedure across repeated samples.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Paired difference of means
    The two rows below are paired: each position is one pair.
    The interval is a one-sample \(t\) interval for the mean of the differences.
    If the interval contains 0, the data are compatible with a mean difference of 0 at that confidence level.
    That is a statement about the interval, not a hypothesis test written out in full.
    """)
    return


@app.cell
def _(mo, np, pd, st):
    x1 = np.array([148, 128, 69, 34, 155, 123, 101, 150, 139, 98])
    x2 = np.array([151, 146, 32, 70, 155, 142, 134, 157, 150, 130])
    differences = x2 - x1
    _n = len(differences)
    _df = _n - 1
    _mean = differences.mean()
    _sd = differences.std(ddof=1)
    _rows = []
    for _level in (0.90, 0.95, 0.99):
        _t = st.t.ppf(1 - (1 - _level) / 2, _df)
        _half = _t * _sd / np.sqrt(_n)
        _rows.append({
            "Confidence": f"{_level:.0%}",
            "Low": _mean - _half,
            "High": _mean + _half,
            "Contains_0": (_mean - _half) <= 0 <= (_mean + _half),
        })
    difference_intervals = pd.DataFrame(_rows)
    mo.Html(
        difference_intervals.round(2).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return difference_intervals, differences, x1, x2


@app.cell
def _(differences, plt, sns, x1, x2):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    sns.kdeplot(x=x1, ax=_axes[0], fill=True, color="#4C78A8", label="x1")
    sns.kdeplot(x=x2, ax=_axes[0], fill=True, color="#54A24B", label="x2")
    _axes[0].legend(frameon=False)
    _axes[0].set(title="Paired samples", xlabel="Value")
    sns.kdeplot(x=differences, ax=_axes[1], fill=True, color="#F58518")
    _axes[1].axvline(0, color="#E45756", linewidth=1.5)
    _axes[1].set(title="Paired differences", xlabel="x2 − x1")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Does the 95% interval for the mean difference contain 0?
    2. Why is the sample size 10 rather than 20?
    """)
    return


@app.cell(hide_code=True)
def _(difference_intervals, differences, mo):
    _row = difference_intervals.loc[difference_intervals["Confidence"] == "95%"].iloc[0]
    _contains = "Yes" if _row.Contains_0 else "No"
    _answers = rf"""
1. **Zero:** {_contains}. The 95% interval is ({_row.Low:.2f}, {_row.High:.2f}). The mean difference is {differences.mean():.2f}.
2. **Sample size:** There are 10 pairs. The interval uses the 10 differences, so \(n = 10\) and the degrees of freedom are 9.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Confidence interval for a normal variance
    If the observations are a normal sample, a confidence interval for the population variance uses chi-square critical values:

    $$\left(\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}},\; \frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}}\right).$$

    The larger chi-square quantile is in the lower endpoint. The interval is not symmetric around \(s^2\).
    The nine measurements below are the example from the original notebook.
    """)
    return


@app.cell
def _(mo, np, pd, st):
    measurements = np.array([8.01, 8.95, 9.65, 9.15, 8.06, 8.95, 8.03, 8.19, 8.03])
    _n = len(measurements)
    _df = _n - 1
    _s2 = measurements.var(ddof=1)
    _rows = []
    for _level in (0.90, 0.95, 0.99):
        _alpha = 1 - _level
        _low = _df * _s2 / st.chi2.ppf(1 - _alpha / 2, _df)
        _high = _df * _s2 / st.chi2.ppf(_alpha / 2, _df)
        _rows.append({"Confidence": f"{_level:.0%}", "Low": _low, "High": _high})
    variance_intervals = pd.DataFrame(_rows)
    print(f"Sample variance s^2 = {_s2:.3f}, with divisor n - 1.")
    mo.Html(
        variance_intervals.round(3).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return measurements, variance_intervals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Is the variance interval centered on the sample variance?
    2. What distributional assumption does this interval use?
    """)
    return


@app.cell(hide_code=True)
def _(measurements, mo, variance_intervals):
    _s2 = measurements.var(ddof=1)
    _row = variance_intervals.loc[variance_intervals["Confidence"] == "95%"].iloc[0]
    _mid = (_row.Low + _row.High) / 2
    _answers = rf"""
1. **Center:** No. The 95% interval is ({_row.Low:.3f}, {_row.High:.3f}). Its midpoint is {_mid:.3f}, while \(s^2\) is {_s2:.3f}.
2. **Assumption:** The observations are treated as a normal sample. The chi-square interval is not a general-purpose interval for every distribution.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. What the confidence level counts
    The population mean in this simulation is 45.
    The next cell draws 30 samples of size 40 and computes a 90% \(t\) interval for each.
    About 27 of the 30 intervals are expected to cover 45. One run of 30 samples will not hit 27 every time.
    The red line in the figure is the known population mean.
    """)
    return


@app.cell
def _(mo, np, pd, plt, st):
    _rng = np.random.default_rng(2026)
    _population = _rng.normal(45, 8, size=20_000)
    _rows = []
    for _i in range(30):
        _sample = _rng.choice(_population, size=40, replace=False)
        _low, _high = st.t.interval(0.90, len(_sample) - 1, loc=_sample.mean(), scale=st.sem(_sample))
        _rows.append({
            "Sample": _i + 1,
            "Mean": _sample.mean(),
            "Low": _low,
            "High": _high,
            "Covers_45": _low <= 45 <= _high,
        })
    coverage_table = pd.DataFrame(_rows)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.errorbar(
        coverage_table["Sample"],
        coverage_table["Mean"],
        yerr=[
            coverage_table["Mean"] - coverage_table["Low"],
            coverage_table["High"] - coverage_table["Mean"],
        ],
        fmt="o",
        color="#4C78A8",
        ecolor="#9ECAE1",
        elinewidth=1,
    )
    _misses = coverage_table.loc[~coverage_table["Covers_45"]]
    _ax.scatter(_misses["Sample"], _misses["Mean"], color="#E45756", zorder=3, label="Interval misses 45")
    _ax.axhline(45, color="#E45756", linewidth=1.5, label="Population mean 45")
    _ax.set(title="Thirty 90% intervals", xlabel="Sample", ylabel="Sample mean")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _covered = int(coverage_table["Covers_45"].sum())
    print(f"{_covered} of 30 intervals contain 45.")
    _fig
    return (coverage_table,)


@app.cell
def _(coverage_table, mo):
    mo.Html(
        coverage_table.round(2).to_html(index=False, border=0, col_space=90)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. If none of these 30 intervals had missed 45, would the procedure be invalid?
    2. Why is a sample of size 40 more useful for this picture than a sample of size 1,000?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **One experiment:** No. Twenty-seven is the expected count, not a quota for every batch of 30 intervals.
2. **Width:** With 1,000 observations the intervals are very narrow and almost all of them cover 45, so the misses that the confidence level allows are hard to see.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A point estimate is one sample number. A confidence interval is a range from a procedure with a stated coverage rate.
    - Divisor \(n\) and divisor \(n-1\) are two sample calculations. Neither one is automatically the known population variance.
    - The standard error estimates the typical sampling error. It is not the sampling error of the sample in hand.
    - A 95% interval from a valid procedure covers the parameter in about 95% of repeated samples. It does not assign a 95% probability to one finished interval.
    - The \(t\) interval for a mean uses \(s/\sqrt{n}\), and \(s\) uses divisor \(n-1\). `scipy.stats.sem` is that standard error.
    - A paired comparison can be reduced to a one-sample interval for the differences.
    - The chi-square variance interval assumes a normal sample, and it is not centered on \(s^2\).
    - These \(t\) and chi-square formulas are tied to their sampling models. A later lesson uses the bootstrap when that model is not the tool you want.

    ## Check your understanding
    1. A sample mean is 12. Is 12 the population mean?
    2. Which interval is wider, 90% or 99%, for the same sample?
    3. A 90% procedure is repeated 200 times. About how many intervals should contain the parameter?
    4. An interval for a mean difference is \((-1.2, 4.0)\). Is 0 inside it?
    5. Why does the upper end of a variance interval divide by the smaller chi-square quantile?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Point estimate:** Not necessarily. It is the estimate produced by this sample.
2. **Width:** The 99% interval is wider.
3. **Coverage count:** About \(0.90 \times 200 = 180\). The count in one set of 200 varies around that value.
4. **Zero:** Yes. Values from \(-1.2\) through \(4.0\) are inside the interval.
5. **Chi-square:** The lower tail quantile is the smaller number. Dividing by a smaller positive number produces the upper endpoint.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer, Chapter 23.
    """)
    return


if __name__ == "__main__":
    app.run()
