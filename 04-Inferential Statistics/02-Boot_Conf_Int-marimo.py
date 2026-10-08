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
    # Bootstrap Confidence Intervals

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Describe a bootstrap sample as a draw with replacement from the observed sample.
    - Build a percentile confidence interval from the bootstrap values of a statistic.
    - Apply that interval to a mean, a median, a spread, and a shape summary.
    - Explain what the interval does not assume about the population distribution.
    - Read a percentile interval for a mean grade in the student file.

    ## 1. Resampling the sample you have
    A **bootstrap sample** has the same size as the original sample and is drawn **with replacement**.
    Repeating that draw produces an approximate sampling distribution for a statistic.
    The **percentile interval** takes the \(\alpha/2\) and \(1-\alpha/2\) percentiles of those bootstrap statistics.
    A 95% interval uses the 2.5th and 97.5th percentiles.

    This lesson uses 4,000 bootstrap replicates. The original notebook used 10,000. The percentile method is the same. The endpoints move a little when the number of replicates changes.

    The percentile interval does not require the statistic to be a mean, and it does not start from a normal formula.
    It still treats the original sample as a stand-in for the population, so a tiny or badly chosen sample remains a weak foundation.
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
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Simulated ages
    The sample is 1,000 ages drawn uniformly from 18 to 85, with seed 2026.
    The uniform draw is the population model for this simulation. The bootstrap procedure below does not use that fact.
    """)
    return


@app.cell
def _(np):
    ages = np.random.default_rng(2026).uniform(18, 85, size=1_000)
    ages
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


@app.cell
def _(ages, np):
    def bootstrap_replicates(sample, statistic, n_replicates=4_000, seed=2026):
        """Draw n_replicates samples with replacement and apply statistic to each."""
        rng = np.random.default_rng(seed)
        draws = rng.choice(np.asarray(sample), size=(n_replicates, len(sample)), replace=True)
        return np.array([statistic(draw) for draw in draws])

    def percentile_interval(replicates, confidence=95):
        """Percentile confidence interval from bootstrap replicates."""
        alpha = 100 - confidence
        return tuple(np.percentile(replicates, [alpha / 2, alpha / 2 + confidence]))

    return bootstrap_replicates, percentile_interval


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mean and median
    For a symmetric population the mean and the median estimate the same center, but their sampling distributions need not have the same width.
    """)
    return


@app.cell
def _(ages, bootstrap_replicates, np):
    mean_replicates = bootstrap_replicates(ages, np.mean)
    median_replicates = bootstrap_replicates(ages, np.median, seed=2027)
    return mean_replicates, median_replicates


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
def _(mean_replicates, median_replicates, mo, pd, percentile_interval):
    center_intervals = pd.DataFrame([
        {"Statistic": "Mean", "Confidence": level, "Low": percentile_interval(mean_replicates, level)[0], "High": percentile_interval(mean_replicates, level)[1]}
        for level in (90, 95, 99)
    ] + [
        {"Statistic": "Median", "Confidence": level, "Low": percentile_interval(median_replicates, level)[0], "High": percentile_interval(median_replicates, level)[1]}
        for level in (90, 95, 99)
    ])
    mo.Html(
        center_intervals.round(2).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (center_intervals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The sample is `np.array([2, 4, 4, 9])`.
    One bootstrap draw, already chosen, is `[4, 2, 4, 4]`.
    1. What is the mean of that bootstrap draw?
    2. Store a 95% percentile interval for the mean of the four-point sample in `low` and `high`. Use `bootstrap_replicates` and `percentile_interval`.
    """)
    return


@app.cell
def _(np):
    practice_sample = np.array([2.0, 4.0, 4.0, 9.0])
    low = None
    high = None
    print("Interval:", low, high)
    return high, low, practice_sample


@app.cell(hide_code=True)
def _(bootstrap_replicates, mo, np, percentile_interval, practice_sample):
    _draw_mean = np.mean([4, 2, 4, 4])
    _low, _high = percentile_interval(bootstrap_replicates(practice_sample, np.mean, seed=7), 95)
    _answers = f"""
1. **Bootstrap mean:** {_draw_mean:.2f}.
2. **Percentile interval:** ({_low:.2f}, {_high:.2f}), using seed 7 inside `bootstrap_replicates`. Another seed moves the endpoints slightly.

```python
low, high = percentile_interval(bootstrap_replicates(practice_sample, np.mean, seed=7), 95)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Spread and shape
    The same percentile function applies to the variance, the standard deviation, the interquartile range, the skewness, and the excess kurtosis.
    Pandas `Series.var` uses divisor \(n-1\). NumPy `var` uses divisor \(n\) unless `ddof` is set.
    The intervals below use divisor \(n-1\) for the variance and the standard deviation, matching Pandas.
    Skewness and excess kurtosis use SciPy, so excess kurtosis is the Fisher definition from the shape lesson.
    """)
    return


@app.cell
def _(ages, bootstrap_replicates, np, st):
    def _variance(sample):
        return np.var(sample, ddof=1)

    def _std(sample):
        return np.std(sample, ddof=1)

    def _iqr(sample):
        return np.subtract(*np.percentile(sample, [75, 25]))

    spread_replicates = {
        "Variance": bootstrap_replicates(ages, _variance, seed=1),
        "Standard deviation": bootstrap_replicates(ages, _std, seed=2),
        "Interquartile range": bootstrap_replicates(ages, _iqr, seed=3),
        "Skewness": bootstrap_replicates(ages, st.skew, seed=4),
        "Excess kurtosis": bootstrap_replicates(ages, st.kurtosis, seed=5),
    }
    return (spread_replicates,)


@app.cell
def _(mo, pd, percentile_interval, spread_replicates):
    spread_intervals = pd.DataFrame([
        {
            "Statistic": name,
            "Low": percentile_interval(values, 95)[0],
            "High": percentile_interval(values, 95)[1],
        }
        for name, values in spread_replicates.items()
    ])
    mo.Html(
        spread_intervals.round(3).to_html(index=False, border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (spread_intervals,)


@app.cell
def _(percentile_interval, plt, spread_replicates):
    _fig, _axes = plt.subplots(2, 3, figsize=(9, 5.2))
    for _ax, (_name, _values) in zip(_axes.ravel(), spread_replicates.items()):
        _low, _high = percentile_interval(_values, 95)
        _ax.hist(_values, bins=30, color="#4C78A8", edgecolor="white")
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
    2. Does a bootstrap interval for the median use a \(t\) critical value?
    3. The age population in this simulation is uniform. Did `percentile_interval` use that fact?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Percentiles:** The 5th and the 95th.
2. **Median:** No. The interval reads percentiles of the bootstrap medians.
3. **Uniform formula:** No. The function resamples the observed ages and takes percentiles. The uniform model was used only to create the sample.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Grades in the student file
    The code reads `../data/student-mat.csv`. Run the notebook with this lesson folder as the working directory.
    `G1`, `G2`, and `G3` are period grades on a 0–20 scale.
    The table gives a 95% percentile interval for each mean, next to the ordinary \(t\) interval.
    The two intervals answer the same question with different approximations. They will be close for these large samples and need not match exactly.
    """)
    return


@app.cell
def _(bootstrap_replicates, mo, np, pd, percentile_interval, st):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    grades = student_file[["G1", "G2", "G3"]].copy()
    _rows = []
    for _name in ["G1", "G2", "G3"]:
        _values = grades[_name].to_numpy()
        _boot = percentile_interval(bootstrap_replicates(_values, np.mean, seed=2026), 95)
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
    2. Are the bootstrap and \(t\) intervals required to have the same endpoints?
    """)
    return


@app.cell(hide_code=True)
def _(grade_intervals, mo):
    _lowest = grade_intervals.loc[grade_intervals["Mean"].idxmin(), "Grade"]
    _answers = rf"""
1. **Lowest mean:** {_lowest}.
2. **Endpoints:** No. One interval uses bootstrap percentiles. The other uses a \(t\) critical value and the standard error.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A bootstrap sample is drawn with replacement and has the original sample size.
    - The percentile interval uses the tails of the bootstrap statistics. A 95% interval uses the 2.5th and 97.5th percentiles.
    - The same construction applies to a mean, a median, a variance, a standard deviation, an interquartile range, skewness, and excess kurtosis.
    - The procedure does not insert a normal or \(t\) formula. It still depends on the observed sample.
    - State the divisor when the statistic is a variance or a standard deviation.
    - For a large sample mean, the percentile interval and the \(t\) interval are two approximations and can differ slightly.
    - Changing the number of replicates changes the Monte Carlo error in the endpoints. It does not change the definition of the interval.

    ## Check your understanding
    1. A sample has 40 rows. How many rows does one bootstrap sample have?
    2. What is replaced: the statistic, or the rows?
    3. Which percentiles are the ends of a 99% percentile interval?
    4. A bootstrap sample contains a value that was not in the original data. What went wrong?
    5. Why can the median interval be wider than the mean interval?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Rows:** 40.
2. **Replacement:** The rows. Each bootstrap sample recomputes the statistic.
3. **99% ends:** The 0.5th percentile and the 99.5th percentile.
4. **Support:** The draw was not taken from the observed sample. Bootstrap values are copies of observed values.
5. **Width:** The median uses fewer features of the sample than the mean, so its bootstrap distribution is often wider. The figure in this lesson is the comparison for these ages.
""")})
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
