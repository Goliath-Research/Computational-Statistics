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
    from scipy.stats import skew

    sns.set_style("whitegrid")
    return mo, np, pd, plt, skew, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootstrap Tests for One Sample

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Build a bootstrap distribution of a sample mean by resampling with replacement.
    - Shift that distribution so its center sits at the hypothesized value.
    - Read a one-sided and a two-sided bootstrap p-value from the shifted distribution.
    - State that this shift is a location adjustment, and that it is the natural bootstrap null for a mean.
    - Apply the same shift, with that limitation named, to a variance and a skewness summary.

    ## 1. The bootstrap distribution of the mean
    A one-sample test asks whether a sample statistic is far from a hypothesized value.
    The bootstrap version used here has two pieces.

    First, resample the observations with replacement and recompute the statistic. That cloud estimates the sampling distribution's shape.
    Second, slide the cloud so that it is centered on the hypothesized value. The p-value is the tail of this shifted cloud beyond the observed statistic.

    Sliding the cloud is a good null for a mean: adding a constant to every observation adds the same constant to the mean, and the spread is unchanged.
    It is only a rough device for a variance or a skewness. Those statistics do not move by a constant when the data are shifted, and this lesson uses the slide only to show the arithmetic from the source notebook.

    The sample is 50 integers drawn from 158, 159, ..., 174. That range is not a normal curve. The bootstrap procedure does not need it to be one.
    This lesson uses 4,000 replicates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The observed mean is 166, and the hypothesized mean is 170.
    1. To center a bootstrap distribution on 170, do you add or subtract 4 if its current center is 166?
    2. Is the p-value read from the original sample, or from the shifted bootstrap values?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Shift:** Add 4. The new center is 166 + 4 = 170.
2. **p-value:** From the shifted bootstrap values. The observed mean is the point you locate in that null distribution.
""")})
    return


@app.cell
def _(np):
    heights = np.random.default_rng(1234).integers(158, 175, size=50)
    observed_mean = float(heights.mean())
    return heights, observed_mean


@app.cell
def _(heights, np, observed_mean, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3.3))
    _ax.hist(heights, bins=np.arange(157.5, 175.5, 1), color="#4C78A8", edgecolor="white")
    _ax.axvline(observed_mean, color="black", linewidth=2, label=f"Mean {observed_mean:.2f}")
    _ax.set(title="Simulated integer heights", xlabel="Height", ylabel="Count")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(heights, np):
    def bootstrap_statistics(sample, statistic, n_replicates=4_000, seed=2026):
        rng = np.random.default_rng(seed)
        draws = rng.choice(np.asarray(sample), size=(n_replicates, len(sample)), replace=True)
        return np.array([statistic(draw) for draw in draws])

    def null_p_value(null_statistics, observed, alternative="two-sided"):
        """Tail probability of a shifted bootstrap distribution."""
        null_statistics = np.asarray(null_statistics, dtype=float)
        left = np.mean(null_statistics <= observed)
        right = np.mean(null_statistics > observed)
        if alternative == "smaller":
            return float(left)
        if alternative == "larger":
            return float(right)
        return float(min(1.0, 2 * min(left, right)))

    mean_replicates = bootstrap_statistics(heights, np.mean)
    return bootstrap_statistics, mean_replicates, null_p_value


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Critical regions
    The hypothesized mean in the picture is 170.
    A left critical region is the lower 5% of the shifted bootstrap means.
    A right critical region is the upper 5%.
    A two-sided region splits 5% into 2.5% on each side.
    The source notebook used the 95th percentile, rather than the 97.5th, as the right edge of the two-sided picture. The right edge below is the 97.5th percentile.
    """)
    return


@app.cell
def _(mean_replicates, np, observed_mean, plt):
    null_at_170 = mean_replicates - mean_replicates.mean() + 170
    _fig, _axes = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
    _specs = [
        ("Left 5%", [np.percentile(null_at_170, 5)]),
        ("Right 5%", [np.percentile(null_at_170, 95)]),
        ("Two-sided 5%", [np.percentile(null_at_170, 2.5), np.percentile(null_at_170, 97.5)]),
    ]
    for _ax, (_title, _cuts) in zip(_axes, _specs):
        _ax.hist(null_at_170, bins=30, color="#9ECAE1", edgecolor="white")
        for _cut in _cuts:
            _ax.axvline(_cut, color="#E45756", linewidth=1.5)
        _ax.axvline(observed_mean, color="black", linewidth=1.5)
        _ax.set_title(_title, fontsize=10)
    _fig.suptitle("Shifted bootstrap means, centered at 170", fontsize=11)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return (null_at_170,)


@app.cell
def _(mo, np, null_at_170, null_p_value, observed_mean, pd):
    mean_tests = pd.DataFrame([
        {
            "H0": "mean = 170",
            "Ha": {"smaller": "mean < 170", "larger": "mean > 170", "two-sided": "mean ≠ 170"}[alternative],
            "Observed_mean": observed_mean,
            "p_value": null_p_value(null_at_170, observed_mean, alternative),
        }
        for alternative in ("smaller", "larger", "two-sided")
    ])
    mean_tests["Decision_at_0.05"] = np.where(mean_tests["p_value"] <= 0.05, "Reject H0", "Do not reject H0")
    mo.Html(
        mean_tests.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (mean_tests,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The smaller and larger p-values add to 1, apart from ties at the observed value, because they are the two sides of the same shifted distribution.
    The two-sided p-value is twice the smaller of those sides, and it is capped at 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A shifted null distribution has 1,000 values. 40 of them are less than or equal to the observed mean.
    1. What is the `smaller` p-value?
    2. What is the two-sided p-value if that left tail is the smaller side?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Left tail:** 40/1000 = 0.04.
2. **Two-sided:** \(2 \times 0.04 = 0.08\).
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Other hypothesized means
    Recentering is cheap once the bootstrap means exist. The same replicates can be slid to 168 or to 166.
    Rejecting 170 does not decide the test of 168. Each hypothesized value has its own null distribution.
    """)
    return


@app.cell
def _(mean_replicates, mo, np, null_p_value, observed_mean, pd):
    _rows = []
    for _value, _alternative in [(168, "two-sided"), (166, "larger")]:
        _null = mean_replicates - mean_replicates.mean() + _value
        _pvalue = null_p_value(_null, observed_mean, _alternative)
        _rows.append({
            "H0": f"mean = {_value}",
            "Ha": "mean ≠ value" if _alternative == "two-sided" else "mean > value",
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    other_mean_tests = pd.DataFrame(_rows)
    mo.Html(
        other_mean_tests.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (other_mean_tests,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Variance and skewness, with a warning
    The source notebook recenters the bootstrap variances on a hypothesized variance, and recenters the bootstrap skewness on 0.
    That copies the mean procedure. A variance does not shift by a constant when the data are shifted, so this is not a null distribution derived from a variance model.
    Read these two rows as a demonstration of the tail arithmetic, not as a finished variance test.
    The variance uses divisor \(n\), matching `numpy.var` in the source notebook. Pandas `DataFrame.var` would use \(n-1\).
    """)
    return


@app.cell
def _(bootstrap_statistics, heights, mo, np, null_p_value, pd, skew):
    variance_replicates = bootstrap_statistics(heights, lambda sample: np.var(sample, ddof=0), seed=11)
    skew_replicates = bootstrap_statistics(heights, skew, seed=12)
    observed_variance = float(np.var(heights, ddof=0))
    observed_skew = float(skew(heights))
    _rows = []
    for _name, _replicates, _observed, _value, _alternative in [
        ("Variance", variance_replicates, observed_variance, 20, "two-sided"),
        ("Variance", variance_replicates, observed_variance, 30, "smaller"),
        ("Skewness", skew_replicates, observed_skew, 0, "smaller"),
    ]:
        _null = _replicates - _replicates.mean() + _value
        _pvalue = null_p_value(_null, _observed, _alternative)
        _rows.append({
            "Statistic": _name,
            "Observed": _observed,
            "H0": f"value = {_value}",
            "Ha": "smaller" if _alternative == "smaller" else "two-sided",
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    shape_tests = pd.DataFrame(_rows)
    mo.Html(
        shape_tests.round(4).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return observed_skew, observed_variance, shape_tests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why is recentering appropriate for the mean?
    2. What changes in the variance row if the divisor changes from \(n\) to \(n-1\)? The hypothesized number, the observed variance, or both the statistic and its bootstrap values?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Mean:** Adding a constant to every observation adds that constant to the mean and leaves the spread alone. Sliding the bootstrap means to the hypothesized mean matches that fact.
2. **Divisor:** Both the observed variance and every bootstrap variance change, because the statistic changed. The hypothesized number 20 or 30 stays whatever you chose to test, but it is no longer being compared with the same statistic.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Bootstrap replicates are draws with replacement. Their means estimate the sampling distribution of the sample mean.
    - A mean test slides that distribution until its center equals the hypothesized mean. The observed mean is then located in the slide.
    - A left p-value is the share of null replicates at or below the observed mean. A right p-value is the share strictly above it. A two-sided p-value doubles the smaller side and does not exceed 1.
    - The two-sided 5% region uses the 2.5th and 97.5th percentiles.
    - Each hypothesized mean is a different slide. A rejection of 170 is not a result about 168.
    - Recentering a variance or a skewness copies the mean arithmetic. It is not, by itself, a null model for those statistics.
    - State the variance divisor. This lesson uses \(n\) in the variance section.

    ## Check your understanding
    1. How many observations are in each bootstrap sample if the original sample has 50?
    2. The bootstrap means are centered at 166.2, and the hypothesis is 170. What constant is added?
    3. Left p-value 0.03 and right p-value 0.97. What is the two-sided p-value?
    4. Does a bootstrap mean test require the height histogram to look normal?
    5. Why is the variance section marked as a warning?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Size:** 50.
2. **Constant:** 170 − 166.2 = 3.8.
3. **Two-sided:** \(2 \times 0.03 = 0.06\).
4. **Normal histogram:** No. The procedure resamples the observed heights.
5. **Warning:** Sliding bootstrap variances to a hypothesized variance is not the same operation as sliding means. The lesson keeps it only as tail arithmetic.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Davison, A. C., and Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press, Chapter 4.
    - Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC, Chapter 16.
    """)
    return


if __name__ == "__main__":
    app.run()
