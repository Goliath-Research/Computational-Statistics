# /// script
# dependencies = [
#     "marimo",
#     "seaborn",
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

    sns.set_style("whitegrid")
    return mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootstrap Tests for Two Samples

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Shift two independent samples onto one common mean and resample each with replacement.
    - Read a p-value for a difference of means, a difference of medians, or a \(t\) statistic from that null.
    - Center paired differences at 0 before bootstrapping a mean difference.
    - Explain why an uncentered bootstrap distribution of the observed differences does not test a mean of 0.

    ## 1. Independent samples
    Class C and Class D are normal samples with population means 85 and 90 and different sample sizes.
    The bootstrap null shifts each class onto the pooled mean, then draws 2,000 bootstrap samples from each shifted class.
    The source notebook used 10,000 draws. The shift-and-resample steps are unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The means are 85 and 91, with sample sizes 4 and 6.
    1. What is the pooled mean?
    2. What is the observed difference of means, first minus second?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Pooled mean:** \((4 \times 85 + 6 \times 91) / 10 = 88.6\).
2. **Observed difference:** \(85 - 91 = -6\).
""")})
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(123)
    class_c = _rng.normal(85, 3, size=100)
    class_d = _rng.normal(90, 3, size=95)

    def tail_p(null_stats, observed, alternative="two-sided"):
        null_stats = np.asarray(null_stats, dtype=float)
        left = float(np.mean(null_stats <= observed))
        right = float(np.mean(null_stats > observed))
        if alternative == "smaller":
            return left
        if alternative == "larger":
            return right
        return float(min(1.0, 2 * min(left, right)))

    def shifted_bootstrap(left, right, n_replicates=2_000, seed=2026):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        pooled = np.concatenate([left, right]).mean()
        left = left - left.mean() + pooled
        right = right - right.mean() + pooled
        rng = np.random.default_rng(seed)
        return (
            rng.choice(left, size=(n_replicates, len(left)), replace=True),
            rng.choice(right, size=(n_replicates, len(right)), replace=True),
        )

    return class_c, class_d, shifted_bootstrap, tail_p


@app.cell
def _(class_c, class_d, mo, np, pd, plt, shifted_bootstrap, sns, tail_p):
    left_draws, right_draws = shifted_bootstrap(class_c, class_d)
    null_means = left_draws.mean(axis=1) - right_draws.mean(axis=1)
    null_medians = np.median(left_draws, axis=1) - np.median(right_draws, axis=1)

    def _t_stat(left, right):
        return (left.mean() - right.mean()) / np.sqrt(
            left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right)
        )

    null_t = np.array([_t_stat(left, right) for left, right in zip(left_draws, right_draws)])
    observed = {
        "Difference of means": class_c.mean() - class_d.mean(),
        "Difference of medians": np.median(class_c) - np.median(class_d),
        "t statistic": _t_stat(class_c, class_d),
    }
    nulls = {
        "Difference of means": null_means,
        "Difference of medians": null_medians,
        "t statistic": null_t,
    }
    independent_results = pd.DataFrame([
        {
            "Statistic": name,
            "Observed": observed[name],
            "p_value": tail_p(nulls[name], observed[name]),
            "Decision_at_0.05": "Reject H0" if tail_p(nulls[name], observed[name]) <= 0.05 else "Do not reject H0",
        }
        for name in observed
    ])
    _fig, _ax = plt.subplots(figsize=(6, 3.3))
    sns.kdeplot(class_c, ax=_ax, fill=True, color="#54A24B", label=f"Class C, n={len(class_c)}")
    sns.kdeplot(class_d, ax=_ax, fill=True, color="#F58518", label=f"Class D, n={len(class_d)}")
    _ax.legend(frameon=False)
    _ax.set(title="Independent classes", xlabel="Score")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return independent_results, null_means


@app.cell
def _(independent_results, mo):
    mo.Html(
        independent_results.round(4).to_html(index=False, border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(independent_results, null_means, plt):
    _observed = independent_results.loc[
        independent_results["Statistic"] == "Difference of means", "Observed"
    ].iloc[0]
    _fig, _ax = plt.subplots(figsize=(6, 3.3))
    _ax.hist(null_means, bins=30, color="#4C78A8", edgecolor="white")
    _ax.axvline(_observed, color="black", linewidth=2, label="Observed mean difference")
    _ax.legend(frameon=False)
    _ax.set(title="Bootstrap null", xlabel="Mean difference", ylabel="Count")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Paired samples
    The function below is the calculation the source notebook sketched and then left commented out.
    It bootstraps the centered differences and returns a two-sided p-value for the mean, or for the median.

    One simulated after-sample sits near the before-sample. The other sits higher.
    The source notebook once compared the observed mean with bootstrap means of the raw differences, without centering. That comparison places the observed mean in the middle of its own bootstrap cloud. The tests below center the differences first.
    """)
    return


@app.cell
def _(np, tail_p):
    def paired_bootstrap(before, after, statistic="mean", n_replicates=2_000, seed=2026):
        before = np.asarray(before, dtype=float)
        after = np.asarray(after, dtype=float)
        differences = after - before
        observed = np.mean(differences) if statistic == "mean" else np.median(differences)
        centered = differences - differences.mean() if statistic == "mean" else differences - np.median(differences)
        rng = np.random.default_rng(seed)
        draws = rng.choice(centered, size=(n_replicates, len(centered)), replace=True)
        null_stats = draws.mean(axis=1) if statistic == "mean" else np.median(draws, axis=1)
        pvalue = tail_p(null_stats, observed)
        return observed, pvalue, null_stats

    return (paired_bootstrap,)


@app.cell
def _(mo, np, paired_bootstrap, pd):
    _rng = np.random.default_rng(12)
    grade_before = _rng.normal(85.5, 3, size=80)
    grade_after_near = _rng.normal(86, 4, size=80)
    grade_after_high = _rng.normal(90, 3, size=80)
    _rows = []
    for _name, _after, _statistic in [
        ("Near, mean", grade_after_near, "mean"),
        ("Near, median", grade_after_near, "median"),
        ("High, mean", grade_after_high, "mean"),
        ("High, median", grade_after_high, "median"),
    ]:
        _observed, _pvalue, _null = paired_bootstrap(grade_before, _after, _statistic, seed=20 + len(_rows))
        _rows.append({
            "Comparison": _name,
            "Observed_difference": _observed,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    paired_results = pd.DataFrame(_rows)
    mo.Html(
        paired_results.round(4).to_html(index=False, border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (paired_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a code task, replace `None` in the next cell.

    ### Try it yourself
    The differences are `np.array([1.0, -2.0, 3.0])`.
    1. Store their mean in `diff_mean`.
    2. Store the centered differences in `centered`.
    """)
    return


@app.cell
def _(np):
    practice_differences = np.array([1.0, -2.0, 3.0])
    diff_mean = None
    centered = None
    print(diff_mean, centered)
    return centered, diff_mean, practice_differences


@app.cell(hide_code=True)
def _(mo, practice_differences):
    _mean = practice_differences.mean()
    _centered = practice_differences - _mean
    _answers = f"""
1. **Mean:** {_mean:.1f}.
2. **Centered values:** {_centered.tolist()}. Their mean is 0.

```python
diff_mean = practice_differences.mean()
centered = practice_differences - diff_mean
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - An independent bootstrap null shifts both samples to the pooled mean, then resamples with replacement.
    - The resampled statistic has to match the observed statistic. A median null does not answer a mean question.
    - A paired bootstrap keeps the pair inside the difference, centers that difference at the null, and resamples.
    - An uncentered bootstrap of the differences is centered on the observed difference, so it cannot test zero.
    - A two-sided p-value doubles the smaller tail and stops at 1.
    - The class scores and grades are simulated.

    ## Check your understanding
    1. Does the independent bootstrap draw from the original classes or from the shifted classes?
    2. What is the mean of the centered differences?
    3. Can the mean row and the median row reach different decisions?
    4. What does a p-value of 0.40 say about the null?
    5. Why does the paired procedure refuse to pool before-scores and after-scores into two unmatched groups?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Source of the draws:** The shifted classes.
2. **Centered mean:** 0.
3. **Decisions:** Yes. They are tests of different statistics.
4. **Large p-value:** Do not reject the null at the 0.05 level. It is not proof that the null is true.
5. **Pairs:** The link within each person is the comparison. Pooling would treat the two occasions as two unrelated groups.
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
