# /// script
# dependencies = [
#     "marimo",
#     "seaborn",
#     "scipy",
#     "statsmodels",
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
    from statsmodels.stats.proportion import proportions_ztest
    from statsmodels.stats.weightstats import ztest

    sns.set_style("whitegrid")
    return mo, np, pd, plt, proportions_ztest, sns, st, ztest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-sample \(z\) and \(t\) Tests

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Write a one-sample proportion test with a direction.
    - Read a one-sample \(z\) test for a mean as a large-sample normal approximation.
    - Read a one-sample \(t\) test as the version that keeps an unknown standard deviation in the reference distribution.
    - Mark a \(t\) statistic on the reference curve and compare it with the critical region.
    - Use "do not reject" when the p-value is above the significance level.

    ## 1. Proportion
    A one-sample proportion test compares a sample of successes and failures with a hypothesized proportion \(p_0\).
    The null standard error is \(\sqrt{p_0(1-p_0)/n}\).
    The alternative is two-sided, `larger`, or `smaller`.

    The survey below is simulated, not a real poll. Each of 100 draws is a success with probability 0.80, using seed 2026.
    The historical comparison in the source notebook is \(p_0 = 0.60\).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The sample proportion is 0.70 and \(p_0 = 0.60\).
    1. Which one-sided alternative matches a research question that asks whether the proportion is higher than 0.60?
    2. If the sample proportion had been 0.40, which one-sided alternative would match "lower than 0.60"?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Higher:** `larger`.
2. **Lower:** `smaller`.
""")})
    return


@app.cell
def _(mo, np, pd, proportions_ztest):
    survey = np.random.default_rng(2026).random(100) < 0.80
    successes = int(survey.sum())
    _rows = []
    for _alternative in ("smaller", "larger", "two-sided"):
        _z, _pvalue = proportions_ztest(successes, len(survey), value=0.60, alternative=_alternative)
        _rows.append({
            "H0": "p = 0.60",
            "Ha": {"smaller": "p < 0.60", "larger": "p > 0.60", "two-sided": "p ≠ 0.60"}[_alternative],
            "z": _z,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    survey_results = pd.DataFrame(_rows)
    print(f"{successes} of {len(survey)} simulated responses are successes.")
    mo.Html(
        survey_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return successes, survey_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The student file asks a similar question about internet access.
    The code reads `../data/student-mat.csv`. Run the notebook with this lesson folder as the working directory.
    """)
    return


@app.cell
def _(mo, pd, proportions_ztest):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    internet_yes = int((student_file["internet"] == "yes").sum())
    internet_n = len(student_file)
    _rows = []
    for _p0, _alternative in [(0.80, "larger"), (0.90, "larger")]:
        _z, _pvalue = proportions_ztest(internet_yes, internet_n, value=_p0, alternative=_alternative)
        _rows.append({
            "H0": f"p = {_p0:.2f}",
            "Ha": "p > p0",
            "Sample_proportion": internet_yes / internet_n,
            "z": _z,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    internet_results = pd.DataFrame(_rows)
    mo.Html(
        internet_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return internet_n, internet_results, internet_yes, student_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. The simulated survey was generated with success probability 0.80. Why is a test of "smaller than 0.60" a poor match to that sample?
    2. Looking at the student table, is the recorded internet proportion above 0.80?
    """)
    return


@app.cell(hide_code=True)
def _(internet_n, internet_results, internet_yes, mo, successes):
    _decision = internet_results.loc[internet_results["H0"] == "p = 0.80", "Decision_at_0.05"].iloc[0]
    _answers = rf"""
1. **Direction:** The sample has {successes} successes out of 100, well above 0.60. The alternative `smaller` looks in the opposite direction, so its p-value stays large.
2. **Internet:** The sample proportion is {internet_yes / internet_n:.3f}. The test of \(p > 0.80\) has decision "{_decision}".
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. \(z\) test for a mean
    `ztest` compares a sample mean with a hypothesized value and uses a standard normal reference.
    The standard error is estimated from the sample. The normal reference is an approximation for a large sample.
    The heights are 200 draws from a normal distribution with mean 165 cm and standard deviation 10 cm.
    The source notebook also compares a grade mean with 11. That comparison is included for `G1`.
    """)
    return


@app.cell
def _(np, pd, plt, student_file, ztest):
    heights = np.random.default_rng(2026).normal(165, 10, size=200)
    _fig, _ax = plt.subplots(figsize=(6, 3.3))
    _ax.hist(heights, bins=20, color="#4C78A8", edgecolor="white")
    _ax.axvline(heights.mean(), color="#E45756", label=f"Mean {heights.mean():.1f}")
    _ax.axvline(170, color="#F58518", linestyle="--", label="170")
    _ax.set(title="Simulated heights", xlabel="Height (cm)", ylabel="Count")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _rows = []
    for _alternative in ("two-sided", "smaller"):
        _z, _pvalue = ztest(heights, value=170, alternative=_alternative)
        _rows.append({
            "Sample": "Heights",
            "H0": "mean = 170",
            "Ha": "mean ≠ 170" if _alternative == "two-sided" else "mean < 170",
            "z": _z,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    _z, _pvalue = ztest(student_file["G1"], value=11, alternative="two-sided")
    _rows.append({
        "Sample": "G1",
        "H0": "mean = 11",
        "Ha": "mean ≠ 11",
        "z": _z,
        "p_value": _pvalue,
        "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
    })
    z_results = pd.DataFrame(_rows)
    _fig
    return heights, z_results


@app.cell
def _(mo, z_results):
    mo.Html(
        z_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. \(t\) test and its reference curve
    `ttest_1samp` uses a \(t\) distribution with \(n-1\) degrees of freedom.
    The figure shades the critical region for \(\alpha = 0.05\) and marks the observed \(t\) statistic.
    A mark inside the shaded region is a rejection at that level. The p-value is the tail area beyond the mark, not the shaded area.
    The sample of 20 heights is drawn from a normal distribution with mean 169 and standard deviation 5.
    """)
    return


@app.cell
def _(np):
    small_heights = np.random.default_rng(7).normal(169, 5, size=20)
    return (small_heights,)


@app.cell
def _(np, pd, plt, small_heights, st):
    def t_reference_figure(sample, hypothesized, alternative="two-sided", alpha=0.05):
        label = {"two-sided": "two-sided", "smaller": "less", "larger": "greater"}[alternative]
        statistic, pvalue = st.ttest_1samp(sample, hypothesized, alternative=label)
        df = len(sample) - 1
        grid = np.linspace(-4, 4, 400)
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.plot(grid, st.t.pdf(grid, df), color="#4C78A8")
        if alternative == "two-sided":
            left, right = st.t.ppf([alpha / 2, 1 - alpha / 2], df)
            shade = (grid <= left) | (grid >= right)
        elif alternative == "smaller":
            left = st.t.ppf(alpha, df)
            shade = grid <= left
        else:
            right = st.t.ppf(1 - alpha, df)
            shade = grid >= right
        ax.fill_between(grid, st.t.pdf(grid, df), where=shade, color="#E45756", alpha=0.35)
        ax.scatter([statistic], [0.01], color="black", zorder=3, label=f"t = {statistic:.2f}")
        decision = "Reject H0" if pvalue <= alpha else "Do not reject H0"
        ax.set(title=f"H0: mean = {hypothesized:g}; p = {pvalue:.4f}; {decision}", xlabel="t", ylabel="Density")
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.close(fig)
        return fig, statistic, pvalue, decision

    _rows = []
    _figures = []
    for _value, _alternative in [(170, "two-sided"), (170, "smaller"), (170, "larger"), (160, "two-sided")]:
        _fig, _statistic, _pvalue, _decision = t_reference_figure(small_heights, _value, _alternative)
        _figures.append(_fig)
        _rows.append({
            "H0": f"mean = {_value}",
            "Ha": {"two-sided": "mean ≠ value", "smaller": "mean < value", "larger": "mean > value"}[_alternative],
            "t": _statistic,
            "p_value": _pvalue,
            "Decision_at_0.05": _decision,
        })
    t_results = pd.DataFrame(_rows)
    t_figures = _figures
    return t_figures, t_reference_figure, t_results


@app.cell
def _(mo, t_results):
    mo.Html(
        t_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(mo, t_figures):
    mo.hstack(t_figures[:2])
    return


@app.cell
def _(mo, t_figures):
    mo.hstack(t_figures[2:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a code task, replace `None` and run the cell in the marimo editor.

    ### Try it yourself
    For `small_heights` and hypothesized mean 165, two-sided:
    1. Store the \(t\) statistic in `practice_t`.
    2. Store the p-value in `practice_p`.
    """)
    return


@app.cell
def _():
    practice_t = None
    practice_p = None
    print(practice_t, practice_p)
    return practice_p, practice_t


@app.cell(hide_code=True)
def _(mo, small_heights, st):
    _t, _p = st.ttest_1samp(small_heights, 165)
    _answers = f"""
1. **Statistic:** {_t:.3f}.
2. **p-value:** {_p:.4f}.

```python
practice_t, practice_p = st.ttest_1samp(small_heights, 165)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A proportion \(z\) test uses the null proportion in the standard error. The alternative picks the tail.
    - A sample generated far above \(p_0\) will not reject a "smaller than \(p_0\)" alternative.
    - `ztest` uses a normal reference with a sample standard error. That is a large-sample approximation.
    - `ttest_1samp` uses \(n-1\) degrees of freedom. The shaded critical region and the p-value are related but not the same area.
    - A p-value above 0.05 means do not reject the hypothesized mean or proportion.
    - The same sample can reject one hypothesized value and not another. The hypothesis is part of the procedure.

    ## Check your understanding
    1. Alternative `larger` looks for a proportion on which side of \(p_0\)?
    2. Where is the \(t\) statistic marked: on the data histogram, or on the reference curve?
    3. The mark sits in the unshaded center. What is the decision at \(\alpha = 0.05\)?
    4. How many degrees of freedom does a \(t\) test with 20 observations use?
    5. Why is the internet test a proportion test rather than a test about the mean of `G1`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Side:** Above \(p_0\).
2. **Curve:** On the \(t\) reference curve. The histogram is a picture of the observations.
3. **Decision:** Do not reject \(H_0\). The mark is outside the critical region.
4. **Degrees of freedom:** 19.
5. **Variable:** Internet access is a yes/no category. `G1` is a numeric grade.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    - [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).
    """)
    return


if __name__ == "__main__":
    app.run()
