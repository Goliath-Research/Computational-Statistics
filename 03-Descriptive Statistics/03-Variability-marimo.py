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
    # Measures of Variability

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Calculate variance and state whether the divisor is \(n\) or \(n-1\).
    - Relate the standard deviation to the variance that uses the same divisor.
    - Distinguish the range from the interquartile range.
    - Calculate a coefficient of variation and name the scale it requires.
    - Compare spread across groups without treating the comparison as a cause.

    ## 1. Two divisors for variance
    Variability describes how spread out the recorded values are.
    For \(x_1,\ldots,x_n\) with mean \(\bar x\), the squared deviations are \((x_i-\bar x)^2\).

    The **descriptive variance** divides their sum by \(n\):

    $$s_0^2=\frac{1}{n}\sum_{i=1}^n(x_i-\bar x)^2.$$

    NumPy's `np.var` uses this divisor unless `ddof` is changed. `ddof=0` means no degrees of freedom are removed.

    Pandas' `Series.var` and `DataFrame.var` default to the **sample variance**, which divides by \(n-1\):

    $$s_1^2=\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar x)^2.$$

    That is `ddof=1`. For \(n>1\), \(s_1^2\) is larger than \(s_0^2\) unless every squared deviation is zero.
    Neither number is a property of a larger population unless the observations were drawn for that purpose.

    The **standard deviation** is the square root of the variance computed with the same divisor.
    Its units match the data. Variance is in squared units.

    ### Try it yourself
    The values are [1, 2, 3, 6].
    1. What is their mean?
    2. What are \(s_0^2\) and \(s_1^2\)?
    3. What is the standard deviation that matches \(s_0^2\)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Mean:** (1 + 2 + 3 + 6) / 4 = 3.
2. **Variances:** The squared deviations are 4, 1, 0, and 9, and their sum is 14. \(s_0^2 = 14/4 = 3.5\). \(s_1^2 = 14/3 \approx 4.6667\).
3. **Matching standard deviation:** \(\sqrt{3.5} \approx 1.8708\). The square of this standard deviation is 3.5, not 4.6667.

```python
values = np.array([1, 2, 3, 6])
descriptive_variance = values.var(ddof=0)
sample_variance = values.var(ddof=1)
descriptive_sd = values.std(ddof=0)
```
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The student file
    The code reads `../data/student-mat.csv`. Run the notebook with this lesson folder as the working directory.
    `G1`, `G2`, and `G3` are grades on a 0–20 scale. Grouped calculations describe the recorded students in each group.
    """)
    return


@app.cell
def _(mo, pd):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    data = student_file[["G1", "G2", "G3", "school", "sex"]].copy()
    print(f"Loaded {len(data)} records.")
    mo.Html(
        data.head().to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (data,)


@app.cell
def _(data, mo, pd):
    _grade_rows = []
    for _grade_name in ["G1", "G2", "G3"]:
        _values = data[_grade_name].to_numpy()
        _grade_rows.append({
            "Grade": _grade_name,
            "Variance_n": _values.var(ddof=0),
            "Variance_n_minus_1": _values.var(ddof=1),
            "SD_n": _values.std(ddof=0),
            "SD_n_minus_1": _values.std(ddof=1),
        })
    mo.Html(
        pd.DataFrame(_grade_rows).round(4).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In each code task, replace the `None` placeholders in the next cell. Use the marimo editor to run the edited cell.

    ### Try it yourself
    For `G3`, calculate the descriptive variance and its matching standard deviation, using divisor \(n\).
    1. Store them in `g3_variance` and `g3_sd`.
    2. Check that `g3_sd ** 2` agrees with `g3_variance` up to rounding.
    """)
    return


@app.cell
def _(data):
    g3_variance = None
    g3_sd = None
    print("Descriptive variance:", g3_variance)
    print("Descriptive standard deviation:", g3_sd)
    return g3_sd, g3_variance


@app.cell(hide_code=True)
def _(data, mo):
    _values = data["G3"].to_numpy()
    _variance = _values.var(ddof=0)
    _sd = _values.std(ddof=0)
    _answers = f"""
1. **G3 with divisor n:** Variance = {_variance:.4f}. Standard deviation = {_sd:.4f}.
2. **Check:** ({_sd:.4f})² = {_sd ** 2:.4f}, which agrees with the variance apart from printed rounding. Pandas' default `.var()` is larger because it divides by n − 1.

```python
g3_values = data["G3"].to_numpy()
g3_variance = g3_values.var(ddof=0)
g3_sd = g3_values.std(ddof=0)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Same center, different spread
    The next two samples are simulated, not student grades. Both come from normal distributions with mean 0.
    Their standard deviations are 10 and 50. The seed makes this draw reproducible.
    A larger standard deviation produces a wider histogram. The sample variances will be near \(10^2\) and \(50^2\), not exactly those population values.
    """)
    return


@app.cell
def _(np, plt):
    generator = np.random.default_rng(2026)
    narrow_sample = generator.normal(0, 10, 2000)
    wide_sample = generator.normal(0, 50, 2000)
    print(f"Narrow descriptive variance = {narrow_sample.var(ddof=0):.2f}")
    print(f"Wide descriptive variance = {wide_sample.var(ddof=0):.2f}")
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.hist(narrow_sample, bins=30, alpha=0.55, color="#E45756", label="sd = 10")
    _ax.hist(wide_sample, bins=30, alpha=0.45, color="#4C78A8", label="sd = 50")
    _ax.set(title="Two simulated samples with the same population mean", xlabel="Value", ylabel="Count")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return narrow_sample, wide_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Grouped summaries
    `groupby` calculates the selected measure inside each recorded group.
    The variance below uses Pandas' default divisor, \(n-1\).
    A larger variance in one group says that the recorded grades in that group are more spread around their own mean.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("**Variance by sex, divisor n − 1**")
    return


@app.cell
def _(data, mo):
    mo.Html(
        data.groupby("sex")[["G1", "G2", "G3"]].var(ddof=1).round(3).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("**Standard deviation by school and sex, divisor n − 1**")
    return


@app.cell
def _(data, mo):
    mo.Html(
        data.groupby(["school", "sex"])[["G1", "G2", "G3"]].std(ddof=1).round(3).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Range and interquartile range
    The **range** is the maximum minus the minimum. It uses only the two extremes, so one unusual value can determine it.

    The **interquartile range** is

    $$\mathrm{IQR}=Q_3-Q_1.$$

    It is the length of the interval containing the central half of the ordered observations, under the same quartile definition used for Q1 and Q3.
    In a vertical box plot, that length is the height of the box, from Q1 to Q3.
    """)
    return


@app.cell
def _(data, mo, np, pd):
    _spread_rows = []
    for _grade_name in ["G1", "G2", "G3"]:
        _values = data[_grade_name].to_numpy()
        _q1, _q3 = np.percentile(_values, [25, 75])
        _spread_rows.append({
            "Grade": _grade_name,
            "Range": _values.max() - _values.min(),
            "IQR": _q3 - _q1,
        })
    spread_table = pd.DataFrame(_spread_rows)
    mo.Html(
        spread_table.round(2).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (spread_table,)


@app.cell
def _(data, plt, sns):
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(data=data[["G1", "G2", "G3"]], ax=_ax)
    _ax.set(title="Box height is the interquartile range", ylabel="Grade")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Suppose one G3 value of 0 is replaced by 100, and every other grade stays the same.
    1. Must the range change?
    2. Must the interquartile range change?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Range:** Yes, if 0 was the minimum and 100 becomes the maximum. The range changes from 20 − 0 to 100 − 0.
2. **Interquartile range:** No. Q1 and Q3 are determined by the central ordered values. Moving one extreme from 0 to 100 can leave both quartiles unchanged.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Coefficient of variation
    The **coefficient of variation** is a standard deviation divided by the mean:

    $$\mathrm{CV}=\frac{s}{\bar x}.$$

    Use the same divisor for \(s\) that you state in the sentence. `scipy.stats.variation` defaults to divisor \(n\).
    `Series.std() / Series.mean()` uses Pandas' divisor \(n-1\) for the standard deviation.

    The ratio is useful when the mean is nonzero and ratios on the measurement scale are meaningful.
    A length of 20 is twice a length of 10. A grade of 20 does not establish twice the measured achievement of a grade of 10, so these grades are not a ratio scale.
    The grade table below applies the formula only to show the arithmetic. It is not a comparison of achievement per grade point.
    A mean near zero makes the ratio unstable. The coefficient has no units because the units cancel.

    Two groups can share a standard deviation and still have different coefficients because their means differ.
    For a standard deviation of 50, the coefficient is \(50/150 \approx 0.3333\) when the mean is 150, and \(50/500 = 0.10\) when the mean is 500.
    The first group has more spread relative to its own mean.
    """)
    return


@app.cell
def _(data, mo, pd):
    _cv_rows = []
    for _grade_name in ["G1", "G2", "G3"]:
        _values = data[_grade_name]
        _cv_rows.append({
            "Grade": _grade_name,
            "CV_divisor_n": _values.std(ddof=0) / _values.mean(),
            "CV_divisor_n_minus_1": _values.std(ddof=1) / _values.mean(),
        })
    mo.Html(
        pd.DataFrame(_cv_rows).round(4).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Group A has mean 150 and standard deviation 50. Group B has mean 500 and standard deviation 50.
    1. Calculate each coefficient of variation.
    2. Which group has the larger relative spread?
    3. Do the two standard deviations differ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Coefficients:** Group A has \(50/150 = 1/3 \approx 0.3333\). Group B has \(50/500 = 0.10\).
2. **Relative spread:** Group A is larger relative to its mean.
3. **Standard deviations:** No. Both are 50. The coefficients differ because the means differ.

```python
cv_group_a = 50 / 150
cv_group_b = 50 / 500
```
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Variance averages squared deviations from the mean. State whether the divisor is \(n\) or \(n-1\).
    - The standard deviation is the square root of the variance that uses the same divisor, so its units match the observations.
    - Two samples can share a mean and differ in spread. A finite simulated variance need not equal the population variance.
    - The range depends on the two extremes. The interquartile range describes the central half.
    - The coefficient of variation compares a standard deviation with the mean when ratios on that scale are meaningful. The grade table illustrates the formula only. Equal standard deviations can produce different coefficients.
    - Grouped measures describe the recorded students in each group.

    ## Check your understanding
    1. For [2, 2, 2], what is the descriptive variance?
    2. If the standard deviation is 4, what is the corresponding variance?
    3. Why can one extreme grade dominate the range?
    4. For mean 20 and standard deviation 5, what is the coefficient of variation?
    5. Which Pandas default divisor does `Series.var()` use?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Constant sample:** 0. Every deviation from the mean is 0.
2. **Variance:** 16, because the standard deviation is the square root of the variance.
3. **Range:** It is calculated from only the maximum and the minimum.
4. **Coefficient:** 5/20 = 0.25, using the stated standard deviation and mean.
5. **Pandas default:** \(n-1\), which is `ddof=1`.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), for the dataset and variable definitions.
    - Nussbaumer Knaflic, C. (2015). *Storytelling with Data*. Wiley, Chapter 2.
    - Unpingco, J. (2019). *Python for Probability, Statistics, and Machine Learning*. Springer.
    """)
    return


if __name__ == "__main__":
    app.run()
