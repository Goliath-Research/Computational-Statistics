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
    from scipy.stats import kurtosis, skew, t

    sns.set_style("whitegrid")
    return kurtosis, mo, np, pd, plt, skew, sns, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Measures of Distribution Shape

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Interpret the sign of the moment skewness as a usual indication of a longer or heavier tail.
    - Distinguish that skewness from Pearson's median-based approximation.
    - Interpret excess kurtosis as tail weight relative to a normal distribution.
    - Explain why changing a normal distribution's standard deviation does not change its kurtosis.
    - Summarize an unordered category with its mode and counts.

    ## 1. Moment skewness
    **Skewness** describes asymmetry around the sample mean.
    For \(x_1,\ldots,x_n\) with mean \(\bar x\) and descriptive standard deviation \(s_0\), the biased moment skewness used by `scipy.stats.skew` is

    $$g_1=\frac{\frac{1}{n}\sum_{i=1}^n(x_i-\bar x)^3}{s_0^3}.$$

    - \(g_1>0\): positive moment skewness usually indicates a longer or heavier right tail.
    - \(g_1<0\): negative moment skewness usually indicates a longer or heavier left tail.
    The sign does not strictly determine tail length.
    - \(g_1\) near 0: the cubed deviations nearly balance. The distribution can still be non-normal. A uniform distribution is symmetric and is not normal.

    A finite normal sample has skewness near 0, not necessarily exactly 0.
    Pearson's second coefficient, \(3(\bar x-\mathrm{median})/s_0\), is a different approximation. It often has the same sign, but it is not the value `scipy.stats.skew` returns.

    ### Try it yourself
    The values are [1, 2, 2, 3, 10].
    1. Is the mean larger than the median?
    2. What does that suggest about the direction of the tail?
    3. Must Pearson's coefficient equal the moment skewness?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Mean and median:** The mean is (1 + 2 + 2 + 3 + 10) / 5 = 3.6. The ordered middle value is 2, so the mean is larger.
2. **Tail:** The value 10 pulls the mean to the right, so the right tail is longer.
3. **Two coefficients:** No. Both should be positive here, but they use different calculations and need not be equal.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Simulated shapes
    Each sample below uses seed 2026. The lognormal sample has a longer right tail.
    Negating it reverses that tail. The normal sample is the symmetric comparison.
    """)
    return


@app.cell
def _(np, plt, skew):
    _generator = np.random.default_rng(2026)
    normal_sample = _generator.normal(0, 1, 20_000)
    right_skew_sample = _generator.lognormal(0, 0.5, 20_000)
    left_skew_sample = -right_skew_sample
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for _axis, _sample, _title, _color in zip(
        _axes,
        [normal_sample, right_skew_sample, left_skew_sample],
        ["Normal sample", "Right tail", "Left tail"],
        ["#E45756", "#4C78A8", "#54A24B"],
    ):
        _axis.hist(_sample, bins=40, color=_color, edgecolor="white")
        _axis.set(title=f"{_title}\nskew = {skew(_sample):.2f}")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return left_skew_sample, normal_sample, right_skew_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a code task, replace each `None` in the next cell and run it in the marimo editor.

    ### Try it yourself
    For `example_values = np.array([1, 2, 2, 3, 10])`, calculate:
    1. `moment_skew`, using `skew`;
    2. `pearson_skew`, using \(3(\bar x-\mathrm{median})/s_0\), with `ddof=0`.

    Print both values and compare their signs.
    """)
    return


@app.cell
def _(np):
    example_values = np.array([1, 2, 2, 3, 10])
    moment_skew = None
    pearson_skew = None
    print("Moment skewness:", moment_skew)
    print("Pearson skewness:", pearson_skew)
    return example_values, moment_skew, pearson_skew


@app.cell(hide_code=True)
def _(example_values, mo, np, skew):
    _moment = skew(example_values)
    _pearson = 3 * (example_values.mean() - np.median(example_values)) / example_values.std(ddof=0)
    _answers = f"""
1. **Moment skewness:** {_moment:.4f}.
2. **Pearson coefficient:** {_pearson:.4f}.
3. **Signs:** Both are positive. The numbers differ because the two formulas are not the same.

```python
moment_skew = skew(example_values)
pearson_skew = 3 * (example_values.mean() - np.median(example_values)) / example_values.std(ddof=0)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Excess kurtosis
    `scipy.stats.kurtosis` returns **excess kurtosis** by default (`fisher=True`).
    A normal distribution has excess kurtosis 0.
    Positive excess kurtosis means heavier tails than a normal distribution.
    Negative excess kurtosis means lighter tails.

    The older names describe that comparison:
    - **Mesokurtic:** excess kurtosis near 0.
    - **Leptokurtic:** excess kurtosis greater than 0.
    - **Platykurtic:** excess kurtosis less than 0.

    Kurtosis is not a measure of whether the drawn peak looks sharp.
    Every normal distribution has the same excess kurtosis, regardless of its standard deviation.
    Changing the standard deviation changes the height and width of the density. A standardized comparison removes that scale effect.

    A uniform distribution has population excess kurtosis \(-1.2\).
    A Student \(t\) distribution with 5 degrees of freedom has population excess kurtosis \(6/(5-4)=6\).
    """)
    return


@app.cell
def _(kurtosis, mo, normal_sample, np, pd, t):
    _generator = np.random.default_rng(2026)
    narrow_normal = _generator.normal(0, 0.5, 100_000)
    wide_normal = _generator.normal(0, 4, 100_000)
    uniform_sample = _generator.uniform(-1, 1, 100_000)
    heavy_tail_sample = t.rvs(df=5, size=100_000, random_state=2026)
    kurtosis_table = pd.DataFrame({
        "Sample": ["Normal sd = 0.5", "Normal sd = 4", "Uniform(-1, 1)", "t with 5 df", "Normal sample from the skewness figure"],
        "Excess_kurtosis": [
            kurtosis(narrow_normal),
            kurtosis(wide_normal),
            kurtosis(uniform_sample),
            kurtosis(heavy_tail_sample),
            kurtosis(normal_sample),
        ],
    })
    mo.Html(
        kurtosis_table.round(3).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return heavy_tail_sample, kurtosis_table, uniform_sample


@app.cell
def _(heavy_tail_sample, np, plt, uniform_sample):
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.4))
    _axes[0].hist(uniform_sample, bins=30, color="#4C78A8", edgecolor="white")
    _axes[0].set(title="Uniform sample: light tails")
    _standardized_t = heavy_tail_sample / np.std(heavy_tail_sample, ddof=0)
    _axes[1].hist(_standardized_t, bins=80, range=(-8, 8), color="#F58518", edgecolor="white")
    _axes[1].set(title="Standardized t sample: heavy tails")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. The table reports excess kurtosis for normal samples with standard deviations 0.5 and 4. Are those two population kurtosis values different?
    2. Which simulated distribution is the heavy-tailed one?
    3. Does excess kurtosis between −1 and 1 prove that a sample came from a normal distribution?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Normal kurtosis:** No. Both populations have excess kurtosis 0. The simulated values differ from 0 only by finite-sample variation.
2. **Heavy tails:** The Student \(t\) sample with 5 degrees of freedom. Its population excess kurtosis is 6.
3. **A bound is not a proof:** No. Values near 0 are compatible with several distributions. This is a descriptive screen, not a normality test.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Grades in the student file
    The code reads `../data/student-mat.csv`. Run the notebook with this lesson folder as the working directory.
    The printed skewness and excess kurtosis describe the recorded grades.
    `G3` includes zeros, and those low grades contribute to its left tail.
    """)
    return


@app.cell
def _(kurtosis, mo, pd, skew):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    data = student_file[
        ["school", "sex", "age", "studytime", "schoolsup", "internet", "G1", "G2", "G3"]
    ].copy()
    grade_shape = pd.DataFrame({
        "Grade": ["G1", "G2", "G3"],
        "Skewness": [skew(data[grade]) for grade in ["G1", "G2", "G3"]],
        "Excess_kurtosis": [kurtosis(data[grade]) for grade in ["G1", "G2", "G3"]],
    })
    print(f"Loaded {len(data)} records.")
    mo.Html(
        grade_shape.round(4).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return data, grade_shape


@app.cell
def _(data, grade_shape, plt, sns):
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for _axis, _grade, _row in zip(_axes, ["G1", "G2", "G3"], grade_shape.itertuples(index=False)):
        sns.kdeplot(data[_grade], fill=True, ax=_axis, color="#4C78A8")
        _axis.set(title=f"{_grade}\nskew {_row.Skewness:.2f}; kurtosis {_row.Excess_kurtosis:.2f}")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    From the grade table:
    1. Which final-grade measure, skewness or excess kurtosis, usually indicates a longer or heavier left tail?
    2. Is the G3 excess kurtosis the sharpness of the kernel-density peak?
    """)
    return


@app.cell(hide_code=True)
def _(grade_shape, mo):
    _g3 = grade_shape.loc[grade_shape["Grade"] == "G3"].iloc[0]
    _answers = f"""
1. **Left tail:** The skewness, {_g3.Skewness:.4f}, is negative. Negative moment skewness usually indicates a longer or heavier left tail. The sign does not strictly determine tail length.
2. **Kurtosis:** No. The excess kurtosis, {_g3.Excess_kurtosis:.4f}, compares tail weight with a normal distribution. The kernel-density peak is a display choice.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Unordered categories
    Skewness and kurtosis use distances from a numeric mean. They are not summaries of an unordered category.
    For `school`, `sex`, `schoolsup`, and `internet`, the relevant descriptive summaries are the counts, the mode, and `describe(include="object")`.
    In that table, `top` is the mode and `freq` is its count.
    """)
    return


@app.cell
def _(data, mo):
    categories = data[["school", "sex", "schoolsup", "internet"]]
    mo.Html(
        categories.describe().to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (categories,)


@app.cell
def _(categories, plt, sns):
    _fig, _axes = plt.subplots(2, 2, figsize=(8, 6))
    for _axis, _column in zip(_axes.ravel(), categories.columns):
        sns.countplot(x=categories[_column], ax=_axis, color="#4C78A8")
        _axis.bar_label(_axis.containers[0], fmt="%.0f", fontsize=8)
        _axis.set(title=f"Mode = {categories[_column].mode().iloc[0]}", xlabel="", ylabel="Count")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Positive moment skewness usually indicates a longer or heavier right tail. Negative moment skewness usually indicates a longer or heavier left tail. The sign does not strictly determine tail length.
    - Skewness near zero means the cubed deviations nearly balance. It does not establish a normal distribution.
    - Pearson's median-based coefficient is an approximation with a different formula.
    - Excess kurtosis compares tail weight with a normal distribution, whose excess kurtosis is 0.
    - Changing only the standard deviation of a normal distribution leaves its excess kurtosis unchanged.
    - A uniform sample is light-tailed. A \(t\) sample with few degrees of freedom is heavy-tailed.
    - Unordered categories are summarized by counts and a mode.

    ## Check your understanding
    1. A sample is [1, 2, 3, 4, 20]. Is its moment skewness positive or negative?
    2. A large normal sample has skewness 0.03. Does that sample have to be exactly symmetric?
    3. What is the population excess kurtosis of every normal distribution?
    4. A density is drawn taller because its standard deviation is smaller. Has its kurtosis increased?
    5. What does `top` report for an object column in `describe`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Sign:** Positive. The value 20 creates the longer right tail.
2. **Finite sample:** No. A symmetric population can produce a small nonzero sample skewness.
3. **Normal kurtosis:** 0.
4. **Scale:** No. The taller peak is the change in scale. Normal excess kurtosis remains 0.
5. **Object summary:** The mode, and `freq` reports how many times that mode occurs.
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
