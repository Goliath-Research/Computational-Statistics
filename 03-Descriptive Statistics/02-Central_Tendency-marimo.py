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
    # Measures of Central Tendency

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Identify the mode, including a distribution with more than one mode.
    - Calculate the arithmetic mean and state how an extreme value moves it.
    - Calculate the median and explain why it need not equal an observed value.
    - Distinguish a quartile from a percentile.
    - Read the median and quartiles in a box plot.

    ## 1. What a central value summarizes
    A measure of central tendency describes one typical value of the recorded observations.
    The three measures in this lesson answer different questions:

    - The **mode** is a most frequent value.
    - The **mean** is the sum of the values divided by how many there are.
    - The **median** is the middle value after sorting.

    Comparisons below describe students in this file. They do not establish why a grade differs between groups.

    ### Try it yourself
    The values are [1, 1, 1, 2, 2, 2, 3].
    1. Which values are modes?
    2. Can a variable have more than one mode?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Modes:** 1 and 2. Each appears three times, and 3 appears once.
2. **More than one mode:** Yes. This sample is bimodal.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The student file
    The code reads `../data/student-mat.csv`. Run the notebook with this lesson folder as the working directory.
    Grades `G1`, `G2`, and `G3` use a 0–20 scale. `studytime` is an ordered category code, not a number of hours.
    """)
    return


@app.cell
def _(mo, pd):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    data = student_file[
        ["school", "sex", "age", "Pstatus", "studytime", "schoolsup", "internet", "G1", "G2", "G3"]
    ].copy()
    print(f"Loaded {len(data)} records.")
    mo.Html(
        data.head().to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mode
    For a categorical variable, the mode is the category with the largest count. The tallest bar shows it.
    If two categories tie for the largest count, both are modes.
    """)
    return


@app.cell
def _(data, mo, pd):
    mode_table = pd.DataFrame({
        "Variable": ["school", "sex", "Pstatus", "schoolsup", "internet"],
        "Mode": [data[column].mode().iloc[0] for column in ["school", "sex", "Pstatus", "schoolsup", "internet"]],
        "Count": [int((data[column] == data[column].mode().iloc[0]).sum()) for column in ["school", "sex", "Pstatus", "schoolsup", "internet"]],
    })
    mo.Html(
        mode_table.to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (mode_table,)


@app.cell
def _(data, plt, sns):
    _fig, _axes = plt.subplots(2, 2, figsize=(7, 5))
    for _axis, _column, _title in zip(
        _axes.ravel(),
        ["sex", "Pstatus", "schoolsup", "internet"],
        ["Sex", "Parent status", "Extra support", "Internet"],
    ):
        sns.countplot(x=data[_column], ax=_axis, color="#4C78A8")
        _axis.bar_label(_axis.containers[0], fmt="%.0f", fontsize=8)
        _axis.set(title=f"{_title}; mode = {data[_column].mode().iloc[0]}", xlabel="", ylabel="Count")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Mean
    For values \(x_1,\ldots,x_n\),

    $$\bar x=\frac{1}{n}\sum_{i=1}^n x_i.$$

    For a finite list of real numbers, the mean lies between the smallest and largest values, including the endpoints when every value is equal.
    One extreme observation can move the mean substantially because every observation enters the sum with equal weight.

    The vertical line on each histogram is that grade's mean.
    """)
    return


@app.cell
def _(data, np, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(8, 3.2), sharey=True)
    for _axis, _grade in zip(_axes, ["G1", "G2", "G3"]):
        _values = data[_grade].to_numpy()
        _mean = _values.mean()
        _axis.hist(_values, bins=10, color="#4C78A8", alpha=0.75, edgecolor="white")
        _axis.axvline(_mean, color="black", linewidth=1.5)
        _axis.set(title=f"{_grade} mean = {_mean:.2f}", xlabel="Grade", ylabel="Count")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Median
    Sort the values. If \(n\) is odd, the median is the middle value.
    If \(n\) is even, the usual sample median is the average of the two middle values, so it need not equal one of the observations.

    The median uses the order of the values. Changing one extreme value to something more extreme does not move the middle in the way it moves the sum.

    In each **Try it yourself** section with a Python task, replace the `None` placeholders in the next code cell.
    Open the notebook in the **marimo editor** to edit and run it.

    ### Try it yourself
    The recorded grades are [10, 12, 14, 15, 12, 16, 17, 18, 16, 20].
    A new value is entered as 180 instead of 18.
    1. Predict which measure moves more, the mean or the median.
    2. Calculate both measures before and after the incorrect value.

    **Write your script in the next cell.** Store the four results in `mean_original`, `median_original`, `mean_with_error`, and `median_with_error`.
    """)
    return


@app.cell
def _(np):
    grades = np.array([10, 12, 14, 15, 12, 16, 17, 18, 16, 20])
    grades_with_error = np.append(grades, 180)
    mean_original = None
    median_original = None
    mean_with_error = None
    median_with_error = None
    print("Original mean and median:", mean_original, median_original)
    print("With 180, mean and median:", mean_with_error, median_with_error)
    return grades, grades_with_error, mean_original, mean_with_error, median_original, median_with_error


@app.cell(hide_code=True)
def _(grades, grades_with_error, mo, np):
    _answers = f"""
1. **Which measure moves more:** The mean. It uses the sum, so replacing the intended 18 with 180 changes the total by 162. The median uses the middle of the ordered list.
2. **Values:** The original mean is {grades.mean():.1f} and the original median is {np.median(grades):.1f}. With 180 included, the mean is {grades_with_error.mean():.1f} and the median is {np.median(grades_with_error):.1f}.

The original list has 10 values, so its median is the average of the 5th and 6th ordered values, 15 and 16. That median, 15.5, is not one of the recorded grades.

```python
mean_original = grades.mean()
median_original = np.median(grades)
mean_with_error = grades_with_error.mean()
median_with_error = np.median(grades_with_error)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Quartiles and percentiles
    **Quartiles** cut the ordered sample into four parts as evenly as the sample size allows.

    - **Q1**, the first quartile, is the 25th percentile: about 25% of the observations are at or below it.
    - **Q2** is the median, the 50th percentile.
    - **Q3**, the third quartile, is the 75th percentile.

    A **percentile** is a cutoff in the ordered values, not one of 100 groups of equal size.
    The 20th percentile is a value at which about 20% of the observations are at or below it.
    With repeated values and a finite sample, several percentile definitions exist. This lesson uses NumPy's default linear interpolation, which matches `Series.quantile` for these grades.
    """)
    return


@app.cell
def _(data, mo, np, pd):
    quartile_rows = []
    for grade_name in ["G1", "G2", "G3"]:
        quartile_rows.append({
            "Grade": grade_name,
            "Q1": np.percentile(data[grade_name], 25),
            "Q2": np.percentile(data[grade_name], 50),
            "Q3": np.percentile(data[grade_name], 75),
        })
    mo.Html(
        pd.DataFrame(quartile_rows).round(2).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    For `G1` in this file:
    1. Calculate the 20th and 80th percentiles.
    2. Are these percentiles categories of students, or cutoffs on the grade scale?

    Store the results in `g1_p20` and `g1_p80`.
    """)
    return


@app.cell
def _(data):
    g1_p20 = None
    g1_p80 = None
    print("20th percentile:", g1_p20)
    print("80th percentile:", g1_p80)
    return g1_p20, g1_p80


@app.cell(hide_code=True)
def _(data, mo, np):
    _p20 = np.percentile(data["G1"], 20)
    _p80 = np.percentile(data["G1"], 80)
    _answers = f"""
1. **Cutoffs:** The 20th percentile is {_p20:.1f}, and the 80th percentile is {_p80:.1f}.
2. **Meaning:** Each is a cutoff on the grade scale. About 20% of the recorded G1 values are at or below {_p20:.1f}.

```python
g1_p20 = np.percentile(data["G1"], 20)
g1_p80 = np.percentile(data["G1"], 80)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Box, boxen, and violin plots
    A standard **box plot** shows five features of the ordered sample:
    - the median, as a line inside the box;
    - Q1 and Q3, as the ends of the box;
    - whiskers extending from the box;
    - points drawn separately when a value falls beyond the whiskers.

    The mean is not part of that standard display. `showmeans=True` adds it.
    Comparing schools with `x` and `hue` repeats the same summaries within each displayed group.

    A **boxen plot** draws more quantiles, so the tails receive more marks than a five-number box plot.
    A **violin plot** adds a kernel density estimate. That smooth curve can extend slightly past the smallest or largest observation; the recorded grades themselves do not.
    """)
    return


@app.cell
def _(data, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(8, 3.2))
    for _axis, _grade in zip(_axes, ["G1", "G2", "G3"]):
        _axis.boxplot(data[_grade], showmeans=True)
        _axis.set(title=_grade, ylabel="Grade")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(data, plt, sns):
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.6))
    sns.boxplot(data=data, x="school", y="G3", hue="sex", ax=_axes[0])
    sns.boxenplot(data=data, x="school", y="G3", hue="sex", ax=_axes[1])
    sns.violinplot(data=data, x="school", y="G3", hue="sex", split=True, ax=_axes[2])
    _axes[0].set_title("Box plot")
    _axes[1].set_title("Boxen plot")
    _axes[2].set_title("Split violin plot")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which line inside the box is the median?
    2. How can the mean be added to a Matplotlib box plot?
    3. Does a difference between two school boxes establish the cause of that difference?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Median:** The line inside the box. In Matplotlib's default style it is orange; the box ends are Q1 and Q3.
2. **Mean:** Pass `showmeans=True` to `boxplot`.
3. **Cause:** No. The plots compare recorded grades in this file.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - The mode is a most frequent value. A sample can have more than one.
    - The mean is the arithmetic average. It lies between the minimum and maximum, and an extreme value can move it.
    - The median is the middle of the ordered values. With an even count, it can fall halfway between two observations.
    - Q1, Q2, and Q3 are the 25th, 50th, and 75th percentiles. A percentile is a cutoff, not a group of equal size.
    - A standard box plot shows the median and quartiles. The mean appears when it is requested.
    - Boxen plots add quantiles, and violin plots add a smoothed density. Grouped plots describe this file's groups.

    ## Check your understanding
    1. For [4, 4, 7, 9], what is the mean?
    2. For [4, 4, 7, 9], what is the median?
    3. Name the modes of [1, 1, 2, 2].
    4. Which box-plot feature is Q2?
    5. Does the 80th percentile name a group of students?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Mean:** (4 + 4 + 7 + 9) / 4 = 6.
2. **Median:** The average of the two middle ordered values, (4 + 7) / 2 = 5.5.
3. **Modes:** 1 and 2.
4. **Q2:** The median, the line inside the box.
5. **Percentile:** No. It is a cutoff in the ordered values.
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
