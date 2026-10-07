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
    # Measures of Frequency

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Distinguish a count from a relative frequency.
    - Read a one-way frequency table and a two-way table.
    - Explain why a bar chart needs a zero baseline.
    - Interpret a normalized cross-tabulation by naming its denominator.
    - Use a histogram for a numeric measurement and a bar chart for categories.
    - Explain what changing histogram bins does and does not change.

    ## 1. Counts and relative frequencies
    A **frequency distribution** assigns every observation to one class and records how many observations fall in each class.
    The classes must not overlap, and together they must cover the observations being summarized.

    For categories such as school or internet access, each category is a class.
    For a numeric measurement, a histogram groups values into bins.

    - A **count** is the number of observations in a class.
    - A **relative frequency** is that count divided by the number of observations.

    $$\text{relative frequency}=\frac{\text{count}}{n}.$$

    Counts add to \(n\). Relative frequencies add to 1, apart from rounding.
    These summaries describe the recorded file. They do not by themselves describe every student.

    ### Try it yourself
    Six quiz scores are [10, 10, 12, 12, 12, 15].
    1. What is the count of 12?
    2. What is the relative frequency of 12?
    3. Do the counts add to 6, and do the relative frequencies add to 1?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Count of 12:** 3.
2. **Relative frequency:** 3/6 = 0.5.
3. **Totals:** The counts 2, 3, and 1 add to 6. The relative frequencies 2/6, 3/6, and 1/6 add to 1.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The student file
    We use `student-mat.csv` from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).
    The code reads `../data/student-mat.csv`. Run the notebook with this lesson folder as the working directory.

    The retained variables are:
    - `school`: GP or MS.
    - `sex`: F or M.
    - `age`: age in years, from 15 to 22 in this file.
    - `Pstatus`: parents living together (`T`) or apart (`A`).
    - `studytime`: ordered categories 1 (<2 hours), 2 (2–5 hours), 3 (5–10 hours), and 4 (>10 hours). These are codes, not exact hours.
    - `schoolsup`: extra educational support, yes or no.
    - `internet`: internet access at home, yes or no.
    - `G1`, `G2`, `G3`: first-period, second-period, and final grades, on a 0–20 scale.
    """)
    return


@app.cell
def _(mo, pd):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    data = student_file[
        ["school", "sex", "age", "Pstatus", "studytime", "schoolsup", "internet", "G1", "G2", "G3"]
    ].copy()
    print(f"Loaded {len(data)} records and {data.shape[1]} columns.")
    mo.Html(
        data.head().to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. One categorical variable
    `value_counts()` gives the count in each category.
    Dividing by the number of rows gives the relative frequency.

    A **bar chart** compares counts by bar length. Its baseline must be zero: a nonzero baseline changes the visual comparison.
    A **pie chart** uses angles to show parts of one whole. Those angles are harder to compare than bar lengths, so this lesson uses bars as the main display.
    """)
    return


@app.cell
def _(data, mo, pd):
    school_counts = data["school"].value_counts()
    school_table = pd.DataFrame({
        "Count": school_counts,
        "Relative_frequency": school_counts / len(data),
    })
    mo.Html(
        school_table.round(4).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (school_table,)


@app.cell
def _(plt, school_table):
    _fig, _ax = plt.subplots(figsize=(5, 3.5))
    school_table["Count"].plot(kind="bar", ax=_ax, rot=0, color=["#4C78A8", "#F58518"])
    _ax.set(title="Students by school", xlabel="School", ylabel="Count")
    _ax.bar_label(_ax.containers[0], fmt="%.0f")
    _ax.set_ylim(0, None)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In each **Try it yourself** section with a Python task, write your script in the code cell immediately below the instructions.
    Replace the `None` placeholders and run the cell. Open this notebook in the **marimo editor** to edit these cells.

    ### Try it yourself
    Using `data`, calculate the count and relative frequency of each `internet` category.
    1. Which category is more common?
    2. What is the relative frequency of `yes`?

    **Write your script in the next cell:**
    - Store the counts in `internet_counts`.
    - Store the relative frequencies in `internet_relative`.
    - Print both results.
    """)
    return


@app.cell
def _(data):
    internet_counts = None
    internet_relative = None
    print("Counts:")
    print(internet_counts)
    print("Relative frequencies:")
    print(internet_relative)
    return internet_counts, internet_relative


@app.cell(hide_code=True)
def _(data, mo):
    _yes = int((data["internet"] == "yes").sum())
    _no = int((data["internet"] == "no").sum())
    _n = len(data)
    _answers = f"""
1. **More common category:** `yes`, with {_yes} students. `no` has {_no}.
2. **Relative frequency of yes:** {_yes}/{_n} = {_yes / _n:.4f}.

```python
internet_counts = data["internet"].value_counts()
internet_relative = internet_counts / len(data)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Two categorical variables
    A **cross-tabulation** counts observations in each combination of two variables.
    `margins=True` adds the row and column totals.

    There are two different normalizations:
    - `normalize=True` divides every cell by the grand total. Each value is a proportion of all students.
    - `normalize="index"` divides each row by that row's total. Each value is a proportion within one school.

    The denominator changes the meaning. A heatmap colors those numbers; it does not create a new summary.
    """)
    return


@app.cell
def _(data, pd):
    school_internet_counts = pd.crosstab(data["school"], data["internet"], margins=True)
    school_internet_within_school = pd.crosstab(data["school"], data["internet"], normalize="index")
    return school_internet_counts, school_internet_within_school


@app.cell(hide_code=True)
def _(mo):
    mo.md("**Counts**")
    return


@app.cell
def _(mo, school_internet_counts):
    mo.Html(
        school_internet_counts.to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("**Proportion within each school**")
    return


@app.cell
def _(mo, school_internet_within_school):
    mo.Html(
        school_internet_within_school.round(4).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(plt, school_internet_within_school, sns):
    _fig, _ax = plt.subplots(figsize=(5, 3.2))
    sns.heatmap(school_internet_within_school, annot=True, fmt=".3f", cmap="Blues", ax=_ax)
    _ax.set(title="Internet access within each school", xlabel="Internet", ylabel="School")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Use the printed tables.
    1. What is the denominator of the GP row in the within-school table?
    2. Which school has the higher proportion of students with internet access?
    3. Does that comparison use the joint table divided by 395, or the within-school table?
    """)
    return


@app.cell(hide_code=True)
def _(mo, school_internet_counts, school_internet_within_school):
    _gp_yes = school_internet_within_school.loc["GP", "yes"]
    _ms_yes = school_internet_within_school.loc["MS", "yes"]
    _answers = f"""
1. **GP denominator:** {int(school_internet_counts.loc["GP", "All"])}, the number of GP students.
2. **Higher proportion:** GP, {_gp_yes:.4f}, compared with MS, {_ms_yes:.4f}.
3. **Table:** The within-school table. Dividing by all 395 students answers a different question: the proportion of the whole file in each school-and-internet combination.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Numeric measurements
    A **histogram** counts numeric values in bins. The bin width and the number of bins change the picture, not the observations.

    `studytime` has four ordered codes. A bar chart of those codes matches the categories directly.
    A kernel density estimate draws a smooth curve from the observations. It is an estimate of shape, not a table of counts.
    """)
    return


@app.cell
def _(mo):
    age_bins = mo.ui.slider(start=4, stop=16, step=1, value=8, show_value=True, label="Age histogram bins")
    age_bins
    return (age_bins,)


@app.cell
def _(age_bins, data, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.4))
    _axes[0].hist(data["age"], bins=age_bins.value, color="#4C78A8", edgecolor="white")
    _axes[0].set(title="Age", xlabel="Age", ylabel="Count")
    data["studytime"].value_counts().sort_index().plot(kind="bar", ax=_axes[1], rot=0, color="#54A24B")
    _axes[1].set(title="Study-time category", xlabel="Category", ylabel="Count")
    _axes[1].set_ylim(0, None)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(data, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for _axis, _grade in zip(_axes, ["G1", "G2", "G3"]):
        _axis.hist(data[_grade], bins=10, color="#4C78A8", edgecolor="white")
        _axis.set(title=_grade, xlabel="Grade", ylabel="Count")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Move the age-bin slider. Do any student's recorded age change?
    2. Why is a bar chart a direct display for `studytime`?
    3. Is a kernel density curve a count of students in a grade interval?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Bins:** No. The slider changes how ages are grouped in the picture. The recorded ages stay the same.
2. **Study time:** The variable has four ordered categories. Each bar is the count of one category.
3. **Density curve:** No. It is a smoothed estimate of shape. The histogram bars are the counts.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A frequency distribution reports the count in each nonoverlapping class.
    - Relative frequency equals that count divided by the number of observations.
    - Bar length is easier to compare when the baseline is zero. Pie angles show parts of one whole.
    - A cross-tabulation needs an explicit denominator: all observations, or one row, or one column.
    - Histogram bins change the display of a numeric variable. They do not change the observations.
    - Ordered categories are summarized directly by category counts. A density curve is not a frequency table.

    ## Check your understanding
    1. Twenty students belong to a category. The file has 200 students. What is the relative frequency?
    2. A bar chart starts at 10 instead of 0. What comparison becomes misleading?
    3. In a within-school internet table, what is the denominator for school MS?
    4. Does changing the number of histogram bins change a student's age?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Relative frequency:** 20/200 = 0.10.
2. **Baseline:** Differences in bar length no longer match differences in count.
3. **MS denominator:** The number of students recorded at school MS.
4. **Bins:** No. Only the grouping in the display changes.
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
