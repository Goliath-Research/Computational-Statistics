# /// script
# dependencies = [
#     "marimo",
#     "seaborn",
#     "statsmodels",
# ]
# ///

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    lesson_dir = Path(__file__).resolve().parent
    if str(lesson_dir) not in sys.path:
        sys.path.insert(0, str(lesson_dir))
    from goliath_research import resampling as rs

    sns.set_style("whitegrid")
    return lesson_dir, mo, np, pd, plt, rs, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Study Cases for Resampling Tests

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Split a numeric column into groups and pass those groups to a resampling test.
    - Read a two-sample difference and a several-group distance as different statistics.
    - Report a bootstrap percentile interval beside the test decision.
    - Translate a Bonferroni homogeneous-subset list into group labels.
    - Keep a paired grade comparison from being treated as three independent classes.

    ## 1. What the helper computes
    The tests come from `goliath_research.resampling` in this lesson folder.
    Run the notebook with this folder as the working directory so `glass.csv` is found. The student file is read from `../data/student-mat.csv`.

    For two groups, the library's statistic is a difference of means.
    For more than two groups, it is a non-negative dispersion of the group means.
    `get_p_value()` defaults to a two-sided tail. For the dispersion, an upper tail matches the statistic more closely. The tables below use the library default, because that is the call in the source notebook, and they name that choice.
    Each test uses the library's 10,000 resamples. The seed is set at the start of the cell so a rerun of that cell repeats.
    A p-value at or below 0.05 rejects the null at that level. A larger p-value means do not reject it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A table has a numeric column `bill` and a label column `day`.
    1. What should the keys of the group dictionary be?
    2. If `day` has four values, is the library statistic a single difference of two means?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Keys:** The distinct values of `day`. Each value holds the `bill` amounts from that day.
2. **Four days:** No. With more than two groups the library uses a dispersion of all the group means.
""")})
    return


@app.cell
def _(np, pd, rs):
    def grouped_values(frame, value, label):
        groups = {}
        for key, part in frame.groupby(label, observed=True):
            groups[str(key)] = part[value].to_numpy(dtype=float)
        return groups

    def resampling_row(groups, design, method, seed):
        np.random.seed(seed)
        arrays = list(groups.values())
        if design == "independent" and method == "bootstrap":
            model = rs.BootstrapIndependentHT(*arrays)
        elif design == "independent":
            model = rs.PermutationIndependentHT(*arrays)
        elif method == "bootstrap":
            model = rs.BootstrapRelatedHT(*arrays)
        else:
            model = rs.PermutationRelatedHT(*arrays)
        pvalue = float(model.get_p_value())
        return {
            "Method": method,
            "Design": design,
            "Statistic": float(model.get_observed_stat()),
            "p_value": pvalue,
            "Decision_at_0.05": "Reject H0" if pvalue <= 0.05 else "Do not reject H0",
            "model": model,
        }

    def interval_frame(model, labels):
        intervals = model.confidence_intervals()
        return pd.DataFrame({
            "Group": list(labels),
            "Low": intervals[:, 0],
            "High": intervals[:, 1],
            "Midpoint": intervals.mean(axis=1),
        })

    def subset_frame(model, labels, alpha=0.01):
        assigned = {label: "" for label in labels}
        for letter, subset in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", model.get_homogeneous_subsets(alpha)):
            for index in subset:
                assigned[labels[index]] += letter
        return pd.DataFrame({"Group": list(assigned), "Subsets_at_0.01": list(assigned.values())})

    return grouped_values, interval_frame, resampling_row, subset_frame


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Restaurant tips
    The tips table is the example shipped with seaborn. `total_bill` is compared across `sex`, `smoker`, and `day`.
    Sex and smoker have two levels, so the statistic is a difference of means.
    Day has more than two levels, so the statistic is the dispersion.
    The intervals are percentile intervals for each group's mean, from the bootstrap draws stored on the model.
    """)
    return


@app.cell
def _(grouped_values, interval_frame, mo, pd, resampling_row, sns):
    tips = sns.load_dataset("tips")
    tip_rows = []
    tip_intervals = []
    for _offset, (_value, _label) in enumerate([("total_bill", "sex"), ("total_bill", "smoker"), ("total_bill", "day")]):
        _groups = grouped_values(tips, _value, _label)
        _bootstrap = resampling_row(_groups, "independent", "bootstrap", seed=100 + _offset)
        _permutation = resampling_row(_groups, "independent", "permutation", seed=200 + _offset)
        for _result in (_bootstrap, _permutation):
            tip_rows.append({"Comparison": _label, **{k: v for k, v in _result.items() if k != "model"}})
        tip_intervals.append(interval_frame(_bootstrap["model"], _groups.keys()).assign(Comparison=_label))
    tip_results = pd.DataFrame(tip_rows)
    tip_interval_table = pd.concat(tip_intervals, ignore_index=True)
    print(f"Loaded {len(tips)} tips.")
    mo.Html(
        tip_results.round(4).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return tip_interval_table, tip_results, tips


@app.cell
def _(mo, tip_interval_table):
    mo.Html(
        tip_interval_table.round(2).to_html(index=False, border=0, col_space=100)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(plt, tip_interval_table):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    _labels = tip_interval_table["Comparison"] + ": " + tip_interval_table["Group"]
    _ax.hlines(_labels, tip_interval_table["Low"], tip_interval_table["High"], color="#4C78A8")
    _ax.scatter(tip_interval_table["Midpoint"], _labels, color="#4C78A8")
    _ax.set(title="Bootstrap percentile intervals for mean total bill", xlabel="Mean total bill")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which tip comparison uses a dispersion rather than a difference of two means?
    2. If the bootstrap and permutation rows disagree, which hypothesis did each one fail to share?
    """)
    return


@app.cell(hide_code=True)
def _(mo, tip_results):
    _day = tip_results.loc[tip_results["Comparison"] == "day", "Decision_at_0.05"].tolist()
    _answers = f"""
1. **Dispersion:** `day`, because that column has more than two levels.
2. **Disagreement:** They did not fail to share a hypothesis about the mean gap. They share the observed statistic and use different null distributions. For `day`, the decisions in this run are {_day}. Read the table rather than forcing the two methods to copy each other.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Glass types
    `glass.csv` is the glass identification file in this lesson folder.
    The comparison is sodium (`Na`) across glass `Type`.
    There are more than two types, so both tests use the dispersion.
    Homogeneous subsets come from the library's Bonferroni routine at level 0.01.
    Groups that share a letter were not separated at that level. The bootstrap subsets and the permutation subsets need not print the same letters.
    """)
    return


@app.cell
def _(grouped_values, lesson_dir, mo, pd, resampling_row, subset_frame):
    glass = pd.read_csv(lesson_dir / "glass.csv")
    sodium = grouped_values(glass, "Na", "Type")
    glass_bootstrap = resampling_row(sodium, "independent", "bootstrap", seed=11)
    glass_permutation = resampling_row(sodium, "independent", "permutation", seed=12)
    glass_results = pd.DataFrame([
        {k: v for k, v in glass_bootstrap.items() if k != "model"},
        {k: v for k, v in glass_permutation.items() if k != "model"},
    ])
    glass_subsets = subset_frame(glass_bootstrap["model"], list(sodium)).merge(
        subset_frame(glass_permutation["model"], list(sodium)),
        on="Group",
        suffixes=("_bootstrap", "_permutation"),
    )
    print(f"Loaded {len(glass)} glass rows and {len(sodium)} types.")
    mo.Html(
        glass_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return glass_results, glass_subsets


@app.cell
def _(glass_subsets, mo):
    mo.Html(
        glass_subsets.to_html(index=False, border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Two glass types share the letter A and one of them also has the letter B. Were those two types separated at level 0.01?
    2. Why might the permutation letters differ from the bootstrap letters?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Shared letter:** No. A shared letter means the routine placed them in a subset that was not split at that level.
2. **Different letters:** The two routines build different null distributions. Bonferroni is applied to each routine's pairwise p-values, so the resulting subsets can differ.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Student grades
    `G1`, `G2`, and `G3` are three grades for the same student. The comparison is related, not a comparison of three independent classes.
    The code reads `../data/student-mat.csv`.
    The subset table is a multiple-comparison summary. It does not replace the single three-grade test in the first table.
    """)
    return


@app.cell
def _(interval_frame, mo, np, pd, resampling_row, subset_frame):
    students = pd.read_csv("../data/student-mat.csv", sep=";")
    grades = {
        "G1": students["G1"].to_numpy(dtype=float),
        "G2": students["G2"].to_numpy(dtype=float),
        "G3": students["G3"].to_numpy(dtype=float),
    }
    np.random.seed(21)
    grade_bootstrap = resampling_row(grades, "related", "bootstrap", seed=21)
    grade_permutation = resampling_row(grades, "related", "permutation", seed=22)
    grade_results = pd.DataFrame([
        {k: v for k, v in grade_bootstrap.items() if k != "model"},
        {k: v for k, v in grade_permutation.items() if k != "model"},
    ])
    grade_subsets = subset_frame(grade_bootstrap["model"], list(grades)).merge(
        subset_frame(grade_permutation["model"], list(grades)),
        on="Group",
        suffixes=("_bootstrap", "_permutation"),
    )
    grade_intervals = interval_frame(grade_bootstrap["model"], grades)
    print(f"Loaded {len(students)} students.")
    mo.Html(
        grade_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return grade_intervals, grade_results, grade_subsets


@app.cell
def _(grade_intervals, grade_subsets, mo):
    mo.vstack([
        mo.Html(
            grade_subsets.to_html(index=False, border=0, col_space=140)
            .replace("<table ", '<table style="width: auto;" ')
        ),
        mo.Html(
            grade_intervals.round(3).to_html(index=False, border=0, col_space=100)
            .replace("<table ", '<table style="width: auto;" ')
        ),
    ])
    return


@app.cell
def _(grade_intervals, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.hlines(grade_intervals["Group"], grade_intervals["Low"], grade_intervals["High"], color="#4C78A8")
    _ax.scatter(grade_intervals["Midpoint"], grade_intervals["Group"], color="#4C78A8", zorder=3)
    _ax.set(title="Bootstrap percentile intervals for the grade means", xlabel="Mean grade")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why are the three grades a related sample?
    2. A subset label puts G2 with G1 and also with G3, while G1 and G3 do not share a letter. What pattern is that?
    """)
    return


@app.cell(hide_code=True)
def _(grade_subsets, mo):
    _answers = f"""
1. **Related:** Each row is one student, so the three grades are repeated measurements on that student.
2. **Middle group:** G2 can sit between G1 and G3. It may share a subset with each neighbor while the two ends do not share a subset with each other. The letters from this run are in the subset table:

```
{grade_subsets.to_string(index=False)}
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - The resampling helpers test a difference of means for two groups and a dispersion of means for more than two.
    - The reported p-value is the library default, a two-sided tail. For a dispersion, that is not the same number as an upper-tail p-value.
    - A bootstrap percentile interval describes one group's mean. It is not the hypothesis test.
    - Bonferroni subsets are a pairwise follow-up. A shared letter means those groups were not separated at the stated level.
    - Bootstrap and permutation answers can differ because their nulls differ.
    - Restaurant bills, glass types, and grades answer different sampling designs. Grades stay paired by student.
    - A p-value above 0.05 means do not reject the null for that statistic. It does not prove the groups have the same mean.

    ## Check your understanding
    1. Which file in this lesson is read from the lesson folder rather than from `../data`?
    2. What changes in the library statistic when a third group is added?
    3. What do the low and high columns of an interval table estimate?
    4. Two groups share a Bonferroni letter. Were they declared different at that level?
    5. Why is a test of G1, G2, and G3 not three independent samples of students?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Local file:** `glass.csv`. The student file is `../data/student-mat.csv`.
2. **Third group:** The statistic changes from a difference of two means to a dispersion of all the means.
3. **Interval:** Each row is a percentile interval for that group's mean, built from the group's bootstrap means.
4. **Shared letter:** No. The shared letter marks a subset that the correction did not split.
5. **Same students:** The three columns are repeated grades for the students in the file.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Cortez, P., and Silva, A. (2008). Using data mining to predict secondary school student performance. In A. Brito and J. Teixeira (Eds.), *Proceedings of 5th Future Business Technology Conference*. The student file is described by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).
    - German, B. (1987). Glass identification data. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/42/glass+identification).
    """)
    return


if __name__ == "__main__":
    app.run()
