# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.25.1"

app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import scipy.stats as st

    lesson_dir = Path(__file__).resolve().parent
    sns.set_style("whitegrid")
    return lesson_dir, mo, np, pd, plt, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlation

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Describe the direction and strength of a linear association with a correlation coefficient.
    - Match a scatter of points to a strong, weak, positive, or negative coefficient.
    - Explain why a curved pattern can have a correlation near zero.
    - State the hypotheses of a test that the linear correlation is zero, and read the p-value beside the coefficient.
    - Contrast Pearson's \(r\) with Spearman's rank correlation and Kendall's \(\tau\).
    - Say what a correlation leaves unsettled, including cause.

    The examples below are teaching data. A coefficient summarizes the pairs in front of you. It does not, by itself, identify a cause or a policy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What a correlation describes
    A **correlation coefficient** is a single number, usually between \(-1\) and \(1\), that describes how two quantitative variables move together in a set of pairs.

    - A **positive** coefficient means that larger values of one variable tend to arrive with larger values of the other.
    - A **negative** coefficient means that larger values of one tend to arrive with smaller values of the other.
    - A coefficient **near zero** means that a straight line through the cloud would be nearly flat. The points can still form a curve, or two clumps, or some other pattern a straight line cannot see.

    Strength and sign answer different questions. A coefficient of \(0.95\) and a coefficient of \(-0.95\) are equally strong. They point in opposite directions.

    The number is about the **pairs that were measured**. Two variables can move together because one influences the other, because the influence runs the other way, or because a third factor moves both. The coefficient does not choose among those explanations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. A store records daily temperature and daily sales of cold drinks. The correlation is \(0.8\). What does the sign say, and what does it leave open?
    2. Why is \(-0.9\) a strong association rather than a weak one?
    3. If a coefficient is \(0.02\), what kind of pattern might the scatter still contain?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Sign:** Higher temperatures tend to arrive with higher cold-drink sales in these records. **What stays open:** The coefficient does not show that heat causes the sales. A holiday, a promotion, or the school calendar could move both series.
2. **Strength:** The size is \(0.9\), close to the maximum of \(1\). The minus sign is the direction, not a penalty on the strength.
3. **A flat line is not an empty plot:** A U-shape, a cluster of groups, or a relationship that appears only after a threshold can all produce a coefficient near zero. The scatter is part of the answer.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Four linear patterns
    A straight-line pattern can be built on purpose. Draw an \(x\), then build \(y\) as a multiple of \(x\) plus noise. The multiple sets the direction and the steepness. The noise loosens the line.

    The next cell uses a seeded generator, `default_rng(2026)`, so the same 40 points appear every time the cell runs. These series are created for this section. They are not read from a file.
    """)
    return


@app.cell
def _(mo, np, pd):
    _rng = np.random.default_rng(2026)
    _n = 40
    _x = _rng.uniform(-3, 3, size=_n)
    _noise_strong_pos, _noise_strong_neg, _noise_weak_pos, _noise_weak_neg = (
        _rng.normal(0, 1, size=_n) for _ in range(4)
    )
    patterns = pd.DataFrame(
        {
            "x": _x,
            "strong_positive": 1.6 * _x + 0.45 * _noise_strong_pos,
            "strong_negative": -1.4 * _x + 0.45 * _noise_strong_neg,
            "weak_positive": 1.0 * _x + 2.0 * _noise_weak_pos,
            "weak_negative": -0.9 * _x + 1.9 * _noise_weak_neg,
        }
    )
    mo.Html(patterns.head().to_html(index=False, border=0))
    return (patterns,)


@app.cell
def _(np, patterns, plt):
    _specs = [
        ("strong_positive", "Strong positive", "#E45756"),
        ("strong_negative", "Strong negative", "#4C78A8"),
        ("weak_positive", "Weak positive", "#F58518"),
        ("weak_negative", "Weak negative", "#54A24B"),
    ]
    _x = patterns["x"].to_numpy()
    _fig, _axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey=False)
    for _ax, (_key, _title, _color) in zip(_axes.ravel(), _specs):
        _y = patterns[_key].to_numpy()
        _ax.scatter(_x, _y, s=28, color=_color, edgecolor="white", linewidth=0.4)
        _slope, _intercept = np.polyfit(_x, _y, 1)
        _order = np.argsort(_x)
        _ax.plot(_x[_order], _slope * _x[_order] + _intercept, color="#333333", linewidth=1.4)
        _r = float(np.corrcoef(_x, _y)[0, 1])
        _ax.set(title=f"{_title} (r = {_r:.2f})", xlabel="x", ylabel="y")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo, np, patterns, pd):
    _x = patterns["x"].to_numpy()
    _rows = []
    for _key in ["strong_positive", "strong_negative", "weak_positive", "weak_negative"]:
        _rows.append({"Series": _key, "Pearson r": float(np.corrcoef(_x, patterns[_key])[0, 1])})
    pattern_correlations = pd.DataFrame(_rows)
    mo.Html(pattern_correlations.round(3).to_html(index=False, border=0))
    return (pattern_correlations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The five pairs below are a smaller version of the same idea. In the next cell, replace `None` with the Pearson correlation of `example_x` and `example_y`.

    `np.corrcoef(example_x, example_y)[0, 1]` returns that coefficient. Run the cell after you edit it.

    Then use the four panels above.
    4. Which series should have a coefficient close to \(-1\), and which feature of the recipe made it negative?
    5. The weak series use the same x values as the strong series. What was changed to loosen the line?
    6. If you change the seed and rebuild the noise, which part of the recipe stays fixed, and which part changes?
    """)
    return


@app.cell
def _(np):
    example_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    example_y = np.array([2.0, 1.0, 4.0, 3.0, 6.0])
    print("x:", example_x)
    print("y:", example_y)
    return example_x, example_y


@app.cell
def _(example_x, example_y):
    example_r = None
    print("Pearson r:", example_r)
    return (example_r,)


@app.cell(hide_code=True)
def _(example_x, example_y, mo, np, pattern_correlations):
    _r = float(np.corrcoef(example_x, example_y)[0, 1])
    _negative = pattern_correlations.loc[
        pattern_correlations["Series"] == "strong_negative", "Pearson r"
    ].iloc[0]
    _weak = pattern_correlations.loc[
        pattern_correlations["Series"] == "weak_positive", "Pearson r"
    ].iloc[0]
    _answers = f"""
1. **Coefficient:** {_r:.3f}.
2. **Sign:** Positive. The points wobble, and the larger x values still tend to have the larger y values.
3. **Code:** `example_r = np.corrcoef(example_x, example_y)[0, 1]`
4. **Strong negative:** The series `strong_negative`, with Pearson r = {_negative:.3f}. It is negative because y was built as a **negative** multiple of x, plus a little noise.
5. **What loosened the line:** Each series has its own noise, and the weak series use a larger noise scale relative to the multiple of x (2.0 against a multiple of 1, versus 0.45 against a multiple of 1.6). The weak positive coefficient in this run is {_weak:.3f}.
6. **Seed:** The formulas stay fixed: which series is positive, which is negative, and how large the noise term is. A new seed draws a new set of points, so the coefficient moves a little. The same seed repeats the same points.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A curve is not "no relationship"
    Pearson's \(r\) answers a narrow question: how well does a **straight line** describe the pairs?

    The course file `corrExamples.csv` sits in this lesson's folder. It holds 30 synthetic pairs used again in the bootstrap and permutation lessons. Two of its columns are easy to misread if you look only at the coefficient:

    - `y_nc` is unstructured noise, loosely attached to `x`.
    - `y_nc2` is a **curve**: the square of `x`, plus a little vertical noise.

    The short names were chosen when the file was saved. "nc" is not a verdict you should trust until you have seen the scatter. This lesson reads the file and does not rewrite it.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd):
    examples = pd.read_csv(lesson_dir / "corrExamples.csv")
    print("Rows:", len(examples), "  Columns:", ", ".join(examples.columns))
    mo.Html(examples.head().round(3).to_html(index=False, border=0))
    return (examples,)


@app.cell
def _(examples, np, plt):
    _x = examples["x"].to_numpy()
    _fig, _axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharex=True)
    for _ax, _column, _title, _color in [
        (_axes[0], "y_nc", "Unstructured cloud", "#4C78A8"),
        (_axes[1], "y_nc2", "Curved pattern", "#E45756"),
    ]:
        _y = examples[_column].to_numpy()
        _ax.scatter(_x, _y, s=32, color=_color, edgecolor="white", linewidth=0.4)
        _slope, _intercept = np.polyfit(_x, _y, 1)
        _order = np.argsort(_x)
        _ax.plot(_x[_order], _slope * _x[_order] + _intercept, color="#333333", linewidth=1.4)
        _r = float(np.corrcoef(_x, _y)[0, 1])
        _ax.set(title=f"{_title} (r = {_r:.2f})", xlabel="x", ylabel=_column)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gray line on each panel is the best straight line for those pairs. On the curve, that line is a poor summary: the points fall and then rise. A correlation near zero reports the failure of the straight line. It does not report that x and `y_nc2` are unrelated. Here y was built from \(x^2\).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why can `y_nc2` show a clear pattern and still have a Pearson correlation near zero?
    2. What would you miss if you ranked the columns by the coefficient and never opened the scatter?
    3. The file has 30 rows. Does a correlation computed from those rows describe every future x that someone might choose?
    """)
    return


@app.cell(hide_code=True)
def _(examples, mo, np):
    _curve_r = float(np.corrcoef(examples["x"], examples["y_nc2"])[0, 1])
    _cloud_r = float(np.corrcoef(examples["x"], examples["y_nc"])[0, 1])
    _answers = f"""
1. **The line is the wrong shape:** `y_nc2` follows a curve, close to the square of x. Pearson's r measures straight-line association, so this run's coefficient is only {_curve_r:.3f}.
2. **A ranking hides the picture:** The unstructured column `y_nc` has r = {_cloud_r:.3f}. A table of coefficients would place it near the curve. The scatters show a cloud in one panel and a U-shape in the other.
3. **Scope:** The coefficient describes these 30 saved pairs. A new sample, or x values far outside this range, can behave differently.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Pearson's \(r\) and a hypothesis test
    For pairs \((x_i, y_i)\), Pearson's \(r\) is the average product of the standardized values. Subtract each variable's mean, divide by its standard deviation, multiply the pairs, and average the products. The result is \(-1\), \(0\), or \(1\) when the points sit on a falling line, show no straight-line association, or sit on a rising line. Values in between describe a noisier line.

    A test asks whether a coefficient as far from zero as the one we observed would be surprising if the **population** linear correlation were zero.

    For each column in the course file, the hypotheses are:

    - \(H_0\): the population linear correlation with `x` is \(0\).
    - \(H_a\): the population linear correlation with `x` is not \(0\).

    The p-value is the evidence against \(H_0\), under the sampling model used by `scipy.stats.pearsonr`. We will use a 5% significance level. A small p-value does not turn a modest coefficient into a strong one, and it does not establish a cause. These columns were simulated, so a small p-value mostly confirms that the straight line we built is still visible in a sample of 30.
    """)
    return


@app.cell
def _(examples, mo, pd, st):
    _rows = []
    for _column in ["y_spc", "y_snc", "y_wpc", "y_wnc", "y_nc", "y_nc2"]:
        _result = st.pearsonr(examples["x"], examples[_column])
        _rows.append(
            {
                "Column": _column,
                "Pearson r": float(_result.statistic),
                "p-value": float(_result.pvalue),
            }
        )
    pearson_table = pd.DataFrame(_rows)
    pearson_table["Reject H0 at 5%"] = pearson_table["p-value"] < 0.05
    mo.Html(pearson_table.round(4).to_html(index=False, border=0))
    return (pearson_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the strong columns and the weak columns as different lessons. `y_spc` and `y_snc` have coefficients near \(1\) and \(-1\), and tiny p-values: the line is both strong and easy to distinguish from zero in this sample. `y_wpc` and `y_wnc` have coefficients near \(0.4\) in size. Their p-values fall just under 0.05, so the test rejects \(H_0\) at 5%, while the association remains modest. Significance is a statement about evidence against zero, not a synonym for "large."

    `y_nc` and `y_nc2` do not reject \(H_0\) at 5%. For the curve, that decision is about the straight line. Section 3 already showed the pattern the line misses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Replace `None` with the **column name** from `pearson_table` that has the most negative Pearson correlation. The names are strings, for example `"y_spc"`.

    Then answer these questions about the tests.
    4. Write \(H_0\) and \(H_a\) for the test reported on `y_wpc`.
    5. `y_wpc` rejects \(H_0\) at the 5% level. Why is that not the same as calling the association strong?
    6. The p-value for `y_nc2` is large. Which mistake would you make if you told someone the variables are unrelated?
    """)
    return


@app.cell
def _():
    strongest_negative_column = None
    print("Most negative column:", strongest_negative_column)
    return (strongest_negative_column,)


@app.cell(hide_code=True)
def _(mo, pearson_table):
    _row = pearson_table.loc[pearson_table["Pearson r"].idxmin()]
    _weak = pearson_table.loc[pearson_table["Column"] == "y_wpc"].iloc[0]
    _curve = pearson_table.loc[pearson_table["Column"] == "y_nc2"].iloc[0]
    _answers = rf"""
1. **Column:** `{_row["Column"]}`.
2. **Coefficient:** {_row["Pearson r"]:.3f}.
3. **Code:** `strongest_negative_column = "{_row["Column"]}"`
4. **Hypotheses:** \(H_0\): the population linear correlation between `x` and `y_wpc` is 0. \(H_a\): that correlation is not 0.
5. **Evidence is not strength:** The p-value is {_weak["p-value"]:.3f}, so at the 5% level the sample is surprising if the linear correlation were zero. The coefficient itself is only {_weak["Pearson r"]:.3f}. A detectable line can still be a loose one.
6. **The curve is still there:** The Pearson p-value for `y_nc2` is {_curve["p-value"]:.3f}, so this test does not reject a zero **linear** correlation. The scatter in Section 3 shows a curved relationship. "No straight-line signal" is not "no relationship."
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Ranks: Spearman and Kendall
    Pearson's \(r\) uses the values themselves, so a few very large observations can pull it. Two rank correlations first replace the values by their order in the sample.

    - **Spearman's rank correlation** is Pearson's \(r\) applied to the ranks. It describes monotone association: as one variable increases, the other tends to increase, or tends to decrease, without requiring a straight line of constant slope.
    - **Kendall's \(\tau\)** looks at pairs of observations. A pair is concordant when the two variables put the observations in the same order, and discordant when they put them in opposite orders. \(\tau\) compares the counts of those pairs.

    Both still lie between \(-1\) and \(1\). A symmetric U-shape is not monotone, so ranks do not rescue `y_nc2`. The left arm falls while the right arm rises. Rank methods answer a different question from Pearson's; they do not detect every picture a scatter can show.

    The hypotheses for each test below are the rank versions of the same idea:

    - \(H_0\): there is no monotone association (the population rank correlation is 0).
    - \(H_a\): there is a monotone association.

    The 5% level is the same rule as in Section 4. The p-value is not a measure of how steep the pattern is.
    """)
    return


@app.cell
def _(examples, mo, pd, st):
    _rows = []
    for _column in ["y_spc", "y_snc", "y_wpc", "y_wnc", "y_nc", "y_nc2"]:
        _spearman = st.spearmanr(examples["x"], examples[_column])
        _kendall = st.kendalltau(examples["x"], examples[_column])
        _rows.append(
            {
                "Column": _column,
                "Spearman": float(_spearman.statistic),
                "Spearman p": float(_spearman.pvalue),
                "Kendall": float(_kendall.statistic),
                "Kendall p": float(_kendall.pvalue),
            }
        )
    rank_table = pd.DataFrame(_rows)
    mo.Html(rank_table.round(4).to_html(index=False, border=0))
    return (rank_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare this table with the Pearson table. The strong lines stay strong, because a steep line is monotone. The curve stays near zero on both rank correlations, because a U-shape is not monotone. The weak lines are the subtle row: Spearman does not reject \(H_0\) at 5% for `y_wpc` or `y_wnc`, even though Pearson did. With 30 points and a loose cloud, the choice of coefficient can change the 5% decision. That is a reason to report the coefficient and the picture, not only a reject-or-not label.

    Ranks are easier to see in a tiny grade book. The next cell uses nine fixed students, not a random draw. History and algebra move together. History and physical education do not.
    """)
    return


@app.cell
def _(mo, np, pd):
    history_grades = np.array([35, 23, 47, 17, 10, 43, 9, 6, 28])
    algebra_grades = np.array([30, 33, 45, 23, 8, 49, 12, 4, 31])
    phys_ed_grades = np.array([15, 23, 27, 17, 30, 14, 19, 36, 28])
    grades = pd.DataFrame(
        {
            "History": history_grades,
            "Algebra": algebra_grades,
            "Physical education": phys_ed_grades,
        }
    )
    mo.Html(grades.to_html(index=False, border=0))
    return algebra_grades, grades, history_grades, phys_ed_grades


@app.cell
def _(algebra_grades, history_grades, phys_ed_grades, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    _axes[0].scatter(history_grades, algebra_grades, s=40, color="#E45756", edgecolor="white")
    _axes[0].set(title="History and algebra", xlabel="History", ylabel="Algebra")
    _axes[1].scatter(history_grades, phys_ed_grades, s=40, color="#4C78A8", edgecolor="white")
    _axes[1].set(title="History and physical education", xlabel="History", ylabel="Physical education")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Replace `None` with Spearman's rank correlation between `history_grades` and `algebra_grades`.

    `st.spearmanr(history_grades, algebra_grades).statistic` is the coefficient.

    Then answer these questions about ranks.
    4. Why does Spearman's coefficient for `y_nc2` stay near zero even though the scatter is a clear curve?
    5. For `y_wpc`, Pearson rejects a zero linear correlation at 5% and Spearman does not. What should you report so a reader can see that disagreement?
    6. The grade book has nine students. What limitation does that sample size put on the history-and-algebra test?
    """)
    return


@app.cell
def _(algebra_grades, history_grades):
    history_algebra_spearman = None
    print("History grades:", history_grades)
    print("Algebra grades:", algebra_grades)
    print("Spearman r:", history_algebra_spearman)
    return (history_algebra_spearman,)


@app.cell(hide_code=True)
def _(algebra_grades, history_grades, mo, rank_table, st):
    _result = st.spearmanr(history_grades, algebra_grades)
    _curve = rank_table.loc[rank_table["Column"] == "y_nc2"].iloc[0]
    _weak = rank_table.loc[rank_table["Column"] == "y_wpc"].iloc[0]
    _answers = f"""
1. **Spearman r:** {_result.statistic:.3f}.
2. **p-value:** {_result.pvalue:.4f}. With only nine students, treat this as a description of this grade book plus a small-sample test, not as a finding about every class.
3. **Code:** `history_algebra_spearman = st.spearmanr(history_grades, algebra_grades).statistic`
4. **Not monotone:** Along a symmetric U-shape, y falls as x increases and then rises. Ranks reward a relationship that keeps heading one way. Spearman's coefficient for `y_nc2` is {_curve["Spearman"]:.3f}.
5. **Show the conflict:** Report both coefficients, both p-values, and the scatter. For `y_wpc`, Spearman is {_weak["Spearman"]:.3f} with p-value {_weak["Spearman p"]:.3f}. A single "significant" label would hide the fact that the other coefficient does not reject at 5%.
6. **Nine pairs:** A large rank correlation in a class of nine is a description of those nine students. It is thin evidence about students who were not in the table.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Three variables, one picture
    A correlation table is a list of pairs. A scatterplot matrix shows every pair at once, which makes it harder to miss a curve or a single unusual point.

    The next figure uses the nine-student grade book. Each diagonal panel is a histogram of one subject. Each off-diagonal panel is the scatter of two subjects. Read the history-and-algebra panel against the history-and-physical-education panel before you trust the coefficients alone.
    """)
    return


@app.cell
def _(grades, plt, sns):
    _grid = sns.pairplot(
        grades,
        diag_kind="hist",
        height=1.8,
        plot_kws={"s": 28, "color": "#4C78A8", "edgecolor": "white"},
    )
    _fig = _grid.figure
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(algebra_grades, history_grades, mo, pd, phys_ed_grades, st):
    _rows = []
    for _name, _values in [("Algebra", algebra_grades), ("Physical education", phys_ed_grades)]:
        _pearson = st.pearsonr(history_grades, _values)
        _spearman = st.spearmanr(history_grades, _values)
        _rows.append(
            {
                "Pair with history": _name,
                "Pearson r": float(_pearson.statistic),
                "Pearson p": float(_pearson.pvalue),
                "Spearman": float(_spearman.statistic),
                "Spearman p": float(_spearman.pvalue),
            }
        )
    grade_correlations = pd.DataFrame(_rows)
    mo.Html(grade_correlations.round(3).to_html(index=False, border=0))
    return (grade_correlations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. From the matrix, which subject moves with history, and which scatter looks unstructured?
    2. The physical-education correlation is negative but its p-value is large. What decision does the 5% rule give, and what does that decision not prove?
    3. Why is the scatterplot matrix still useful after the table has already reported the coefficients?
    """)
    return


@app.cell(hide_code=True)
def _(grade_correlations, mo):
    _algebra = grade_correlations.loc[grade_correlations["Pair with history"] == "Algebra"].iloc[0]
    _pe = grade_correlations.loc[grade_correlations["Pair with history"] == "Physical education"].iloc[0]
    _answers = f"""
1. **The picture:** Algebra moves with history (Pearson r = {_algebra["Pearson r"]:.3f}). The history-and-physical-education scatter does not show a tight line.
2. **Do not reject:** The Pearson p-value for physical education is {_pe["Pearson p"]:.3f}, which is above 0.05, so this test does not reject a zero linear correlation. That is not proof that the correlation is exactly zero, and it is not proof that the two subjects are unrelated in every class.
3. **The matrix can show what the number compresses:** One unusual student, a clump, or a bend would change how you read a coefficient of {_algebra["Pearson r"]:.2f}. The table cannot display that shape.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    A correlation coefficient is a compact description of paired numbers. The sign is the direction of a straight-line or monotone pattern. The absolute value is the strength. A test of "zero association" then asks whether a coefficient of that size would be surprising if there were no linear, or no monotone, association in the population the sample represents.

    Three distinctions keep the number honest.

    - **Shape.** Pearson's \(r\) sees a straight line. Spearman's coefficient and Kendall's \(\tau\) see a monotone pattern. A symmetric curve can hide from all three. The scatter is not optional.
    - **Evidence versus strength.** In the course file, the weak positive column rejects a zero Pearson correlation at 5% with a coefficient near \(0.4\). The strong columns have coefficients near \(1\) and \(-1\). Both facts belong in the sentence.
    - **Cause.** Grades can move together because they record the same students. Temperature and drink sales can move together because of the season. The coefficient records the co-movement. It does not name the mechanism, and it does not predict what would happen if someone changed only one variable.

    The later lessons keep these pairs and ask a different question: how far the coefficient might move if we drew a new sample of the same size, and how a permutation test builds a world in which the pairing is arbitrary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. A classmate says, "The p-value is 0.03, so the variables are strongly related." What two separate facts should they report instead?
    2. You need to know whether a relationship keeps going in one direction, and a few extreme values worry you. Which coefficient is built for that question?
    3. What is one explanation of a large correlation that is not "x causes y"?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Two facts:** Report the coefficient, so a reader can see the strength and the sign, and report the p-value, so a reader can see the evidence against a zero association. A p-value of 0.03 can sit next to a modest coefficient, as `y_wpc` does.
2. **Ranks:** Spearman's rank correlation, or Kendall's \(\tau\), describes monotone association and replaces the raw values by their order. A curve that rises and then falls still needs the scatter.
3. **Another explanation:** A third factor can move both variables. The two variables can also be repeated measures on the same people, as history grades and algebra grades are.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer. Correlation and regression.
    - SciPy documentation: `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and `scipy.stats.kendalltau`.
    """)
    return


if __name__ == "__main__":
    app.run()
