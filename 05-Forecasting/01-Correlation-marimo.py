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
    - Read a coefficient together with its scatter, and with the p-value when a test is reported.

    The examples below are teaching data. A coefficient summarizes the pairs in front of you.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What a correlation describes
    A **correlation coefficient** is a single number, usually between \(-1\) and \(1\), that describes how two quantitative variables move together in a set of pairs.

    - A **positive** coefficient means that larger values of one variable tend to arrive with larger values of the other.
    - A **negative** coefficient means that larger values of one tend to arrive with smaller values of the other.
    - A **Pearson coefficient near zero** indicates little linear association. The scatter may still show a curve, two clumps, or another nonlinear pattern.

    Strength and sign answer different questions. A coefficient of \(0.95\) and a coefficient of \(-0.95\) are equally strong. They point in opposite directions. The number describes the **pairs that were measured**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. A store records daily temperature and daily sales of cold drinks. The correlation is \(0.8\). Do higher temperatures tend to arrive with higher sales or with lower sales in these records?
    2. Why is \(-0.9\) a strong association rather than a weak one?
    3. If a coefficient is \(0.02\), name one pattern the scatter might still show.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Direction:** Higher. A positive coefficient means larger temperatures tend to arrive with larger cold-drink sales in these records.
2. **Strength:** The size is \(0.9\), close to the maximum of \(1\). The minus sign is the direction, not a penalty on the strength.
3. **A coefficient near zero does not mean no pattern:** A U-shape, or two separate clumps, can produce a coefficient near zero. The scatter is part of the answer.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Four linear patterns
    A straight-line pattern can be built on purpose. Draw an \(x\), then build \(y\) as a multiple of \(x\) plus noise. The multiple sets the direction and the steepness. The noise loosens the line. Strength labels are descriptive and depend on context; here, “weaker” means weaker than the two strong examples.

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
    mo.Html(patterns.head().round(4).to_html(index=False, border=0))
    return (patterns,)


@app.cell
def _(np, patterns, plt):
    _specs = [
        ("strong_positive", "Strong positive", "#E45756"),
        ("strong_negative", "Strong negative", "#4C78A8"),
        ("weak_positive", "Weaker positive", "#F58518"),
        ("weak_negative", "Weaker negative", "#54A24B"),
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
        _ax.set(title=f"{_title} (r = {_r:.3f})", xlabel="x", ylabel="y")
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
    mo.Html(pattern_correlations.round(4).to_html(index=False, border=0))
    return (pattern_correlations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Use the four panels and the table above.
    1. Which series has a coefficient close to \(-1\)?
    2. What in the recipe made that series negative?
    3. The weak series use the same x values as the strong series. What was changed to loosen the line?
    """)
    return


@app.cell(hide_code=True)
def _(mo, pattern_correlations):
    _negative = pattern_correlations.loc[
        pattern_correlations["Series"] == "strong_negative", "Pearson r"
    ].iloc[0]
    _weak = pattern_correlations.loc[
        pattern_correlations["Series"] == "weak_positive", "Pearson r"
    ].iloc[0]
    _answers = f"""
1. **Series:** `strong_negative`, with Pearson r = {_negative:.3f}.
2. **Sign of the recipe:** y was built as a negative multiple of x, plus a little noise.
3. **Noise:** The weak series use a larger noise scale relative to the multiple of x (2.0 against a multiple of 1, versus 0.45 against a multiple of 1.6). The weak positive coefficient in this run is {_weak:.3f}.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Using `patterns`, calculate Pearson’s correlation for `x` with `strong_positive` and with `weak_positive`. Replace each `None` in the next cell with your code. Which association is stronger?
    """)
    return


@app.cell
def _():
    student_strong_result = None
    print("student_strong_result:", student_strong_result)
    student_weak_result = None
    print("student_weak_result:", student_weak_result)
    return (student_strong_result, student_weak_result)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_strong_result = st.pearsonr(patterns["x"], patterns["strong_positive"])
student_weak_result = st.pearsonr(patterns["x"], patterns["weak_positive"])
```
The coefficients are approximately 0.9837 and 0.4386. The strong-positive series has the larger absolute coefficient. Each result also contains a p-value.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A curve is not "no relationship"
    Pearson's \(r\) answers a narrow question: how well does a **straight line** describe the pairs?

    The course file `corrExamples.csv` holds 30 synthetic pairs used again in the bootstrap and permutation lessons. The next cell generates these data using the original Jupyter recipe and seed, then automatically saves them in this lesson's folder as `corrExamples.csv` if that file is missing. An existing file is preserved without prompting. Two of its columns are easy to misread if you look only at the coefficient:

    - `y_nc` is generated independently of `x`; its sample correlation need not be exactly zero.
    - `y_nc2` is a **curve**: the square of `x`, plus a little vertical noise.

    The short names were chosen when the file was saved. "nc" is not a verdict you should trust until you have seen the scatter. The analysis below always uses the original generated data in memory. If an existing CSV has different contents, a message explains that it was preserved; later notebooks will use the saved file.
    """)
    return


@app.cell
def _(lesson_dir, mo, np, pd):
    _csv_path = lesson_dir / "corrExamples.csv"
    # Preserve the original Jupyter dataset without changing NumPy's global seed.
    _rng = np.random.RandomState(30)
    _x = _rng.uniform(-3, 3, size=30)
    _noise = _rng.uniform(3, 5, size=30)
    _noise_weak = _rng.uniform(10, 30, size=30)
    _saved_examples = pd.DataFrame(
        {
            "x": _x,
            "y_spc": 1.5 * _x + _noise,
            "y_snc": -0.9 * _x + _noise + 30,
            "y_wpc": 1.7 * _x + _noise_weak,
            "y_wnc": -1.7 * _x + 1.8 * _noise_weak,
            "y_nc": _rng.uniform(10, 20, size=30) * _noise,
            "y_nc2": _x * _x + _noise,
        }
    )
    if not _csv_path.exists():
        _saved_examples.to_csv(_csv_path, index=False)
        print("Created:", _csv_path.name)
    else:
        print("Preserved existing file:", _csv_path.name)
        try:
            _existing = pd.read_csv(_csv_path)
            _same = (
                list(_existing.columns) == list(_saved_examples.columns)
                and _existing.shape == _saved_examples.shape
                and np.allclose(_existing.to_numpy(dtype=float), _saved_examples.to_numpy(), rtol=1e-12, atol=1e-12)
            )
        except (OSError, ValueError, TypeError, pd.errors.ParserError):
            _same = False
        if not _same:
            print("The existing CSV differs from this lesson's generated data or could not be read. It was not overwritten; later notebooks will use that existing file.")
    examples = _saved_examples
    print("Rows:", len(examples), "  Columns:", ", ".join(examples.columns))
    mo.Html(examples.head().round(4).to_html(index=False, border=0))
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
        _ax.set(title=f"{_title} (r = {_r:.3f})", xlabel="x", ylabel=_column)
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
    1. Why can `y_nc2` show a clear curve and still have a Pearson correlation near zero?
    2. The unstructured column and the curve can have similar coefficients. What do the two scatters show that those two numbers do not?
    3. The file has 30 rows. Does the coefficient describe those 30 pairs, or every x someone might choose later?
    """)
    return


@app.cell(hide_code=True)
def _(examples, mo, np):
    _curve_r = float(np.corrcoef(examples["x"], examples["y_nc2"])[0, 1])
    _cloud_r = float(np.corrcoef(examples["x"], examples["y_nc"])[0, 1])
    _answers = f"""
1. **The line is the wrong shape:** `y_nc2` follows a curve, close to the square of x. Pearson's r measures straight-line association, so this run's coefficient is only {_curve_r:.3f}.
2. **Two pictures:** The unstructured column `y_nc` has r = {_cloud_r:.3f}. One scatter is a cloud. The other is a U-shape.
3. **These 30 pairs:** The coefficient describes the saved rows. A new sample can differ.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate Pearson’s correlation for `examples["x"]` and `examples["y_nc2"]`. Replace `None` with your code. Explain why a coefficient near zero does not rule out the relationship shown in the scatter.
    """)
    return


@app.cell
def _():
    student_curve_result = None
    print("student_curve_result:", student_curve_result)
    return (student_curve_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_curve_result = st.pearsonr(examples["x"], examples["y_nc2"])
```
The coefficient is approximately 0.1414. Pearson summarizes linear association; the scatter shows a U-shaped relationship.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Pearson's \(r\) and a hypothesis test
    For pairs \((x_i, y_i)\), Pearson's \(r\) is the sample covariance divided by the product of the sample standard deviations. The result is \(-1\), \(0\), or \(1\) when the points sit on a falling line, show no straight-line association, or sit on a rising line. Values in between describe a noisier line.

    A test asks whether a coefficient as far from zero as the one we observed would be surprising if the **population** linear correlation were zero.

    For each column in the course file, the hypotheses are:

    - \(H_0\): the population linear correlation with `x` is \(0\).
    - \(H_a\): the population linear correlation with `x` is not \(0\).

    The p-value is the probability, under \(H_0\) and the test assumptions, of obtaining a correlation at least as far from zero as the observed one. Computing Pearson's coefficient does not require normality. The default p-value from `scipy.stats.pearsonr` uses a normal-theory null distribution for independent observation pairs; its exact justification assumes a bivariate normal population. These synthetic examples illustrate the calculation, rather than establish that assumption. We will use a 5% significance level. A small p-value does not turn a modest coefficient into a strong one. These columns were simulated, so a small p-value mostly confirms that the straight line we built is still visible in a sample of 30.
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
    All numeric tables use `.round(4)` for consistent display. A p-value shown as `0.0000` is smaller than the displayed precision; it does **not** mean the p-value is exactly zero. Calculations and the 5% decisions use the original, unrounded values.

    Read the strong columns and the weak columns as different lessons. `y_spc` and `y_snc` have coefficients near \(1\) and \(-1\), and tiny p-values: the line is both strong and easy to distinguish from zero in this sample. `y_wpc` and `y_wnc` have coefficients near \(0.4\) in size. Their p-values fall just under 0.05, so the test rejects \(H_0\) at 5%, while the association remains modest. Significance is a statement about evidence against zero, not a synonym for "large."

    `y_nc` and `y_nc2` do not reject \(H_0\) at 5%. For the curve, that decision is about the straight line. Section 3 already showed the pattern the line misses.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Use `pearson_table`.
    1. Which column has the most negative Pearson correlation?
    2. Write \(H_0\) and \(H_a\) for the test reported on `y_wpc`.
    3. `y_wpc` rejects \(H_0\) at the 5% level. Is its coefficient close to \(1\), or only modest?
    """)
    return


@app.cell(hide_code=True)
def _(mo, pearson_table):
    _row = pearson_table.loc[pearson_table["Pearson r"].idxmin()]
    _weak = pearson_table.loc[pearson_table["Column"] == "y_wpc"].iloc[0]
    _answers = rf"""
1. **Column:** `{_row["Column"]}`, with Pearson r = {_row["Pearson r"]:.3f}.
2. **Hypotheses:** \(H_0\): the population linear correlation between `x` and `y_wpc` is 0. \(H_a\): that correlation is not 0.
3. **Modest:** The coefficient is {_weak["Pearson r"]:.3f}. The p-value is {_weak["p-value"]:.3f}, so the sample is surprising if the linear correlation were zero. A detectable line can still be a loose one.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. The p-value for `y_nc2` is large. Does this test reject a zero linear correlation at 5%?
    2. Section 3 showed a curve for `y_nc2`. Is "no straight-line signal" the same sentence as "no pattern in the scatter"?
    """)
    return


@app.cell(hide_code=True)
def _(mo, pearson_table):
    _curve = pearson_table.loc[pearson_table["Column"] == "y_nc2"].iloc[0]
    _answers = f"""
1. **Do not reject:** The Pearson p-value for `y_nc2` is {_curve["p-value"]:.3f}, which is above 0.05.
2. **No:** The test is about a straight line. The scatter is still a curve.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Test whether the population linear correlation between `x` and `y_wpc` is zero. Use the arrays in `examples`. Replace `None` with the SciPy call, state both hypotheses, and interpret the coefficient and p-value separately at the 5% significance level.
    """)
    return


@app.cell
def _():
    student_pearson_result = None
    print("student_pearson_result:", student_pearson_result)
    return (student_pearson_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_pearson_result = st.pearsonr(examples["x"], examples["y_wpc"])
```
\(H_0: \rho=0\); \(H_a: \rho\ne0\). The coefficient is approximately 0.3758 and the p-value is 0.0407. Reject the null hypothesis at 5%, subject to the test assumptions. The sample association remains modest.
''')}, lazy=True)
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

    - \(H_0\): the population rank correlation is \(0\).
    - \(H_a\): the population rank correlation is not \(0\).

    The 5% level is the same rule as in Section 4. A zero rank correlation does not rule out every relationship. The p-value is not a measure of how steep the pattern is. SciPy's default Spearman p-value is approximate; for samples as small as the 30 pairs and nine students here, permutation testing is preferable for inference. The later permutation lesson develops that approach.
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
    Compare this table with the Pearson table. The strong linear patterns stay strong because their observations largely preserve the same or opposite order. Steepness alone does not determine correlation strength. The curve stays near zero on both rank correlations, because a U-shape is not monotone. The weak lines are the subtle row: Spearman does not reject \(H_0\) at 5% for `y_wpc` or `y_wnc`, even though Pearson did. With 30 points and a loose cloud, the choice of coefficient can change the 5% decision. That is a reason to report the coefficient and the picture, not only a reject-or-not label.

    Ranks are easier to see in a tiny grade book. The next cell uses nine fixed students, not a random draw. History and algebra show a strong positive sample association. History and physical education show a weaker negative sample association; their correlation tests do not reject zero at the 5% level. Failure to reject does not establish that there is no association.
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
    Use the rank table and the grade-book scatter.
    1. Why does Spearman's coefficient for `y_nc2` stay near zero even though the scatter is a clear curve?
    2. For `y_wpc`, Pearson rejects a zero linear correlation at 5% and Spearman does not. Which two numbers should you report for Spearman?
    3. The grade book has nine students. Does the history-and-algebra scatter describe those nine students, or every class?
    """)
    return


@app.cell(hide_code=True)
def _(mo, rank_table):
    _curve = rank_table.loc[rank_table["Column"] == "y_nc2"].iloc[0]
    _weak = rank_table.loc[rank_table["Column"] == "y_wpc"].iloc[0]
    _answers = f"""
1. **Not monotone:** Along a symmetric U-shape, y falls and then rises. Ranks reward a relationship that keeps heading one way. Spearman's coefficient for `y_nc2` is {_curve["Spearman"]:.3f}.
2. **Coefficient and p-value:** Spearman is {_weak["Spearman"]:.3f} with p-value {_weak["Spearman p"]:.3f}. Report both, plus the scatter.
3. **These nine:** The scatter describes the nine rows in the table.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate Spearman’s and Kendall’s correlations between history and algebra grades. Replace the two `None` values with your code. Compare their signs and explain what each coefficient measures.
    """)
    return


@app.cell
def _():
    student_spearman_result = None
    print("student_spearman_result:", student_spearman_result)
    student_kendall_result = None
    print("student_kendall_result:", student_kendall_result)
    return (student_spearman_result, student_kendall_result)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_spearman_result = st.spearmanr(history_grades, algebra_grades)
student_kendall_result = st.kendalltau(history_grades, algebra_grades)
```
Spearman’s coefficient is 0.9000 (approximate p-value 0.0009); Kendall’s coefficient is 0.7222 (p-value 0.0059). Both describe positive ordinal association. Spearman correlates ranks; Kendall compares concordant and discordant pairs. Their magnitudes need not match. With nine observations, prefer permutation inference for Spearman.
''')}, lazy=True)
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
        _kendall = st.kendalltau(history_grades, _values)
        _rows.append(
            {
                "Pair with history": _name,
                "Pearson r": float(_pearson.statistic),
                "Pearson p": float(_pearson.pvalue),
                "Spearman": float(_spearman.statistic),
                "Spearman p": float(_spearman.pvalue),
                "Kendall": float(_kendall.statistic),
                "Kendall p": float(_kendall.pvalue),
            }
        )
    grade_correlations = pd.DataFrame(_rows)
    mo.Html(grade_correlations.round(4).to_html(index=False, border=0))
    return (grade_correlations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. From the matrix, which subject moves with history?
    2. Which pair shows a weaker negative association?
    3. The physical-education Pearson p-value is above 0.05. Does the 5% rule reject a zero linear correlation?
    """)
    return


@app.cell(hide_code=True)
def _(grade_correlations, mo):
    _algebra = grade_correlations.loc[grade_correlations["Pair with history"] == "Algebra"].iloc[0]
    _pe = grade_correlations.loc[grade_correlations["Pair with history"] == "Physical education"].iloc[0]
    _answers = f"""
1. **Algebra:** Pearson r = {_algebra["Pearson r"]:.3f}. The history-and-algebra panel rises.
2. **Physical education:** The sample association is negative, with Pearson r = {_pe["Pearson r"]:.3f}; the points are more dispersed than for history and algebra.
3. **Do not reject:** The Pearson p-value is {_pe["Pearson p"]:.3f}, which is above 0.05.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return





@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Using `grades`, create a scatterplot matrix with histograms on the diagonal. Replace `None` with the Seaborn call. Compare the history–algebra panel with the history–physical-education panel.
    """)
    return


@app.cell
def _():
    student_pairplot = None
    print("student_pairplot:", student_pairplot)
    return (student_pairplot,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_pairplot = sns.pairplot(grades, diag_kind="hist")
```
History and algebra show a strong positive sample association. History and physical education show a weaker negative sample association; their correlation tests do not reject zero at 5%.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    A correlation coefficient is a compact description of paired numbers. The sign is the direction of a straight-line or monotone pattern. The absolute value is the strength. A test of zero population correlation then asks whether a coefficient at least that far from zero would be surprising under the null hypothesis and the test assumptions. A zero coefficient does not rule out every form of dependence.

    Three distinctions keep the number honest.

    - **Shape.** Pearson's \(r\) sees a straight line. Spearman's coefficient and Kendall's \(\tau\) see a monotone pattern. A symmetric U-shaped relationship can have all three population coefficients equal to zero. The scatter is not optional.
    - **Evidence versus strength.** In the course file, the weak positive column rejects a zero Pearson correlation at 5% with a coefficient near \(0.4\). The strong columns have coefficients near \(1\) and \(-1\). Both facts belong in the sentence.
    - **Cause.** Grades can move together because factors such as study habits affect both subjects. Temperature and drink sales can move together because of the season. The coefficient records the co-movement. It does not name the mechanism, and it does not predict what would happen if someone changed only one variable.

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
3. **Another explanation:** A third factor can move both variables. For example, study habits can affect both history and algebra grades. Measuring both grades on the same students establishes the pairing; it does not itself explain the correlation.
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
