# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.25.1"

app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns

    lesson_dir = Path(__file__).resolve().parent
    if str(lesson_dir) not in sys.path:
        sys.path.insert(0, str(lesson_dir))
    from goliath_research import paired_resampling as rs

    sns.set_style("whitegrid")
    return lesson_dir, mo, np, pd, plt, rs, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootstrap Confidence Intervals for Correlation

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Treat a correlation as one estimate from one sample of pairs.
    - Resample **pairs**, so each x stays with its own y.
    - Build a percentile confidence interval from those resampled correlations.
    - Interpret the estimate, interval width, and whether the interval includes zero, without treating the bootstrap distribution as a null distribution.
    - Apply the interval to the course file, to student grades, and to the employee file.
    - Explain the approximate repeated-sampling meaning and limitations of a 95% confidence interval.

    This lesson estimates a population Pearson correlation and its uncertainty using paired bootstrap percentile confidence intervals. The following permutation lesson covers hypothesis testing and p-values under an independence null.

    Every interval below uses 4,000 paired resamples and seed 2026. The seed makes the table repeatable. All numeric tables use `.round(4)` for display; calculations use unrounded values. More resamples shrink the Monte Carlo wobble in the interval endpoints. They do not add observations to the original sample.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One number from one sample
    `corrExamples.csv` has 30 synthetic pairs. The column `y_spc` was built as a strong positive line plus noise. Pearson's \(r\) for `x` and `y_spc` is one number computed from these 30 rows.

    Ordinary paired bootstrapping assumes independent, identically distributed observational pairs. Clustering, repeated measurements across rows, and time dependence require a resampling design that respects that structure. Remove rows with missing values first; the helper does not check for them.

    A new draw of 30 pairs from the same process would produce a different coefficient. The bootstrap is a way to ask how far that coefficient tends to move, using only the pairs we already have.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd, rs):
    examples = pd.read_csv(lesson_dir / "corrExamples.csv")
    observed_strong = rs.pearson_r(examples["x"], examples["y_spc"])
    print(f"Observed Pearson r for x and y_spc: {observed_strong:.3f}")
    print("Rows:", len(examples))
    mo.Html(examples[["x", "y_spc"]].head().round(4).to_html(index=False, border=0))
    return examples, observed_strong


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The printed coefficient for `x` and `y_spc` is about 0.983.
    1. If we treat 0.983 as more than a description of these 30 rows, what population quantity is being estimated?
    2. Why would a second sample of 30 pairs from the same process not give exactly 0.983?
    3. Does 0.983 tell you the noise scale that was used to build `y_spc`?
    """)
    return


@app.cell(hide_code=True)
def _(mo, observed_strong):
    _answers = f"""
1. **The target:** A population linear correlation: the correlation we would see if we could average over the whole process that produced pairs like these, not only these 30 rows.
2. **Sampling:** Each sample draws different noise and a different set of x values. The formula for r then returns a different number.
3. **No:** {observed_strong:.3f} says the points hug a rising line. It does not recover the noise scale.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Using `examples`, calculate Pearson’s coefficient between `x` and `y_snc` with `rs.pearson_r`. Replace `None` with your code. Interpret its sign and magnitude.
    """)
    return


@app.cell
def _():
    student_observed_r = None
    print("student_observed_r:", student_observed_r)
    return (student_observed_r,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_observed_r = rs.pearson_r(examples["x"], examples["y_snc"])
```
The coefficient is approximately −0.9496: a strong negative sample linear association.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Resample pairs, not separate columns
    A bootstrap copy of the sample draws **rows** with replacement. If row 7 is drawn, both its x and its y come along. Some rows appear more than once. Some rows are left out. Each copy has the same length as the original sample. We recompute Pearson's \(r\) on that copy. Doing this 4,000 times builds a bootstrap distribution of the coefficient.

    Drawing a new x from the x column and, separately, a new y from the y column answers a different question. That breaks the pairs. The correlation in those broken samples sits near zero even when the original correlation is large, because the matching that created the line has been destroyed. That cloud is closer to a null distribution than to a picture of how the observed coefficient varies. Independent-column resampling simulates independence, a stronger condition than zero Pearson correlation. This lesson does not use it for the interval. `bootstrap_pearson` keeps each pair together.

    For these examples, the observed coefficient sits near the **center** of a paired bootstrap distribution. Bootstrap distributions can also be biased or skewed, so this is not a universal centering guarantee. This distribution estimates sampling uncertainty rather than imposing a zero-correlation null.
    """)
    return


@app.cell
def _(examples, rs):
    strong_bootstrap = rs.bootstrap_pearson(examples["x"], examples["y_spc"], num_samples=4_000, seed=2026)
    print("Bootstrap replicates:", strong_bootstrap.size)
    print(f"Bootstrap mean: {strong_bootstrap.mean():.3f}")
    return (strong_bootstrap,)


@app.cell
def _(observed_strong, plt, rs, sns, strong_bootstrap):
    _low, _high = rs.percentile_interval(strong_bootstrap, confidence=95)
    _fig, _ax = plt.subplots(figsize=(6.2, 2.8))
    sns.kdeplot(strong_bootstrap, fill=True, color="#4C78A8", ax=_ax)
    _ax.axvline(observed_strong, color="#E45756", linewidth=2, label=f"Observed r = {observed_strong:.3f}")
    _ax.axvline(_low, color="#333333", linestyle="--", label=f"95% interval [{_low:.3f}, {_high:.3f}]")
    _ax.axvline(_high, color="#333333", linestyle="--")
    _ax.set(title="Paired bootstrap distribution of the correlation", xlabel="Pearson r", ylabel="Density")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why must a bootstrap copy take x and y from the same row?
    2. Why is the observed correlation near the middle of the blue curve, rather than out in a tail?
    3. The dashed lines on the figure are the 95% percentile interval. Do they include 0?
    """)
    return


@app.cell(hide_code=True)
def _(mo, observed_strong, rs, strong_bootstrap):
    _low, _high = rs.percentile_interval(strong_bootstrap, confidence=95)
    _answers = f"""
1. **The pair is the observation:** The correlation is a fact about matched pairs. A row is one pair.
2. **Re-estimates cluster:** The bootstrap copies re-estimate the same coefficient. They cluster around the original estimate, {observed_strong:.3f}. The bootstrap mean in this run is {strong_bootstrap.mean():.3f}.
3. **No:** The interval is ({_low:.3f}, {_high:.3f}). Both endpoints are positive.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. If someone resampled the x column and the y column separately, would the pairs stay intact?
    2. Where would those correlations sit: near the observed correlation, or near 0?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **No:** Separate draws invent matches that never occurred.
2. **Near 0:** Breaking the pairs destroys the association. That cloud is not the sampling uncertainty of the coefficient we observed.
""")
        },
        lazy=True,
    )
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Generate 4,000 paired bootstrap correlations for `x` and `y_snc`, using seed 2026. Replace `None` with your code. Around which observed coefficient should the distribution cluster?
    """)
    return


@app.cell
def _():
    student_negative_bootstrap = None
    print("student_negative_bootstrap:", student_negative_bootstrap)
    return (student_negative_bootstrap,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_negative_bootstrap = rs.bootstrap_pearson(examples["x"], examples["y_snc"], num_samples=4_000, seed=2026)
```
The distribution should cluster near the observed coefficient, approximately −0.9496. The resampling preserves each matched pair.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A percentile interval
    If a resampled column is constant, that copy has no correlation; the helper leaves it out of the interval. This does not happen with the data in this lesson.

    Sort the 4,000 bootstrap correlations. The 95% percentile interval runs from the 2.5th percentile to the 97.5th percentile. About 95% of the bootstrap coefficients fall between those two endpoints.

    Read the estimate and interval together. The sign and absolute value of Pearson's r describe the estimated direction and strength of linear association. The interval endpoints show the range of population correlations compatible with this approximate procedure; its width describes estimated uncertainty.

    An interval entirely above zero supports a positive population linear correlation; an interval entirely below zero supports a negative one. An interval containing zero does not exclude zero correlation, but does not prove zero correlation or independence. With only 30 rows, a coefficient near 0.4 can still have an interval containing zero. Each interval is pointwise; the tables do not provide simultaneous 95% coverage across all comparisons.

    A 95% confidence level describes intended long-run coverage: if independent samples were repeatedly drawn from the same population and an interval constructed for each, approximately 95% would contain the fixed population correlation when the method performs as intended. It does not assign a 95% probability to the population correlation lying in this particular interval.

    Percentile intervals are approximate, and small samples, influential observations, bias, or skewness can impair coverage. Increasing the number of bootstrap resamples reduces simulation variability; it does not fix an unrepresentative original sample or guarantee coverage. Excluding zero can be interpreted as an approximate interval-based test, but this lesson reports intervals rather than bootstrap p-values. It need not agree with a separate permutation test.

    Thirty rows is a small sample; read the interval as an approximate summary.
    """)
    return


@app.cell
def _(examples, pd, rs):
    _rows = []
    for _column, _label in [
        ("y_spc", "Strong positive"),
        ("y_snc", "Strong negative"),
        ("y_wpc", "Weak positive"),
        ("y_wnc", "Weak negative"),
        ("y_nc", "Unstructured"),
        ("y_nc2", "Curved"),
    ]:
        _observed = rs.pearson_r(examples["x"], examples[_column])
        _low, _high = rs.percentile_interval(rs.bootstrap_pearson(examples["x"], examples[_column], num_samples=4_000, seed=2026))
        _rows.append(
            {
                "Column": _column,
                "Pattern": _label,
                "Pearson r": _observed,
                "95% low": _low,
                "95% high": _high,
                "Excludes 0": bool(_low > 0 or _high < 0),
            }
        )
    example_intervals = pd.DataFrame(_rows)
    return (example_intervals,)


@app.cell
def _(example_intervals, mo):
    mo.Html(example_intervals.round(4).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. For the strong negative column, the interval lies entirely below zero. What does that interval suggest about the direction of the population linear correlation?
    2. The curved column's interval covers zero. Does that rule out a zero straight-line correlation, or a curve?
    3. The weak positive coefficient is near 0.4. Does its 95% interval still cover 0?
    """)
    return


@app.cell(hide_code=True)
def _(example_intervals, mo):
    _strong = example_intervals.loc[example_intervals["Column"] == "y_snc"].iloc[0]
    _weak = example_intervals.loc[example_intervals["Column"] == "y_wpc"].iloc[0]
    _curve = example_intervals.loc[example_intervals["Column"] == "y_nc2"].iloc[0]
    _answers = f"""
1. **Direction:** The interval ({_strong["95% low"]:.3f}, {_strong["95% high"]:.3f}) excludes 0, so these resamples are hard to reconcile with a population linear correlation of 0. The estimate itself is {_strong["Pearson r"]:.3f}, a strong negative line.
2. **The straight line only:** The interval ({_curve["95% low"]:.3f}, {_curve["95% high"]:.3f}) covers 0, so the bootstrap does not rule out a zero linear correlation. It does not decide whether a curve is present.
3. **Yes:** `y_wpc` has r = {_weak["Pearson r"]:.3f} and interval ({_weak["95% low"]:.3f}, {_weak["95% high"]:.3f}). With 30 rows, a modest line can still have an interval that covers 0.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    For `x` and `y_wpc`, generate 4,000 paired bootstrap correlations with seed 2026, then calculate the 95% percentile interval. Replace both `None` values with your code. Does the interval exclude zero?
    """)
    return


@app.cell
def _():
    student_weak_bootstrap = None
    print("student_weak_bootstrap:", student_weak_bootstrap)
    student_weak_interval = None
    print("student_weak_interval:", student_weak_interval)
    return (student_weak_bootstrap, student_weak_interval)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_weak_bootstrap = rs.bootstrap_pearson(examples["x"], examples["y_wpc"], num_samples=4_000, seed=2026)
student_weak_interval = rs.percentile_interval(student_weak_bootstrap, confidence=95)
```
For the original course data, the interval is approximately (−0.0214, 0.6923). It includes zero; this does not prove that the population correlation equals zero.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Student grades
    `student-mat.csv` records 395 students in a mathematics course. `G1`, `G2`, and `G3` are the first-period grade, the second-period grade, and the final grade, each on a 0–20 scale. The file is semicolon-separated. Each row is one student, so the pair `(G1, G3)` is two grades from the same person.

    Question, before the intervals: are earlier grades linearly associated with later grades in the population these students represent?

    Each row is one student. A large positive Pearson correlation indicates that students with higher grades in one period tend to have higher grades in another, with an approximately linear association.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd):
    students = pd.read_csv(lesson_dir / "student-mat.csv", sep=";")
    print("Students:", len(students))
    mo.Html(students[["G1", "G2", "G3"]].head().to_html(index=False, border=0))
    return (students,)


@app.cell
def _(mo, pd, rs, students):
    _rows = []
    for _left, _right in [("G1", "G2"), ("G2", "G3"), ("G1", "G3")]:
        _observed = rs.pearson_r(students[_left], students[_right])
        _low, _high = rs.percentile_interval(rs.bootstrap_pearson(students[_left], students[_right], num_samples=4_000, seed=2026))
        _rows.append(
            {
                "Pair": f"{_left} and {_right}",
                "Pearson r": _observed,
                "95% low": _low,
                "95% high": _high,
                "Excludes 0": bool(_low > 0 or _high < 0),
            }
        )
    grade_intervals = pd.DataFrame(_rows)
    mo.Html(grade_intervals.round(4).to_html(index=False, border=0))
    return (grade_intervals,)


@app.cell
def _(plt, sns, students):
    _fig, _ax = plt.subplots(figsize=(5.8, 2.8))
    sns.scatterplot(data=students, x="G2", y="G3", s=28, color="#4C78A8", edgecolor="white", ax=_ax)
    _ax.set(title="Final grade against the second-period grade", xlabel="G2", ylabel="G3")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Use `grade_intervals`.
    1. Which pair has the largest Pearson correlation?
    2. Why is the row, rather than a randomly matched pair of grades, the right unit to resample?
    3. These students are one mathematics course in the file. Does the interval describe every mathematics student?
    """)
    return


@app.cell(hide_code=True)
def _(grade_intervals, mo):
    _row = grade_intervals.loc[grade_intervals["Pearson r"].idxmax()]
    _g2g3 = grade_intervals.loc[grade_intervals["Pair"] == "G2 and G3"].iloc[0]
    _answers = f"""
1. **Pair:** {_row["Pair"]}, with r = {_row["Pearson r"]:.3f} and 95% interval ({_row["95% low"]:.3f}, {_row["95% high"]:.3f}).
2. **The student is the pair:** G2 and G3 on the same row are two measurements of one person. Resampling rows keeps that person intact. G2 and G3 have r = {_g2g3["Pearson r"]:.3f}.
3. **No:** The interval describes the linear association in the population this course file can represent.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the observed Pearson coefficient and a 95% paired bootstrap interval for `students["G1"]` and `students["G3"]`. Use 4,000 resamples and seed 2026. Replace each `None` with your code, then interpret the estimate and interval separately.
    """)
    return


@app.cell
def _():
    student_grade_r = None
    print("student_grade_r:", student_grade_r)
    student_grade_bootstrap = None
    print("student_grade_bootstrap:", student_grade_bootstrap)
    student_grade_interval = None
    print("student_grade_interval:", student_grade_interval)
    return (student_grade_r, student_grade_bootstrap, student_grade_interval)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_grade_r = rs.pearson_r(students["G1"], students["G3"])
student_grade_bootstrap = rs.bootstrap_pearson(students["G1"], students["G3"], num_samples=4_000, seed=2026)
student_grade_interval = rs.percentile_interval(student_grade_bootstrap, confidence=95)
```
The coefficient describes sample linear association. The interval describes uncertainty in the corresponding population correlation under the sampling assumptions. Check whether zero lies inside your calculated interval.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Tenure, pay, and distance
    `HR-Employee-Attrition.csv` is a teaching file of 1,470 employee records. Two questions, each with its own interval:

    - Is monthly income linearly associated with years at the company?
    - Is distance from home linearly associated with years at the company?

    The interval interpretation is the same as in Section 3: an interval that excludes 0 is hard to reconcile with a zero linear correlation. An interval that covers 0 says a zero linear correlation is still compatible with these resamples.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd, rs):
    employees = pd.read_csv(lesson_dir / "HR-Employee-Attrition.csv")
    _rows = []
    for _other, _label in [("MonthlyIncome", "Years and monthly income"), ("DistanceFromHome", "Years and distance from home")]:
        _observed = rs.pearson_r(employees["YearsAtCompany"], employees[_other])
        _low, _high = rs.percentile_interval(
            rs.bootstrap_pearson(employees["YearsAtCompany"], employees[_other], num_samples=4_000, seed=2026)
        )
        _rows.append(
            {
                "Question": _label,
                "Pearson r": _observed,
                "95% low": _low,
                "95% high": _high,
                "Excludes 0": bool(_low > 0 or _high < 0),
            }
        )
    employee_intervals = pd.DataFrame(_rows)
    print("Employees:", len(employees))
    mo.Html(employee_intervals.round(4).to_html(index=False, border=0))
    return employee_intervals, employees


@app.cell
def _(employees, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    _axes[0].scatter(
        employees["YearsAtCompany"], employees["MonthlyIncome"], s=12, color="#4C78A8", alpha=0.45, edgecolor="none"
    )
    _axes[0].set(title="Income and years at the company", xlabel="Years at company", ylabel="Monthly income")
    _axes[1].scatter(
        employees["YearsAtCompany"],
        employees["DistanceFromHome"],
        s=12,
        color="#54A24B",
        alpha=0.45,
        edgecolor="none",
    )
    _axes[1].set(title="Distance and years at the company", xlabel="Years at company", ylabel="Distance from home")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which interval excludes 0, and which covers 0?
    2. Years and monthly income have a positive coefficient and an interval that excludes 0. Do the two columns move together in this file?
    3. The distance interval covers 0. Does that prove distance and tenure have correlation exactly 0?
    """)
    return


@app.cell(hide_code=True)
def _(employee_intervals, mo):
    _pay = employee_intervals.loc[employee_intervals["Question"] == "Years and monthly income"].iloc[0]
    _distance = employee_intervals.loc[employee_intervals["Question"] == "Years and distance from home"].iloc[0]
    _answers = f"""
1. **The two intervals:** Years and monthly income have r = {_pay["Pearson r"]:.3f} and interval ({_pay["95% low"]:.3f}, {_pay["95% high"]:.3f}), which excludes 0. Years and distance have r = {_distance["Pearson r"]:.3f} and interval ({_distance["95% low"]:.3f}, {_distance["95% high"]:.3f}), which covers 0.
2. **Yes:** The interval describes co-movement of years and monthly income in these records.
3. **No:** The interval says a zero linear correlation is still compatible with these resamples. This interval does not provide sufficient evidence of a nonzero population linear correlation; it does not establish its absence.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate a 95% paired bootstrap interval for `employees["YearsAtCompany"]` and `employees["DistanceFromHome"]`. Use 4,000 resamples and seed 2026. Replace both `None` values with your code. Explain why an interval containing zero does not establish independence.
    """)
    return


@app.cell
def _():
    student_distance_bootstrap = None
    print("student_distance_bootstrap:", student_distance_bootstrap)
    student_distance_interval = None
    print("student_distance_interval:", student_distance_interval)
    return (student_distance_bootstrap, student_distance_interval)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_distance_bootstrap = rs.bootstrap_pearson(employees["YearsAtCompany"], employees["DistanceFromHome"], num_samples=4_000, seed=2026)
student_distance_interval = rs.percentile_interval(student_distance_bootstrap, confidence=95)
```
Including zero means the interval does not exclude a zero population linear correlation. It neither proves zero correlation nor rules out nonlinear dependence.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    A paired bootstrap asks how a correlation moves when the sample of pairs is redrawn from itself. Keep each row together. Recompute the coefficient. The percentile interval is a band around that estimate.

    Read the estimate for direction and strength, and the interval for uncertainty. Then check whether zero is included. In this lesson the strong synthetic lines and the grade pairs produce intervals far from 0. The weak columns have coefficients near 0.4, and their 95% intervals still cover 0: thirty rows leave a modest line compatible with a zero population correlation. The unstructured column and the curve also leave 0 inside the straight-line interval. Years at the company show a positive sample association with monthly income. The interval for tenure and distance includes zero, so this analysis does not provide sufficient evidence of a nonzero population linear correlation. The next lesson tests whether the pairing matters by shuffling one column. Its decision need not match this interval.

    None of those intervals names a cause. The next lesson shuffles one column under an independence null, breaks the pairs, and compares the observed correlation with that null distribution. A statistic based on Pearson correlation is not sensitive to every form of dependence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Where should the observed correlation sit in a paired bootstrap curve, and why?
    2. What does an interval that excludes 0 say, and what does it not say about strength?
    3. Why is resampling the two columns independently the wrong bootstrap for this interval?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Usually near the estimate in these examples:** Bias or skewness can shift the center. The bootstrap re-estimates the same coefficient from copies of the sample, so the copies cluster around the estimate you started with.
2. **Evidence, then size:** Excluding 0 says a zero linear correlation is hard to reconcile with these resamples. The coefficient describes the estimated direction and strength. The interval width describes uncertainty in the estimate; exclusion alone does not establish a strong association.
3. **Independent columns break the pairs:** That simulation destroys the association and pulls the correlations toward 0. It does not describe the sampling uncertainty of the correlation you observed.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    - Student grades: Cortez, P., and Silva, A. Student Performance, UCI Machine Learning Repository. The employee table is a published teaching file of personnel records, not a census of one workforce we studied ourselves.
    """)
    return


if __name__ == "__main__":
    app.run()
