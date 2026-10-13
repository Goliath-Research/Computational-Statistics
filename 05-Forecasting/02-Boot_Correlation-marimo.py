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
    from goliath_research import resampling as rs

    sns.set_style("whitegrid")
    return lesson_dir, mo, np, pd, plt, rs, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootstrap intervals for a correlation

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Treat a correlation as one estimate from one sample of pairs.
    - Resample **pairs**, so each x stays with its own y.
    - Build a percentile confidence interval from those resampled correlations.
    - Use the interval as evidence about a population correlation of zero, without treating the bootstrap cloud as a null distribution.
    - Apply the interval to the course file, to student grades, and to the employee file.
    - State what the interval does not establish, including cause.

    Every interval below uses 4,000 paired resamples and seed 2026. The seed makes the table repeatable. More resamples shrink the Monte Carlo wobble in the interval endpoints. They do not add observations to the original sample.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One number from one sample
    `corrExamples.csv` has 30 synthetic pairs. The column `y_spc` was built as a strong positive line plus noise. Pearson's \(r\) for `x` and `y_spc` is one number computed from these 30 rows.

    A new draw of 30 pairs from the same process would produce a different coefficient. The bootstrap is a way to ask how far that coefficient tends to move, using only the pairs we already have.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd, rs):
    examples = pd.read_csv(lesson_dir / "corrExamples.csv")
    observed_strong = rs.pearson_r(examples["x"], examples["y_spc"])
    print(f"Observed Pearson r for x and y_spc: {observed_strong:.3f}")
    print("Rows:", len(examples))
    mo.Html(examples[["x", "y_spc"]].head().round(3).to_html(index=False, border=0))
    return examples, observed_strong


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. What is being estimated when we treat 0.983 as more than a description of these 30 rows?
    2. Why would a second sample of 30 pairs from the same process not give exactly the same coefficient?
    3. Does a coefficient this large, by itself, tell you how the y values were generated?
    """)
    return


@app.cell(hide_code=True)
def _(mo, observed_strong):
    _answers = f"""
1. **The target:** A population linear correlation: the correlation we would see if we could average over the whole process that produced pairs like these, not only these 30 rows.
2. **Sampling:** Each sample draws different noise and a different set of x values. The formula for r then returns a different number.
3. **The recipe stays hidden:** {observed_strong:.3f} says the points hug a rising line. It does not recover the noise scale, and it does not say whether x was assigned by an experiment or simply recorded.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Resample pairs, not separate columns
    A bootstrap copy of the sample draws **rows** with replacement. If row 7 is drawn, both its x and its y come along. Some rows appear more than once. Some rows are left out. Each copy has the same length as the original sample. We recompute Pearson's \(r\) on that copy. Doing this 4,000 times builds a bootstrap distribution of the coefficient.

    Drawing a new x from the x column and, separately, a new y from the y column answers a different question. That breaks the pairs. The correlation in those broken samples sits near zero even when the original correlation is large, because the matching that created the line has been destroyed. That cloud is closer to a null distribution than to a picture of how the observed coefficient varies. This lesson does not do that. `bootstrap_pearson` keeps each pair together.

    The observed coefficient should sit near the **center** of a paired bootstrap distribution. That is what a sampling distribution of an estimate looks like. It is not a test that places the estimate in a tail.
    """)
    return


@app.cell
def _(examples, rs):
    strong_bootstrap = rs.bootstrap_pearson(examples["x"], examples["y_spc"])
    print("Bootstrap replicates:", strong_bootstrap.size)
    print(f"Bootstrap mean: {strong_bootstrap.mean():.3f}")
    return (strong_bootstrap,)


@app.cell
def _(observed_strong, plt, rs, sns, strong_bootstrap):
    _low, _high = rs.percentile_interval(strong_bootstrap, confidence=95)
    _fig, _ax = plt.subplots(figsize=(6.8, 3.6))
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
    `strong_bootstrap` holds the 4,000 paired correlations. Replace `None` with the 95% percentile interval, as a pair of endpoints.

    `rs.percentile_interval(strong_bootstrap, confidence=95)` returns `(low, high)`.
    """)
    return


@app.cell
def _():
    strong_positive_interval = None
    print("95% interval:", strong_positive_interval)
    return (strong_positive_interval,)


@app.cell(hide_code=True)
def _(mo, rs, strong_bootstrap):
    _low, _high = rs.percentile_interval(strong_bootstrap, confidence=95)
    _answers = f"""
1. **Interval:** ({_low:.3f}, {_high:.3f}).
2. **Where zero sits:** Both endpoints are positive, so 0 is outside the interval.
3. **Code:** `strong_positive_interval = rs.percentile_interval(strong_bootstrap, confidence=95)`
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Why must a bootstrap copy take x and y from the same row?
    2. Why is the observed correlation near the middle of the blue curve, rather than out in a tail?
    3. What would a near-zero cloud of correlations be describing if someone had resampled the two columns independently?
    """)
    return


@app.cell(hide_code=True)
def _(mo, observed_strong, strong_bootstrap):
    _answers = f"""
1. **The pair is the observation:** The correlation is a fact about matched pairs. A row is one pair. Splitting the columns invents matches that never occurred.
2. **It is a sampling distribution of the estimate:** The bootstrap copies re-estimate the same coefficient. They cluster around the original estimate, {observed_strong:.3f}. The bootstrap mean in this run is {strong_bootstrap.mean():.3f}.
3. **A broken pairing:** Independent draws from the two columns destroy the association. The resulting correlations describe a world in which the matching is arbitrary, not the uncertainty of the coefficient we actually observed.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A percentile interval
    Sort the 4,000 bootstrap correlations. The 95% percentile interval runs from the 2.5th percentile to the 97.5th percentile. About 95% of the bootstrap coefficients fall between those two endpoints.

    Decision rule, stated before looking at the other columns: if this interval lies entirely above 0 or entirely below 0, the bootstrap sample is hard to reconcile with a population linear correlation of 0. If the interval covers 0, these resamples do not rule out a zero linear correlation. The width of the interval is the uncertainty. The distance of the estimate from 0 is the strength. With only 30 rows, a coefficient near 0.4 can still have an interval that covers 0. Covering 0 means "a flat line is still compatible with these resamples," not "the estimate was zero."

    This is not a proof that 95% of future intervals from this process cover the truth. The percentile interval is an approximation that behaves well when the estimator is well behaved and the sample carries the shape of the population. Thirty rows is a small sample. Read the interval as a careful summary of that uncertainty, not as a guarantee.
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
        _low, _high = rs.percentile_interval(rs.bootstrap_pearson(examples["x"], examples[_column]))
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
    mo.Html(example_intervals.round(3).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. For the strong negative column, the interval lies entirely below zero. What decision does the rule in this section give?
    2. The curved column's interval covers zero. What question does that answer, and which question does it leave open?
    3. The weak positive coefficient is near 0.4, but its interval can still cover 0. What does that say about a sample of 30?
    """)
    return


@app.cell(hide_code=True)
def _(example_intervals, mo):
    _strong = example_intervals.loc[example_intervals["Column"] == "y_snc"].iloc[0]
    _weak = example_intervals.loc[example_intervals["Column"] == "y_wpc"].iloc[0]
    _curve = example_intervals.loc[example_intervals["Column"] == "y_nc2"].iloc[0]
    _answers = f"""
1. **Decision:** The interval ({_strong["95% low"]:.3f}, {_strong["95% high"]:.3f}) excludes 0, so these resamples are hard to reconcile with a population linear correlation of 0. The estimate itself is {_strong["Pearson r"]:.3f}, a strong negative line.
2. **The curve:** The interval ({_curve["95% low"]:.3f}, {_curve["95% high"]:.3f}) covers 0, so the bootstrap does not rule out a zero **linear** correlation. Section 3 of the correlation lesson showed that `y_nc2` is a curve. A straight-line interval cannot declare that curve absent.
3. **Thirty rows are a wide net:** `y_wpc` has r = {_weak["Pearson r"]:.3f} and interval ({_weak["95% low"]:.3f}, {_weak["95% high"]:.3f}). The estimate is not zero, and the interval still reaches the other side of 0. A modest line in a sample this small is compatible with a population correlation of 0.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Student grades
    `student-mat.csv` records 395 students in a mathematics course. `G1`, `G2`, and `G3` are the first-period grade, the second-period grade, and the final grade, each on a 0–20 scale. The file is semicolon-separated. Each row is one student, so the pair `(G1, G3)` is two grades from the same person.

    Question, before the intervals: are earlier grades linearly associated with later grades in the population these students represent?

    The bootstrap interval does not say that the first-period grade causes the final grade. The same student brings study habits, prior preparation, and the same course into every period. The pairing is the student. A high correlation means the ordering of students is stable across periods.
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
        _low, _high = rs.percentile_interval(rs.bootstrap_pearson(students[_left], students[_right]))
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
    mo.Html(grade_intervals.round(3).to_html(index=False, border=0))
    return (grade_intervals,)


@app.cell
def _(plt, sns, students):
    _fig, _ax = plt.subplots(figsize=(6.2, 3.6))
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
    Replace `None` with the pair name from `grade_intervals` that has the largest Pearson correlation. Use the label in the Pair column, such as `"G1 and G2"`.
    """)
    return


@app.cell
def _():
    strongest_grade_pair = None
    print("Strongest pair:", strongest_grade_pair)
    return (strongest_grade_pair,)


@app.cell(hide_code=True)
def _(grade_intervals, mo):
    _row = grade_intervals.loc[grade_intervals["Pearson r"].idxmax()]
    _answers = f"""
1. **Pair:** {_row["Pair"]}.
2. **Coefficient:** {_row["Pearson r"]:.3f}, with 95% interval ({_row["95% low"]:.3f}, {_row["95% high"]:.3f}).
3. **Code:** `strongest_grade_pair = "{_row["Pair"]}"`
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Why is the row, rather than a randomly matched pair of grades, the right unit to resample?
    2. All three intervals exclude zero and the coefficients are large. What does that say about the ordering of students, and what does it not say about cause?
    3. These students are one mathematics course in the file, not every mathematics student. How should that limit the sentence you write?
    """)
    return


@app.cell(hide_code=True)
def _(grade_intervals, mo):
    _g2g3 = grade_intervals.loc[grade_intervals["Pair"] == "G2 and G3"].iloc[0]
    _answers = f"""
1. **The student is the pair:** G2 and G3 on the same row are two measurements of one person. Resampling rows keeps that person intact.
2. **Stable ordering, not a mechanism:** G2 and G3 have r = {_g2g3["Pearson r"]:.3f}. Students who rank high in the second period tend to rank high at the end. The interval does not isolate the effect of changing the second-period grade while holding everything else fixed.
3. **Scope:** The interval estimates the linear association in the population this course file can represent. It is not a result about every school.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Tenure, pay, and distance
    `HR-Employee-Attrition.csv` is a teaching file of 1,470 employee records. Two questions, each with its own interval:

    - Is monthly income linearly associated with years at the company?
    - Is distance from home linearly associated with years at the company?

    The decision rule is the same as in Section 3: an interval that excludes 0 is hard to reconcile with a zero linear correlation. Pay and tenure are assigned together by an employer. An interval that excludes 0 would say the two columns move together in this file. It would not say that staying one more year causes a raise of a specific amount. An interval that covers 0 for distance would say this file does not show a linear association. It would not prove that commuting never matters for anyone.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd, rs):
    employees = pd.read_csv(lesson_dir / "HR-Employee-Attrition.csv")
    _rows = []
    for _other, _label in [("MonthlyIncome", "Years and monthly income"), ("DistanceFromHome", "Years and distance from home")]:
        _observed = rs.pearson_r(employees["YearsAtCompany"], employees[_other])
        _low, _high = rs.percentile_interval(
            rs.bootstrap_pearson(employees["YearsAtCompany"], employees[_other])
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
    mo.Html(employee_intervals.round(3).to_html(index=False, border=0))
    return employee_intervals, employees


@app.cell
def _(employees, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
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
    ### Check your understanding
    1. Which interval excludes 0, and which covers 0?
    2. A positive association between years and income is visible in the scatter. Why is that not an estimate of the raise a particular employee would get from one more year?
    3. The distance interval covers 0. Why is the careful sentence "no linear association shows up in this file" rather than "distance and tenure are unrelated"?
    """)
    return


@app.cell(hide_code=True)
def _(employee_intervals, mo):
    _pay = employee_intervals.loc[employee_intervals["Question"] == "Years and monthly income"].iloc[0]
    _distance = employee_intervals.loc[employee_intervals["Question"] == "Years and distance from home"].iloc[0]
    _answers = f"""
1. **The two intervals:** Years and monthly income have r = {_pay["Pearson r"]:.3f} and interval ({_pay["95% low"]:.3f}, {_pay["95% high"]:.3f}), which excludes 0. Years and distance have r = {_distance["Pearson r"]:.3f} and interval ({_distance["95% low"]:.3f}, {_distance["95% high"]:.3f}), which covers 0.
2. **The line is not a raise:** Income and tenure are set together. People in different jobs, hired in different years, are mixed in the file. The interval describes co-movement in these records. It does not hold job, performance, and hours fixed and then add one year.
3. **A covered zero is not a proof of independence:** The interval says a zero linear correlation is still compatible with these resamples. A curved pattern, or an association in some other workforce, could still exist. This teaching file does not show a straight-line signal.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    A paired bootstrap asks how a correlation moves when the sample of pairs is redrawn from itself. Keep each row together. Recompute the coefficient. The percentile interval is a band around that estimate.

    Use the band in two steps. First read whether 0 is inside it. That is the evidence about a zero linear correlation. Then read the estimate itself. That is the strength. In this lesson the strong synthetic lines and the grade pairs produce intervals far from 0. The weak columns have coefficients near 0.4, and their 95% intervals still cover 0: thirty rows leave a modest line compatible with a zero population correlation. The unstructured column and the curve also leave 0 inside the straight-line interval. Years at the company move with monthly income in the employee file and do not move, in a straight line, with distance from home. The next lesson's permutation test asks a sharper question about the same weak columns, and the two procedures need not agree at the boundary.

    None of those intervals names a cause. The next lesson builds the other picture on purpose: shuffle one column, break the pairs, and see where the observed correlation sits in that null world.
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
1. **Near the center:** The bootstrap re-estimates the same coefficient from copies of the sample, so the copies cluster around the estimate you started with.
2. **Evidence, then size:** Excluding 0 says a zero linear correlation is hard to reconcile with these resamples. The strength is the coefficient and the width of the interval, not the fact of exclusion alone.
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
