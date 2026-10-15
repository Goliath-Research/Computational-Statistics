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
    import math
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
    return lesson_dir, math, mo, np, pd, plt, rs, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Permutation tests for a correlation

    ## Learning goals
    By the end of this lesson, you should be able to:
    - State a null hypothesis that says the pairing of the two variables is arbitrary.
    - Build a null distribution by shuffling one variable and leaving the other fixed.
    - Explain why a sample of 4,000 shuffles stands in for the full list of permutations.
    - Read a two-sided permutation p-value, including the add-one correction.
    - Separate a small p-value from a large correlation.
    - See that a test of Pearson's \(r\) can miss a curved pattern.
    - Apply the test to grades and to the employee file, and report the coefficient beside the decision.

    The shuffles below use seed 2026 and 4,000 draws. The same data can produce a slightly different p-value with a different seed. Near α = 0.05, Monte Carlo variability can change the decision. Choose the number of shuffles in advance; use more shuffles for greater precision rather than selecting a favorable seed. Result tables use `.round(4)`; decisions use unrounded p-values. Each decision uses an individual 5% level, without correction for multiple comparisons. The resolution of 4,000 shuffles is about \(1/4001\): that is the smallest p-value this procedure can report.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The null hypothesis is about the pairing
    Pearson's \(r\) measures straight-line association in matched pairs. A permutation test asks what values that coefficient would take if the matching were arbitrary.

    For a chosen pair of columns:

    - \(H_0\): X and Y are independent; under the sampling assumptions, the pairings are exchangeable.
    - \(H_a\): X and Y are dependent. Pearson's coefficient is the chosen statistic and may miss nonlinear dependence.

    Independence implies zero Pearson correlation when correlation exists, but zero correlation does not imply independence. This shuffle test is not a general test of zero correlation for arbitrary dependent populations. Unrestricted shuffling requires exchangeable pairings under the null; time dependence, clustering, or repeated observations across rows can require a different permutation scheme. Shuffling y leaves the x values, and the set of y values, exactly as they were. Only the matches change. If the original correlation is typical of those shuffled correlations, the data do not push us off \(H_0\).

    We will use a 5% significance level. A decision to reject \(H_0\) is not a measurement of strength.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. What does a shuffle of y change, and what does it leave untouched?
    2. Why is \(H_0\) a statement about pairings rather than a statement that every y equals every x?
    3. At the 5% level, what will we compare the p-value with, once we have computed it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Matches only:** The list of x values and the list of y values stay the same. The shuffle assigns the y values to different x values.
2. **Association, not equality:** \(H_0\) states independence under exchangeable pairings; it does not merely state zero linear correlation. The two columns can still have different means and different spreads.
3. **The rule:** Reject \(H_0\) when the p-value is below 0.05. Otherwise do not reject it. The coefficient is still reported beside that decision.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Why the test uses a sample of shuffles
    Three items can be ordered in \(3! = 6\) ways. Ten items can be ordered in \(10! = 3{,}628{,}800\) ways. The course file has 30 rows, and \(30!\) is far too large to list. The employee file has 1,470 rows. Listing every shuffle is not the method.

    The test draws 4,000 random shuffles. Their correlations approximate the full permutation null distribution. Each shuffle uses every observed y value exactly once; independently drawn shuffles may repeat a permutation. For n labeled observations there are n! index permutations; tied values can produce fewer distinct value arrangements.

    **Procedure**:

    - **step 1**: calculate the observed sample correlation, \(r_{\mathrm{obs}}\).

    - **step 2**: repeat **b** times: keep x fixed, shuffle y, then calculate and save the shuffled correlation.

    - **step 3**: count the shuffled correlations whose absolute values are at least as large as \(\lvert r_{\mathrm{obs}}\rvert\). Call this count **k** and calculate the approximate two-sided p-value:

      $p = \frac{k+1}{b+1}.$

    - **step 4**: compare p with the significance level **α**. If **p < α**, reject H₀; otherwise, do not reject H₀.

    The helper includes a small floating-point tolerance so numerically equal statistics count as equally extreme.

    Adding one keeps the p-value from being reported as zero. With 4,000 shuffles the smallest value is \(1/4001 \approx 0.00025\). A result that lands on that floor means none of the shuffles matched the observed correlation in size. It does not mean the probability is exactly zero.
    """)
    return


@app.cell
def _(math, mo, pd):
    _counts = pd.DataFrame(
        {
            "Items": [3, 6, 10],
            "Permutations": [math.factorial(3), math.factorial(6), math.factorial(10)],
        }
    )
    mo.Html(_counts.to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why is a list of all shuffles of 30 rows not a practical test?
    2. What does the add-one correction stop the procedure from printing?
    3. If the p-value equals \(1/4001\), how many of the 4,000 shuffles were at least as extreme as the observed correlation?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The list is enormous:** \(30!\) dwarfs 4,000, and \(1470!\) is larger still. The test uses a random sample of shuffles instead of the full list.
2. **It stops a printed zero:** A coefficient more extreme than every shuffle would otherwise look like a p-value of 0. The correction reports \(1/4001\) in that case.
3. **None of them:** The numerator is 1, from the observed statistic itself. Every shuffle was strictly smaller in absolute value.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. One test, read against the null cloud
    The column `y_spc` in `corrExamples.csv` is a strong positive line. The hypotheses are the ones in Section 1, applied to `x` and `y_spc`. The picture to expect, before looking, is the opposite of the bootstrap picture from the previous lesson. Shuffled correlations should sit near 0. An observed correlation far into a tail is unusual under the independence null.

    The red line on the next figure is the observed coefficient. The blue curve is the null distribution.
    """)
    return


@app.cell
def _(lesson_dir, pd, rs):
    examples = pd.read_csv(lesson_dir / "corrExamples.csv")
    observed_strong = rs.pearson_r(examples["x"], examples["y_spc"])
    strong_null = rs.permutation_pearson(examples["x"], examples["y_spc"], num_samples=4_000, seed=2026)
    strong_p = rs.permutation_p_value(observed_strong, strong_null)
    print(f"Observed r = {observed_strong:.3f}")
    print(f"Permutation p-value = {strong_p:.4f}")
    return examples, observed_strong, strong_null, strong_p


@app.cell
def _(observed_strong, plt, sns, strong_null, strong_p):
    _fig, _ax = plt.subplots(figsize=(6.2, 2.8))
    sns.kdeplot(strong_null, fill=True, color="#4C78A8", ax=_ax)
    _ax.axvline(observed_strong, color="#E45756", linewidth=2, label=f"Observed r = {observed_strong:.3f}")
    _ax.axvline(0, color="#333333", linestyle=":", label="0")
    _ax.set(
        title=f"Null correlations for y_spc (p = {strong_p:.4f})",
        xlabel="Pearson r after shuffling y",
        ylabel="Density",
    )
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Use the printed p-value and the figure.
    1. Is the p-value for `y_spc` below 0.05?
    2. Does the 5% rule reject \(H_0\)?
    3. Is the observed correlation in the center of the blue curve, or far from it?
    """)
    return


@app.cell(hide_code=True)
def _(mo, strong_p):
    _reject = strong_p < 0.05
    _answers = f"""
1. **Yes:** The p-value is {strong_p:.4f}.
2. **{"Reject" if _reject else "Do not reject"}:** {strong_p:.4f} is {"below" if _reject else "not below"} 0.05.
3. **Far from the center:** The blue curve is the null cloud around 0. The observed correlation sits out in the tail.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why is this blue curve centered near 0, while a paired bootstrap curve is centered near the observed correlation?
    2. The p-value sits on the floor of what 4,000 shuffles can report. How many of those shuffles were at least as large, in absolute value, as the observed correlation?
    """)
    return


@app.cell(hide_code=True)
def _(mo, observed_strong, strong_null, strong_p):
    _answers = f"""
1. **Different questions:** The shuffle breaks the pairs, so the null correlations describe no association. Their center is near 0 (this run's null mean is {strong_null.mean():.3f}). The paired bootstrap keeps the pairs and re-estimates the observed correlation, {observed_strong:.3f}.
2. **None:** The p-value is {strong_p:.4f}, the smallest value 4,000 shuffles can return. Every shuffled correlation was smaller in absolute value than the observed one.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the observed Pearson correlation for `x` and `y_snc` in `examples`, then generate 4,000 shuffled correlations with seed 2026. Replace both `None` values with your code. What does the shuffle preserve?
    """)
    return


@app.cell
def _():
    student_observed_r = None
    print("student_observed_r:", student_observed_r)
    student_null = None
    print("student_null:", student_null)
    return (student_observed_r, student_null)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_observed_r = rs.pearson_r(examples["x"], examples["y_snc"])
student_null = rs.permutation_pearson(examples["x"], examples["y_snc"], num_samples=4_000, seed=2026)
```
The observed correlation is approximately −0.9496. Shuffling preserves the values in each column and changes only their pairing.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Strength is not the same as the decision
    The same hypotheses and the same 5% rule apply to every column. Read the coefficient and the p-value as a pair.

    A weak line can reject \(H_0\) with a coefficient near 0.4 even when the paired bootstrap interval from the previous lesson still covers 0. The two procedures answer different questions, and with 30 rows they can disagree at the boundary. Even independent data can reject \(H_0\) by chance at the chosen significance level; an unstructured-looking scatter does not guarantee non-rejection. The curved column `y_nc2` is the trap: the scatter is a real pattern, and a test of independence using Pearson's **linear-association statistic** can still fail to reject. Look back at that scatter before you translate a large p-value into "nothing is going on."
    """)
    return


@app.cell
def _(examples, mo, pd, rs):
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
        _null = rs.permutation_pearson(examples["x"], examples[_column], num_samples=4_000, seed=2026)
        _p = rs.permutation_p_value(_observed, _null)
        _rows.append(
            {
                "Column": _column,
                "Pattern": _label,
                "Pearson r": _observed,
                "Permutation p": _p,
                "Reject at 5%": bool(_p < 0.05),
            }
        )
    permutation_table = pd.DataFrame(_rows)
    mo.Html(permutation_table.round(4).to_html(index=False, border=0))
    return (permutation_table,)


@app.cell
def _(examples, np, plt):
    _x = examples["x"].to_numpy()
    _y = examples["y_nc2"].to_numpy()
    _fig, _ax = plt.subplots(figsize=(5.8, 2.8))
    _ax.scatter(_x, _y, s=32, color="#E45756", edgecolor="white", linewidth=0.4)
    _slope, _intercept = np.polyfit(_x, _y, 1)
    _order = np.argsort(_x)
    _ax.plot(_x[_order], _slope * _x[_order] + _intercept, color="#333333", linewidth=1.4)
    _ax.set(title="y_nc2 is a curve; the line is a poor summary", xlabel="x", ylabel="y_nc2")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Is this permutation test built to detect straight-line association, or every pattern a scatter can show?
    2. In the weak positive row, is the coefficient near 0.4 or near 1?
    3. Does the 5% rule reject \(H_0\) for that row?
    """)
    return


@app.cell(hide_code=True)
def _(mo, permutation_table):
    _weak = permutation_table.loc[permutation_table["Column"] == "y_wpc"].iloc[0]
    _answers = f"""
1. **Straight-line association:** The statistic is Pearson's r.
2. **Near 0.4:** `y_wpc` has r = {_weak["Pearson r"]:.3f}.
3. **{"Reject" if _weak["Permutation p"] < 0.05 else "Do not reject"}:** The permutation p-value is {_weak["Permutation p"]:.4f}.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Does the curved row reject \(H_0\) at 5%?
    2. The scatter of `y_nc2` is a U-shape. Are "no straight-line signal" and "no pattern" the same sentence?
    3. The unstructured row also has a large p-value. Which display separates that cloud from the curve?
    """)
    return


@app.cell(hide_code=True)
def _(mo, permutation_table):
    _curve = permutation_table.loc[permutation_table["Column"] == "y_nc2"].iloc[0]
    _cloud = permutation_table.loc[permutation_table["Column"] == "y_nc"].iloc[0]
    _answers = f"""
1. **No:** `y_nc2` has p = {_curve["Permutation p"]:.3f}.
2. **No:** The test only discusses the straight line. The scatter is a U-shape built from the square of x.
3. **The scatter:** `y_nc` has r = {_cloud["Pearson r"]:.3f} and a large p-value, and its scatter is a cloud. The p-values alone do not separate a cloud from a curve.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    For `x` and `y_wpc`, calculate the observed coefficient, generate 4,000 shuffles with seed 2026, and count equally or more extreme absolute correlations. Use the add-one formula to calculate the p-value and interpret it at α = 0.05. Replace each `None` with your code. Use the tolerance shown in the answer to handle floating-point ties.
    """)
    return


@app.cell
def _():
    student_weak_r = None
    print("student_weak_r:", student_weak_r)
    student_weak_null = None
    print("student_weak_null:", student_weak_null)
    student_extreme_count = None
    print("student_extreme_count:", student_extreme_count)
    student_weak_p = None
    print("student_weak_p:", student_weak_p)
    return (student_weak_r, student_weak_null, student_extreme_count, student_weak_p)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_weak_r = rs.pearson_r(examples["x"], examples["y_wpc"])
student_weak_null = rs.permutation_pearson(examples["x"], examples["y_wpc"], num_samples=4_000, seed=2026)
_tolerance = 100 * np.finfo(float).eps * abs(student_weak_r)
student_extreme_count = np.count_nonzero(np.abs(student_weak_null) >= abs(student_weak_r) - _tolerance)
student_weak_p = (student_extreme_count + 1) / (len(student_weak_null) + 1)
```
For the original course data, r ≈ 0.3758 and p ≈ 0.0460. Reject independence at 5% using Pearson’s statistic. The modest coefficient and the small p-value describe different aspects of the result.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Grades from the same students
    For each pair of grade columns in `student-mat.csv`, the hypotheses are:

    - \(H_0\): the two grade variables are independent, with exchangeable pairings across students under the null.
    - \(H_a\): the two grade variables are dependent.

    A positive Pearson coefficient together with a small p-value indicates a positive sample linear association that is unusual under the independence null. The two-sided p-value alone does not establish direction. It would not say that changing the first-period grade, and nothing else, would change the final grade by a stated amount. The same student sits in every column.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd, rs):
    students = pd.read_csv(lesson_dir / "student-mat.csv", sep=";")
    _rows = []
    for _left, _right in [("G1", "G2"), ("G2", "G3"), ("G1", "G3")]:
        _observed = rs.pearson_r(students[_left], students[_right])
        _p = rs.permutation_p_value(_observed, rs.permutation_pearson(students[_left], students[_right], num_samples=4_000, seed=2026))
        _rows.append(
            {
                "Pair": f"{_left} and {_right}",
                "Pearson r": _observed,
                "Permutation p": _p,
                "Reject at 5%": bool(_p < 0.05),
            }
        )
    grade_tests = pd.DataFrame(_rows)
    print("Students:", len(students))
    mo.Html(grade_tests.round(4).to_html(index=False, border=0))
    return grade_tests, students


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. All three tests reject \(H_0\), and the coefficients are large. Which part of that sentence is the strength: the coefficient, or the small p-value?
    2. Why does shuffling G3, while leaving G2 in its original student order, match the null hypothesis?
    3. These are 395 students in one course file. Does the result describe every course?
    """)
    return


@app.cell(hide_code=True)
def _(grade_tests, mo):
    _row = grade_tests.loc[grade_tests["Pair"] == "G1 and G3"].iloc[0]
    _answers = f"""
1. **The coefficient:** G1 and G3 have r = {_row["Pearson r"]:.3f}. The p-value {_row["Permutation p"]:.4f} is the evidence against an arbitrary pairing.
2. **The null breaks the student:** Under H0 the final grade is matched to an arbitrary classmate's second-period grade. The shuffle does that and nothing else.
3. **No:** The result describes this course file.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Test independence between `students["G1"]` and `students["G3"]` using Pearson’s statistic, 4,000 shuffles, and seed 2026. Replace each `None` with your code. Report direction and strength from the coefficient, and evidence against independence from the p-value.
    """)
    return


@app.cell
def _():
    student_grade_r = None
    print("student_grade_r:", student_grade_r)
    student_grade_null = None
    print("student_grade_null:", student_grade_null)
    student_grade_p = None
    print("student_grade_p:", student_grade_p)
    return (student_grade_r, student_grade_null, student_grade_p)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_grade_r = rs.pearson_r(students["G1"], students["G3"])
student_grade_null = rs.permutation_pearson(students["G1"], students["G3"], num_samples=4_000, seed=2026)
student_grade_p = rs.permutation_p_value(student_grade_r, student_grade_null)
```
Compare the p-value with α = 0.05. A small p-value is evidence against independence, not a measure of strength or a causal effect.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Pay and distance, tested by shuffling
    Two questions in `HR-Employee-Attrition.csv`, with hypotheses stated first.

    Years at the company and monthly income:

    - \(H_0\): tenure and monthly income are independent, with exchangeable pairings under the null.
    - \(H_a\): tenure and monthly income are dependent.

    Years at the company and distance from home: the same pair of hypotheses, with distance in place of income.

    A positive coefficient together with rejection of the income null indicates positive sample linear association unusual under the independence null. Failing to reject the distance null would say this file's straight-line coefficient is typical of shuffled pairings.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd, rs):
    employees = pd.read_csv(lesson_dir / "HR-Employee-Attrition.csv")
    _rows = []
    for _other, _label in [("MonthlyIncome", "Years and monthly income"), ("DistanceFromHome", "Years and distance from home")]:
        _observed = rs.pearson_r(employees["YearsAtCompany"], employees[_other])
        _p = rs.permutation_p_value(
            _observed, rs.permutation_pearson(employees["YearsAtCompany"], employees[_other], num_samples=4_000, seed=2026)
        )
        _rows.append(
            {
                "Question": _label,
                "Pearson r": _observed,
                "Permutation p": _p,
                "Reject at 5%": bool(_p < 0.05),
            }
        )
    employee_tests = pd.DataFrame(_rows)
    print("Employees:", len(employees))
    mo.Html(employee_tests.round(4).to_html(index=False, border=0))
    return employee_tests, employees


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which null is rejected, and which is not?
    2. For income, is the strength the coefficient or the smallness of the p-value?
    3. The distance test does not reject \(H_0\). Does that prove the correlation is exactly zero?
    """)
    return


@app.cell(hide_code=True)
def _(employee_tests, mo):
    _pay = employee_tests.loc[employee_tests["Question"] == "Years and monthly income"].iloc[0]
    _distance = employee_tests.loc[employee_tests["Question"] == "Years and distance from home"].iloc[0]
    _answers = f"""
1. **Decisions:** Years and monthly income have r = {_pay["Pearson r"]:.3f} and p = {_pay["Permutation p"]:.4f}, so the 5% rule rejects H0. Years and distance have r = {_distance["Pearson r"]:.3f} and p = {_distance["Permutation p"]:.3f}, so the rule does not reject H0.
2. **The coefficient:** Years and monthly income have r = {_pay["Pearson r"]:.3f}. The p-value {_pay["Permutation p"]:.4f} says the matching is unusual among shuffles. The strength is the coefficient.
3. **No:** A p-value of {_distance["Permutation p"]:.3f} says the observed coefficient is ordinary under random pairings. The correlation could still be a small nonzero number.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Test independence between tenure and distance from home using Pearson’s statistic, 4,000 shuffles, and seed 2026. Replace each `None` with your code. Explain what a failure to reject would mean.
    """)
    return


@app.cell
def _():
    student_distance_r = None
    print("student_distance_r:", student_distance_r)
    student_distance_null = None
    print("student_distance_null:", student_distance_null)
    student_distance_p = None
    print("student_distance_p:", student_distance_p)
    return (student_distance_r, student_distance_null, student_distance_p)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_distance_r = rs.pearson_r(employees["YearsAtCompany"], employees["DistanceFromHome"])
student_distance_null = rs.permutation_pearson(employees["YearsAtCompany"], employees["DistanceFromHome"], num_samples=4_000, seed=2026)
student_distance_p = rs.permutation_p_value(student_distance_r, student_distance_null)
```
Failure to reject means this statistic provides insufficient evidence against independence under the test assumptions. It does not prove independence or zero correlation.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    A permutation test builds the world the null describes. Shuffle one variable, recompute Pearson's \(r\), and see whether the original coefficient is extreme in that world. The null cloud is centered near 0. That is the opposite geometry from a paired bootstrap, which often clusters near the estimate, although bias or skewness can shift its center.

    Report three things together: the coefficient, the p-value, and the scatter. The sign and absolute value of the coefficient describe estimated direction and strength. The p-value is the evidence against an arbitrary pairing, on a scale whose smallest notch with 4,000 shuffles is \(1/4001\). The scatter is what both numbers can miss. In this lesson the strong lines and the grade pairs are extreme under shuffling. The weak lines are smaller coefficients that can still be extreme. The unstructured cloud is not extreme. The curve is a real pattern that this linear test does not catch.

    A rejected null is not a cause. Income and tenure move together in the employee teaching file. The distance test does not provide sufficient evidence against independence using Pearson's statistic; it does not prove zero correlation or independence. Neither sentence says what would happen if one column were changed and everything else were held fixed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Which distribution is centered near the observed correlation: the paired bootstrap, or the permutation null?
    2. Why can a p-value below 0.05 sit next to a correlation of only about 0.4?
    3. What extra display stops you from calling `y_nc2` unrelated?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The paired bootstrap usually clusters near the estimate.** Bias or skewness can shift its center. It re-estimates the observed association. The permutation null breaks the pairs and centers near 0.
2. **Evidence is not size:** In a sample of 30, a coefficient of about 0.4 can be unusual among shuffles and still describe a loose line. The coefficient carries the strength.
3. **The scatter:** `y_nc2` bends. Pearson's r and its permutation p-value only discuss the straight line.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses*. Springer.
    - Student grades: Cortez, P., and Silva, A. Student Performance, UCI Machine Learning Repository. The employee table is a published teaching file of personnel records.
    """)
    return


if __name__ == "__main__":
    app.run()
