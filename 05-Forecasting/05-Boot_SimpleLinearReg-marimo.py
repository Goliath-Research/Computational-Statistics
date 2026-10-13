# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "statsmodels",
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
    import statsmodels.api as sm

    lesson_dir = Path(__file__).resolve().parent
    if str(lesson_dir) not in sys.path:
        sys.path.insert(0, str(lesson_dir))
    from goliath_research import resampling as rs

    sns.set_style("whitegrid")
    return lesson_dir, mo, np, pd, plt, rs, sm, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootstrap intervals for a regression line

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Refit a simple line on resampled pairs from `Salary_Data.csv`.
    - Read a percentile interval for the slope at more than one confidence level.
    - Decide whether a zero slope is compatible with that interval, without treating the bootstrap cloud as a null distribution.
    - See that a bundle of refitted lines, and the band around the mean, is wider at the edges of the data than in the middle.
    - State what the interval does not establish about an individual's raise.

    The resamples use 4,000 paired draws and seed 2026. The file is read as `Salary_Data.csv`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The fitted line, restated
    Each row is one pair: years of experience and salary. The least-squares line estimates the mean salary at a given experience in this 30-row teaching file.

    Hypotheses for the slope, carried forward from the regression lesson:

    - \(H_0\): the population slope is 0.
    - \(H_a\): the population slope is not 0.

    This lesson does not shuffle the pairs. It keeps each pair and asks how the fitted slope moves when the 30 rows are redrawn with replacement. The decision rule, stated now, is: if a percentile interval for the slope lies entirely above 0 or entirely below 0, a zero slope is hard to reconcile with these resamples at that confidence level. If the interval covers 0, a zero slope remains compatible with them.
    """)
    return


@app.cell
def _(lesson_dir, np, pd, rs, sm):
    salaries = pd.read_csv(lesson_dir / "Salary_Data.csv")
    experience = salaries["YearsExperience"].to_numpy(dtype=float)
    salary = salaries["Salary"].to_numpy(dtype=float)
    ols_result = sm.OLS(salary, sm.add_constant(experience)).fit()
    slope = float(np.asarray(ols_result.params)[1])
    intercept = float(np.asarray(ols_result.params)[0])
    # The helper uses the same least-squares formulas. The two slopes should match.
    _check_slope, _check_intercept = rs.ols_slope_intercept(experience, salary)
    print(f"statsmodels slope = {slope:.2f}, intercept = {intercept:.2f}")
    print(f"helper slope      = {_check_slope:.2f}, intercept = {_check_intercept:.2f}")
    print("Rows:", len(salaries))
    return experience, intercept, ols_result, salaries, salary, slope


@app.cell
def _(experience, intercept, plt, salary, slope):
    _order = experience.argsort()
    _fig, _ax = plt.subplots(figsize=(6.2, 3.6))
    _ax.scatter(experience, salary, s=36, color="#4C78A8", edgecolor="white", linewidth=0.4)
    _ax.plot(
        experience[_order],
        intercept + slope * experience[_order],
        color="#E45756",
        linewidth=2,
    )
    _ax.set(title="Least-squares line for the salary file", xlabel="Years of experience", ylabel="Salary")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. What is held together when this lesson resamples the file?
    2. State the decision rule that turns a slope interval into evidence about \(H_0\).
    3. Why should the helper's slope match the statsmodels slope before any resampling begins?
    """)
    return


@app.cell(hide_code=True)
def _(intercept, mo, slope):
    _answers = f"""
1. **The pair:** A resampled row keeps that person's years of experience with that person's salary.
2. **The rule:** If the percentile interval excludes 0, a zero slope is hard to reconcile with these resamples. If it covers 0, a zero slope remains compatible with them.
3. **Same arithmetic:** Both are ordinary least squares on the same 30 pairs. This run's slope is {slope:.2f} and the intercept is {intercept:.2f}. Resampling comes next. It is not a second definition of the line.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Resampling pairs refits the line
    A bootstrap copy draws 30 rows with replacement and refits the slope and the intercept. Four thousand copies produce two clouds: one for the slope, one for the intercept. Each cloud is a sampling distribution of an estimate. The original estimate belongs near the center.

    That geometry is not a null distribution. A null distribution for "no linear association" would break the pairs, as the permutation lesson did, and would sit near a slope of 0. Do not read the red line on the next figure as a test statistic sitting in a tail. It marks the estimate the bootstrap is varying.
    """)
    return


@app.cell
def _(experience, rs, salary):
    boot_slopes, boot_intercepts = rs.bootstrap_slope_intercept(experience, salary)
    print("Bootstrap fits:", boot_slopes.size)
    print(f"Mean bootstrap slope: {boot_slopes.mean():.2f}")
    return boot_intercepts, boot_slopes


@app.cell
def _(boot_intercepts, boot_slopes, intercept, plt, slope, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    sns.kdeplot(boot_slopes, fill=True, color="#4C78A8", ax=_axes[0])
    _axes[0].axvline(slope, color="#E45756", linewidth=2, label=f"Estimate {slope:.0f}")
    _axes[0].set(title="Bootstrap slopes", xlabel="Slope", ylabel="Density")
    _axes[0].legend(frameon=False)
    sns.kdeplot(boot_intercepts, fill=True, color="#54A24B", ax=_axes[1])
    _axes[1].axvline(intercept, color="#E45756", linewidth=2, label=f"Estimate {intercept:.0f}")
    _axes[1].set(title="Bootstrap intercepts", xlabel="Intercept", ylabel="Density")
    _axes[1].legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Replace `None` with the 95% percentile interval for the slope. `rs.percentile_interval(boot_slopes, confidence=95)` returns `(low, high)`.
    """)
    return


@app.cell
def _():
    slope_interval = None
    print("95% interval for the slope:", slope_interval)
    return (slope_interval,)


@app.cell(hide_code=True)
def _(boot_slopes, mo, rs):
    _low, _high = rs.percentile_interval(boot_slopes, confidence=95)
    _answers = f"""
1. **Interval:** ({_low:.2f}, {_high:.2f}), in salary units per year.
2. **Zero:** Both endpoints are positive, so 0 is outside this interval.
3. **Code:** `slope_interval = rs.percentile_interval(boot_slopes, confidence=95)`
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Why is the red line in the middle of each cloud?
    2. What would you have to do to the pairs to build a null cloud centered near a slope of 0?
    3. The intercept cloud is an uncertainty statement about the fitted value at 0 years of experience. Why is that a less interesting target than the slope in this file?
    """)
    return


@app.cell(hide_code=True)
def _(boot_slopes, mo, slope):
    _answers = f"""
1. **Re-estimates cluster:** Each bootstrap copy refits the same line. The copies gather around the original slope, {slope:.2f}. This run's bootstrap mean is {boot_slopes.mean():.2f}.
2. **Break the pairs:** Shuffle salary against experience, or otherwise match them at random. That is a permutation null, not this bootstrap.
3. **Zero years is off to the side:** The intercept is the line's height at 0 years of experience. The file's question is the change per year, which is the slope.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Does the interval contain zero?
    The same bootstrap sample can be summarized at more than one confidence level. A 90% interval uses the 5th and 95th percentiles. A 95% interval uses the 2.5th and 97.5th. A 99% interval uses the 0.5th and 99.5th. Higher confidence makes a wider interval. It does not use more employees. There are still 30 rows.

    Apply the decision rule from Section 1 to each row of the table. Also count the bootstrap slopes that fall on the zero side of the estimate. That count is a description of this bootstrap sample. It is not a permutation p-value, because these slopes were not generated under a broken pairing.
    """)
    return


@app.cell
def _(boot_slopes, mo, pd, rs, slope):
    _rows = []
    for _level in (90, 95, 99):
        _low, _high = rs.percentile_interval(boot_slopes, confidence=_level)
        _rows.append(
            {
                "Confidence": f"{_level}%",
                "Low": _low,
                "High": _high,
                "Excludes 0": bool(_low > 0 or _high < 0),
            }
        )
    slope_intervals = pd.DataFrame(_rows)
    print("Bootstrap slopes less than or equal to 0:", int((boot_slopes <= 0).sum()), "of", boot_slopes.size)
    mo.Html(slope_intervals.round(2).to_html(index=False, border=0))
    return (slope_intervals,)


@app.cell
def _(boot_slopes, plt, rs, slope):
    _fig, _ax = plt.subplots(figsize=(6.8, 3.6))
    _ax.hist(boot_slopes, bins=40, color="#4C78A8", edgecolor="white")
    _ax.axvline(slope, color="#E45756", linewidth=2, label="Estimated slope")
    for _level, _color in [(90, "#F58518"), (95, "#333333"), (99, "#54A24B")]:
        _low, _high = rs.percentile_interval(boot_slopes, confidence=_level)
        _ax.axvline(_low, color=_color, linestyle="--", label=f"{_level}%")
        _ax.axvline(_high, color=_color, linestyle="--")
    _ax.set(title="Percentile intervals for the slope", xlabel="Slope", ylabel="Bootstrap fits")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Which of the three intervals exclude 0?
    2. Why is the 99% interval the widest even though it uses the same 4,000 slopes?
    3. Why is "none of the bootstrap slopes were negative" not the same sentence as "the permutation p-value is zero"?
    """)
    return


@app.cell(hide_code=True)
def _(boot_slopes, mo, slope_intervals):
    _excluded = ", ".join(slope_intervals.loc[slope_intervals["Excludes 0"], "Confidence"].tolist())
    _answers = f"""
1. **All three:** {_excluded} exclude 0 in this run. A zero slope is outside even the widest of these intervals.
2. **The percentiles move out:** 99% keeps more of the bootstrap slopes inside the interval. The sample of employees does not grow.
3. **The pairing was kept:** {(boot_slopes <= 0).sum()} of the {boot_slopes.size} bootstrap slopes are at or below 0. Those fits still come from intact pairs, so they describe uncertainty around the estimated slope. A permutation p-value would count shuffles that break the pairs.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Where the line is better anchored
    Drawing every one of the 4,000 lines on one panel makes a solid ribbon and hides the points. The next figure draws 40 of the bootstrap lines, equally spaced through the seeded sample, plus the original line and the data.

    The band after that uses all 4,000 fits. At each experience value on a grid, it marks the 2.5th and 97.5th percentiles of the fitted means. Uncertainty in the slope swings the ends of the line more than the middle, and the middle of this file is where most of the points sit. Predictions at the edge of the observed years are the ones the resampled lines disagree about most.
    """)
    return


@app.cell
def _(boot_intercepts, boot_slopes, experience, intercept, np, plt, salary, slope):
    _grid = np.linspace(experience.min(), experience.max(), 80)
    _fig, _ax = plt.subplots(figsize=(6.6, 3.8))
    _step = max(len(boot_slopes) // 40, 1)
    for _s, _b in zip(boot_slopes[::_step][:40], boot_intercepts[::_step][:40]):
        _ax.plot(_grid, _b + _s * _grid, color="#9ecae1", linewidth=1, alpha=0.45)
    _ax.plot(_grid, intercept + slope * _grid, color="#E45756", linewidth=2, label="Original fit")
    _ax.scatter(experience, salary, s=28, color="#333333", zorder=3, label="Records")
    _ax.set(title="40 bootstrap lines", xlabel="Years of experience", ylabel="Salary")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(boot_intercepts, boot_slopes, experience, intercept, np, plt, salary, slope):
    _grid = np.linspace(experience.min(), experience.max(), 80)
    _fitted = boot_intercepts[:, None] + boot_slopes[:, None] * _grid[None, :]
    _low = np.percentile(_fitted, 2.5, axis=0)
    _high = np.percentile(_fitted, 97.5, axis=0)
    _fig, _ax = plt.subplots(figsize=(6.6, 3.8))
    _ax.fill_between(_grid, _low, _high, color="#4C78A8", alpha=0.25, label="95% band of fitted means")
    _ax.plot(_grid, intercept + slope * _grid, color="#E45756", linewidth=2, label="Original fit")
    _ax.scatter(experience, salary, s=28, color="#333333", zorder=3)
    _ax.set(title="Bootstrap band for the fitted mean", xlabel="Years of experience", ylabel="Salary")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Replace `None` with `True` if the bundle of lines spreads more at the ends of the experience range than near the middle, and `False` if it spreads evenly.
    """)
    return


@app.cell
def _():
    wider_at_edges = None
    print("Wider at the edges:", wider_at_edges)
    return (wider_at_edges,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Answer:** `True`.
2. **Why:** A small change in slope moves the line more at the far left and far right than near the center of the points. The resampled lines agree most where the data anchor them.
3. **Code:** `wider_at_edges = True`
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Why draw 40 lines on the scatter instead of all 4,000?
    2. The band is a percentile of fitted means. How is that different from a percentile interval for a single new person's salary?
    3. The band is for experience values inside the file's range. What new problem appears if you extend the grid far beyond the largest year in the file?
    """)
    return


@app.cell(hide_code=True)
def _(experience, mo):
    _answers = f"""
1. **A readable picture:** Four thousand lines paint a solid region and cover the points. Forty lines show the disagreement. The band uses all 4,000 fits without drawing each one.
2. **The mean, not one person:** The band describes uncertainty in the fitted line. A single new salary also includes the residual scatter around that line, which is wider than the uncertainty in the mean.
3. **Extrapolation:** The file runs from {experience.min():.1f} to {experience.max():.1f} years. Outside that range the straight line is a guess, and the bootstrap only reshuffles years that were already observed. It does not create information about 30 years of experience.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    Resampling the salary pairs and refitting the line gives a cloud of slopes around the original estimate of about 9,450 salary units per year. The 90%, 95%, and 99% percentile intervals from that cloud all stay positive in this run, so a zero slope is outside even the widest of the three. That is evidence against \(H_0\) from a sampling distribution of the slope. It is not a permutation test, and the bootstrap picture should keep the estimate in the center.

    The bundle of lines is tighter through the middle of the experience range and fans out at the edges. A prediction near the center of the file is the one these 30 rows agree on. A prediction at the edge, or beyond the years in the file, leans on a slope that the resamples still wiggle.

    None of the intervals is a raise. The slope describes how mean recorded salary differs across the experience values in a 30-row teaching file. It does not say what would happen to one person who worked one more year with job, hours, and employer held fixed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Why does the estimated slope belong in the center of the bootstrap histogram?
    2. What does a 99% interval use that a 90% interval does not?
    3. What causal claim does the salary slope still not support?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **The cloud re-estimates it:** Paired resamples vary the same least-squares slope. They gather around the slope from the original 30 rows.
2. **Wider percentiles, not more rows:** The 99% interval reaches farther into the same list of 4,000 slopes. Both intervals are computed from the same 30 employees.
3. **Not an individual's raise:** The interval does not estimate the effect of adding one year of experience while everything else about the person stays fixed.
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
    """)
    return


if __name__ == "__main__":
    app.run()
