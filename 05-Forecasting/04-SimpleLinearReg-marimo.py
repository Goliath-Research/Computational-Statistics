# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "scipy",
#     "scikit-learn",
#     "statsmodels",
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
    import statsmodels.api as sm
    import scipy.stats as st
    from sklearn.linear_model import LinearRegression

    lesson_dir = Path(__file__).resolve().parent
    sns.set_style("whitegrid")
    return LinearRegression, lesson_dir, mo, np, pd, plt, sm, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple linear regression

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Describe a fitted line as an estimate of the mean response at a given x, not as a cause.
    - Read the 30-row salary teaching file and name the explanatory variable and the response.
    - Connect Pearson's correlation to the direction of the line, and state a test for that correlation before reading it.
    - Fit the same straight line with scikit-learn and with statsmodels.
    - State a hypothesis for the slope before reading its t test.
    - Decide whether a squared term earns a place in the model.

    `Salary_Data.csv` is the file in this lesson's folder. The Jupyter notebook refers to `salary_data.csv`. On a case-sensitive system those are different names. This lesson reads `Salary_Data.csv` and does not rename the file.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The line estimates a mean
    Simple linear regression looks for a straight line

    \[
    \text{mean response at } x = \text{intercept} + \text{slope} \times x.
    \]

    For this file the response is salary and x is years of experience. The fitted line estimates the mean salary among people in this file who have a given amount of experience. It is a summary of these 30 rows.

    The slope is not automatically the raise a particular person would receive for working one more year. That would require a change in experience with job, employer, and everything else held fixed. This file does not contain that comparison. It contains 30 paired records in a teaching table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. What does the line claim about people who have 6 years of experience in this file?
    2. What comparison would you need before calling the slope a raise?
    3. Why does a tight line in a 30-row teaching file still leave the labor market mostly unmeasured?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A mean:** The line estimates the average salary, in the units recorded in the file, at 6 years of experience among records like these.
2. **A held-fixed comparison:** You would need people, or the same people, who differ in experience while job, hours, and employer stay fixed. A column of recorded years is not that experiment.
3. **Thirty rows:** The file is a small teaching table. A line that fits it well describes this table. It is not a census of wages.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The 30 salaries
    Each row is one person's years of experience and salary. Experience is the explanatory variable: the line uses it to describe the mean salary. Salary is the response: it is the variable whose mean the line estimates. Neither name means that experience was assigned by an experiment.
    """)
    return


@app.cell
def _(lesson_dir, mo, pd):
    salaries = pd.read_csv(lesson_dir / "Salary_Data.csv")
    experience = salaries["YearsExperience"].to_numpy(dtype=float)
    salary = salaries["Salary"].to_numpy(dtype=float)
    print("Rows:", len(salaries))
    mo.Html(salaries.head().to_html(index=False, border=0))
    return experience, salaries, salary


@app.cell
def _(experience, plt, salary):
    _fig, _ax = plt.subplots(figsize=(5.8, 2.8))
    _ax.scatter(experience, salary, s=36, color="#4C78A8", edgecolor="white", linewidth=0.4)
    _ax.set(title="Salary against years of experience", xlabel="Years of experience", ylabel="Salary")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which column is x, and which column is the response?
    2. The points rise. What does that suggest about the sign of the correlation, before you calculate it?
    3. Why is "30 employees from one company we studied" a stronger claim than this file can support?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Roles:** `YearsExperience` is x. `Salary` is the response.
2. **Sign:** A cloud that rises from left to right suggests a positive correlation.
3. **Provenance:** This is a 30-row teaching file shipped with the lesson. The lesson does not identify an employer, a year, or a sampling design. The line describes the file.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Correlation before the line
    The line's slope will be positive only if the correlation is positive. Start with the correlation test, and state it before reading the number.

    - \(H_0\): the population linear correlation between experience and salary is 0.
    - \(H_a\): that correlation is not 0.

    The 5% level is the rejection rule. A small p-value says a flat relationship is a poor description of a population like this sample. The coefficient says whether the line is tight. The test does not say that gaining a year of experience causes the salary difference.
    """)
    return


@app.cell
def _(experience, salary, st):
    salary_correlation = st.pearsonr(experience, salary)
    print(f"Pearson r = {salary_correlation.statistic:.4f}")
    print(f"p-value = {salary_correlation.pvalue:.4g}")
    return (salary_correlation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Replace `None` with the Pearson correlation. `salary_correlation.statistic` is that number.

    Then answer these questions about the test.
    4. Write \(H_0\) in a sentence that names both columns.
    5. The correlation is about 0.98. What does that say about a straight line, beyond the p-value?
    6. Why is the correlation test not yet an answer in salary units per year?
    """)
    return


@app.cell
def _():
    salary_r = None
    print("Pearson r:", salary_r)
    return (salary_r,)


@app.cell(hide_code=True)
def _(mo, salary_correlation):
    _answers = f"""
1. **Coefficient:** {salary_correlation.statistic:.4f}.
2. **p-value:** {salary_correlation.pvalue:.4g}. At the 5% level, reject a zero linear correlation.
3. **Code:** `salary_r = salary_correlation.statistic`
4. **Null:** In the population this file represents, the linear correlation between years of experience and salary is 0.
5. **The line is tight:** r = {salary_correlation.statistic:.3f} means the points stay close to a rising line. The p-value only says "not flat."
6. **Units:** A correlation has no salary units. The slope, in salary units per year, is the regression question in the next section.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Fitting the line with scikit-learn
    `LinearRegression` expects x as a column, shape `(n, 1)`, even when there is only one predictor. The fitted attributes are `coef_[0]`, the slope, and `intercept_`. `score(x, y)` is \(R^2\): the fraction of the variance of salary that the line accounts for in this file.

    \(R^2\) near 1 means the residuals are small compared with the spread of salary **in this table**. It is not a promise about the next person hired somewhere else.
    """)
    return


@app.cell
def _(LinearRegression, experience, salary):
    experience_column = experience.reshape(-1, 1)
    salary_model = LinearRegression().fit(experience_column, salary)
    print(f"Slope: {salary_model.coef_[0]:.2f} salary units per year")
    print(f"Intercept: {salary_model.intercept_:.2f}")
    print(f"R-squared: {salary_model.score(experience_column, salary):.3f}")
    return experience_column, salary_model


@app.cell
def _(experience, experience_column, plt, salary, salary_model):
    _fitted = salary_model.predict(experience_column)
    _order = experience.argsort()
    _fig, _ax = plt.subplots(figsize=(5.8, 2.8))
    _ax.scatter(experience, salary, s=36, color="#4C78A8", edgecolor="white", linewidth=0.4, label="Recorded")
    _ax.plot(experience[_order], _fitted[_order], color="#E45756", linewidth=2, label="Fitted mean")
    _ax.set(title="Fitted mean salary", xlabel="Years of experience", ylabel="Salary")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Replace `None` with the model's predicted **mean** salary at 6 years of experience.

    `float(salary_model.predict(np.array([[6.0]]))[0])` is that prediction. The double brackets give scikit-learn the column shape it expects.

    Then answer these questions about the fit.
    4. Why does scikit-learn want `experience` reshaped to one column?
    5. If \(R^2\) is about 0.96, what fraction of the salary variance in this file is still residual?
    6. The prediction at 6 years uses the line. What would make that prediction a poor guide for an x far outside the years in the file?
    """)
    return


@app.cell
def _():
    fitted_at_6_years = None
    print("Predicted mean salary at 6 years:", fitted_at_6_years)
    return (fitted_at_6_years,)


@app.cell(hide_code=True)
def _(experience, experience_column, mo, np, salary, salary_model):
    _prediction = float(salary_model.predict(np.array([[6.0]]))[0])
    _r2 = float(salary_model.score(experience_column, salary))
    _answers = f"""
1. **Prediction:** {_prediction:.2f}, in the salary units of the file.
2. **What it is:** The fitted mean at 6 years, not a guarantee for one person and not a causal raise.
3. **Code:** `fitted_at_6_years = float(salary_model.predict(np.array([[6.0]]))[0])`
4. **Shape:** The estimator is written for any number of predictors. One predictor is still a column of shape ({len(experience)}, 1).
5. **What is left:** R-squared is {_r2:.3f}, so about {1 - _r2:.3f} of the variance of salary in this file is not on the line.
6. **Outside the data:** The line is anchored by the years that appear in the file, roughly {experience.min():.1f} to {experience.max():.1f}. Far outside that range, the mean may bend, and this table has no rows there.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The same line in statsmodels, and a test of the slope
    statsmodels fits the line by ordinary least squares and adds a test for each coefficient. The design matrix needs a column of ones so the intercept has a place to sit. `sm.add_constant` adds that column.

    Hypotheses for the slope, stated before the table:

    - \(H_0\): the population slope is 0. Mean salary does not change linearly with years of experience.
    - \(H_a\): the population slope is not 0.

    The t statistic is the estimate divided by its standard error. The p-value is the evidence against \(H_0\) under the usual linear-model assumptions: independent rows, a linear mean, and residuals whose spread does not depend on x in a serious way. Thirty teaching rows will not prove those assumptions. We use the test as a disciplined summary and we still look at the scatter.

    The intercept has its own test, against a mean salary of 0 at zero years of experience. Zero salary is not an interesting benchmark here. The slope test is the one that matches the question in this section.
    """)
    return


@app.cell
def _(experience, salary, sm):
    design = sm.add_constant(experience)
    ols_result = sm.OLS(salary, design).fit()
    return design, ols_result


@app.cell
def _(mo, np, ols_result, pd):
    _params = np.asarray(ols_result.params, dtype=float)
    _tvalues = np.asarray(ols_result.tvalues, dtype=float)
    _pvalues = np.asarray(ols_result.pvalues, dtype=float)
    coefficient_table = pd.DataFrame(
        {
            "Term": ["Intercept", "Years of experience"],
            "Estimate": _params,
            "t": _tvalues,
            "p-value": _pvalues,
        }
    )
    print(f"R-squared: {ols_result.rsquared:.3f}")
    mo.Html(coefficient_table.round(4).to_html(index=False, border=0))
    return (coefficient_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    scikit-learn and statsmodels are fitting the same least-squares line. The slope and intercept should match the numbers in Section 4, up to rounding. The new information is the standard-error test, not a second geometry.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Write \(H_0\) for the slope in words.
    2. The slope's p-value is tiny. What does that add to the news that the correlation was 0.98?
    3. Why is a tiny p-value for the intercept not the result to lead with?
    """)
    return


@app.cell(hide_code=True)
def _(coefficient_table, mo, ols_result):
    _slope = coefficient_table.loc[coefficient_table["Term"] == "Years of experience"].iloc[0]
    _intercept = coefficient_table.loc[coefficient_table["Term"] == "Intercept"].iloc[0]
    _answers = f"""
1. **Null:** The population slope is 0. Mean salary does not change linearly with years of experience.
2. **Units and a standard error:** The slope is {_slope["Estimate"]:.2f} salary units per year, and its t test gives p = {_slope["p-value"]:.4g}. The correlation already said the line was tight and not flat. The slope puts that line in salary units and compares the estimate with its standard error. R-squared is {ols_result.rsquared:.3f}.
3. **The intercept's benchmark is 0 salary:** Its estimate is {_intercept["Estimate"]:.2f} and its p-value is tiny because 0 salary at 0 years is far from this line. That is not the scientific question.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. A squared term
    A curve would say the salary gap for one more year is different at 2 years of experience than at 10. One way to allow that is to add the square of experience as a second predictor:

    \[
    \text{mean salary} = b_0 + b_1 \times \text{years} + b_2 \times \text{years}^2.
    \]

    Hypotheses for the new term, before the fit:

    - \(H_0\): \(b_2 = 0\). The square does not change the mean once the straight line is included.
    - \(H_a\): \(b_2 \neq 0\).

    Two summaries matter together. \(R^2\) says whether the residuals shrink. The p-value for \(b_2\) says whether that coefficient is distinguishable from zero in this sample. A squared term that leaves \(R^2\) essentially unchanged, with a large p-value, has not earned a place. Extra flexibility that does not improve the fit is not a better description.
    """)
    return


@app.cell
def _(experience, pd, salary, sm):
    squared = pd.DataFrame(
        {
            "YearsExperience": experience,
            "YearsExperienceSquared": experience ** 2,
        }
    )
    squared_design = sm.add_constant(squared)
    squared_result = sm.OLS(salary, squared_design).fit()
    return squared_design, squared_result


@app.cell
def _(mo, np, ols_result, pd, squared_result):
    _params = np.asarray(squared_result.params, dtype=float)
    _pvalues = np.asarray(squared_result.pvalues, dtype=float)
    squared_table = pd.DataFrame(
        {
            "Term": ["Intercept", "Years of experience", "Years squared"],
            "Estimate": _params,
            "p-value": _pvalues,
        }
    )
    print(f"Straight-line R-squared: {ols_result.rsquared:.4f}")
    print(f"With a squared term:     {squared_result.rsquared:.4f}")
    mo.Html(squared_table.round(4).to_html(index=False, border=0))
    return (squared_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Replace `None` with `True` if the squared term is supported by a meaningful rise in \(R^2\) and a p-value below 0.05, and `False` otherwise.

    Then answer these questions about the squared term.
    4. What pattern in the scatter would have made you expect a squared term to matter?
    5. The straight-line \(R^2\) was already about 0.96. Why is "the p-value of the square is large" still necessary, rather than stopping at \(R^2\)?
    6. A future file with a bend in it would need a different conclusion. What, in this file, blocks that conclusion?
    """)
    return


@app.cell
def _():
    squared_term_supported = None
    print("Squared term supported:", squared_term_supported)
    return (squared_term_supported,)


@app.cell(hide_code=True)
def _(mo, np, ols_result, squared_result):
    _p = float(np.asarray(squared_result.pvalues, dtype=float)[2])
    _supported = bool((_p < 0.05) and (squared_result.rsquared - ols_result.rsquared > 0.01))
    _answers = f"""
1. **Decision:** `{_supported}`.
2. **The two numbers:** Straight-line R-squared is {ols_result.rsquared:.4f}. With the square it is {squared_result.rsquared:.4f}. The squared term's p-value is {_p:.3f}.
3. **Code:** `squared_term_supported = False`
4. **A bend:** The vertical gap for one more year would change as experience increases: a curve, a flattening, or a steepening that a single slope cannot follow.
5. **R-squared almost has to rise:** Adding a predictor cannot lower the in-sample R-squared. Here it moves from {ols_result.rsquared:.4f} to {squared_result.rsquared:.4f}. The p-value, {_p:.3f}, asks whether that extra coefficient is distinguishable from zero. It is not, at any conventional level.
6. **This table does not bend enough:** The square does not improve the description of these 30 rows. A different range of experience could still curve. This fit does not show that curve.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    The salary file shows a tight, rising line. Pearson's correlation is about 0.98, and the 5% test rejects a zero linear correlation. The least-squares slope is about 9,450 salary units per year of experience, with an intercept near 25,800. scikit-learn and statsmodels report that same line. statsmodels adds a t test: the slope is far from zero relative to its standard error. \(R^2\) is about 0.957, so most of the variance of salary **in this table** lies on the line.

    A squared term does not change that summary. \(R^2\) barely moves, and the new coefficient's p-value is about 0.91. The straight line is the description these 30 rows support.

    The line estimates a mean in a teaching file. It does not measure the raise an individual would get from one more year of work, and it does not describe employers who are not in the table. The next lesson keeps the pairs together and asks how far the slope moves when the 30 rows are resampled.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check your understanding
    1. Which number answers "how tight is the line," and which number answers "how many salary units per year"?
    2. Why fit the line twice, with scikit-learn and with statsmodels?
    3. What claim about cause does this lesson refuse?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **Two numbers:** The correlation, or R-squared, describes tightness. The slope describes salary units per year.
2. **Same line, extra test:** Both fits are least squares, so the slope and intercept match. statsmodels is there for the coefficient tests and the squared term. scikit-learn is there as the prediction interface.
3. **Not a raise:** The lesson does not claim that working one more year causes a salary increase of the slope, with everything else held fixed.
""")
        },
        lazy=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer. Simple linear regression.
    - scikit-learn: `sklearn.linear_model.LinearRegression`.
    - statsmodels: `statsmodels.api.OLS`.
    """)
    return


if __name__ == "__main__":
    app.run()
