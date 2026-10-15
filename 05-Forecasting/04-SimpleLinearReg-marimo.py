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
    - Describe a fitted line as an estimate of the mean response at a given x.
    - Read the 30-row salary teaching file and name the explanatory variable and the response.
    - Connect Pearson's correlation to the direction of the line, and state a test for that correlation before reading it.
    - Fit the same straight line with scikit-learn and with statsmodels.
    - State a hypothesis for the slope before reading its t test.
    - Decide whether a squared term earns a place in the model.

    Read `salary_data.csv` from this lesson’s folder. Numeric tables use `.round(4)` for consistent display; calculations use unrounded values. A p-value displayed as `0.0000` is not exactly zero: it is smaller than the displayed precision.
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

    The slope is the change in that estimated mean when experience increases by one year, in the units recorded in the file. This file contains 30 paired records in a teaching table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. What does the line estimate for people who have 6 years of experience in this file: one person's salary, or a mean salary?
    2. How many rows are in the teaching file?
    3. Does a line fitted to these 30 rows describe every employer?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Show answers": mo.md(r"""
1. **A mean:** The line estimates the average salary, in the units recorded in the file, at 6 years of experience among records like these.
2. **Thirty rows:** The file is a small teaching table.
3. **No:** A line that fits this table describes this table.
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
    salaries = pd.read_csv(lesson_dir / "salary_data.csv")
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

    The 5% level is the rejection rule. A small p-value provides evidence against zero population linear correlation under the test assumptions. It does not establish a strong relationship or rule out nonlinear patterns. The coefficient describes estimated direction and strength. The conventional exact Pearson test assumes independent pairs from a bivariate normal population; computing the coefficient itself does not require normality.
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
    Use the printed correlation and p-value.
    1. Is the Pearson correlation positive?
    2. Is the p-value below 0.05?
    3. A correlation near 0.98 says the points stay close to a line. Does that number have salary units?
    """)
    return


@app.cell(hide_code=True)
def _(mo, salary_correlation):
    _answers = f"""
1. **Yes:** r = {salary_correlation.statistic:.4f}.
2. **Yes:** The p-value is {salary_correlation.pvalue:.4g}, so at the 5% level reject a zero linear correlation.
3. **No:** A correlation has no salary units. The slope, in salary units per year, comes in the next section.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Write \(H_0\) in a sentence that names both columns.
    2. The correlation is about 0.98. Are the points close to a rising line, or spread in a loose cloud?
    """)
    return


@app.cell(hide_code=True)
def _(mo, salary_correlation):
    _answers = f"""
1. **Null:** In the population this file represents, the linear correlation between years of experience and salary is 0.
2. **Close to a rising line:** r = {salary_correlation.statistic:.3f}. The p-value provides evidence against zero population linear correlation under the test assumptions. The coefficient says the points stay close to it.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate Pearson’s correlation and its p-value using `experience` and `salary`. Replace `None` with your code. Interpret strength and statistical evidence separately.
    """)
    return


@app.cell
def _():
    student_correlation = None
    print("student_correlation:", student_correlation)
    return (student_correlation,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_correlation = st.pearsonr(experience, salary)
```
The coefficient describes direction and strength. Compare the p-value with α = 0.05 under the stated assumptions; significance alone does not describe strength.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Fitting the line with scikit-learn
    `LinearRegression` expects x as a column, shape `(n, 1)`, even when there is only one predictor. The fitted attributes are `coef_[0]`, the slope, and `intercept_`. `score(x, y)` is \(R^2\): the fraction of the variance of salary that the line accounts for in this file.

    For this training-data least-squares fit with an intercept:

    $$
    R^2 = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}.
    $$

    The denominator is the squared error from predicting the sample mean for everyone. R-squared compares the fitted model with that baseline, rather than counting observations on a line.

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
    The cell above prints the slope and the intercept.
    1. Is the predicted value at 6 years one person's salary, or the fitted mean?
    2. If \(R^2\) is about 0.96, does the fitted model account for most or little of the observed variation in salary?
    3. The line is anchored by the years that appear in the file. Is a prediction far outside that range using rows the file contains?
    """)
    return


@app.cell(hide_code=True)
def _(experience, experience_column, mo, np, salary, salary_model):
    _prediction = float(salary_model.predict(np.array([[6.0]]))[0])
    _r2 = float(salary_model.score(experience_column, salary))
    _answers = f"""
1. **The fitted mean:** At 6 years the line gives {_prediction:.2f}, in the salary units of the file.
2. **Most:** R-squared is {_r2:.4f}, so the model accounts for about {100 * _r2:.2f}% of the observed variation in salary. The residual sum of squares is about {100 * (1 - _r2):.2f}% of the total sum of squares. This is not the percentage of observations lying on the line.
3. **No:** The years in the file run from about {experience.min():.1f} to {experience.max():.1f}. Far outside that range, the table has no rows.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Fit a scikit-learn linear regression using `experience` and `salary`, then predict salary at 6 years. Replace both `None` values with your code. Why must the predictor input have two dimensions?
    """)
    return


@app.cell
def _():
    student_model = None
    print("student_model:", student_model)
    student_prediction = None
    print("student_prediction:", student_prediction)
    return (student_model, student_prediction)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_model = LinearRegression().fit(experience.reshape(-1, 1), salary)
student_prediction = student_model.predict(np.array([[6.0]]))[0]
```
The input shape is (number of observations, number of predictors). The fitted value estimates mean salary at 6 years under the model; it is also a point prediction for an individual, whose salary can differ from that mean.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The same line in statsmodels, and a test of the slope
    statsmodels fits the line by ordinary least squares and adds a test for each coefficient. The design matrix needs a column of ones so the intercept has a place to sit. `sm.add_constant` adds that column.

    Hypotheses for the slope, stated before the table:

    - \(H_0\): the population slope is 0. Mean salary does not change linearly with years of experience.
    - \(H_a\): the population slope is not 0.

    The t statistic is the estimate divided by its standard error. The p-value is the evidence against \(H_0\) under the conventional linear-model assumptions: a correctly specified linear conditional mean, independent errors with constant variance, and normally distributed errors for exact finite-sample t inference. Normality concerns the errors conditional on x, not x or y themselves. Fitting the least-squares line does not require normality. Thirty teaching rows will not prove those assumptions. We use the test as a disciplined summary and we still look at the scatter.

    For simple ordinary least-squares regression with an intercept and the same observations, the conventional two-sided slope test and Pearson zero-correlation test have the same t statistic and p-value. They are two descriptions of the same evidence, not independent confirmations. The slope adds salary units per year:

    $$
    t = r \sqrt{\frac{n-2}{1-r^2}}.
    $$

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
2. **Units, not independent confirmation:** The slope is {_slope["Estimate"]:.2f} salary units per year. Its conventional two-sided t test gives p = {_slope["p-value"]:.4g}, the same as the Pearson test on these observations. The slope expresses the relationship in salary units per year. R-squared is {ols_result.rsquared:.4f}.
3. **The intercept's benchmark is 0 salary:** Its estimate is {_intercept["Estimate"]:.2f} and its p-value is tiny because 0 salary at 0 years is far from this line. That is not the scientific question.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inspecting residuals
    A residual is the recorded salary minus the fitted salary. Plot residuals against fitted values: a curved pattern can indicate a missing nonlinear term, changing vertical spread can indicate nonconstant error variance, and isolated points can merit inspection. A patternless plot does not prove all assumptions; independence also depends on the sampling design.
    """)
    return


@app.cell
def _(ols_result, plt):
    _fig, _ax = plt.subplots(figsize=(5.8, 2.8))
    _ax.scatter(ols_result.fittedvalues, ols_result.resid, s=36, color="#4C78A8", edgecolor="white", linewidth=0.4)
    _ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    _ax.set(title="Residuals against fitted salary", xlabel="Fitted salary", ylabel="Residual")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Fit the same line with statsmodels, including an intercept, then calculate its residuals. Replace each `None` with your code. Compare its coefficients with scikit-learn and inspect the residual plot above.
    """)
    return


@app.cell
def _():
    student_design = None
    print("student_design:", student_design)
    student_ols = None
    print("student_ols:", student_ols)
    student_residuals = None
    print("student_residuals:", student_residuals)
    return (student_design, student_ols, student_residuals)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_design = sm.add_constant(experience)
student_ols = sm.OLS(salary, student_design).fit()
student_residuals = salary - student_ols.predict(student_design)
```
The fitted coefficients agree with scikit-learn up to floating-point rounding. Residuals are observed minus fitted values. In a least-squares fit with an intercept, their sum is approximately zero.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. A squared term
    A curve would say the salary gap for one more year is different at 2 years of experience than at 10. One way to allow that is to add the square of experience as a second predictor:

    \[
    \text{mean salary} = \beta_0 + \beta_1 \times \text{years} + \beta_2 \times \text{years}^2.
    \]

    Hypotheses for the new term, before the fit:

    - \(H_0\): \(\beta_2 = 0\). The square does not change the mean once the straight line is included.
    - \(H_a\): \(\beta_2 \neq 0\).

    This model is linear in its coefficients, although its relationship with experience can be curved. There is one underlying explanatory variable represented by two terms.

    Adding a term to this nested least-squares model cannot decrease training R-squared. A small increase alone does not justify the extra term. A large p-value for the squared term means insufficient evidence of curvature under this model; it does not prove that curvature is absent. Consider residual patterns and, when prediction is the goal, performance on new data as well.

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
    The cell above prints both R-squared values and the table of p-values.
    1. Did \(R^2\) rise by more than a little when the squared term was added?
    2. Is the squared term's p-value below 0.05?
    3. What pattern in the scatter would have made you expect a squared term to matter?
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, ols_result, squared_result):
    _p = float(np.asarray(squared_result.pvalues, dtype=float)[2])
    _answers = f"""
1. **No:** Straight-line R-squared is {ols_result.rsquared:.4f}. With the square it is {squared_result.rsquared:.4f}.
2. **No:** The squared term's p-value is {_p:.3f}.
3. **A bend:** The vertical gap for one more year would change as experience increases: a curve, a flattening, or a steepening that a single slope cannot follow.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Extract the squared-term p-value and calculate the increase in training R-squared. Replace both `None` values with your code. Does a small increase in training R-squared by itself establish that the squared term is useful?
    """)
    return


@app.cell
def _():
    student_squared_p = None
    print("student_squared_p:", student_squared_p)
    student_r2_increase = None
    print("student_r2_increase:", student_r2_increase)
    return (student_squared_p, student_r2_increase)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r'''
```python
student_squared_p = squared_result.pvalues.iloc[2]
student_r2_increase = squared_result.rsquared - ols_result.rsquared
```
No. Training R-squared cannot decrease when this term is added. Interpret the p-value under the model assumptions, inspect residual patterns, and consider prediction on new data when relevant.
''')}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    The salary file shows a tight, rising line. Pearson's correlation is about 0.98, and the 5% test rejects a zero linear correlation. The least-squares slope is about 9,450 salary units per year of experience, with an intercept near 25,800. scikit-learn and statsmodels report that same line. statsmodels adds a t test: the slope is far from zero relative to its standard error. \(R^2\) is about 0.957, so the fitted model accounts for about 95.7% of the observed variation in salary in this table. This is training fit, not validated predictive performance.

    A squared term changes that summary very little. \(R^2\) barely moves, and the new coefficient's p-value is about 0.91. There is insufficient evidence that the squared term improves the conditional-mean model. This does not prove that every form of curvature is absent.

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
