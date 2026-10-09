# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "scipy",
#     "statsmodels",
# ]
# ///

import marimo

__generated_with = "0.25.1"

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as st
    from statsmodels.stats.proportion import proportions_ztest
    from statsmodels.stats.weightstats import ztest

    sns.set_style("whitegrid")
    return mo, np, pd, plt, proportions_ztest, sns, st, ztest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-sample Hypothesis Tests

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Formulate null and alternative hypotheses for a statistical question.
    - Apply chi-square tests to assess a categorical distribution or an association between categorical variables.
    - Apply one-sample hypothesis tests for population proportions and means.
    - Interpret test results and communicate conclusions in the context of the problem.

    ## 1. A decision, not a proof
    A hypothesis test compares a sample with a stated model.
    The **null hypothesis** \(H_0\) is the model you test.
    The **alternative** \(H_a\) is the departure you are looking for.
    The **p-value** is the probability, under \(H_0\), of a result at least as far from \(H_0\) as the one observed, in the direction of \(H_a\).

    This lesson uses significance level 0.05.
    A p-value at or below 0.05 is a reason to reject \(H_0\) at that level.
    A larger p-value is not evidence that \(H_0\) is true. The result is "do not reject \(H_0\)".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A test uses \(\alpha = 0.05\), and the p-value is 0.08.
    1. Do you reject \(H_0\)?
    2. Does the result prove that \(H_0\) is true?
    3. What would a p-value of 0.01 change?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Decision:** Do not reject \(H_0\). The p-value is larger than 0.05.
2. **Proof:** No. Failing to reject \(H_0\) means this sample did not supply enough evidence against it.
3. **Smaller p-value:** 0.01 is below 0.05, so the same rule rejects \(H_0\).
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Chi-square goodness of fit
    The goodness-of-fit test asks whether a sample's categorical distribution matches a specified distribution.
    \(H_0\): the population follows the reference proportions.
    \(H_a\): at least one population proportion differs.

    The data describe a fictitious reference population and two towns. We treat the reference proportions as the specified distribution. Town X has proportions close to the reference; Town Y has a much larger proportion in one category.

    You can describe the sample using frequencies or proportions. The test must retain the sample size. With observed proportions \(q_i\), reference proportions \(p_i\), and sample size \(n\),

    $$\chi^2=n\sum_i\frac{(q_i-p_i)^2}{p_i}.$$

    Equivalently, use observed frequencies \(O_i=nq_i\) and expected frequencies \(E_i=np_i\) in `chisquare`. Passing percentages directly would scale the calculation to a total of 100 instead of the actual sample size.

    Observations must be independent. The usual approximation guideline is that each expected frequency is at least 5. With four specified category probabilities, the test has three degrees of freedom.
    """)
    return


@app.cell
def _(np, pd):
    race_order = ["asian", "black", "hispanic", "white"]
    reference_counts = pd.Series({"asian": 10_000, "black": 50_000, "hispanic": 60_000, "white": 100_000}).loc[race_order]
    reference_proportions = reference_counts / reference_counts.sum()
    town_x_counts = pd.Series({"asian": 8, "black": 25, "hispanic": 30, "white": 60}).loc[race_order]
    town_y_counts = pd.Series({"asian": 8, "black": 25, "hispanic": 30, "white": 300}).loc[race_order]
    return race_order, reference_proportions, town_x_counts, town_y_counts


@app.cell
def _(mo, pd, reference_proportions, town_x_counts, town_y_counts):
    frequency_table = pd.DataFrame({
        "Reference_proportion": reference_proportions,
        "Town_X_proportion": town_x_counts / town_x_counts.sum(),
        "Town_Y_proportion": town_y_counts / town_y_counts.sum(),
    })
    mo.Html(
        frequency_table.round(3).to_html(border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (frequency_table,)


@app.cell
def _(frequency_table, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.5))
    frequency_table.plot(kind="bar", ax=_ax, rot=0, color=["#4C78A8", "#F58518", "#E45756"])
    _ax.set(title="Relative frequencies", ylabel="Proportion", xlabel="")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo, pd, reference_proportions, st, town_x_counts, town_y_counts):
    def _gof(counts):
        expected = reference_proportions.to_numpy() * counts.sum()
        stat, pvalue = st.chisquare(counts.to_numpy(), expected)
        return stat, pvalue, expected

    _rows = []
    for _name, _counts in [("Town X", town_x_counts), ("Town Y", town_y_counts)]:
        _stat, _pvalue, _expected = _gof(_counts)
        _rows.append({
            "Town": _name,
            "n": int(_counts.sum()),
            "Chi_square": _stat,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    gof_table = pd.DataFrame(_rows)
    mo.Html(
        gof_table.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (gof_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A reference distribution puts probability 0.25 on each of four categories.
    A sample of 40 observations has counts [10, 10, 10, 10].
    1. What are the four expected counts?
    2. What is the chi-square statistic?
    3. Town X and Town Y can have similar-looking bars and different p-values. What else enters the statistic?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Expected counts:** \(40 \times 0.25 = 10\) in each category.
2. **Statistic:** 0, because every observed count equals its expected count.
3. **Sample size:** The statistic uses the gaps between counts and expected counts. A larger sample makes the same proportional gap harder to attribute to chance. Town Y is larger and more concentrated than Town X.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A shop normally receives 40% of its orders online, 35% in person, and 25% by phone. A random sample of 200 recent orders contains the following counts, in that order.
    """)
    return


@app.cell
def _(np):
    order_counts = np.array([100, 60, 40])
    return (order_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Do these orders provide evidence that the population proportions differ from 40%, 35%, and 25%? State the hypotheses, calculate the test statistic and p-value, and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_order_result = None
    print("Test result:", student_order_result)
    return (student_order_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_order_result = st.chisquare(order_counts, f_exp=200 * np.array([0.40, 0.35, 0.25]))
```
The chi-square statistic is approximately 8.429 and the p-value is 0.0148. Reject \(H_0\): the sample provides evidence that the order proportions have changed.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Chi-square test of independence
    Independence means that knowing one category does not change the probabilities of the other.
    \(H_0\): the two categorical variables are independent.
    \(H_a\): they are associated.

    `chi2_contingency` builds the expected counts from the row and column totals. Pass the table of counts, without the margin totals.

    The first table describes 1,000 simulated community-centre visitors, classified by visit period (Morning, Afternoon, Evening) and preferred activity (Art, Music, Sport). The two variables are generated independently. Even independent variables can produce an uneven table by chance. At significance level 0.05, repeated independent simulations can still reject independence about 5% of the time.
    The second table records degree level and sex for 92 people who earned foreign-language degrees. We ask whether the distribution of degree levels differs between the two groups.

    For each cell, the expected count is its row total multiplied by its column total, divided by the overall total. The degrees of freedom are \((r-1)(c-1)\), where \(r\) and \(c\) are the numbers of rows and columns. Each person contributes to one cell; observations must be independent.
    """)
    return


@app.cell
def _(np, pd):
    _rng = np.random.default_rng(10)
    visitors = pd.DataFrame({
        "visit_period": _rng.choice(["Morning", "Afternoon", "Evening"], size=1_000, p=[0.20, 0.30, 0.50]),
        "activity": _rng.choice(["Art", "Music", "Sport"], size=1_000, p=[0.40, 0.20, 0.40]),
    })
    visitor_counts = pd.crosstab(visitors["visit_period"], visitors["activity"]).reindex(index=["Morning", "Afternoon", "Evening"], columns=["Art", "Music", "Sport"])
    return visitor_counts, visitors


@app.cell
def _(mo, pd, st, visitor_counts):
    _stat, _pvalue, _df, _expected = st.chi2_contingency(visitor_counts)
    visitor_result = pd.DataFrame({
        "Chi_square": [_stat],
        "df": [_df],
        "p_value": [_pvalue],
        "Decision_at_0.05": ["Reject H0" if _pvalue <= 0.05 else "Do not reject H0"],
    })
    mo.Html(
        visitor_counts.to_html(border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (visitor_result,)


@app.cell
def _(mo, visitor_result):
    mo.Html(
        visitor_result.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(plt, visitor_counts):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    visitor_counts.plot(kind="bar", ax=_ax, rot=0, color=["#4C78A8", "#F58518", "#54A24B"])
    _ax.set(title="Community-centre visitors", ylabel="Count", xlabel="Visit period")
    _ax.legend(frameon=False, title="Activity")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo, pd, st):
    degree_counts = pd.DataFrame(
        {"Bachelors": [10, 6], "Masters": [20, 9], "Doctorate": [30, 17]},
        index=["Male", "Female"],
    )
    _stat, _pvalue, _df, _expected = st.chi2_contingency(degree_counts)
    degree_result = pd.DataFrame({
        "Chi_square": [_stat],
        "df": [_df],
        "p_value": [_pvalue],
        "Decision_at_0.05": ["Reject H0" if _pvalue <= 0.05 else "Do not reject H0"],
    })
    mo.Html(
        degree_counts.to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return degree_counts, degree_result


@app.cell
def _(degree_result, mo):
    mo.Html(
        degree_result.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this degree table, the smallest **expected** count is approximately 5.57, so all expected counts meet the usual guideline of at least 5. The guideline concerns expected counts, rather than the observed counts. A large p-value means the sample does not provide enough evidence of an association; it does not establish independence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why are the row and column totals left out of `chi2_contingency`?
    2. Visit period and activity preference were simulated independently. What decision fits that design if the p-value is large?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Margins:** The totals are sums of the cells. Including them would count the same observations twice. The function rebuilds the expected counts from those totals.
2. **Independent simulation:** Do not reject \(H_0\). A large p-value agrees with the way the table was generated. It does not prove independence from one sample.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A community centre surveys independent visitors about their preferred activity. The table records activity preference by visit period.
    """)
    return


@app.cell
def _(mo, pd):
    activity_counts = pd.DataFrame({"Art": [30, 15], "Sport": [20, 35], "Music": [25, 25]}, index=["Morning", "Evening"])
    mo.Html(activity_counts.to_html(border=0))
    return (activity_counts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Is activity preference associated with visit period in the population? State the hypotheses, calculate the test statistic and p-value, and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_activity_result = None
    print("Test result:", student_activity_result)
    return (student_activity_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_activity_result = st.chi2_contingency(activity_counts)
```
The chi-square statistic is approximately 9.091 and the p-value is 0.0106. Reject \(H_0\): activity preference and visit period appear associated. The result also includes the degrees of freedom and expected counts.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. One-sample test for a proportion
    For a large sample of yes/no outcomes, the \(z\) statistic compares the sample proportion \(\hat p\) with a hypothesized proportion \(p_0\):

    $$z=\frac{\hat p-p_0}{\sqrt{\hat p(1-\hat p)/n}}.$$

    This implementation uses `proportions_ztest` with its default **sample-based variance**. The denominator estimates the standard error from the sample proportion \(\hat p\). A different convention uses \(p_0\) in the variance; the calculations in this lesson consistently use the sample-based version.
    The alternatives are `two-sided`, `larger`, and `smaller`.

    Observations must be independent, and the normal approximation needs sufficiently many expected successes and failures. A common guideline is \(np_0 \geq 10\) and \(n(1-p_0) \geq 10\).
    The code reads `../data/student-mat.csv`. Run the notebook with this lesson folder as the working directory.
    This file records students at two Portuguese secondary schools. The `internet` column says whether each student has internet access at home. We count `yes` as a success. Generalizing beyond these students requires an appropriate sampling design.

    We show three research questions: is the population proportion above 0.80, above 0.90, or different from 0.85? Each row corresponds to a different alternative. These are teaching examples, rather than a reason to choose an alternative after inspecting the results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exploring the student data
    Each row describes one student. First, read the semicolon-separated file and inspect the first five rows. The full dataset remains available for the calculations; the preview displays only five students.
    """)
    return


@app.cell
def _(mo, pd):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    print("Rows and columns:", student_file.shape)
    mo.Html(student_file.head().to_html(index=False, border=0))
    return (student_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `internet` column describes **internet access at home**, rather than how often students use the internet. Inspect its values and count the students in each category.
    """)
    return


@app.cell
def _(mo, student_file):
    print("Internet categories:", student_file["internet"].unique())
    internet_counts = student_file["internet"].value_counts().rename_axis("Internet access").reset_index(name="Students")
    mo.Html(internet_counts.to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Encode `yes` as 1 and `no` as 0. The sum then counts the students with access, and the mean is the sample proportion with access. This binary sample is the input to our reusable proportion-test function.
    """)
    return


@app.cell
def _(student_file):
    internet = (student_file["internet"] == "yes").astype(int).to_numpy()
    internet_yes = int(internet.sum())
    internet_n = len(internet)
    print("Students with internet access:", internet_yes)
    print("Number of students:", internet_n)
    print("Sample proportion:", round(internet.mean(), 3))
    return internet, internet_n, internet_yes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A reusable proportion-test function
    `one_sample_prop` takes a sample coded as 0 and 1, a hypothesized population proportion, a significance level, and an alternative. It prints the test name, both hypotheses, the sample proportion, the statistic, the p-value, and the decision. It also returns the statistic and p-value for later use.

    The function uses `proportions_ztest` with its default sample-based variance. The significance level defaults to 0.05; the hypothesized proportion is a required argument.
    """)
    return


@app.cell
def _(np, proportions_ztest):
    def one_sample_prop(sample, population_prop, alpha=0.05, alternative="two-sided"):
        """Report a one-sample proportion z test for a sample coded as 0 and 1."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- One-sample z test for a proportion ---")
        print(f"H0: p = {population_prop:g}")
        print(f"Ha: p {signs[alternative]} {population_prop:g}")
        print(f"Sample proportion = {np.mean(sample):.3f}")
        z_stat, pval = proportions_ztest(
            np.sum(sample), len(sample), value=population_prop, alternative=alternative
        )
        print(f"z statistic = {z_stat:.3f}; p-value = {pval:.4f}")
        if pval <= alpha:
            print(f"p-value ≤ {alpha:g}: Reject H0.")
        else:
            print(f"p-value > {alpha:g}: Do not reject H0.")
        return z_stat, pval
    return (one_sample_prop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the population proportion above 80%?
    Let \(p\) be the population proportion of students with internet access at home.
    We test \(H_0:p=0.80\) against \(H_a:p>0.80\), at \(\alpha=0.05\).
    The alternative looks for a proportion above 0.80, so this is a right-tailed test.
    """)
    return


@app.cell
def _(internet, one_sample_prop):
    internet_80_z, internet_80_p = one_sample_prop(internet, 0.80, alternative="larger")
    return internet_80_p, internet_80_z


@app.cell(hide_code=True)
def _(internet_80_p, mo):
    mo.md(rf"""
    The p-value is **{internet_80_p:.4f}**, below 0.05. Reject \(H_0\): the sample provides evidence that the population proportion exceeds 80%, under the sampling and independence assumptions of the test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the population proportion above 90%?
    Now test \(H_0:p=0.90\) against \(H_a:p>0.90\), again at \(\alpha=0.05\).
    The sample has not changed, but the reference proportion has. The sample proportion is below 0.90, so it points in the opposite direction from this alternative.
    """)
    return


@app.cell
def _(internet, one_sample_prop):
    internet_90_z, internet_90_p = one_sample_prop(internet, 0.90, alternative="larger")
    return internet_90_p, internet_90_z


@app.cell(hide_code=True)
def _(internet_90_p, mo):
    mo.md(rf"""
    The p-value is **{internet_90_p:.4f}**, above 0.05. Do not reject \(H_0\). The sample does not provide evidence that the population proportion exceeds 90%. This result does **not** establish that the population proportion is 90% or less.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does the population proportion differ from 85%?
    Test \(H_0:p=0.85\) against \(H_a:p\ne0.85\), at \(\alpha=0.05\).
    This is a two-sided question: either a lower or a higher proportion would count as a departure from the null hypothesis. A test can detect evidence against equality; it cannot prove equality.
    """)
    return


@app.cell
def _(internet, one_sample_prop):
    internet_85_z, internet_85_p = one_sample_prop(internet, 0.85)
    return internet_85_p, internet_85_z


@app.cell(hide_code=True)
def _(internet_85_p, mo):
    mo.md(rf"""
    The p-value is **{internet_85_p:.4f}**, above 0.05. Do not reject \(H_0\): there is insufficient evidence that the population proportion differs from 85%. This does not prove it equals 85%.
    """)
    return


@app.cell
def _(internet_80_z, internet_80_p, internet_90_z, internet_90_p, internet_85_z, internet_85_p, mo, pd):
    proportion_results = pd.DataFrame({
        "H0": ["p = 0.80", "p = 0.90", "p = 0.85"],
        "Ha": ["p > 0.80", "p > 0.90", "p ≠ 0.85"],
        "z": [internet_80_z, internet_90_z, internet_85_z],
        "p_value": [internet_80_p, internet_90_p, internet_85_p],
    })
    proportion_results["Decision_at_0.05"] = ["Reject H0" if _p <= 0.05 else "Do not reject H0" for _p in proportion_results["p_value"]]
    mo.Html(proportion_results.round(4).to_html(index=False, border=0))
    return (proportion_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Forty successes in 100 trials are compared with \(p_0 = 0.50\), two-sided.
    1. Is the sample proportion above or below 0.50?
    2. If the p-value is 0.045, what is the decision at the 0.05 level?
    3. In the student table, is the sample proportion greater than 0.90?
    """)
    return


@app.cell(hide_code=True)
def _(internet_n, internet_yes, mo, proportion_results):
    _row = proportion_results.loc[proportion_results["H0"] == "p = 0.90"].iloc[0]
    _answers = rf"""
1. **Sample proportion:** 40/100 = 0.40, which is below 0.50.
2. **Decision:** Reject \(H_0\), because 0.045 is below 0.05.
3. **Student file:** The sample proportion is {internet_yes / internet_n:.3f}. The one-sided test of \(H_0: p = 0.90\) against \(p > 0.90\) has decision "{_row['Decision_at_0.05']}".
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a random sample of 120 independent library visitors, 80 say they use the library’s online catalogue.
    """)
    return


@app.cell
def _():
    catalogue_successes = 80
    catalogue_n = 120
    return catalogue_successes, catalogue_n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Does this sample provide evidence that more than 60% of library visitors use the online catalogue? State the hypotheses, calculate the test statistic and p-value, and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_catalogue_result = None
    print("Test result:", student_catalogue_result)
    return (student_catalogue_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_catalogue_result = proportions_ztest(catalogue_successes, catalogue_n, value=0.60, alternative="larger")
```
The statistic is approximately 1.549 and the p-value is 0.0607. Do not reject \(H_0\): this sample does not supply enough evidence that the population proportion exceeds 60%.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. One-sample \(z\) test for a mean
    The \(z\) test compares a sample mean with a hypothesized mean.
    `statsmodels` `ztest` estimates the standard error from the sample and treats that estimate as the denominator of a standard normal statistic.
    Here the statistic is \(z=(\bar x-\mu_0)/(s/\sqrt n)\). A negative value means the sample mean is below the hypothesized mean. The p-value measures how extreme that value is under the chosen alternative.

    This is a large-sample normal approximation using the **sample** standard deviation. A classical test with a known population standard deviation instead uses \(\sigma/\sqrt n\). The `ztest` calls below do not take a known population standard deviation. Observations must be independent; a large sample does not fix biased sampling.
    A \(t\) test keeps the same location comparison and uses a \(t\) reference distribution, which is the usual choice when the population standard deviation is unknown, including for smaller samples.

    Our `one_sample_ztest` function prints a complete test report and returns the statistic and p-value.

    The heights below represent 200 fictitious students, generated from a normal distribution with mean 165 cm and standard deviation 10 cm. The seed makes the sample reproducible. The test estimates the standard deviation from these heights.

    We compare the population mean with 170 cm. The three alternatives illustrate how the same statistic gives different p-values. In an actual investigation, choose the alternative before examining the sample.
    """)
    return


@app.cell
def _(ztest, np):
    def one_sample_ztest(sample, population_value, alpha=0.05, alternative="two-sided"):
        """Report a one-sample z test for a population mean."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- One-sample z test for a mean ---")
        print(f"H0: mean = {population_value:g}")
        print(f"Ha: mean {signs[alternative]} {population_value:g}")
        print(f"Sample mean = {np.mean(sample):.3f}")
        z_stat, pval = ztest(sample, value=population_value, alternative=alternative)
        print(f"z statistic = {z_stat:.3f}; p-value = {pval:.4f}")
        if pval <= alpha:
            print(f"p-value ≤ {alpha:g}: Reject H0.")
        else:
            print(f"p-value > {alpha:g}: Do not reject H0.")
        return z_stat, pval
    return (one_sample_ztest,)


@app.cell
def _(np, one_sample_ztest, pd, plt):
    heights = np.random.default_rng(2026).normal(165, 10, size=200)
    _fig, _ax = plt.subplots(figsize=(6, 3.3))
    _ax.hist(heights, bins=20, color="#4C78A8", edgecolor="white")
    _ax.axvline(heights.mean(), color="#E45756", linewidth=2, label=f"Sample mean {heights.mean():.2f}")
    _ax.axvline(170, color="#F58518", linestyle="--", label="Hypothesized mean 170")
    _ax.set(title="Simulated heights", xlabel="Height (cm)", ylabel="Count")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _rows = []
    for _alternative in ("two-sided", "smaller", "larger"):
        _z, _pvalue = one_sample_ztest(heights, 170, alternative=_alternative)
        _rows.append({
            "Ha": {"two-sided": "mean ≠ 170", "smaller": "mean < 170", "larger": "mean > 170"}[_alternative],
            "z": _z,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    z_results = pd.DataFrame(_rows)
    _fig
    return heights, z_results


@app.cell
def _(mo, z_results):
    mo.Html(
        z_results.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo, z_results):
    _two = z_results.iloc[0]
    _lower = z_results.iloc[1]
    mo.md(f"""
    The two-sided test has p-value {_two['p_value']:.4f}: **{_two['Decision_at_0.05']}**. {"The sample provides evidence that the population mean differs from 170 cm." if _two['p_value'] <= 0.05 else "The sample does not provide enough evidence that the population mean differs from 170 cm."}

    The lower-tailed test has p-value {_lower['p_value']:.4f}: **{_lower['Decision_at_0.05']}**. {"The sample provides evidence that the population mean is below 170 cm." if _lower['p_value'] <= 0.05 else "The sample does not provide enough evidence that the population mean is below 170 cm."}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The sample mean is below 170.
    1. Which alternative, `larger` or `smaller`, is the direction the sample points?
    2. Can the two-sided p-value be smaller than both one-sided p-values?
    """)
    return


@app.cell(hide_code=True)
def _(heights, mo, z_results):
    _answers = f"""
1. **Direction:** `smaller`, because the sample mean is {heights.mean():.2f}, below 170.
2. **Two-sided p-value:** No. The two-sided p-value counts both tails, so it is larger than the one-sided p-value in the observed direction. Compare the three rows in the table.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A bottling line aims for a mean fill volume of 500 mL. The following simulated random sample represents 150 independent bottles. The seed keeps the lesson data reproducible.
    """)
    return


@app.cell
def _(np):
    bottle_volumes = np.random.default_rng(2027).normal(502, 8, size=150)
    print("First 10 bottle volumes (mL):", bottle_volumes[:10].round(2))
    return (bottle_volumes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Does the sample provide evidence that the population mean fill volume differs from 500 mL? State the hypotheses, calculate the test statistic and p-value, and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_bottle_result = None
    print("Test result:", student_bottle_result)
    return (student_bottle_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_bottle_result = ztest(bottle_volumes, value=500, alternative="two-sided")
```
The statistic is approximately 2.858 and the p-value is 0.0043. Reject \(H_0\): the sample provides evidence that the population mean fill volume differs from 500 mL.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. One-sample \(t\) test for a mean
    `scipy.stats.ttest_1samp` is the one-sample \(t\) test.
    Its alternative labels are `two-sided`, `less`, and `greater`. Our reporting function `one_sample_ttest` accepts `two-sided`, `smaller`, and `larger`, and translates them for SciPy. It also reports the degrees of freedom.
    The next sample has 20 heights drawn from a normal distribution with mean 169 and standard deviation 5.
    These are 20 different fictitious students. The histogram shows this smaller sample.

    The statistic is \(t=(\bar x-\mu_0)/(s/\sqrt n)\), with \(n-1\) degrees of freedom. Estimating the population standard deviation adds uncertainty: the \(t\) distribution has heavier tails than the normal distribution.

    Observations must be independent. For a small sample, the population should be approximately normal, without strong skewness or extreme outliers. A \(t\) test can also be used for larger samples; 30 is not a strict cutoff.
    """)
    return


@app.cell
def _(st, np):
    def one_sample_ttest(sample, population_value, alpha=0.05, alternative="two-sided"):
        """Report a one-sample t test for a population mean."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- One-sample t test for a mean ---")
        print(f"H0: mean = {population_value:g}")
        print(f"Ha: mean {signs[alternative]} {population_value:g}")
        print(f"Sample mean = {np.mean(sample):.3f}")
        scipy_alternative = {"two-sided": "two-sided", "smaller": "less", "larger": "greater"}[alternative]
        t_stat, pval = st.ttest_1samp(sample, population_value, alternative=scipy_alternative)
        print(f"t statistic = {t_stat:.3f}; p-value = {pval:.4f}")
        print(f"Degrees of freedom = {len(sample) - 1}")
        if pval <= alpha:
            print(f"p-value ≤ {alpha:g}: Reject H0.")
        else:
            print(f"p-value > {alpha:g}: Do not reject H0.")
        return t_stat, pval
    return (one_sample_ttest,)


@app.cell
def _(np, one_sample_ttest, pd, plt):
    small_heights = np.random.default_rng(7).normal(169, 5, size=20)
    _fig, _ax = plt.subplots(figsize=(6, 3.3))
    _ax.hist(small_heights, bins=8, color="#54A24B", edgecolor="white")
    _ax.axvline(small_heights.mean(), color="#E45756", linewidth=2, label=f"Sample mean {small_heights.mean():.2f}")
    _ax.axvline(170, color="#F58518", linestyle="--", label="Hypothesized mean 170")
    _ax.set(title="Small height sample", xlabel="Height (cm)", ylabel="Count")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _rows = []
    for _alternative, _label in [("two-sided", "mean ≠ 170"), ("smaller", "mean < 170"), ("larger", "mean > 170")]:
        _t, _pvalue = one_sample_ttest(small_heights, 170, alternative=_alternative)
        _rows.append({
            "Ha": _label,
            "t": _t,
            "df": len(small_heights) - 1,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    t_results = pd.DataFrame(_rows)
    _fig
    return small_heights, t_results


@app.cell
def _(mo, t_results):
    mo.Html(
        t_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo, t_results):
    _two = t_results.iloc[0]
    _lower = t_results.iloc[1]
    mo.md(f"""
    The two-sided test has p-value {_two['p_value']:.4f}: **{_two['Decision_at_0.05']}**. {"The sample provides evidence that the population mean differs from 170 cm." if _two['p_value'] <= 0.05 else "The sample does not provide enough evidence that the population mean differs from 170 cm."}

    The lower-tailed test has p-value {_lower['p_value']:.4f}: **{_lower['Decision_at_0.05']}**. {"The sample provides evidence that the population mean is below 170 cm." if _lower['p_value'] <= 0.05 else "The sample does not provide enough evidence that the population mean is below 170 cm."}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. How many degrees of freedom does this \(t\) test use?
    2. A rule of thumb says "use \(t\) below 30 and \(z\) above 30." Is that a proof that the \(z\) test becomes exact at 30?
    """)
    return


@app.cell(hide_code=True)
def _(mo, small_heights):
    _answers = rf"""
1. **Degrees of freedom:** {len(small_heights) - 1}.
2. **Cutoff:** No. Thirty is a rough classroom boundary. With an unknown population standard deviation, the \(t\) reference matches the normal-sample derivation at every sample size. The normal approximation to that \(t\) distribution improves as \(n\) grows.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following fixed sample records completion times, in minutes, for 12 independently selected participants completing a task. Assume the population of completion times is approximately normal.
    """)
    return


@app.cell
def _(np):
    task_times = np.array([26, 29, 25, 31, 27, 28, 24, 30, 26, 27, 29, 25])
    print("Task times (minutes):", task_times)
    return (task_times,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Do these completion times provide evidence that the population mean is less than 30 minutes? State the hypotheses, calculate the test statistic and p-value, and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_task_result = None
    print("Test result:", student_task_result)
    return (student_task_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_task_result = st.ttest_1samp(task_times, popmean=30, alternative="less")
```
The statistic is approximately −4.371 and the p-value is 0.00056. Reject \(H_0\): the sample provides evidence that the population mean completion time is below 30 minutes.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Hypotheses express the population question and the direction of the proposed difference.
    - Chi-square goodness of fit assesses a categorical distribution; chi-square independence assesses association between categorical variables.
    - One-sample proportion and mean tests compare sample evidence with a hypothesized population value.
    - The p-value and significance level determine whether to reject the null hypothesis. Failing to reject does not establish that it is true.
    - Statistical conclusions must answer the original question and account for the test's assumptions.

    ## Check your understanding
    1. Expected counts are [10, 10] and observed counts are [10, 10]. What is \(\chi^2\)?
    2. A contingency test includes a row called `total`. What should you do with that row?
    3. Sample proportion 0.62, hypothesized proportion 0.50, alternative `larger`. Which side of 0.50 agrees with \(H_a\)?
    4. The one-sided p-value in the observed direction is 0.03. Is the two-sided p-value smaller than 0.03?
    5. A sample of size 12 has an unknown population standard deviation. Which reference distribution does `ttest_1samp` use?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Statistic:** 0.
2. **Total row:** Remove it. The test needs the cell counts, and the function computes the totals itself.
3. **Direction:** Above 0.50. The alternative `larger` looks for a proportion above the hypothesized value.
4. **Two-sided:** No. It is about twice the one-sided p-value when the statistic is on one side of the null.
5. **Reference:** A \(t\) distribution with 11 degrees of freedom.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    - [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), for the internet variable.
    """)
    return


if __name__ == "__main__":
    app.run()
