# /// script
# dependencies = [
#     "marimo",
#     "seaborn",
#     "scipy",
#     "statsmodels",
# ]
# ///

import marimo

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
    - State a null hypothesis, an alternative, and a significance level before looking at the p-value.
    - Use a chi-square goodness-of-fit test on counts, with expected counts from a reference distribution.
    - Use a chi-square test of independence on a contingency table.
    - Use a one-sample \(z\) test for a proportion.
    - Choose a one-sample \(z\) or \(t\) test for a mean and name what each treats as known.
    - Say "do not reject" when the p-value is above the significance level.

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
    The goodness-of-fit test asks whether sample **counts** match a reference distribution.
    \(H_0\): the counts come from the reference proportions.
    \(H_a\): they do not.

    The categories below are fictitious. A reference population of 220,000 labels and two towns are constructed so the arithmetic is visible.
    Town X uses nearly the reference proportions. Town Y puts many more labels in one category.

    The chi-square statistic uses counts:

    $$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}, \qquad E_i = n \hat\pi_i.$$

    The original notebook multiplied relative frequencies by 100 and passed those numbers to `chisquare`. That rescaling is not a sample size, so the p-value does not answer the count question. This lesson uses the town counts and expected counts \(n\hat\pi_i\).
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
    ## 3. Chi-square test of independence
    Independence means that knowing one category does not change the probabilities of the other.
    \(H_0\): the two categorical variables are independent.
    \(H_a\): they are associated.

    `chi2_contingency` builds the expected counts from the row and column totals. Pass the table of counts, without the margin totals.

    The first table simulates 1,000 voters with race and party drawn separately, so the generating model is independence. A large p-value is the result that model leads you to expect. It is still one sample.
    The second table is a fixed cross-classification of degrees. "Bachelors" corrects the spelling in the source notebook.
    """)
    return


@app.cell
def _(np, pd):
    _rng = np.random.default_rng(10)
    voters = pd.DataFrame({
        "race": _rng.choice(["black", "hispanic", "white"], size=1_000, p=[0.20, 0.30, 0.50]),
        "party": _rng.choice(["democrat", "independent", "republican"], size=1_000, p=[0.40, 0.20, 0.40]),
    })
    voter_counts = pd.crosstab(voters["race"], voters["party"])
    return voter_counts, voters


@app.cell
def _(mo, pd, st, voter_counts):
    _stat, _pvalue, _df, _expected = st.chi2_contingency(voter_counts)
    voter_result = pd.DataFrame({
        "Chi_square": [_stat],
        "df": [_df],
        "p_value": [_pvalue],
        "Decision_at_0.05": ["Reject H0" if _pvalue <= 0.05 else "Do not reject H0"],
    })
    mo.Html(
        voter_counts.to_html(border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (voter_result,)


@app.cell
def _(mo, voter_result):
    mo.Html(
        voter_result.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(plt, voter_counts):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    voter_counts.plot(kind="bar", ax=_ax, rot=0, color=["#4C78A8", "#F58518", "#54A24B"])
    _ax.set(title="Simulated voters", ylabel="Count", xlabel="Race")
    _ax.legend(frameon=False, title="Party")
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
    The degree table has small counts. The chi-square p-value is an approximation that is less reliable when some expected counts are very small. Read it as an illustration of the calculation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why are the row and column totals left out of `chi2_contingency`?
    2. The voter race and party were simulated independently. What decision fits that design if the p-value is large?
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
    ## 4. One-sample test for a proportion
    For a large sample of yes/no outcomes, the \(z\) statistic compares the sample proportion \(\hat p\) with a hypothesized proportion \(p_0\):

    $$z = \frac{\hat p - p_0}{\sqrt{p_0(1-p_0)/n}}.$$

    `proportions_ztest` uses this statistic. The alternative may be two-sided, `larger`, or `smaller`.
    The code reads `../data/student-mat.csv`. Run the notebook with this lesson folder as the working directory.
    The question is about the proportion of students recorded with internet access at home.
    """)
    return


@app.cell
def _(mo, pd, proportions_ztest):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    internet_yes = int((student_file["internet"] == "yes").sum())
    internet_n = len(student_file)
    _rows = []
    for _p0, _alternative in [(0.80, "larger"), (0.90, "larger"), (0.85, "two-sided")]:
        _z, _pvalue = proportions_ztest(internet_yes, internet_n, value=_p0, alternative=_alternative)
        _rows.append({
            "H0": f"p = {_p0:.2f}",
            "Ha": {"larger": "p > p0", "smaller": "p < p0", "two-sided": "p ≠ p0"}[_alternative],
            "z": _z,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    proportion_results = pd.DataFrame(_rows)
    print(f"Internet = yes for {internet_yes} of {internet_n} students; sample proportion = {internet_yes / internet_n:.3f}.")
    mo.Html(
        proportion_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return internet_n, internet_yes, proportion_results


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
    ## 5. One-sample \(z\) test for a mean
    The \(z\) test compares a sample mean with a hypothesized mean.
    `statsmodels` `ztest` estimates the standard error from the sample and treats that estimate as the denominator of a standard normal statistic.
    That approximation is aimed at large samples. The normal model for the observations is a separate assumption.
    A \(t\) test keeps the same location comparison and uses a \(t\) reference distribution, which is the usual choice when the population standard deviation is unknown, including for smaller samples.

    The heights below are 200 draws from a normal distribution with mean 165 cm and standard deviation 10 cm.
    """)
    return


@app.cell
def _(np, pd, plt, ztest):
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
        _z, _pvalue = ztest(heights, value=170, alternative=_alternative)
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
    ## 6. One-sample \(t\) test for a mean
    `scipy.stats.ttest_1samp` is the one-sample \(t\) test.
    Its alternative labels are `two-sided`, `less`, and `greater`.
    The next sample has 20 heights drawn from a normal distribution with mean 169 and standard deviation 5.
    The original notebook drew that small sample and then plotted the large sample from the \(z\) section. The histogram here is the sample of 20.
    """)
    return


@app.cell
def _(np, pd, plt, st):
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
    for _alternative, _label in [("two-sided", "mean ≠ 170"), ("less", "mean < 170"), ("greater", "mean > 170")]:
        _t, _pvalue = st.ttest_1samp(small_heights, 170, alternative=_alternative)
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
    ## Conclusions
    - Write \(H_0\), \(H_a\), and the significance level before interpreting the p-value.
    - A p-value above the significance level means do not reject \(H_0\). It does not prove \(H_0\).
    - Goodness of fit compares observed counts with expected counts \(n\hat\pi_i\). Relative frequencies alone hide the sample size.
    - Independence uses a contingency table of counts. Leave the margin totals out of the test call.
    - A one-sample proportion \(z\) test compares \(\hat p\) with \(p_0\) using the null standard error.
    - A one-sample \(z\) test for a mean, as implemented by `ztest`, treats the estimated standard error as a normal denominator. The \(t\) test uses a \(t\) reference distribution when the population standard deviation is unknown.
    - The alternative controls which tail is the p-value. A two-sided test is not the smaller of the two one-sided tests.
    - Small expected counts make the chi-square approximation less trustworthy.

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
