# /// script
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "seaborn", "scipy", "statsmodels"]
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
    from scipy import stats as st
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    sns.set_style("whitegrid")
    return AnovaRM, mo, np, pairwise_tukeyhsd, pd, plt, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparing More Than Two Groups with Python
    ## Learning goals
    By the end of this lesson, you should be able to:
    - Distinguish independent groups from repeated measurements on the same subjects.
    - Apply analysis of variance to compare population means across several groups or conditions.
    - Use pairwise comparisons to identify differences between population means.
    - Apply rank-based procedures to independent and repeated-measures data.
    - Interpret test results and communicate conclusions in the context of the problem.

    We will compare separate groups of students, then measurements from the same students under several conditions. The examples use simulated scores and seeded random generators. The design determines whether the observations should be treated as independent groups or repeated measurements.

    ## 1. One-way analysis of variance
    One-way analysis of variance (ANOVA) compares the population means of independent groups. Although its name refers to variance, the hypothesis concerns **means**. The F statistic compares variability between group means with variability within the groups.

    $$H_0:\mu_1=\mu_2=\mu_3,\qquad H_a:\text{at least one population mean differs}.$$

    A rejection does not identify which pairs differ. A non-rejection does not establish equality.

    The ordinary one-way procedure assumes independent observations, approximately normal errors within groups, and equal population variances. Students in different groups are not matched, and each contributes one score. Normality matters most for small samples or pronounced outliers.

    ### Simulated scores from three independent groups
    We generate 50 scores with population mean 80, 48 with mean 82, and 52 with mean 83. All three populations have standard deviation 5, matching the equal-variance model used here. The sample spreads will still differ because of sampling variation.
    """)
    return


@app.cell
def _(np):
    _rng = np.random.RandomState(50)
    g1_grades = _rng.normal(80, 5, size=50)
    g2_grades = _rng.normal(82, 5, size=48)
    g3_grades = _rng.normal(83, 5, size=52)
    print("First 10 Group 1 scores:", g1_grades[:10].round(2))
    print("First 10 Group 2 scores:", g2_grades[:10].round(2))
    print("First 10 Group 3 scores:", g3_grades[:10].round(2))
    return g1_grades, g2_grades, g3_grades


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the score distributions
    The kernel density curves are smooth summaries of the observed scores. Their locations show where scores concentrate, and their widths show spread. Each curve has area one; density is not a student count. Overlap describes the samples but does not determine whether the population means differ. ANOVA also accounts for sample size and within-group variability.
    """)
    return


@app.cell
def _(g1_grades, g2_grades, g3_grades, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=g1_grades, fill=True, label="Group 1", ax=_ax)
    sns.kdeplot(x=g2_grades, fill=True, label="Group 2", ax=_ax)
    sns.kdeplot(x=g3_grades, fill=True, label="Group 3", ax=_ax)
    _ax.set(title="Scores from three independent groups", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Organizing the observations
    The combined DataFrame uses one row per student. `Group` identifies the group and `Grades` contains the observed score. Unequal group sizes are allowed. The summary table reports each sample size, mean, and sample standard deviation.
    """)
    return


@app.cell
def _(g1_grades, g2_grades, g3_grades, mo, pd):
    dfG1 = pd.DataFrame({"Group": "Group 1", "Grades": g1_grades})
    dfG2 = pd.DataFrame({"Group": "Group 2", "Grades": g2_grades})
    dfG3 = pd.DataFrame({"Group": "Group 3", "Grades": g3_grades})
    data = pd.concat([dfG1, dfG2, dfG3], ignore_index=True)
    mo.Html(data.head(10).round(2).to_html(index=False, border=0))
    return data, dfG1, dfG2, dfG3


@app.cell
def _(data, mo):
    group_summary = data.groupby("Group")["Grades"].agg(n="size", Mean="mean", SD="std").reset_index()
    mo.Html(group_summary.round(3).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A reusable ANOVA report
    `ANOVA` accepts the score arrays, prints the hypotheses and sample summaries, and reports F, its degrees of freedom, the p-value, and the decision. It returns `(F, p_value)` so the result can be reused. Sample standard deviations use divisor n−1. We use significance level 0.05 unless another value is supplied.
    """)
    return


@app.cell
def _(np, st):
    def ANOVA(*arrays, alpha=0.05):
        """Report an ordinary independent-groups ANOVA and return F and p-value."""
        arrays = [np.asarray(a) for a in arrays]
        print("--- One-way ANOVA ---")
        print("H0: all population means are equal")
        print("Ha: at least one population mean differs")
        for i, a in enumerate(arrays, start=1):
            print(f"Group {i}: n = {len(a)}, mean = {a.mean():.3f}, sample SD = {a.std(ddof=1):.3f}")
        statistic, p_value = st.f_oneway(*arrays)
        print(f"F = {statistic:.4g}; p-value = {p_value:.4g}")
        print(f"Degrees of freedom: {len(arrays) - 1}, {sum(len(a) for a in arrays) - len(arrays)}")
        print(f"Significance level = {alpha:g}")
        print("There is sufficient evidence to reject the null hypothesis." if p_value <= alpha else "There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value
    return (ANOVA,)


@app.cell
def _(g1_grades, g2_grades, g3_grades, ANOVA):
    anova_three = ANOVA(g1_grades, g2_grades, g3_grades)
    return (anova_three,)


@app.cell
def _(anova_three, mo):
    _p = anova_three[1]
    mo.md("At 0.05, the scores provide evidence that at least one of the three population means differs. This does not identify the differing pair." if _p <= 0.05 else "At 0.05, the scores do not provide sufficient evidence that the three population means differ. This does not establish that they are equal.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding a fourth independent group
    We now generate 44 scores from a fourth population with mean 60 and standard deviation 5. This adds a new group to the comparison; it does not change the original three groups. The next graph shows how its observed scores compare with theirs.
    """)
    return


@app.cell
def _(np, data, mo, pd):
    g4_grades = np.random.default_rng(51).normal(60, 5, size=44)
    dfG4 = pd.DataFrame({"Group": "Group 4", "Grades": g4_grades})
    data2 = pd.concat([data, dfG4], ignore_index=True)
    print("First 10 Group 4 scores:", g4_grades[:10].round(2))
    mo.Html(data2.groupby("Group")["Grades"].agg(n="size", Mean="mean", SD="std").reset_index().round(3).to_html(index=False, border=0))
    return data2, g4_grades


@app.cell
def _(g1_grades, g2_grades, g3_grades, g4_grades, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=g1_grades, fill=True, label="Group 1", ax=_ax)
    sns.kdeplot(x=g2_grades, fill=True, label="Group 2", ax=_ax)
    sns.kdeplot(x=g3_grades, fill=True, label="Group 3", ax=_ax)
    sns.kdeplot(x=g4_grades, fill=True, label="Group 4", ax=_ax)
    _ax.set(title="Scores from four independent groups", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Group 4 is concentrated at lower scores. Test whether all four population means are equal; the alternative is that at least one differs.
    """)
    return


@app.cell
def _(g1_grades, g2_grades, g3_grades, g4_grades, ANOVA):
    anova_four = ANOVA(g1_grades, g2_grades, g3_grades, g4_grades)
    return (anova_four,)


@app.cell
def _(anova_four, mo):
    _p = anova_four[1]
    mo.md("Reject equality of all four population means at 0.05. The plot suggests where a difference may lie, but pairwise comparisons are needed to identify supported mean differences." if _p <= 0.05 else "There is insufficient evidence to reject equality of all four population means at 0.05.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A new problem: delivery services
    Three independent services record delivery times in minutes. The observations below belong to different deliveries, with no matching across services. For this exercise, assume the ordinary ANOVA model is appropriate.
    """)
    return


@app.cell
def _(np, mo, pd):
    delivery_A = np.array([28, 31, 29, 32, 30, 27, 33, 29])
    delivery_B = np.array([34, 32, 35, 33, 36, 31, 34, 35])
    delivery_C = np.array([30, 33, 31, 34, 32, 29, 35, 31])
    delivery_data = pd.DataFrame({"Time": np.concatenate([delivery_A, delivery_B, delivery_C]), "Service": np.repeat(["A", "B", "C"], 8)})
    mo.Html(pd.DataFrame({"Service A (min)": delivery_A, "Service B (min)": delivery_B, "Service C (min)": delivery_C}).to_html(index=False, border=0))
    return delivery_A, delivery_B, delivery_C, delivery_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Do the data provide evidence that the population mean delivery times differ among the three services? State the hypotheses, calculate the test result, and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_delivery_anova = None
    return (student_delivery_anova,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_delivery_anova = ANOVA(delivery_A, delivery_B, delivery_C)
    ```
    H0: μA = μB = μC; Ha: at least one population mean differs. Here F ≈ 8.167 and p ≈ 0.002378. Reject at 0.05: there is evidence of a difference, but this test does not identify which service pairs differ.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Tukey pairwise comparisons
    Tukey's honestly significant difference procedure compares every pair of population means while accounting for the family of pairwise comparisons. The ordinary procedure used here shares the independent normal-error and common-variance assumptions of ordinary ANOVA. With unequal sample sizes, it uses the Tukey–Kramer form.

    For each pair, H0 is equality of its population means and Ha is a difference. The table reports the estimated difference as **group 2 minus group 1**, an adjusted p-value, lower and upper simultaneous confidence limits, and `reject`. An interval excluding zero indicates a supported difference at the selected level.

    Tukey comparisons can be specified as part of the analysis plan; an ANOVA rejection is not a mathematical requirement for computing them. Here we show the three-group comparison first, then use the four-group example to investigate its overall difference. Do not search among procedures for whichever gives the smallest p-value.
    """)
    return


@app.cell
def _(data, pairwise_tukeyhsd):
    tukey_test = pairwise_tukeyhsd(endog=data["Grades"], groups=data["Group"], alpha=0.05)
    return (tukey_test,)


@app.cell
def _(tukey_test, mo, pd):
    tukey_three_table = pd.DataFrame(tukey_test.summary().data[1:], columns=tukey_test.summary().data[0])
    mo.Html(tukey_three_table.to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the Tukey graph
    The graph places each sample mean on the score axis and shows Tukey **comparison intervals** around it. These are constructed for comparing groups; they are not the pairwise difference intervals printed in the table, nor ordinary separate confidence intervals for each mean. Use the adjusted p-values, difference intervals, and `reject` column in the table for the pairwise decisions.
    """)
    return


@app.cell
def _(tukey_test, plt):
    _fig = tukey_test.plot_simultaneous(figsize=(8, 4))
    _fig.axes[0].set(title="Tukey comparison intervals: three groups", xlabel="Score")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(tukey_test, mo):
    _labels = [str(a) + " versus " + str(b) for a, b, _, _, _, _, rejected in tukey_test.summary().data[1:] if rejected]
    mo.md("At 0.05, Tukey supports these population mean differences: " + "; ".join(_labels) + "." if _labels else "At 0.05, none of the three pairwise comparisons provides sufficient evidence of a population mean difference. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now compare all four groups. The Tukey table identifies which population mean pairs have evidence of a difference after accounting for the six comparisons.
    """)
    return


@app.cell
def _(data2, pairwise_tukeyhsd):
    tukey_test2 = pairwise_tukeyhsd(endog=data2["Grades"], groups=data2["Group"], alpha=0.05)
    return (tukey_test2,)


@app.cell
def _(tukey_test2, mo, pd):
    tukey_four_table = pd.DataFrame(tukey_test2.summary().data[1:], columns=tukey_test2.summary().data[0])
    mo.Html(tukey_four_table.to_html(index=False, border=0))
    return


@app.cell
def _(tukey_test2, plt):
    _fig = tukey_test2.plot_simultaneous(figsize=(8, 4))
    _fig.axes[0].set(title="Tukey comparison intervals: four groups", xlabel="Score")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(tukey_test2, mo):
    _labels = [str(a) + " versus " + str(b) for a, b, _, _, _, _, rejected in tukey_test2.summary().data[1:] if rejected]
    mo.md("At 0.05, Tukey supports these population mean differences: " + "; ".join(_labels) + "." if _labels else "At 0.05, no pairwise population mean difference is supported.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    For the delivery services, compare the population mean delivery times pair by pair at significance level 0.05. Which pairs have evidence of a difference after accounting for multiple comparisons?
    """)
    return


@app.cell
def _():
    student_delivery_tukey = None
    return (student_delivery_tukey,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_delivery_tukey = pairwise_tukeyhsd(delivery_data["Time"], delivery_data["Service"], alpha=0.05)
    ```
    Read `student_delivery_tukey.summary()`. Service A versus B has a supported difference; A versus C and B versus C do not at 0.05. Unflagged comparisons do not establish equality.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Repeated-measures ANOVA
    A repeated-measures design records the same subjects under every condition. Students are independent of other students, but measurements from the same student may be dependent. The null is equality of the population condition means; the alternative is that at least one differs.

    The normal-error model applies to within-subject contrasts. For more than two conditions, ordinary repeated-measures ANOVA assumes **sphericity**: the population variances of differences between all pairs of conditions are equal. `AnovaRM` supports complete, balanced repeated-measures data and does not automatically correct for violations of sphericity. The simulated model below satisfies this assumption by construction.

    ### Simulated scores from the same 50 students
    Each student has a baseline score from a normal population with mean 60 and standard deviation 5. We add an independent normal error with standard deviation 4 for each test. Test 1 has no population shift; Tests 2 and 3 add one point. Using a shared baseline makes the measurements from the same student related. Every row represents the same student across all tests.
    """)
    return


@app.cell
def _(np, mo, pd):
    _rng = np.random.RandomState(50)
    student_baseline = _rng.normal(60, 5, size=50)
    test1 = student_baseline + _rng.normal(0, 4, size=50)
    test2 = student_baseline + 1 + _rng.normal(0, 4, size=50)
    test3 = student_baseline + 1 + _rng.normal(0, 4, size=50)
    repeated = pd.DataFrame({"Student": np.arange(1, 51), "Test 1": test1, "Test 2": test2, "Test 3": test3})
    mo.Html(repeated.head(10).round(2).to_html(index=False, border=0))
    return student_baseline, test1, test2, test3, repeated


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These density curves summarize each test separately. They show location and spread, but do not display the within-student links. The repeated-measures calculation uses those links, even though they are not visible in a density plot.
    """)
    return


@app.cell
def _(test1, test2, test3, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=test1, fill=True, label="Test 1", ax=_ax)
    sns.kdeplot(x=test2, fill=True, label="Test 2", ax=_ax)
    sns.kdeplot(x=test3, fill=True, label="Test 3", ax=_ax)
    _ax.set(title="Scores under three repeated conditions", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A reusable repeated-measures report
    `ANOVA_RM` accepts one array per condition. The value at position i in every array must belong to the same subject. It creates subject identifiers from the supplied data, reshapes the scores into long format, and fits the repeated-measures model. Never reorder each condition separately: that would destroy the matching.

    The report includes condition summaries, F, numerator and denominator degrees of freedom, the p-value, and the decision. The function returns `(F, p_value)`.
    """)
    return


@app.cell
def _(AnovaRM, np, pd):
    def ANOVA_RM(*arrays, alpha=0.05):
        """Report a complete one-factor repeated-measures ANOVA and return F and p-value."""
        arrays = [np.asarray(a) for a in arrays]
        print("--- Repeated-measures ANOVA ---")
        print("H0: all population condition means are equal")
        print("Ha: at least one population condition mean differs")
        frame = pd.DataFrame({"subject": np.arange(len(arrays[0]))})
        for i, a in enumerate(arrays, start=1):
            print(f"Condition {i}: n = {len(a)}, mean = {a.mean():.3f}, sample SD = {a.std(ddof=1):.3f}")
            frame[f"test{i}"] = a
        long = frame.melt(id_vars="subject", var_name="test", value_name="score")
        fit = AnovaRM(long, depvar="score", subject="subject", within=["test"]).fit()
        row = fit.anova_table.iloc[0]
        statistic, p_value = float(row["F Value"]), float(row["Pr > F"])
        print(f"F = {statistic:.4g}; p-value = {p_value:.4g}")
        print(f"Degrees of freedom: {row['Num DF']:g}, {row['Den DF']:g}")
        print(f"Significance level = {alpha:g}")
        print("There is sufficient evidence to reject the null hypothesis." if p_value <= alpha else "There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value
    return (ANOVA_RM,)


@app.cell
def _(test1, test2, test3, ANOVA_RM):
    rm_three = ANOVA_RM(test1, test2, test3)
    return (rm_three,)


@app.cell
def _(rm_three, mo):
    _p = rm_three[1]
    mo.md("The repeated-measures test provides evidence of different population condition means at 0.05. It does not identify which pairs differ." if _p <= 0.05 else "The repeated-measures test does not provide sufficient evidence of different population condition means at 0.05. It does not establish that the means are equal.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding a fourth condition for the same students
    Test 4 uses the same student baselines, adds ten points, and has independent normal errors with standard deviation 4. These are additional measurements on the original students, not a fourth independent group.
    """)
    return


@app.cell
def _(np, student_baseline, repeated, mo):
    test4 = student_baseline + 10 + np.random.default_rng(53).normal(0, 4, size=50)
    repeated_four = repeated.assign(**{"Test 4": test4})
    mo.Html(repeated_four.head(10).round(2).to_html(index=False, border=0))
    return test4, repeated_four


@app.cell
def _(test1, test2, test3, test4, ANOVA_RM):
    rm_four = ANOVA_RM(test1, test2, test3, test4)
    return (rm_four,)


@app.cell
def _(rm_four, mo):
    _p = rm_four[1]
    mo.md("Reject equality of the four population condition means at 0.05. The observed pattern includes higher Test 4 scores, but this overall test does not supply pairwise decisions or establish a causal improvement." if _p <= 0.05 else "There is insufficient evidence to reject equality of the four population condition means at 0.05.")
    return


@app.cell
def _(test1, test2, test3, test4, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=test1, fill=True, label="Test 1", ax=_ax)
    sns.kdeplot(x=test2, fill=True, label="Test 2", ax=_ax)
    sns.kdeplot(x=test3, fill=True, label="Test 3", ax=_ax)
    sns.kdeplot(x=test4, fill=True, label="Test 4", ax=_ax)
    _ax.set(title="Scores under four repeated conditions", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A new problem: interface layouts
    The same 24 employees complete a comparable task with three interface layouts. Times are in seconds. We simulate employee baseline times with population mean 60 and standard deviation 5. Layout B adds three seconds and Layout C subtracts three seconds; all layouts have independent errors with standard deviation 3. The shared employee baseline retains the pairing, and the model satisfies sphericity.
    """)
    return


@app.cell
def _(np, mo, pd):
    _rng = np.random.default_rng(2026)
    _employee_baseline = _rng.normal(60, 5, size=24)
    layout_A = _employee_baseline + _rng.normal(0, 3, size=24)
    layout_B = _employee_baseline + 3 + _rng.normal(0, 3, size=24)
    layout_C = _employee_baseline - 3 + _rng.normal(0, 3, size=24)
    layout_frame = pd.DataFrame({"Employee": np.arange(1, 25), "Layout A (s)": layout_A, "Layout B (s)": layout_B, "Layout C (s)": layout_C})
    mo.Html(layout_frame.head(10).round(2).to_html(index=False, border=0))
    return layout_A, layout_B, layout_C


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Do the data provide evidence of different population mean completion times across layouts? State the hypotheses, calculate the test result, and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_layout_rm = None
    return (student_layout_rm,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_layout_rm = ANOVA_RM(layout_A, layout_B, layout_C)
    ```
    H0: μA = μB = μC; Ha: at least one population mean differs. Here F ≈ 19.726 and p ≈ 0.0000006514. Reject at 0.05: the data provide evidence of a condition mean difference. This overall result does not identify every differing pair.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Kruskal–Wallis H test
    Kruskal–Wallis compares independent groups using the ranks of their pooled observations. It does not require normality. The usual null is identical population distributions; the procedure looks for systematic differences in group rank locations.

    This is not generally a test of means. A median interpretation requires a common-shape, common-spread location model. Without that model, a rejection does not specifically establish different population medians, and the test is not equally sensitive to every possible distribution difference.

    Observations must be independent and rankable. The usual p-value is a chi-square approximation; groups should not be very small. The statistic is **H**, not F. We reuse the independent class scores, first with three groups and then with four.
    """)
    return


@app.cell
def _(np, st):
    def KW(*arrays, alpha=0.05):
        """Report the independent-groups Kruskal–Wallis test and return H and p-value."""
        arrays = [np.asarray(a) for a in arrays]
        print("--- Kruskal–Wallis H test ---")
        print("H0: all population distributions are identical")
        print("Ha: the groups differ in rank location")
        for i, a in enumerate(arrays, start=1):
            print(f"Group {i}: n = {len(a)}, median = {np.median(a):.3f}, mean = {a.mean():.3f}, sample SD = {a.std(ddof=1):.3f}")
        statistic, p_value = st.kruskal(*arrays)
        print(f"H = {statistic:.4g}; p-value = {p_value:.4g}")
        print(f"Significance level = {alpha:g}")
        print("There is sufficient evidence to reject the null hypothesis." if p_value <= alpha else "There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value
    return (KW,)


@app.cell
def _(g1_grades, g2_grades, g3_grades, KW):
    kw_three = KW(g1_grades, g2_grades, g3_grades)
    return (kw_three,)


@app.cell
def _(kw_three, mo):
    _p = kw_three[1]
    mo.md("At 0.05, the rank comparison supplies evidence against identical population distributions for the three independent groups." if _p <= 0.05 else "At 0.05, the rank comparison does not provide sufficient evidence against identical population distributions for the three independent groups.")
    return


@app.cell
def _(g1_grades, g2_grades, g3_grades, g4_grades, KW):
    kw_four = KW(g1_grades, g2_grades, g3_grades, g4_grades)
    return (kw_four,)


@app.cell
def _(kw_four, mo):
    _p = kw_four[1]
    mo.md("At 0.05, the rank comparison supplies evidence against identical population distributions. Under the equal-shape location model used to generate these scores, this supports a location difference. It does not identify which pairs differ." if _p <= 0.05 else "At 0.05, the rank comparison does not provide sufficient evidence against identical population distributions.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Use a rank-based comparison of the independent delivery-service samples. State the null hypothesis and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_delivery_kw = None
    return (student_delivery_kw,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_delivery_kw = KW(delivery_A, delivery_B, delivery_C)
    ```
    The null is identical population delivery-time distributions. Here H ≈ 9.925 and p ≈ 0.006995. Reject at 0.05: the samples show a rank-location difference. Without a common-shape model, this is not specifically a conclusion about means or medians.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Friedman test
    Friedman is a rank procedure for repeated measurements. It ranks the conditions **within each subject**, then compares the rank patterns across subjects. This accounts for subjects who consistently score higher or take longer than others.

    Under the null, condition labels are interchangeable within each subject: there is no systematic condition effect on the within-subject ranks. The alternative is a systematic difference among conditions. This is not generally a test of equal population means. Subjects must be independent of one another, measurements must be rankable, and every subject must have one observation under every condition.

    SciPy reports a chi-square statistic and an approximate p-value. Its documentation cautions that the chi-square approximation is reliable for more than ten subjects and more than six conditions. We retain the three- and four-condition examples to teach the procedure, but their approximate p-values should not be treated as exact; a within-subject permutation calculation is an option when more accurate inference is required.

    The scientist and economist who developed this procedure was **Milton Friedman**. We reuse the same matched test arrays; array positions must continue to represent the same students.
    """)
    return


@app.cell
def _(np, st):
    def Friedman(*arrays, alpha=0.05):
        """Report the repeated-measures Friedman rank test and return statistic and p-value."""
        arrays = [np.asarray(a) for a in arrays]
        print("--- Friedman repeated-measures rank test ---")
        print("H0: condition labels are interchangeable within subjects")
        print("Ha: there is a systematic condition effect on within-subject ranks")
        for i, a in enumerate(arrays, start=1):
            print(f"Condition {i}: n = {len(a)}, median = {np.median(a):.3f}, mean = {a.mean():.3f}, sample SD = {a.std(ddof=1):.3f}")
        statistic, p_value = st.friedmanchisquare(*arrays)
        print(f"Friedman chi-square statistic = {statistic:.4g}; approximate p-value = {p_value:.4g}")
        print(f"Significance level = {alpha:g}")
        print("There is sufficient evidence to reject the null hypothesis using this approximation." if p_value <= alpha else "There is insufficient evidence to reject the null hypothesis using this approximation.")
        return statistic, p_value
    return (Friedman,)


@app.cell
def _(test1, test2, test3, Friedman):
    friedman_three = Friedman(test1, test2, test3)
    return (friedman_three,)


@app.cell
def _(friedman_three, mo):
    _p = friedman_three[1]
    mo.md("Using the chi-square approximation, there is evidence of a condition effect on within-student ranks at 0.05. This is not specifically a population mean conclusion." if _p <= 0.05 else "Using the chi-square approximation, there is insufficient evidence of a condition effect on within-student ranks at 0.05. This does not establish equality of the conditions.")
    return


@app.cell
def _(test1, test2, test3, test4, Friedman):
    friedman_four = Friedman(test1, test2, test3, test4)
    return (friedman_four,)


@app.cell
def _(friedman_four, mo):
    _p = friedman_four[1]
    mo.md("Using the chi-square approximation, reject the no-condition-effect null at 0.05. The within-student ranks show a systematic condition difference, but this overall test does not identify each differing pair." if _p <= 0.05 else "Using the chi-square approximation, there is insufficient evidence of a systematic condition difference at 0.05.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Compare the interface layouts using a rank-based procedure that preserves the employee matching. Interpret the approximate result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_layout_friedman = None
    return (student_layout_friedman,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_layout_friedman = Friedman(layout_A, layout_B, layout_C)
    ```
    The statistic is 19 and the approximate p-value is 0.00007485. Using this approximation, reject the no-condition-effect null at 0.05. The completion times show a systematic difference in within-employee ranks. This does not specifically establish differences in population means. With three conditions, retain the approximation caveat discussed above.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Separate groups and repeated measurements require different analyses. Repeated-measures procedures preserve the matching within each subject.
    - One-way and repeated-measures ANOVA assess equality of population means. A rejection means at least one differs, without identifying every differing pair.
    - Tukey comparisons assess population mean differences pair by pair while accounting for multiple comparisons. Their adjusted p-values and simultaneous difference intervals support the pairwise conclusions.
    - Kruskal–Wallis uses pooled ranks for independent groups. Friedman uses within-subject ranks for repeated conditions. Their conclusions are not generally conclusions about population means.
    - Graphs describe the samples and help interpret the question being tested. Statistical significance requires the test result and an appropriate model or null assumption.
    - A p-value above the significance level means insufficient evidence to reject the null, rather than proof that the groups or conditions are equal. Interpret every result in context and recognize approximate p-values.

    ## Check your understanding
    1. A researcher compares scores from three different classes. Each student belongs to one class and no students are matched across classes. Is this an independent-groups or repeated-measures design?
    2. The same employees complete a task using three interfaces. Every row contains one employee's times under all three interfaces. Is this an independent-groups or repeated-measures design?
    3. An ANOVA comparison of three population mean delivery times gives p = 0.01. At significance level 0.05, what can you conclude? Does it identify the differing service pairs?
    4. Which procedure taught here could compare those delivery-service population means pair by pair while accounting for multiple comparisons?
    5. The same ANOVA instead gives p = 0.20. Does this establish equal population mean delivery times?
    6. A researcher wants a rank-based comparison of three independent groups. Which procedure taught here fits that design?
    7. A researcher wants a rank-based comparison of three conditions measured on the same people. Which procedure taught here preserves that matching?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    1. Independent groups: each observation belongs to a separate, unmatched group.
    2. Repeated measures: each employee contributes a measurement under every interface.
    3. Reject equality of all three population means at 0.05. There is evidence that at least one differs, but the ANOVA result does not identify each differing pair.
    4. Tukey's pairwise comparisons, when their model assumptions are appropriate.
    5. No. There is insufficient evidence to reject equal population means; equality has not been established.
    6. Kruskal–Wallis.
    7. Friedman. Retain the within-person matching and consider the accuracy of its approximate p-value for the study size.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    - SciPy documentation: [one-way ANOVA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html), [Kruskal–Wallis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html), and [Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).
    - statsmodels documentation: [repeated-measures ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html) and [Tukey comparisons](https://www.statsmodels.org/stable/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html).
    """)
    return


if __name__ == "__main__":
    app.run()
