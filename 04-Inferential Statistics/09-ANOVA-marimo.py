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
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    sns.set_style("whitegrid")
    return AnovaRM, mo, np, pairwise_tukeyhsd, pd, plt, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tests for More Than Two Samples

    ## Learning goals
    By the end of this lesson, you should be able to:
    - State the one-way ANOVA null as equality of all the means.
    - Name the independence, normality, and equal-variance assumptions that ANOVA uses.
    - Use Tukey's pairwise intervals after an ANOVA rejection, and say which pairs the test flags.
    - Distinguish a repeated-measures ANOVA from independent groups.
    - Use Kruskal-Wallis and Friedman as rank procedures, and avoid calling their statistics \(F\).

    ## 1. One-way ANOVA
    ANOVA compares several group means in one test.
    \(H_0\): every group has the same population mean.
    \(H_a\): at least one mean differs.
    A rejection does not say which group differs. That is a later comparison.

    The assumptions used by the ordinary one-way test are independent observations, normal populations, and equal variances.
    The groups below are simulated normal samples. Groups 1–3 have nearby means. Group 4 is centered much lower, so the four-group test has a difference large enough to detect.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Three group means are 10, 10, and 10.
    1. Does that pattern agree with the ANOVA null?
    2. If a fourth mean is 2, does the null still describe all four groups?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Three equal means:** Yes. The null says all of the means in the test are equal.
2. **Fourth mean:** No. One of the four means differs from the others, which is the alternative.
""")})
    return


@app.cell
def _(np, pd):
    _rng = np.random.default_rng(50)
    groups = {
        "Group 1": _rng.normal(80, 6, size=50),
        "Group 2": _rng.normal(82, 4, size=48),
        "Group 3": _rng.normal(83, 5, size=52),
        "Group 4": _rng.normal(60, 5, size=44),
    }
    grade_frame = pd.concat(
        [pd.DataFrame({"Group": name, "Grade": values}) for name, values in groups.items()],
        ignore_index=True,
    )
    return grade_frame, groups


@app.cell
def _(grade_frame, mo):
    summary = (
        grade_frame.groupby("Group")["Grade"]
        .agg(n="size", Mean="mean", SD=lambda s: s.std(ddof=1))
        .reset_index()
    )
    mo.Html(
        summary.round(2).to_html(index=False, border=0, col_space=100)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (summary,)


@app.cell
def _(grade_frame, plt, sns):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    sns.kdeplot(data=grade_frame, x="Grade", hue="Group", fill=True, common_norm=False, ax=_ax)
    _ax.set(title="Independent groups")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(groups, mo, np, pd, st):
    _rows = []
    for _names in (["Group 1", "Group 2", "Group 3"], ["Group 1", "Group 2", "Group 3", "Group 4"]):
        _samples = [groups[name] for name in _names]
        _stat, _pvalue = st.f_oneway(*_samples)
        _rows.append({
            "Groups": ", ".join(_names),
            "F": _stat,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    anova_results = pd.DataFrame(_rows)
    mo.Html(
        anova_results.round(4).to_html(index=False, border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (anova_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Tukey pairwise comparisons
    Tukey's test asks which pairs of means differ, with an adjustment for the number of pairs.
    It is the follow-up after ANOVA rejects equal means. If ANOVA does not reject, the pairwise table is not a second chance to hunt for a difference.
    The plot shows simultaneous intervals. A pair whose contrast is flagged has `reject` equal to True in the table.
    """)
    return


@app.cell
def _(grade_frame, mo, pairwise_tukeyhsd, pd, plt):
    four_groups = grade_frame.copy()
    tukey = pairwise_tukeyhsd(four_groups["Grade"], four_groups["Group"], alpha=0.05)
    tukey_table = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    _fig = tukey.plot_simultaneous(figsize=(6.5, 3.4))
    _fig.tight_layout()
    plt.close(_fig)
    mo.Html(
        tukey_table.to_html(index=False, border=0, col_space=90)
        .replace("<table ", '<table style="width: auto;" ')
    )
    _fig
    return (tukey_table,)


@app.cell
def _(grade_frame, pairwise_tukeyhsd, plt):
    _three = grade_frame[grade_frame["Group"] != "Group 4"]
    _tukey = pairwise_tukeyhsd(_three["Grade"], _three["Group"], alpha=0.05)
    _fig = _tukey.plot_simultaneous(figsize=(6.5, 3.2))
    _fig.suptitle("Groups 1–3")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. How many pairwise comparisons are there among four groups?
    2. Why can one pair be flagged when the ANOVA null is "all means are equal"?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Pairs:** 6. The pairs are 1–2, 1–3, 1–4, 2–3, 2–4, and 3–4.
2. **Follow-up:** ANOVA says at least one mean differs. Tukey asks which pairs carry that difference. A pair of the nearby groups can remain unflagged while each pair involving Group 4 is flagged.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Repeated measures
    A repeated-measures design records the same subjects under each condition.
    The null hypothesis is that the condition means are equal.
    The test assumes a normal model for the observations and sphericity of the differences among conditions.
    It is not a test that all of the distributions are identical. That broader sentence belongs to the rank procedures below.

    Tests 1–3 are nearby. Test 4 is centered at 70. Every test has 50 subjects, and subject \(i\) is the same index in each test. The scores were simulated separately, so the pairing is a teaching device.
    """)
    return


@app.cell
def _(AnovaRM, np, pd):
    _rng = np.random.default_rng(50)
    repeated = pd.DataFrame({
        "subject": np.arange(50),
        "Test 1": _rng.normal(60, 5, size=50),
        "Test 2": _rng.normal(61, 4, size=50),
        "Test 3": _rng.normal(61, 5, size=50),
        "Test 4": _rng.normal(70, 4, size=50),
    })
    return (repeated,)


@app.cell
def _(AnovaRM, mo, pd, repeated):
    _rows = []
    for _columns in (["Test 1", "Test 2", "Test 3"], ["Test 1", "Test 2", "Test 3", "Test 4"]):
        _long = repeated.melt(id_vars="subject", value_vars=_columns, var_name="test", value_name="score")
        _fit = AnovaRM(_long, depvar="score", subject="subject", within=["test"]).fit()
        _stat = float(_fit.anova_table["F Value"].iloc[0])
        _pvalue = float(_fit.anova_table["Pr > F"].iloc[0])
        _rows.append({
            "Conditions": ", ".join(_columns),
            "F": _stat,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    repeated_results = pd.DataFrame(_rows)
    mo.Html(
        repeated_results.round(4).to_html(index=False, border=0, col_space=150)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (repeated_results,)


@app.cell
def _(plt, repeated, sns):
    _long = repeated.melt(id_vars="subject", var_name="test", value_name="score")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    sns.kdeplot(data=_long, x="score", hue="test", fill=True, common_norm=False, ax=_ax)
    _ax.set(title="Repeated measurements")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Kruskal-Wallis and Friedman
    The **Kruskal-Wallis** test ranks independent samples.
    \(H_0\): the distributions are the same.
    The statistic is \(H\), not \(F\). The source notebook printed it with an \(F\) label.

    The **Friedman** test ranks conditions inside each subject.
    \(H_0\): the condition distributions are the same apart from subject effects.
    It is the rank analogue of repeated-measures ANOVA, not a second copy of Kruskal-Wallis.
    """)
    return


@app.cell
def _(groups, mo, np, pd, repeated, st):
    _rows = []
    for _label, _samples in [
        ("Groups 1–3", [groups[name] for name in ["Group 1", "Group 2", "Group 3"]]),
        ("Groups 1–4", [groups[name] for name in ["Group 1", "Group 2", "Group 3", "Group 4"]]),
    ]:
        _stat, _pvalue = st.kruskal(*_samples)
        _rows.append({
            "Data": _label,
            "Test": "Kruskal-Wallis",
            "Statistic": _stat,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    for _columns in (["Test 1", "Test 2", "Test 3"], ["Test 1", "Test 2", "Test 3", "Test 4"]):
        _stat, _pvalue = st.friedmanchisquare(*[repeated[name].to_numpy() for name in _columns])
        _rows.append({
            "Data": ", ".join(_columns),
            "Test": "Friedman",
            "Statistic": _stat,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    rank_results = pd.DataFrame(_rows)
    mo.Html(
        rank_results.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (rank_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which rank test is for independent groups?
    2. Which rank test needs the same subject in every condition?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Independent groups:** Kruskal-Wallis.
2. **Same subject:** Friedman.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - One-way ANOVA tests equality of means. A rejection says that at least one mean differs.
    - The ordinary ANOVA model assumes independent observations, normal populations, and equal variances.
    - Tukey compares pairs after that rejection and adjusts for the set of pairs.
    - Repeated-measures ANOVA keeps the subject link. Its null is equality of the condition means.
    - Kruskal-Wallis ranks independent samples. Friedman ranks conditions within each subject.
    - Their statistics are not \(F\) statistics.
    - A p-value above 0.05 means do not reject that test's null. Nearby simulated groups are a case where that can happen. A distant extra group is a case where rejection is expected.
    - The scores are simulated.

    ## Check your understanding
    1. ANOVA rejects. Does that identify the group with the different mean?
    2. How many Tukey pairs do five groups produce?
    3. What design feature makes a repeated-measures test different from a one-way test?
    4. Why is the Kruskal-Wallis statistic not labeled \(F\)?
    5. Can equal sample sizes be required by Friedman when Kruskal-Wallis allows unequal sizes?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Which group:** No. The alternative is "at least one difference." Pairwise comparisons address which pairs.
2. **Five groups:** \(5 \times 4 / 2 = 10\) pairs.
3. **Design:** The same subjects are measured in every condition.
4. **Label:** \(H\) comes from a rank procedure. \(F\) comes from the normal mean model.
5. **Sizes:** Friedman needs one observation per subject per condition, so the conditions have the same length. Kruskal-Wallis allows different group sizes.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    """)
    return


if __name__ == "__main__":
    app.run()
