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
    from statsmodels.stats.weightstats import ztest

    sns.set_style("whitegrid")
    return mo, np, pd, plt, sns, st, ztest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-sample Hypothesis Tests

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Distinguish an independent-sample comparison from a paired comparison.
    - Use a Welch \(t\) test and a two-sample \(z\) test for a difference of means.
    - State the rank hypothesis of a Mann-Whitney test and the distribution hypothesis of a two-sample Kolmogorov-Smirnov test.
    - Use a paired \(t\) test and a Wilcoxon signed-rank test on paired differences.
    - Choose the test from the design and the hypothesis, not from which p-value is smaller.

    ## 1. Independent samples
    Two samples are **independent** when the observations in one sample are not matched to the observations in the other.
    A test of means then compares the two group means.
    A rank test compares a different feature of the two samples. It is not a second opinion on the same sentence.

    Class A and Class B below are small normal samples with similar means.
    Class C and Class D are larger normal samples whose population means differ by 10 points.
    The standard deviations in the tables use divisor \(n-1\).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Twenty students are measured before a course and again after the course.
    1. Are those two sets of grades independent samples or paired samples?
    2. Class A has 23 students and Class B has 25 different students. Which design is that?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Before and after:** Paired. Each student contributes two grades.
2. **Two classes:** Independent. The class sizes differ, and the students are not matched one to one.
""")})
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(123)
    class_a = _rng.normal(86, 6, size=23)
    class_b = _rng.normal(88, 5, size=25)
    class_c = _rng.normal(80, 3, size=100)
    class_d = _rng.normal(90, 3, size=95)
    return class_a, class_b, class_c, class_d


@app.cell
def _(class_a, class_b, class_c, class_d, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    sns.kdeplot(class_a, ax=_axes[0], fill=True, color="#4C78A8", label="Class A")
    sns.kdeplot(class_b, ax=_axes[0], fill=True, color="#F58518", label="Class B")
    sns.kdeplot(class_c, ax=_axes[1], fill=True, color="#54A24B", label="Class C")
    sns.kdeplot(class_d, ax=_axes[1], fill=True, color="#E45756", label="Class D")
    _axes[0].set(title="Similar classes", xlabel="Score")
    _axes[1].set(title="Separated classes", xlabel="Score")
    for _ax in _axes:
        _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(class_a, class_b, class_c, class_d, mo, pd):
    def _row(name, sample):
        return {"Sample": name, "n": len(sample), "Mean": sample.mean(), "SD": sample.std(ddof=1)}

    sample_summary = pd.DataFrame([
        _row("Class A", class_a),
        _row("Class B", class_b),
        _row("Class C", class_c),
        _row("Class D", class_d),
    ])
    mo.Html(
        sample_summary.round(2).to_html(index=False, border=0, col_space=90)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (sample_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Welch \(t\) and two-sample \(z\)
    The Welch \(t\) test allows the two populations to have different variances. `ttest_ind(..., equal_var=False)` is that test.
    \(H_0\): the two population means are equal.
    The alternative `less` says the first sample's mean is smaller.

    `ztest` uses a normal reference for the difference of means, with standard errors estimated from the samples. It is a large-sample approximation. Class C and Class D are the samples large enough for that illustration. Class A and Class B stay with the Welch test.
    """)
    return


@app.cell
def _(class_a, class_b, class_c, class_d, mo, pd, st, ztest):
    _rows = []
    for _name, _left, _right, _alternative in [
        ("A vs B", class_a, class_b, "two-sided"),
        ("A vs B", class_a, class_b, "less"),
        ("C vs D", class_c, class_d, "two-sided"),
        ("C vs D", class_c, class_d, "less"),
    ]:
        _stat, _pvalue = st.ttest_ind(_left, _right, equal_var=False, alternative=_alternative)
        _rows.append({
            "Comparison": _name,
            "Test": "Welch t",
            "Ha": "means differ" if _alternative == "two-sided" else "first mean is smaller",
            "Statistic": _stat,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    for _alternative, _label in [("two-sided", "means differ"), ("smaller", "first mean is smaller"), ("larger", "first mean is larger")]:
        _stat, _pvalue = ztest(class_c, class_d, value=0, alternative=_alternative)
        _rows.append({
            "Comparison": "C vs D",
            "Test": "Two-sample z",
            "Ha": _label,
            "Statistic": _stat,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    mean_results = pd.DataFrame(_rows)
    mo.Html(
        mean_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (mean_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Class C was simulated with a lower population mean than Class D. Which alternative matches "Class C is smaller"?
    2. Why is a non-rejection for Class A versus Class B not a proof that the two teaching methods are equal?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Direction:** The alternative that the first mean is smaller. In the Welch table that row is `less`. In `ztest` the same direction is `smaller`.
2. **Non-rejection:** The samples are small and the population means are close. Failing to reject equal means means the difference was not large enough to detect. It does not prove the means are equal.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mann-Whitney and Kolmogorov-Smirnov
    The **Mann-Whitney** test ranks the pooled observations.
    Its null hypothesis is that a random draw from the first sample is as likely to exceed a random draw from the second sample as the reverse.
    That is a statement about ranks, not a second copy of the Welch sentence. It does not assume normality.
    The source notebook printed the null as "mean ranks are equal." The rank-probability statement above is the hypothesis SciPy tests.

    The **two-sample Kolmogorov-Smirnov** test compares the two empirical distribution functions.
    \(H_0\): the two samples come from the same continuous distribution.
    It can react to differences in spread or shape, not only to a shift in the mean. It is not a test about mean ranks. The source function reused the mean-rank wording for this test. That wording does not match the Kolmogorov-Smirnov hypothesis.
    """)
    return


@app.cell
def _(class_a, class_b, class_c, class_d, mo, pd, st):
    _rows = []
    for _test, _function in [("Mann-Whitney", st.mannwhitneyu), ("Kolmogorov-Smirnov", st.ks_2samp)]:
        for _name, _left, _right, _alternative in [
            ("A vs B", class_a, class_b, "two-sided"),
            ("C vs D", class_c, class_d, "two-sided"),
            ("C vs D", class_c, class_d, "less"),
        ]:
            _stat, _pvalue = _function(_left, _right, alternative=_alternative)
            _rows.append({
                "Comparison": _name,
                "Test": _test,
                "Alternative": _alternative,
                "Statistic": _stat,
                "p_value": _pvalue,
                "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
            })
    rank_results = pd.DataFrame(_rows)
    mo.Html(
        rank_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (rank_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which test compares entire distribution functions?
    2. Can Class C versus Class D reject both a mean test and a rank test for different reasons?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Distribution functions:** The two-sample Kolmogorov-Smirnov test.
2. **Different hypotheses:** Yes. The samples were generated with different means, so several tests can reject. The rejection still belongs to the hypothesis that test stated. A rank rejection is not automatically a mean rejection.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Paired samples
    Pairing keeps each before-grade attached to its after-grade.
    The paired \(t\) test is a one-sample \(t\) test on the differences.
    \(H_0\): the mean difference is 0.

    The Wilcoxon signed-rank test ranks the absolute differences and then restores the signs.
    \(H_0\): the differences are symmetric around 0.
    It does not assume that the differences are normal.

    The first after-sample is drawn from a much higher distribution than the before-sample.
    The second after-sample is drawn close to the before-sample. The pairs are still row by row: index \(i\) before is matched with index \(i\) after. These rows were simulated separately, so the pairing is a teaching device rather than a real student's two scores.
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(123)
    grade_before = _rng.normal(60, 7, size=20)
    grade_after = _rng.normal(85, 5, size=20)
    grade_after_close = _rng.normal(64, 5, size=20)
    return grade_after, grade_after_close, grade_before


@app.cell
def _(grade_after, grade_after_close, grade_before, mo, pd, st):
    _rows = []
    for _name, _after in [("Large shift", grade_after), ("Small shift", grade_after_close)]:
        for _test, _function, _alternative, _label in [
            ("Paired t", st.ttest_rel, "two-sided", "mean difference ≠ 0"),
            ("Paired t", st.ttest_rel, "less", "before mean is smaller"),
            ("Wilcoxon", st.wilcoxon, "two-sided", "differences not symmetric about 0"),
            ("Wilcoxon", st.wilcoxon, "less", "before tends to be smaller"),
        ]:
            _stat, _pvalue = _function(grade_before, _after, alternative=_alternative)
            _rows.append({
                "Data": _name,
                "Test": _test,
                "Ha": _label,
                "Statistic": _stat,
                "p_value": _pvalue,
                "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
            })
    paired_results = pd.DataFrame(_rows)
    mo.Html(
        paired_results.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (paired_results,)


@app.cell
def _(grade_after, grade_before, plt):
    _differences = grade_after - grade_before
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    _axes[0].scatter(grade_before, grade_after, color="#4C78A8")
    _axes[0].plot([50, 100], [50, 100], color="#E45756", linewidth=1)
    _axes[0].set(title="Before and after", xlabel="Before", ylabel="After")
    _axes[1].hist(_differences, bins=8, color="#F58518", edgecolor="white")
    _axes[1].axvline(0, color="#E45756")
    _axes[1].set(title="After − before", xlabel="Difference", ylabel="Count")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The differences are 2, 0, and 5.
    1. What is the mean difference?
    2. Why can a paired test not be replaced by a test that pools all six numbers and ignores which before-grade belongs to which after-grade?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Mean difference:** (2 + 0 + 5) / 3 = 2.333.
2. **Matching:** The pair is the unit. Pooling the six numbers treats a before-grade as exchangeable with someone else's after-grade and throws away the within-student comparison.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Independent samples are unmatched. Paired samples keep a link between the two measurements.
    - The Welch test and the two-sample \(z\) test are tests about means. The \(z\) test uses a normal approximation and is aimed at large samples.
    - Mann-Whitney compares ranks. Its null is a balance of which sample ranks higher, not a normal mean model.
    - Kolmogorov-Smirnov compares distribution functions. A difference in shape can matter even when a sentence about means is the wrong summary.
    - The paired \(t\) test is a one-sample test of the mean difference. The Wilcoxon signed-rank test is a rank test of symmetry around 0.
    - A p-value above 0.05 means do not reject that test's null hypothesis. It does not prove the null, and it does not answer a different test's question.
    - These simulated classes show the procedures. They are not a study of teaching methods.

    ## Check your understanding
    1. Two groups of unrelated students: independent or paired?
    2. Which test in this lesson allows unequal variances in a comparison of means?
    3. What does the Kolmogorov-Smirnov null say is the same?
    4. A paired \(t\) test rejects and the Wilcoxon test does not. Did one of them compute the wrong p-value?
    5. Class sizes 100 and 95 are used for the \(z\) illustration. Why not the classes of size 23 and 25?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Design:** Independent.
2. **Unequal variances:** The Welch \(t\) test, `ttest_ind` with `equal_var=False`.
3. **Kolmogorov-Smirnov:** The two continuous distributions.
4. **Disagreement:** Not necessarily. The tests have different null hypotheses. A mean difference and symmetry of the differences are different claims.
5. **Sample size:** The normal reference in `ztest` is a large-sample approximation. The smaller classes stay with the \(t\) test.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    """)
    return


if __name__ == "__main__":
    app.run()
