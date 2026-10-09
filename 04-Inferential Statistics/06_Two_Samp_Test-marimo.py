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
    import scipy.stats as st
    from statsmodels.stats import weightstats as stests
    sns.set_style("whitegrid")
    return mo, np, pd, plt, sns, st, stests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-Sample Hypothesis Tests with Python

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Distinguish independent samples from paired samples.
    - Formulate hypotheses for comparisons of population means or distributions.
    - Apply two-sample tests to compare population means, ranked observations, or distributions.
    - Interpret test results and communicate conclusions in the context of the problem.

    We will first compare separate classes, then compare measurements from the same students before and after a course. Each procedure answers a particular population question. In every example, state that question and the hypotheses before calculating the test, then interpret the result in context.

    ## 1. Two independent samples: comparing means
    Independent samples contain observations from separate groups, without a within-person or matched-pair link. Independence also requires that observations within each group do not depend on one another; different sample sizes alone do not establish independence.

    A paired design instead links two measurements from the same person or matched unit. We will study that design later.

    ### Independent-samples t test
    Two statistics classes use different teaching methods. We simulate 23 Class A scores from a normal population with mean 86 and standard deviation 6, and 25 Class B scores from a normal population with mean 88 and standard deviation 5. These are teaching data, rather than evidence that a particular method causes better scores.

    The research question is whether the **population mean scores** differ. The Welch t test estimates the standard error separately for each group:

    $$t=\frac{\bar x_1-\bar x_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}}.$$

    It allows unequal population variances. Observations must be independent, with an appropriate sampling design. For small samples, the populations should be approximately normal without severe outliers. A t test remains usable for large samples; there is no strict cutoff at 30.
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(123)
    class_a = _rng.normal(86, 6, size=23)
    class_b = _rng.normal(88, 5, size=25)
    return class_a, class_b


@app.cell
def _(class_a, class_b, mo, pd):
    class_ab_summary = pd.DataFrame({"Class": ["A", "B"], "n": [len(class_a), len(class_b)], "Mean": [class_a.mean(), class_b.mean()], "SD": [class_a.std(ddof=1), class_b.std(ddof=1)]})
    mo.Html(class_ab_summary.round(3).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the score distributions
    The curves below are kernel density estimates: smooth summaries of the observed scores. The horizontal axis shows scores; the vertical axis shows density, rather than numbers of students. Each curve has total area one, so groups with different sample sizes can be compared.

    Look at where the scores concentrate and how widely they vary. Overlap does not tell us whether the population means differ: the test also accounts for sample variability and sample size. Smoothing can suggest features that are uncertain in these small samples.
    """)
    return


@app.cell
def _(class_a, class_b, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=class_a, fill=True, color="royalblue", label="Class A", ax=_ax)
    sns.kdeplot(x=class_b, fill=True, color="salmon", label="Class B", ax=_ax)
    _ax.set(title="Class A and Class B scores", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Class A and Class B have considerable overlap in their observed scores. The graph describes the samples; the Welch test below evaluates evidence about their population means.

    ### A reusable reporting function
    The function prints the hypotheses, sample summaries, test statistic, p-value, significance level, and decision. It also returns the numerical results for reuse. Supply the two samples in the order used in your hypotheses. Choose `two-sided` for a difference, `smaller` for population 1 below population 2, or `larger` for population 1 above population 2. Specify the direction from the research question before examining the result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `two_ind_ttest` uses statsmodels’ Welch procedure (`usevar="unequal"`), reports sample standard deviations with divisor n−1, and prints the Welch degrees of freedom without rounding them to an integer. All reporting functions accept `two-sided`, `smaller`, or `larger`, where the first sample determines the direction.
    """)
    return


@app.cell
def _(np, stests):
    def two_ind_ttest(sample1, sample2, alpha=0.05, alternative="two-sided"):
        """Report a Welch independent-samples t test and return its numerical results."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- Welch independent-samples t test ---")
        print("H0: μ1 = μ2")
        print(f"Ha: μ1 {signs[alternative]} μ2")
        print(f"Sample 1: n = {len(sample1)}, mean = {np.mean(sample1):.3f}, SD = {np.std(sample1, ddof=1):.3f}")
        print(f"Sample 2: n = {len(sample2)}, mean = {np.mean(sample2):.3f}, SD = {np.std(sample2, ddof=1):.3f}")
        statistic, p_value, df = stests.ttest_ind(sample1, sample2, usevar="unequal", alternative=alternative)
        print(f"t statistic = {statistic:.4g}; p-value = {p_value:.4g}")
        print(f"Degrees of freedom: {df:.3f}")
        print(f"Significance level: {alpha:g}")
        if p_value <= alpha:
            print("There is sufficient evidence to reject the null hypothesis.")
        else:
            print("There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value, df

    return (two_ind_ttest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do the population mean scores of A and B differ?
    Test \(H_0:\mu_A=\mu_B\) against \(H_a:\mu_A\neq\mu_B\), at significance level 0.05.
    """)
    return


@app.cell
def _(class_a, class_b, two_ind_ttest):
    class_ab_t_two = two_ind_ttest(class_a, class_b, alternative="two-sided")
    return (class_ab_t_two,)


@app.cell(hide_code=True)
def _(class_ab_t_two, mo):
    _p = class_ab_t_two[1]
    _text = "The sample provides evidence that the population mean scores of Class A and Class B differ." if _p <= 0.05 else "The sample does not provide sufficient evidence that the population mean scores of Class A and Class B differ."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the population mean score of A lower than B?
    Test \(H_0:\mu_A=\mu_B\) against \(H_a:\mu_A<\mu_B\). In an investigation, select the alternative from the research question before examining the scores. The different alternatives here illustrate the procedure.
    """)
    return


@app.cell
def _(class_a, class_b, two_ind_ttest):
    class_ab_t_lower = two_ind_ttest(class_a, class_b, alternative="smaller")
    return (class_ab_t_lower,)


@app.cell(hide_code=True)
def _(class_ab_t_lower, mo):
    _p = class_ab_t_lower[1]
    _text = "The sample provides evidence that Class A has a lower population mean score than Class B." if _p <= 0.05 else "The sample does not provide sufficient evidence that Class A has a lower population mean score than Class B."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following fixed datasets record delivery times, in minutes, for independent deliveries handled by two services. Each observation is a different delivery. These data will be reused in the independent-sample exercises.
    """)
    return


@app.cell
def _(np):
    service_x = np.array([28.4, 31.2, 29.7, 32.5, 30.1, 27.8, 33.3, 29.0])
    service_y = np.array([34.1, 32.8, 35.6, 33.5, 36.2, 31.9, 34.8, 35.0])
    print("Service X delivery times:", service_x)
    print("Service Y delivery times:", service_y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Assume the delivery-time populations are approximately normal. Do the samples provide evidence that the population mean delivery times differ? State the hypotheses, calculate the test result, and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_service_mean_result = None
    print("Test result:", student_service_mean_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_service_mean_result = stests.ttest_ind(service_x, service_y, usevar="unequal")
    ```
    Test H0: μX = μY against Ha: μX ≠ μY. The result contains the statistic, p-value, and degrees of freedom. Here t ≈ −4.644, df ≈ 12.929, and p ≈ 0.000466. Reject the null hypothesis: there is evidence that the population mean delivery times differ.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Two independent samples: a z test
    The two-sample z test also compares population means. A classical z test uses known population variances. The **statsmodels implementation used here estimates a pooled variance from the samples** and uses a normal reference distribution. It therefore assumes equal population variances and is a large-sample approximation, not a known-variance calculation.

    Normal populations are not necessary for a suitable large-sample approximation, but independent observations, finite variances, and representative sampling remain important. Thirty observations per group is not a guarantee of accuracy. Welch's t test is also available for large samples, particularly when variances differ.

    We now simulate 100 scores for Class C from a normal population with mean 80 and standard deviation 3, and 95 scores for Class D from a normal population with mean 90 and standard deviation 3. The common generating standard deviation matches the pooled-variance assumption; the test still estimates that variance from the samples.
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(124)
    class_c = _rng.normal(80, 3, size=100)
    class_d = _rng.normal(90, 3, size=95)
    return class_c, class_d


@app.cell
def _(class_c, class_d, mo, pd):
    class_cd_summary = pd.DataFrame({"Class": ["C", "D"], "n": [len(class_c), len(class_d)], "Mean": [class_c.mean(), class_d.mean()], "SD": [class_c.std(ddof=1), class_d.std(ddof=1)]})
    mo.Html(class_cd_summary.round(3).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing the new classes visually
    These density curves summarize Class C and Class D. Compare their locations and spreads before reading the test result. Their common generating standard deviation was specified in the simulation; the sample spreads need not be identical.
    """)
    return


@app.cell
def _(class_c, class_d, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=class_c, fill=True, color="limegreen", label="Class C", ax=_ax)
    sns.kdeplot(x=class_d, fill=True, color="orange", label="Class D", ax=_ax)
    _ax.set(title="Class C and Class D scores", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Class D scores lie mostly to the right of the Class C scores, indicating higher observed scores. The z test will assess the population mean comparison under the assumptions stated above. The graph itself is not a hypothesis test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `two_ind_ztest` reports the hypotheses, sample summaries, z statistic, p-value, and decision. `value=0` specifies a hypothesized difference of zero, and `usevar="pooled"` makes the equal-variance assumption explicit.
    """)
    return


@app.cell
def _(np, stests):
    def two_ind_ztest(sample1, sample2, alpha=0.05, alternative="two-sided"):
        """Report a independent-samples z test and return its numerical results."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- Independent-samples z test ---")
        print("H0: μ1 = μ2")
        print(f"Ha: μ1 {signs[alternative]} μ2")
        print(f"Sample 1: n = {len(sample1)}, mean = {np.mean(sample1):.3f}, SD = {np.std(sample1, ddof=1):.3f}")
        print(f"Sample 2: n = {len(sample2)}, mean = {np.mean(sample2):.3f}, SD = {np.std(sample2, ddof=1):.3f}")
        statistic, p_value = stests.ztest(sample1, sample2, value=0, usevar="pooled", alternative=alternative)
        print(f"z statistic = {statistic:.4g}; p-value = {p_value:.4g}")

        print(f"Significance level: {alpha:g}")
        if p_value <= alpha:
            print("There is sufficient evidence to reject the null hypothesis.")
        else:
            print("There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value

    return (two_ind_ztest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do the population mean scores of C and D differ?
    Test \(H_0:\mu_C=\mu_D\) against \(H_a:\mu_C\neq\mu_D\), at significance level 0.05.
    """)
    return


@app.cell
def _(class_c, class_d, two_ind_ztest):
    class_cd_z_two = two_ind_ztest(class_c, class_d, alternative="two-sided")
    return (class_cd_z_two,)


@app.cell(hide_code=True)
def _(class_cd_z_two, mo):
    _p = class_cd_z_two[1]
    _text = "The sample provides evidence that Class C and Class D have different population mean scores." if _p <= 0.05 else "The sample does not provide sufficient evidence that Class C and Class D have different population mean scores."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the population mean score of C lower than D?
    Test \(H_0:\mu_C=\mu_D\) against \(H_a:\mu_C<\mu_D\), at significance level 0.05.
    """)
    return


@app.cell
def _(class_c, class_d, two_ind_ztest):
    class_cd_z_lower = two_ind_ztest(class_c, class_d, alternative="smaller")
    return (class_cd_z_lower,)


@app.cell(hide_code=True)
def _(class_cd_z_lower, mo):
    _p = class_cd_z_lower[1]
    _text = "The sample provides evidence that Class C has a lower population mean score than Class D." if _p <= 0.05 else "The sample does not provide sufficient evidence that Class C has a lower population mean score than Class D."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the population mean score of C higher than D?
    Test \(H_0:\mu_C=\mu_D\) against \(H_a:\mu_C>\mu_D\), at significance level 0.05.
    """)
    return


@app.cell
def _(class_c, class_d, two_ind_ztest):
    class_cd_z_upper = two_ind_ztest(class_c, class_d, alternative="larger")
    return (class_cd_z_upper,)


@app.cell(hide_code=True)
def _(class_cd_z_upper, mo):
    _p = class_cd_z_upper[1]
    _text = "The sample provides evidence that Class C has a higher population mean score than Class D." if _p <= 0.05 else "The sample does not provide sufficient evidence that Class C has a higher population mean score than Class D."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a separate exercise, two production lines provide independent random samples of package weights, in grams. The simulated populations have the same standard deviation. Both samples contain 150 packages.
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(2027)
    line_a_weights = _rng.normal(500, 8, size=150)
    line_b_weights = _rng.normal(502, 8, size=150)
    print("First 10 line A weights (g):", line_a_weights[:10].round(2))
    print("First 10 line B weights (g):", line_b_weights[:10].round(2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Assume equal population variances. Do the package samples provide evidence that the population mean weight from line A is lower than line B? State the hypotheses, calculate the result, and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_package_result = None
    print("Test result:", student_package_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_package_result = stests.ztest(line_a_weights, line_b_weights, value=0, usevar="pooled", alternative="smaller")
    ```
    Test H0: μA = μB against Ha: μA < μB. Here z ≈ −3.354 and p ≈ 0.000399. Reject the null hypothesis: the samples provide evidence that line A has a lower population mean package weight than line B.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mann–Whitney U test
    The Mann–Whitney U test compares two independent samples using the ranks of their combined observations. It does not require normality. The usual null hypothesis is that the population distributions are the same.

    The U statistic reflects how often observations from the first group exceed those from the second, with half credit for ties. A smaller alternative looks for lower observations in the first group; a larger alternative looks for higher observations.

    This is **not generally a test of equal means or equal medians**, and it is not equally sensitive to every possible distribution difference. A location or median interpretation needs additional assumptions, such as distributions with the same shape and spread that differ only by a shift.

    Observations must be independent, and the measurements must be rankable. Ties affect the calculation: SciPy's automatic method chooses between exact and approximate procedures according to the data. A nonparametric test still has assumptions.

    We reuse the class scores to compare the questions asked by the different tests. The descriptive means printed below do not change the rank-test hypothesis.
    """)
    return


@app.cell
def _(np, st):
    def m_w(sample1, sample2, alpha=0.05, alternative="two-sided"):
        """Report a mann–whitney u test and return its numerical results."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- Mann–Whitney U test ---")
        print("H0: the population distributions are the same")
        alternatives = {"two-sided": "the population distributions differ in rank tendency", "smaller": "observations from population 1 tend to be lower", "larger": "observations from population 1 tend to be higher"}
        print("Ha:", alternatives[alternative])
        print(f"Sample 1: n = {len(sample1)}, mean = {np.mean(sample1):.3f}, SD = {np.std(sample1, ddof=1):.3f}")
        print(f"Sample 2: n = {len(sample2)}, mean = {np.mean(sample2):.3f}, SD = {np.std(sample2, ddof=1):.3f}")
        scipy_alternative = {"two-sided": "two-sided", "smaller": "less", "larger": "greater"}[alternative]
        statistic, p_value = st.mannwhitneyu(sample1, sample2, alternative=scipy_alternative, method="auto")
        print(f"U statistic = {statistic:.4g}; p-value = {p_value:.4g}")

        print(f"Significance level: {alpha:g}")
        if p_value <= alpha:
            print("There is sufficient evidence to reject the null hypothesis.")
        else:
            print("There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value

    return (m_w,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do Class A and Class B differ in rank tendency?
    Use a two-sided Mann–Whitney comparison at significance level 0.05.
    """)
    return


@app.cell
def _(class_a, class_b, m_w):
    class_ab_mw = m_w(class_a, class_b, alternative="two-sided")
    return (class_ab_mw,)


@app.cell(hide_code=True)
def _(class_ab_mw, mo):
    _p = class_ab_mw[1]
    _text = "The sample provides evidence that the score distributions of Classes A and B differ in rank tendency." if _p <= 0.05 else "The sample does not provide sufficient evidence that the score distributions of Classes A and B differ in rank tendency."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do Class C and Class D differ in rank tendency?
    Use the same two-sided procedure for the larger classes.
    """)
    return


@app.cell
def _(class_c, class_d, m_w):
    class_cd_mw = m_w(class_c, class_d, alternative="two-sided")
    return (class_cd_mw,)


@app.cell(hide_code=True)
def _(class_cd_mw, mo):
    _p = class_cd_mw[1]
    _text = "The sample provides evidence that the score distributions of Classes C and D differ in rank tendency." if _p <= 0.05 else "The sample does not provide sufficient evidence that the score distributions of Classes C and D differ in rank tendency."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do Class C scores tend to be lower than Class D scores?
    The directional alternative concerns lower observations from the first population, rather than directly comparing its mean.
    """)
    return


@app.cell
def _(class_c, class_d, m_w):
    class_cd_mw_lower = m_w(class_c, class_d, alternative="smaller")
    return (class_cd_mw_lower,)


@app.cell(hide_code=True)
def _(class_cd_mw_lower, mo):
    _p = class_cd_mw_lower[1]
    _text = "The sample provides evidence that scores from Class C tend to be lower than scores from Class D." if _p <= 0.05 else "The sample does not provide sufficient evidence that scores from Class C tend to be lower than scores from Class D."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Do the independent delivery samples provide evidence that service X delivery times tend to be lower than service Y delivery times, based on their ranks? State the hypotheses, calculate the result, and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_service_rank_result = None
    print("Test result:", student_service_rank_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_service_rank_result = st.mannwhitneyu(service_x, service_y, alternative="less")
    ```
    The null hypothesis is that the population distributions are the same. Here U = 3 and p ≈ 0.000544. Reject the null hypothesis, providing evidence that service X delivery times tend to be lower. This is not automatically a claim about population means.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Two-sample Kolmogorov–Smirnov test
    The two-sample Kolmogorov–Smirnov test compares **cumulative distribution functions**, rather than ranks or means. In the two-sided case,

    $$H_0:F_1(x)=F_2(x)\text{ for all }x,\qquad H_a:F_1(x)\neq F_2(x)\text{ for at least one }x.$$

    The statistic is the largest vertical gap between the two empirical cumulative distribution functions (ECDFs). Differences in location, spread, or shape can contribute to that gap. The usual p-value calculation assumes independent observations from continuous distributions; ties or discrete measurements need special care.

    ### Understanding the one-sided direction
    A population with lower values has a **higher CDF** at a given score threshold. SciPy's `greater` alternative means \(F_1(x)>F_2(x)\) for at least one threshold, so it corresponds to evidence in the direction of lower values in population 1. SciPy's `less` means the reverse CDF direction.

    Our `k_s` wrapper keeps the lesson's observation-based labels: `smaller` maps to SciPy's `greater`, and `larger` maps to SciPy's `less`. It prints the actual CDF hypotheses to avoid ambiguity. A rejection identifies a CDF departure in that direction; it does not prove that one CDF is larger at every threshold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the cumulative-distribution graph
    At any score on the horizontal axis, the curve height is the proportion of that class scoring at or below it. For example, a height of 0.75 means 75% of the sample has a score no greater than that threshold.

    Compare the curves vertically at the same score. The largest vertical separation is the two-sided Kolmogorov–Smirnov statistic. Unlike the density curves, this graph displays the quantity the test compares directly.
    """)
    return


@app.cell
def _(class_c, class_d, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.ecdfplot(x=class_c, label="Class C", ax=_ax)
    sns.ecdfplot(x=class_d, label="Class D", ax=_ax)
    _ax.set(title="Class C and Class D: empirical cumulative distributions", xlabel="Score", ylabel="Proportion at or below score")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At thresholds between the two groups, a greater proportion of Class C has already scored at or below the threshold. Its curve is therefore higher, even though its scores are lower. This explains the reversal between observation direction and SciPy’s CDF alternative. The graph shows a large separation; the test below supplies its p-value.
    """)
    return


@app.cell
def _(np, st):
    def k_s(sample1, sample2, alpha=0.05, alternative="two-sided"):
        """Report a two-sample kolmogorov–smirnov test and return its numerical results."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- Two-sample Kolmogorov–Smirnov test ---")
        if alternative == "two-sided":
            print("H0: F1(x) = F2(x) for every x")
            print("Ha: F1(x) ≠ F2(x) for at least one x")
        elif alternative == "smaller":
            print("H0: F1(x) ≤ F2(x) for every x")
            print("Ha: F1(x) > F2(x) for at least one x (direction of lower values in population 1)")
        else:
            print("H0: F1(x) ≥ F2(x) for every x")
            print("Ha: F1(x) < F2(x) for at least one x (direction of higher values in population 1)")
        print(f"Sample 1: n = {len(sample1)}, mean = {np.mean(sample1):.3f}, SD = {np.std(sample1, ddof=1):.3f}")
        print(f"Sample 2: n = {len(sample2)}, mean = {np.mean(sample2):.3f}, SD = {np.std(sample2, ddof=1):.3f}")
        scipy_alternative = {"two-sided": "two-sided", "smaller": "greater", "larger": "less"}[alternative]
        statistic, p_value = st.ks_2samp(sample1, sample2, alternative=scipy_alternative, method="auto")
        print(f"D statistic = {statistic:.4g}; p-value = {p_value:.4g}")

        print(f"Significance level: {alpha:g}")
        if p_value <= alpha:
            print("There is sufficient evidence to reject the null hypothesis.")
        else:
            print("There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value

    return (k_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do Class A and Class B have different score distributions?
    The two-sided question concerns the entire distributions, not only their means.
    """)
    return


@app.cell
def _(class_a, class_b, k_s):
    class_ab_ks = k_s(class_a, class_b, alternative="two-sided")
    return (class_ab_ks,)


@app.cell(hide_code=True)
def _(class_ab_ks, mo):
    _p = class_ab_ks[1]
    _text = "The sample provides evidence that the population score distributions of Classes A and B differ." if _p <= 0.05 else "The sample does not provide sufficient evidence that the population score distributions of Classes A and B differ."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do Class C and Class D have different score distributions?
    Compare their full distribution functions with a two-sided test.
    """)
    return


@app.cell
def _(class_c, class_d, k_s):
    class_cd_ks = k_s(class_c, class_d, alternative="two-sided")
    return (class_cd_ks,)


@app.cell(hide_code=True)
def _(class_cd_ks, mo):
    _p = class_cd_ks[1]
    _text = "The sample provides evidence that the population score distributions of Classes C and D differ." if _p <= 0.05 else "The sample does not provide sufficient evidence that the population score distributions of Classes C and D differ."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the CDF departure in the direction of lower Class C scores?
    Test whether the first CDF exceeds the second at any threshold. The ECDF plot shows why lower observations produce a higher curve.
    """)
    return


@app.cell
def _(class_c, class_d, k_s):
    class_cd_ks_lower = k_s(class_c, class_d, alternative="smaller")
    return (class_cd_ks_lower,)


@app.cell(hide_code=True)
def _(class_cd_ks_lower, mo):
    _p = class_cd_ks_lower[1]
    _text = "The sample provides evidence that the Class C population CDF exceeds the Class D population CDF at some threshold." if _p <= 0.05 else "The sample does not provide sufficient evidence that the Class C population CDF exceeds the Class D population CDF at some threshold."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Treat the delivery-time measurements as continuous observations. Do the samples provide evidence that the two population delivery-time distributions differ? State the hypotheses, calculate the result, and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_service_distribution_result = None
    print("Test result:", student_service_distribution_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_service_distribution_result = st.ks_2samp(service_x, service_y)
    ```
    The null hypothesis is equality of the population distribution functions. Here D = 0.75 and p ≈ 0.01865. Reject the null hypothesis: there is evidence that the delivery-time distributions differ. The test does not identify the reason for the difference by itself.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Paired samples: the paired t test
    A paired comparison keeps each person's before-score matched with that same person's after-score. The pairs are independent of other pairs; the measurements **within** a pair can be dependent.

    For our reporting functions, define \(d_i=\text{sample1}_i-\text{sample2}_i\). With before as sample 1 and after as sample 2, negative differences indicate improvement. The paired t test is a one-sample t test of these differences:

    $$t=\frac{\bar d}{s_d/\sqrt n},\qquad df=n-1.$$

    The two-sided null is \(H_0:\mu_d=0\). For a small sample, the **differences** should be approximately normal without severe outliers; the separate before and after distributions do not each need to be normal.

    ### Simulated before and after scores
    We simulate 20 students' before-scores from a normal population with mean 60 and standard deviation 7. Each after-score uses the same student's before-score plus a simulated change. One scenario has a mean gain of 25 points; the other has a mean gain of 1 point. Changes have standard deviation 5. These are two alternative teaching scenarios, not independent groups or a causal study.
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(125)
    grade_before = _rng.normal(60, 7, size=20)
    grade_after = grade_before + _rng.normal(25, 5, size=20)
    grade_after_close = grade_before + _rng.normal(1, 5, size=20)
    return grade_after, grade_after_close, grade_before


@app.cell
def _(grade_after, grade_after_close, grade_before, mo, np, pd):
    paired_scores = pd.DataFrame({"Student": np.arange(1, len(grade_before) + 1), "Before": grade_before, "After: large gain": grade_after, "After: small gain": grade_after_close})
    mo.Html(paired_scores.head(5).round(2).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seeing the pairing
    In the central scatterplot below, each point is one student: the horizontal coordinate is the before-score and the vertical coordinate is that same student’s after-score. The dashed diagonal represents no change. Points above it indicate improved scores; points below it indicate decreased scores.

    The plots along the edges show the separate before and after score distributions. They summarize each measurement, but do not display the individual changes. The central plot preserves those links.
    """)
    return


@app.cell
def _(grade_after, grade_before, plt, sns):
    _grid = sns.jointplot(x=grade_before, y=grade_after, color="royalblue", height=6)
    _low = min(grade_before.min(), grade_after.min()) - 5
    _high = max(grade_before.max(), grade_after.max()) + 5
    _grid.ax_joint.plot([_low, _high], [_low, _high], linestyle="--", color="orangered", label="No change")
    _grid.ax_joint.set(xlim=(_low, _high), ylim=(_low, _high), aspect="equal")
    _grid.set_axis_labels("Before score", "After score: large-gain scenario")
    _grid.ax_joint.legend()
    _grid.fig.suptitle("Paired student scores", y=1.02)
    plt.close(_grid.fig)
    _grid.fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The points lie above the no-change line in this large-gain sample: every simulated student improved. To analyze paired data, we now reduce each pair to one difference.

    ### The differences used in the paired tests
    The histograms below use **before minus after**, matching the order of the function arguments. Negative differences indicate improved scores; positive differences indicate decreased scores. The dashed line at zero marks no change, and bar heights count students.

    Compare the large-gain and small-gain scenarios. Look at the location relative to zero, the spread, and any unusual differences. The paired t test evaluates the population mean of these differences. With only 20 observations, the histograms cannot establish that the population of differences is normal.
    """)
    return


@app.cell
def _(grade_after, grade_after_close, grade_before, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _ax, _after, _title in zip(_axes, [grade_after, grade_after_close], ["Large-gain scenario", "Small-gain scenario"]):
        _ax.hist(grade_before - _after, bins=8, color="mediumseagreen", edgecolor="white")
        _ax.axvline(0, color="orangered", linestyle="--", label="No change")
        _ax.set(title=_title, xlabel="Before − after (points)", ylabel="Number of students")
        _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The large-gain differences are well below zero. The small-gain differences lie on both sides of zero, so a small observed improvement must be considered alongside its variability. We will test each scenario separately; these are alternative examples using the same baseline students, not two independent groups.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `two_rel_ttest` reports both sample summaries, the mean paired difference, the t statistic, degrees of freedom, and decision. Pair matching is positional: observation i in sample 1 belongs to observation i in sample 2. Changing the row order independently would destroy the pairing.
    """)
    return


@app.cell
def _(np, st):
    def two_rel_ttest(sample1, sample2, alpha=0.05, alternative="two-sided"):
        """Report a paired-samples t test and return its numerical results."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- Paired-samples t test ---")
        print("H0: mean(sample1 − sample2) = 0")
        print(f"Ha: mean(sample1 − sample2) {signs[alternative]} 0")
        print(f"Sample 1: n = {len(sample1)}, mean = {np.mean(sample1):.3f}, SD = {np.std(sample1, ddof=1):.3f}")
        print(f"Sample 2: n = {len(sample2)}, mean = {np.mean(sample2):.3f}, SD = {np.std(sample2, ddof=1):.3f}")
        differences = np.asarray(sample1) - np.asarray(sample2)
        scipy_alternative = {"two-sided": "two-sided", "smaller": "less", "larger": "greater"}[alternative]
        result = st.ttest_rel(sample1, sample2, alternative=scipy_alternative)
        statistic, p_value = result.statistic, result.pvalue
        print(f"t statistic = {statistic:.4g}; p-value = {p_value:.4g}")
        print(f"Mean paired difference (sample1 − sample2): {differences.mean():.3f}")
        print(f"Degrees of freedom: {len(differences) - 1}")
        print(f"Significance level: {alpha:g}")
        if p_value <= alpha:
            print("There is sufficient evidence to reject the null hypothesis.")
        else:
            print("There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value

    return (two_rel_ttest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do population mean scores change in the large-gain scenario?
    Test a two-sided mean difference of zero.
    """)
    return


@app.cell
def _(grade_after, grade_before, two_rel_ttest):
    paired_t_large = two_rel_ttest(grade_before, grade_after, alternative="two-sided")
    return (paired_t_large,)


@app.cell(hide_code=True)
def _(mo, paired_t_large):
    _p = paired_t_large[1]
    _text = "The sample provides evidence that the population mean before and after scores differ in the large-gain scenario." if _p <= 0.05 else "The sample does not provide sufficient evidence that the population mean before and after scores differ in the large-gain scenario."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is there a population mean improvement in the large-gain scenario?
    With before first and after second, improvement corresponds to a negative mean difference and the `smaller` alternative.
    """)
    return


@app.cell
def _(grade_after, grade_before, two_rel_ttest):
    paired_t_large_lower = two_rel_ttest(grade_before, grade_after, alternative="smaller")
    return (paired_t_large_lower,)


@app.cell(hide_code=True)
def _(mo, paired_t_large_lower):
    _p = paired_t_large_lower[1]
    _text = "The sample provides evidence that the population mean after-score exceeds the population mean before-score in the large-gain scenario." if _p <= 0.05 else "The sample does not provide sufficient evidence that the population mean after-score exceeds the population mean before-score in the large-gain scenario."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do population mean scores change in the small-gain scenario?
    Use the same students’ before-scores with the small-gain after-scores.
    """)
    return


@app.cell
def _(grade_after_close, grade_before, two_rel_ttest):
    paired_t_close = two_rel_ttest(grade_before, grade_after_close, alternative="two-sided")
    return (paired_t_close,)


@app.cell(hide_code=True)
def _(mo, paired_t_close):
    _p = paired_t_close[1]
    _text = "The sample provides evidence that the population mean before and after scores differ in the small-gain scenario." if _p <= 0.05 else "The sample does not provide sufficient evidence that the population mean before and after scores differ in the small-gain scenario."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A separate fixed dataset records task-completion times, in seconds, for the same ten workers before and after practice. Keep each worker’s measurements in the same row. These data are reused in the paired exercises.
    """)
    return


@app.cell
def _(mo, np, pd):
    task_before = np.array([45, 48, 50, 46, 52, 49, 47, 53, 51, 44])
    task_after = np.array([43, 43, 51, 42, 49, 48, 41, 55, 44, 47])
    task_table = pd.DataFrame({"Worker": np.arange(1, 11), "Before (s)": task_before, "After (s)": task_after})
    mo.Html(task_table.to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Assume the population of paired time differences is approximately normal. Is there evidence that practice reduces the population mean completion time? State the hypotheses, calculate the result, and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_task_mean_result = None
    print("Test result:", student_task_mean_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_task_mean_result = st.ttest_rel(task_before, task_after, alternative="greater")
    ```
    With before first, a positive mean difference indicates a reduction in time. Test H0: μd = 0 against Ha: μd > 0. Here t ≈ 2.031 with 9 degrees of freedom and p ≈ 0.03641. Reject the null hypothesis: the data provide evidence of reduced population mean completion time after practice.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Wilcoxon signed-rank test
    The Wilcoxon signed-rank test uses the signs and ranks of the **absolute paired differences**. It does not require normally distributed differences.

    For a two-sided test, the null hypothesis is that the difference distribution is symmetric about zero. Under a symmetric location-shift model, the test assesses whether the centre of that distribution is zero. It is not a general test of equality of the two marginal distributions, and it is not an assumption-free replacement for the paired t test.

    Pairs must be independent of other pairs, and the differences must be meaningful and rankable. A symmetric difference distribution is needed for the usual location interpretation. If differences are strongly asymmetric, a different procedure may be appropriate.

    Zero differences and tied absolute differences affect the reference distribution. We leave SciPy's method selection on automatic and use its default treatment of zero differences. When subtracting rounded measurements, rounding errors can create artificial rank differences; calculate differences at the precision of the measurements when that issue arises. Our fixed exercise times are integers, so their differences are exact.

    The `wilcoxon` reporting function uses sample1 minus sample2. A `smaller` alternative therefore corresponds to an increase from before to after when before is supplied first.
    """)
    return


@app.cell
def _(np, st):
    def wilcoxon(sample1, sample2, alpha=0.05, alternative="two-sided"):
        """Report a wilcoxon signed-rank test and return its numerical results."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        print("--- Wilcoxon signed-rank test ---")
        print("H0: the paired-difference distribution is symmetric about zero")
        alternatives = {"two-sided": "the paired-difference centre differs from zero", "smaller": "the paired-difference centre is below zero", "larger": "the paired-difference centre is above zero"}
        print("Ha:", alternatives[alternative], "(under a symmetric location model)")
        print(f"Sample 1: n = {len(sample1)}, mean = {np.mean(sample1):.3f}, SD = {np.std(sample1, ddof=1):.3f}")
        print(f"Sample 2: n = {len(sample2)}, mean = {np.mean(sample2):.3f}, SD = {np.std(sample2, ddof=1):.3f}")
        differences = np.asarray(sample1) - np.asarray(sample2)
        scipy_alternative = {"two-sided": "two-sided", "smaller": "less", "larger": "greater"}[alternative]
        statistic, p_value = st.wilcoxon(differences, alternative=scipy_alternative, method="auto")
        print(f"Signed-rank statistic = {statistic:.4g}; p-value = {p_value:.4g}")
        print(f"Median paired difference (sample1 − sample2): {np.median(differences):.3f}")
        print(f"Significance level: {alpha:g}")
        if p_value <= alpha:
            print("There is sufficient evidence to reject the null hypothesis.")
        else:
            print("There is insufficient evidence to reject the null hypothesis.")
        return statistic, p_value

    return (wilcoxon,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is there a paired location change in the large-gain scenario?
    Use a two-sided signed-rank test under the symmetric-difference location model.
    """)
    return


@app.cell
def _(grade_after, grade_before, wilcoxon):
    wilcoxon_large = wilcoxon(grade_before, grade_after, alternative="two-sided")
    return (wilcoxon_large,)


@app.cell(hide_code=True)
def _(mo, wilcoxon_large):
    _p = wilcoxon_large[1]
    _text = "The sample provides evidence that the paired difference distribution has a nonzero location in the large-gain scenario, under the symmetric location model." if _p <= 0.05 else "The sample does not provide sufficient evidence that the paired difference distribution has a nonzero location in the large-gain scenario, under the symmetric location model."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the paired change in the direction of improved scores?
    Before minus after is negative for an improvement, so use `smaller`.
    """)
    return


@app.cell
def _(grade_after, grade_before, wilcoxon):
    wilcoxon_large_lower = wilcoxon(grade_before, grade_after, alternative="smaller")
    return (wilcoxon_large_lower,)


@app.cell(hide_code=True)
def _(mo, wilcoxon_large_lower):
    _p = wilcoxon_large_lower[1]
    _text = "The sample provides evidence that the paired difference location is below zero in the large-gain scenario, under the symmetric location model." if _p <= 0.05 else "The sample does not provide sufficient evidence that the paired difference location is below zero in the large-gain scenario, under the symmetric location model."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is there a paired location change in the small-gain scenario?
    The calculation uses the same paired design and signed-rank procedure.
    """)
    return


@app.cell
def _(grade_after_close, grade_before, wilcoxon):
    wilcoxon_close = wilcoxon(grade_before, grade_after_close, alternative="two-sided")
    return (wilcoxon_close,)


@app.cell(hide_code=True)
def _(mo, wilcoxon_close):
    _p = wilcoxon_close[1]
    _text = "The sample provides evidence that the paired difference distribution has a nonzero location in the small-gain scenario, under the symmetric location model." if _p <= 0.05 else "The sample does not provide sufficient evidence that the paired difference distribution has a nonzero location in the small-gain scenario, under the symmetric location model."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Assume a symmetric location model for the workers’ paired time differences. Do the data provide evidence of a reduction in typical completion time after practice? State the hypotheses, calculate the result, and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_task_rank_result = None
    print("Test result:", student_task_rank_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_task_rank_result = st.wilcoxon(task_before - task_after, alternative="greater", method="auto")
    ```
    With before minus after, reduced times correspond to a positive location shift. The null is a difference distribution symmetric about zero. The one-sided signed-rank statistic is 44.5 and p ≈ 0.04785. Reject the null hypothesis at 0.05: the data provide evidence of a reduction in paired location under the symmetric-difference model. This is not specifically a population mean conclusion.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - The sampling design determines whether to compare independent groups or analyze differences within matched pairs. Pairing must be preserved throughout the calculation.
    - Hypotheses describe the population question. Mean tests compare population means; Mann–Whitney compares ranked observations, and Kolmogorov–Smirnov compares population distributions.
    - Welch’s t test allows unequal population variances. The pooled z procedure used here assumes equal population variances and uses a large-sample normal approximation.
    - For paired observations, the paired t test examines the population mean difference. Wilcoxon signed-rank examines the difference location under a symmetric-difference model. Nonparametric procedures still have assumptions.
    - Graphs help us understand the observations and the quantities being tested. A hypothesis test adds an assessment of sampling uncertainty; a graph alone does not supply a significance decision.
    - Compare the p-value with the chosen significance level and answer the original question. Insufficient evidence to reject a null hypothesis does not establish equality. These simulated examples illustrate the procedures rather than establish teaching effects.

    ### Try it yourself
    1. Scores from two unrelated classes: independent or paired? Scores from the same students before and after a course?
    2. Does Welch's t test require equal population variances?
    3. Does a Mann–Whitney rejection automatically establish a difference in population means?
    4. Why can lower observations produce a higher cumulative distribution function?
    5. For a paired t test with a small sample, which distribution should be approximately normal?
    6. With before as sample 1 and after as sample 2, which direction represents improved scores? Which represents reduced completion times?
    7. Does a nonparametric test eliminate the need to check assumptions?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    1. The unrelated classes are independent; repeated measurements on the same students are paired.
    2. No. Welch's test estimates the variances separately.
    3. No. Its rank-based hypothesis differs from a mean comparison.
    4. At a fixed threshold, a population with lower values can have a larger proportion at or below that threshold.
    5. The population distribution of paired differences, rather than each separate measurement distribution.
    6. Improved scores give before minus after < 0 (`smaller`); reduced completion times give before minus after > 0 (`larger`).
    7. No. Sampling design, independence, and the assumptions specific to the chosen procedure still matter.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    - SciPy documentation: [Mann–Whitney U](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html), [Kolmogorov–Smirnov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html), and [Wilcoxon signed-rank](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
    - statsmodels documentation: [independent-samples t test](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttest_ind.html) and [z test](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html).
    """)
    return


if __name__ == "__main__":
    app.run()
