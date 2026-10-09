# /// script
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "seaborn"]
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
    sns.set_style("whitegrid")
    return mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Resampling Tests for Several Groups and Conditions
    ## Learning goals
    By the end of this lesson, you should be able to:
    - Apply bootstrap and permutation procedures to compare several independent groups.
    - Apply resampling procedures to repeated measurements while preserving subject matching.
    - Formulate population hypotheses and use a statistic that summarizes differences among means.
    - Interpret resampling test results and communicate conclusions in context.

    We will compare separate student groups and repeated scores from the same students. The procedures use the observed data to generate statistics under a specified null hypothesis. These are custom resampling tests of mean differences, not calculations of the ordinary ANOVA F statistic.

    Every example uses 4,000 resamples and a seeded generator. More resamples improve numerical precision, but do not add observations to the original data. The examples use simulated data to illustrate the procedures.

    ## 1. Bootstrap for independent groups
    We simulate 50 scores from a population with mean 80 and standard deviation 6, 48 with mean 82 and standard deviation 4, and 52 with mean 83 and standard deviation 5. Students in different groups are not matched. The independent bootstrap below allows different population spreads.

    $$H_0:\mu_1=\mu_2=\mu_3,\qquad H_a:\text{at least one population mean differs}.$$
    """)
    return


@app.cell
def _(np):
    _rng = np.random.RandomState(50)
    g1_grades = _rng.normal(80, 6, size=50)
    g2_grades = _rng.normal(82, 4, size=48)
    g3_grades = _rng.normal(83, 5, size=52)
    print("First 10 Group 1 scores:", g1_grades[:10].round(2))
    print("First 10 Group 2 scores:", g2_grades[:10].round(2))
    print("First 10 Group 3 scores:", g3_grades[:10].round(2))
    return g1_grades, g2_grades, g3_grades


@app.cell
def _(g1_grades, g2_grades, g3_grades, mo, pd):
    group_summary = pd.DataFrame({"Group": [1, 2, 3], "n": [len(g1_grades), len(g2_grades), len(g3_grades)], "Mean": [g1_grades.mean(), g2_grades.mean(), g3_grades.mean()], "Sample SD": [g1_grades.std(ddof=1), g2_grades.std(ddof=1), g3_grades.std(ddof=1)]})
    mo.Html(group_summary.round(3).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the score distributions
    The kernel density curves show where observed scores concentrate and how widely they vary. Each curve has total area one; density is not a student count. Overlap does not establish equal population means. The resampling test assesses the mean gaps relative to the variation expected under its null.
    """)
    return


@app.cell
def _(g1_grades, g2_grades, g3_grades, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=g1_grades, fill=True, alpha=0.25, label="Sample 1", ax=_ax)
    sns.kdeplot(x=g2_grades, fill=True, alpha=0.25, label="Sample 2", ax=_ax)
    sns.kdeplot(x=g3_grades, fill=True, alpha=0.25, label="Sample 3", ax=_ax)
    _ax.set(title="Original scores: three independent groups", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A statistic for several mean differences
    We retain the weighted dispersion of group means:

    $$T=\sqrt{\sum_{j=1}^{k}\frac{n_j}{N}(\bar x_j-\bar x_{\mathrm{grand}})^2},\qquad N=\sum_{j=1}^k n_j.$$

    The grand mean weights each group by its sample size. The statistic is zero when all sample means are equal and increases as they spread apart. It remains in the units of the measurements. Small values support the equality pattern; **large values** provide evidence against it, so use the upper tail.

    `dispersion` applies this same calculation to original arrays or to resample DataFrames. For a DataFrame, it returns one statistic per column and calculates a new weighted grand mean for each resample. The observed and resampled statistics must use exactly the same definition.
    """)
    return


@app.cell
def _(np):
    def dispersion(*samples):
        """Weighted dispersion of means; also calculate one statistic per resample column."""
        arrays = [np.asarray(sample) for sample in samples]
        sizes = np.array([len(a) for a in arrays])
        means = np.stack([a.mean(axis=0) for a in arrays])
        grand = np.average(means, axis=0, weights=sizes)
        return np.sqrt(np.average((means - grand) ** 2, axis=0, weights=sizes))
    return (dispersion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imposing the equal-mean null
    Shift each original group by subtracting its own mean and adding the combined mean. The shifted groups share a mean while retaining their separate shapes, spreads, and sample sizes. Resample each shifted group from its **own** observations, with replacement. This is an approximate equal-mean bootstrap test, not a test that all population distributions are identical.
    """)
    return


@app.cell
def _(g1_grades, g2_grades, g3_grades, mo, np, pd):
    overall_mean_grades = np.concatenate((g1_grades, g2_grades, g3_grades)).mean()
    g1_grades_sh = g1_grades - g1_grades.mean() + overall_mean_grades
    g2_grades_sh = g2_grades - g2_grades.mean() + overall_mean_grades
    g3_grades_sh = g3_grades - g3_grades.mean() + overall_mean_grades
    mo.Html(pd.DataFrame({"Shifted group": [1, 2, 3], "Mean": [g1_grades_sh.mean(), g2_grades_sh.mean(), g3_grades_sh.mean()]}).round(3).to_html(index=False, border=0))
    return g1_grades_sh, g2_grades_sh, g3_grades_sh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The shifted curves below have the same mean, although their shapes and spreads can differ. Shifting does not make the scores identical.
    """)
    return


@app.cell
def _(g1_grades_sh, g2_grades_sh, g3_grades_sh, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=g1_grades_sh, fill=True, alpha=0.25, label="Sample 1", ax=_ax)
    sns.kdeplot(x=g2_grades_sh, fill=True, alpha=0.25, label="Sample 2", ax=_ax)
    sns.kdeplot(x=g3_grades_sh, fill=True, alpha=0.25, label="Sample 3", ax=_ax)
    _ax.set(title="Shifted groups: equal sample means", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generating bootstrap samples
    `generate_samples` keeps one resample per DataFrame column and uses NumPy to generate the draws together. Every column has the original group's sample size. Its default seed makes calls reproducible; use distinct seeds for independently generated group resamples, as below.
    """)
    return


@app.cell
def _(np, pd):
    def generate_samples(sample_data, num_samples=4_000, seed=2026):
        """Bootstrap a sample with replacement; return one resample per DataFrame column."""
        sample_size = len(sample_data)
        rng = np.random.default_rng(seed)
        draws = rng.choice(sample_data, replace=True, size=sample_size * num_samples)
        return pd.DataFrame(draws.reshape(sample_size, num_samples), columns=["S" + str(k) for k in range(num_samples)])
    return (generate_samples,)


@app.cell
def _(g1_grades_sh, g2_grades_sh, g3_grades_sh, generate_samples, mo):
    df_1 = generate_samples(g1_grades_sh, seed=2026)
    df_2 = generate_samples(g2_grades_sh, seed=2027)
    df_3 = generate_samples(g3_grades_sh, seed=2028)
    mo.Html(df_1.iloc[:5, :5].round(2).to_html(border=0))
    return df_1, df_2, df_3


@app.cell
def _(dispersion, g1_grades, g2_grades, g3_grades, df_1, df_2, df_3):
    test_stat_bootI = dispersion(g1_grades, g2_grades, g3_grades)
    sample_distribution_bootI = dispersion(df_1, df_2, df_3)
    print("Observed dispersion:", round(float(test_stat_bootI), 3))
    return test_stat_bootI, sample_distribution_bootI


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the null distribution
    Each histogram bar counts resamples whose statistic lies in that range. Nonzero statistics occur even under equal population means because the resampled sample means fluctuate. Compare the observed dispersion with the **upper** end of this simulated distribution, rather than with zero alone.
    """)
    return


@app.cell
def _(sample_distribution_bootI, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_bootI, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Bootstrap null: three-group mean dispersion", xlabel="Statistic under the null", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculating and reporting the test
    `get_p_value` uses inclusive tails. For these distance statistics, every test call uses `larger`: the upper-tail estimate is (number of resampled statistics at least as large as observed + 1)/(number of resamples + 1). The finite-simulation correction avoids reporting zero when no resample reaches the observed statistic. It does not make a bootstrap test exact.

    `graph_hyp_test` prints the hypotheses, observed statistic, p-value, significance level, and decision. The blue KDE summarizes the null values, the dashed red line marks the approximate critical boundary, the red tail marks the critical region, and the black line marks the observed statistic. The p-value is calculated from the resampled values, not the smoothed curve. The critical boundary is a visual approximation; the decision uses the p-value.
    """)
    return


@app.cell
def _(np):
    def get_p_value(sample_distribution, obs_value, alternative='larger'):
        """Estimate a resampling p-value using inclusive tails and a plus-one correction."""
        values = np.asarray(sample_distribution)
        left = (np.count_nonzero(values <= obs_value) + 1) / (len(values) + 1)
        right = (np.count_nonzero(values >= obs_value) + 1) / (len(values) + 1)
        if alternative == 'smaller':
            return float(left)
        if alternative == 'larger':
            return float(right)
        return float(min(1, 2 * min(left, right)))
    return (get_p_value,)


@app.cell
def _(get_p_value, np, plt, sns):
    def graph_hyp_test(sample_value, sample_distribution, alpha=0.05, alternative='larger', null_hypothesis='H0: population difference = 0', alternative_hypothesis='Ha: population difference ≠ 0'):
        """Report a resampling test and plot its null distribution and critical tails."""
        p_val = get_p_value(sample_distribution, sample_value, alternative)
        print('--- Resampling hypothesis test ---')
        print(null_hypothesis)
        print(alternative_hypothesis)
        print(f'Observed statistic = {sample_value:.4g}; p-value = {p_val:.4g}')
        print(f'Significance level = {alpha:g}')
        print('There is sufficient evidence to reject the null hypothesis.' if p_val <= alpha else 'There is insufficient evidence to reject the null hypothesis.')
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(x=sample_distribution, color='skyblue', cut=0, clip=(0, None), ax=ax)
        x, y = ax.lines[0].get_data()
        ax.fill_between(x, y, color='skyblue', alpha=0.35)
        if alternative == 'two-sided':
            boundaries = np.percentile(sample_distribution, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        elif alternative == 'smaller':
            boundaries = [np.percentile(sample_distribution, 100 * alpha)]
        else:
            boundaries = [np.percentile(sample_distribution, 100 * (1 - alpha))]
        for boundary in boundaries:
            ax.axvline(boundary, color='orangered', linestyle='--', label='Approximate critical boundary')
        x, y = ax.lines[0].get_data()
        if alternative in ('two-sided', 'smaller'):
            ax.fill_between(x, y, where=x <= boundaries[0], color='orangered', alpha=0.4)
        if alternative in ('two-sided', 'larger'):
            ax.fill_between(x, y, where=x >= boundaries[-1], color='orangered', alpha=0.4)
        ax.axvline(sample_value, color='black', linewidth=2, label='Observed statistic')
        ax.set(title='Resampling null distribution and critical regions', xlabel='Statistic', ylabel='Density')
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        fig.tight_layout()
        plt.close(fig)
        return fig
    return (graph_hyp_test,)


@app.cell
def _(test_stat_bootI, sample_distribution_bootI, graph_hyp_test):
    graph_hyp_test(test_stat_bootI, sample_distribution_bootI, alternative="larger", null_hypothesis="H0: all population group means are equal", alternative_hypothesis="Ha: at least one population mean differs")
    return


@app.cell
def _(test_stat_bootI, sample_distribution_bootI, get_p_value, mo):
    _p = get_p_value(sample_distribution_bootI, test_stat_bootI, alternative="larger")
    mo.md("Three independent groups, bootstrap: there is sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Three independent groups, bootstrap: there is insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding a fourth independent group
    We generate 44 scores from a population with mean 60 and standard deviation 5. Rebuild the null for **all four groups**, shifting every group to the four-group grand mean. Reusing a three-group construction without updating it would not represent this four-group null.
    """)
    return


@app.cell
def _(np):
    g4_grades = np.random.default_rng(51).normal(60, 5, size=44)
    print("First 10 Group 4 scores:", g4_grades[:10].round(2))
    return (g4_grades,)


@app.cell
def _(g1_grades, g2_grades, g3_grades, g4_grades, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=g1_grades, fill=True, alpha=0.25, label="Sample 1", ax=_ax)
    sns.kdeplot(x=g2_grades, fill=True, alpha=0.25, label="Sample 2", ax=_ax)
    sns.kdeplot(x=g3_grades, fill=True, alpha=0.25, label="Sample 3", ax=_ax)
    sns.kdeplot(x=g4_grades, fill=True, alpha=0.25, label="Sample 4", ax=_ax)
    _ax.set(title="Original scores: four independent groups", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `bootstrap_independent` automates the same steps: shift each group to the combined mean, resample it separately, then return a list of DataFrames. Its small loop runs over groups, not individual resamples.
    """)
    return


@app.cell
def _(generate_samples, np):
    def bootstrap_independent(*samples, num_samples=4_000, seed=2029):
        """Shift groups to a common mean and bootstrap each group separately."""
        grand = np.concatenate(samples).mean()
        return [generate_samples(np.asarray(a) - np.mean(a) + grand, num_samples=num_samples, seed=seed + j) for j, a in enumerate(samples)]
    return (bootstrap_independent,)


@app.cell
def _(bootstrap_independent, dispersion, g1_grades, g2_grades, g3_grades, g4_grades):
    ind_boot_four = bootstrap_independent(g1_grades, g2_grades, g3_grades, g4_grades)
    test_stat_bootI4 = dispersion(g1_grades, g2_grades, g3_grades, g4_grades)
    sample_distribution_bootI4 = dispersion(*ind_boot_four)
    return test_stat_bootI4, sample_distribution_bootI4


@app.cell
def _(sample_distribution_bootI4, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_bootI4, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Bootstrap null: four-group mean dispersion", xlabel="Statistic under the null", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(test_stat_bootI4, sample_distribution_bootI4, graph_hyp_test):
    graph_hyp_test(test_stat_bootI4, sample_distribution_bootI4, alternative="larger", null_hypothesis="H0: all four population group means are equal", alternative_hypothesis="Ha: at least one population mean differs")
    return


@app.cell
def _(test_stat_bootI4, sample_distribution_bootI4, get_p_value, mo):
    _p = get_p_value(sample_distribution_bootI4, test_stat_bootI4, alternative="larger")
    mo.md("Four independent groups, bootstrap: there is sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Four independent groups, bootstrap: there is insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A new problem: delivery services
    Three independent services record delivery times in minutes. Every observation belongs to a different delivery, with no matching between services. We prepare an equal-mean bootstrap distribution using the procedures just taught.
    """)
    return


@app.cell
def _(np, pd, mo, bootstrap_independent, dispersion):
    delivery_A = np.array([28, 31, 29, 32, 30, 27, 33, 29])
    delivery_B = np.array([34, 32, 35, 33, 36, 31, 34, 35])
    delivery_C = np.array([30, 33, 31, 34, 32, 29, 35, 31])
    mo.Html(pd.DataFrame({"Service A (min)": delivery_A, "Service B (min)": delivery_B, "Service C (min)": delivery_C}).to_html(index=False, border=0))
    delivery_observed = dispersion(delivery_A, delivery_B, delivery_C)
    delivery_boot_null = dispersion(*bootstrap_independent(delivery_A, delivery_B, delivery_C, seed=2033))
    return delivery_A, delivery_B, delivery_C, delivery_observed, delivery_boot_null


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the bootstrap p-value for the prepared delivery-time dispersion. State the hypotheses and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_delivery_boot_p = None
    return (student_delivery_boot_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_delivery_boot_p = get_p_value(delivery_boot_null, delivery_observed, alternative="larger")
    ```
    H0: μA = μB = μC; Ha: at least one population mean differs. The seeded calculation gives p ≈ 0.000250. Reject at 0.05: the data provide evidence of different population mean delivery times. The overall test does not identify which pairs differ.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bootstrap for related measurements
    We simulate scores from the same 50 students under three tests. Each student has a normal baseline score with mean 60 and standard deviation 5. Each test adds independent normal errors with standard deviation 4; Tests 2 and 3 also add one point. The shared student baseline makes repeated scores related.

    Every row must refer to the same student across conditions. The equal-mean null is equality of the population condition means.

    ### Mean absolute gap among condition means
    Retain the average absolute difference across every pair of condition means:

    $$G=\frac{1}{\binom{k}{2}}\sum_{i<j}|\bar x_i-\bar x_j|.$$

    For three conditions there are three pairs; for four there are six. The statistic is zero when all sample means agree. Larger values indicate a greater overall mean separation, so again use the upper tail.
    """)
    return


@app.cell
def _(np, mo, pd):
    _rng = np.random.RandomState(50)
    student_baseline = _rng.normal(60, 5, size=50)
    test1 = student_baseline + _rng.normal(0, 4, size=50)
    test2 = student_baseline + 1 + _rng.normal(0, 4, size=50)
    test3 = student_baseline + 1 + _rng.normal(0, 4, size=50)
    mo.Html(pd.DataFrame({"Student": np.arange(1, 51), "Test 1": test1, "Test 2": test2, "Test 3": test3}).head(10).round(2).to_html(index=False, border=0))
    return student_baseline, test1, test2, test3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The density curves show the separate score distributions. They do not display which scores belong to the same student. The block bootstrap retains those links in its resampling calculation.
    """)
    return


@app.cell
def _(test1, test2, test3, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=test1, fill=True, alpha=0.25, label="Sample 1", ax=_ax)
    sns.kdeplot(x=test2, fill=True, alpha=0.25, label="Sample 2", ax=_ax)
    sns.kdeplot(x=test3, fill=True, alpha=0.25, label="Sample 3", ax=_ax)
    _ax.set(title="Repeated scores: three conditions", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(np):
    def mean_gap(*samples):
        """Average absolute pairwise mean gap; support arrays and resample columns."""
        means = [np.asarray(a).mean(axis=0) for a in samples]
        gaps = [np.abs(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))]
        return np.mean(gaps, axis=0)
    return (mean_gap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Resampling students as blocks
    Shift each condition to the combined mean. Then generate **one common table of subject indices** and use it for every condition. If a resample selects Student 7, it includes Student 7's shifted scores in all conditions. Sampling each condition with a different index table would mix different students and discard their dependence.

    The function below implements your index-based block procedure. It returns one resample DataFrame per condition, using the same selected rows in every frame.
    """)
    return


@app.cell
def _(generate_samples, np, pd):
    def bootstrap_related(*samples, num_samples=4_000, seed=2036):
        """Bootstrap whole subjects after shifting all condition means to a common value."""
        grand = np.concatenate(samples).mean()
        indices = generate_samples(np.arange(len(samples[0])), num_samples=num_samples, seed=seed)
        return [pd.DataFrame((np.asarray(a) - np.mean(a) + grand)[indices.to_numpy()], columns=indices.columns) for a in samples]
    return (bootstrap_related,)


@app.cell
def _(bootstrap_related, mean_gap, test1, test2, test3, mo):
    df_t1, df_t2, df_t3 = bootstrap_related(test1, test2, test3)
    test_stat_bootR = mean_gap(test1, test2, test3)
    sample_distribution_bootR = mean_gap(df_t1, df_t2, df_t3)
    mo.Html(df_t1.iloc[:5, :5].round(2).to_html(border=0))
    return test_stat_bootR, sample_distribution_bootR


@app.cell
def _(sample_distribution_bootR, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_bootR, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Block bootstrap null: three-condition mean gap", xlabel="Statistic under the null", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(test_stat_bootR, sample_distribution_bootR, graph_hyp_test):
    graph_hyp_test(test_stat_bootR, sample_distribution_bootR, alternative="larger", null_hypothesis="H0: all population condition means are equal", alternative_hypothesis="Ha: at least one condition mean differs")
    return


@app.cell
def _(test_stat_bootR, sample_distribution_bootR, get_p_value, mo):
    _p = get_p_value(sample_distribution_bootR, test_stat_bootR, alternative="larger")
    mo.md("Three related conditions, block bootstrap: there is sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Three related conditions, block bootstrap: there is insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding a fourth condition
    Test 4 uses the same students and adds 50 points to their baselines, with independent normal errors of standard deviation 4. This deliberately large shift illustrates an overall mean difference; it is a simulated score scale, not a percentage constrained to 100.

    Rebuild all four shifted conditions and resample their subject rows together. Each condition must retain its own shifted values.
    """)
    return


@app.cell
def _(np, student_baseline):
    test4 = student_baseline + 50 + np.random.default_rng(54).normal(0, 4, size=50)
    return (test4,)


@app.cell
def _(test1, test2, test3, test4, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=test1, fill=True, alpha=0.25, label="Sample 1", ax=_ax)
    sns.kdeplot(x=test2, fill=True, alpha=0.25, label="Sample 2", ax=_ax)
    sns.kdeplot(x=test3, fill=True, alpha=0.25, label="Sample 3", ax=_ax)
    sns.kdeplot(x=test4, fill=True, alpha=0.25, label="Sample 4", ax=_ax)
    _ax.set(title="Repeated scores: four conditions", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(bootstrap_related, mean_gap, test1, test2, test3, test4):
    related_boot_four = bootstrap_related(test1, test2, test3, test4, seed=2037)
    test_stat_bootR4 = mean_gap(test1, test2, test3, test4)
    sample_distribution_bootR4 = mean_gap(*related_boot_four)
    return test_stat_bootR4, sample_distribution_bootR4


@app.cell
def _(sample_distribution_bootR4, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_bootR4, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Block bootstrap null: four-condition mean gap", xlabel="Statistic under the null", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(test_stat_bootR4, sample_distribution_bootR4, graph_hyp_test):
    graph_hyp_test(test_stat_bootR4, sample_distribution_bootR4, alternative="larger", null_hypothesis="H0: all four population condition means are equal", alternative_hypothesis="Ha: at least one condition mean differs")
    return


@app.cell
def _(test_stat_bootR4, sample_distribution_bootR4, get_p_value, mo):
    _p = get_p_value(sample_distribution_bootR4, test_stat_bootR4, alternative="larger")
    mo.md("Four related conditions, block bootstrap: there is sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Four related conditions, block bootstrap: there is insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A new problem: interface layouts
    The same 24 employees complete a comparable task using three layouts. We simulate employee baseline times with mean 60 seconds and standard deviation 5. Layout B adds three seconds, Layout C subtracts three seconds, and each measurement has independent normal errors with standard deviation 3. Each table row represents one employee.
    """)
    return


@app.cell
def _(np, pd, mo, mean_gap, bootstrap_related):
    _rng = np.random.default_rng(2026)
    _baseline = _rng.normal(60, 5, size=24)
    layout_A = _baseline + _rng.normal(0, 3, size=24)
    layout_B = _baseline + 3 + _rng.normal(0, 3, size=24)
    layout_C = _baseline - 3 + _rng.normal(0, 3, size=24)
    mo.Html(pd.DataFrame({"Employee": np.arange(1, 25), "A (s)": layout_A, "B (s)": layout_B, "C (s)": layout_C}).head(10).round(2).to_html(index=False, border=0))
    layout_observed = mean_gap(layout_A, layout_B, layout_C)
    layout_boot_null = mean_gap(*bootstrap_related(layout_A, layout_B, layout_C, seed=2038))
    return layout_A, layout_B, layout_C, layout_observed, layout_boot_null


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the block-bootstrap p-value for the prepared interface-layout mean gap. State the hypotheses and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_layout_boot_p = None
    return (student_layout_boot_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_layout_boot_p = get_p_value(layout_boot_null, layout_observed, alternative="larger")
    ```
    H0: μA = μB = μC; Ha: at least one condition mean differs. The seeded calculation gives p ≈ 0.000250. Reject at 0.05: the repeated measurements provide evidence of different population mean completion times.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Permutations for independent groups
    Pool the **original, unshifted** scores, shuffle them, and divide them into the original group sizes. Each score appears exactly once in each rearrangement. Repeated rearrangements may repeat an allocation.

    This procedure requires interchangeable group labels under the null: identical population distributions, or a randomized assignment that justifies the rearrangements. Equal means alone do not guarantee interchangeable labels when shapes or variances differ. Our statistic focuses on differences in means, but the permutation null is stronger than the bootstrap equal-mean null.

    `shuffle_k_samples` preserves your list of DataFrames, one per group and one permutation per column. NumPy shuffles each pooled column independently, avoiding one Python function call per column. The observed statistic stays the same; the mechanism generating the null statistics changes.
    """)
    return


@app.cell
def _(np, pd):
    def shuffle_k_samples(*samples, num_samples=4_000, seed=2039):
        """Shuffle pooled observations into original group sizes; one permutation per column."""
        pool = np.concatenate(samples)
        rng = np.random.default_rng(seed)
        repeated = np.broadcast_to(pool[:, None], (len(pool), num_samples))
        shuffled = rng.permuted(repeated, axis=0)
        cuts = np.cumsum([len(a) for a in samples])[:-1]
        columns = ["S" + str(k) for k in range(num_samples)]
        return [pd.DataFrame(a, columns=columns) for a in np.split(shuffled, cuts, axis=0)]
    return (shuffle_k_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 independent groups
    Rearrange the original observations, preserve each group size, and recalculate the dispersion. The histogram shows the dispersion expected under interchangeable group labels.
    """)
    return


@app.cell
def _(g1_grades, g2_grades, g3_grades, shuffle_k_samples, dispersion):
    ind_perm_3 = shuffle_k_samples(g1_grades, g2_grades, g3_grades, seed=2042)
    test_stat_permI = dispersion(g1_grades, g2_grades, g3_grades)
    sample_distribution_permI = dispersion(*ind_perm_3)
    return test_stat_permI, sample_distribution_permI


@app.cell
def _(sample_distribution_permI, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_permI, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Independent permutation null: 3-group dispersion", xlabel="Statistic under the null", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(test_stat_permI, sample_distribution_permI, graph_hyp_test):
    graph_hyp_test(test_stat_permI, sample_distribution_permI, alternative="larger", null_hypothesis="H0: all population group distributions are identical", alternative_hypothesis="Ha: the mean-dispersion statistic departs from interchangeable labels")
    return


@app.cell
def _(test_stat_permI, sample_distribution_permI, get_p_value, mo):
    _p = get_p_value(sample_distribution_permI, test_stat_permI, alternative="larger")
    mo.md("3 independent groups, permutation: there is sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "3 independent groups, permutation: there is insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 independent groups
    Rearrange the original observations, preserve each group size, and recalculate the dispersion. The histogram shows the dispersion expected under interchangeable group labels.
    """)
    return


@app.cell
def _(g1_grades, g2_grades, g3_grades, g4_grades, shuffle_k_samples, dispersion):
    ind_perm_4 = shuffle_k_samples(g1_grades, g2_grades, g3_grades, g4_grades, seed=2043)
    test_stat_permI4 = dispersion(g1_grades, g2_grades, g3_grades, g4_grades)
    sample_distribution_permI4 = dispersion(*ind_perm_4)
    return test_stat_permI4, sample_distribution_permI4


@app.cell
def _(sample_distribution_permI4, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_permI4, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Independent permutation null: 4-group dispersion", xlabel="Statistic under the null", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(test_stat_permI4, sample_distribution_permI4, graph_hyp_test):
    graph_hyp_test(test_stat_permI4, sample_distribution_permI4, alternative="larger", null_hypothesis="H0: all population group distributions are identical", alternative_hypothesis="Ha: the mean-dispersion statistic departs from interchangeable labels")
    return


@app.cell
def _(test_stat_permI4, sample_distribution_permI4, get_p_value, mo):
    _p = get_p_value(sample_distribution_permI4, test_stat_permI4, alternative="larger")
    mo.md("4 independent groups, permutation: there is sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "4 independent groups, permutation: there is insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A rejection supplies evidence against the interchangeable-label null. In these examples, the statistic targets the observed separation of group means. Do not interpret the bootstrap and permutation results as interchangeable tests of the same null when population spreads can differ.
    """)
    return


@app.cell
def _(delivery_A, delivery_B, delivery_C, shuffle_k_samples, dispersion):
    delivery_perm_null = dispersion(*shuffle_k_samples(delivery_A, delivery_B, delivery_C, seed=2044))
    return (delivery_perm_null,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Assume service labels are interchangeable under identical population delivery-time distributions. Calculate the permutation p-value for the prepared dispersion and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_delivery_perm_p = None
    return (student_delivery_perm_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_delivery_perm_p = get_p_value(delivery_perm_null, delivery_observed, alternative="larger")
    ```
    Here p ≈ 0.004749. Reject the interchangeable-label null at 0.05. The samples provide evidence against identical population delivery-time distributions through their mean dispersion. This is not an assumption-free equal-means test.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Permutations for related measurements
    Shuffle condition labels **within each subject**, using the original unshifted measurements. A student with scores 60, 64, and 58 keeps those same three scores; only their condition assignments change. No score moves to a different student.

    The null requires within-subject condition labels to be interchangeable, as justified by a suitable randomized design or a joint distribution invariant under those rearrangements. Equal condition means alone are not sufficient when the within-subject distributions or dependence structures differ. The shared-baseline, equal-error model used here satisfies this condition under no condition effect.

    `shuffle_k_related_samples` retains your three-dimensional layout: subject × condition × permutation. NumPy shuffles along the condition axis independently for every subject and permutation, then returns the original list-of-DataFrames format. This replaces the nested subject and permutation loops without changing which observations may be swapped.
    """)
    return


@app.cell
def _(np, pd):
    def shuffle_k_related_samples(*arrays, num_samples=4_000, seed=2045):
        """Shuffle conditions within each subject; return one permutation DataFrame per condition."""
        data = np.column_stack(arrays)
        rng = np.random.default_rng(seed)
        repeated = np.broadcast_to(data[:, :, None], (len(data), len(arrays), num_samples))
        shuffled = rng.permuted(repeated, axis=1)
        columns = ["S" + str(k) for k in range(num_samples)]
        return [pd.DataFrame(shuffled[:, k, :], columns=columns) for k in range(len(arrays))]
    return (shuffle_k_related_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 related conditions
    Generate rearrangements within students and calculate the mean absolute gap for each permutation. No centering or separate resampling of conditions is used.
    """)
    return


@app.cell
def _(test1, test2, test3, shuffle_k_related_samples, mean_gap):
    rel_perm_3 = shuffle_k_related_samples(test1, test2, test3, seed=2048)
    test_stat_permR = mean_gap(test1, test2, test3)
    sample_distribution_permR = mean_gap(*rel_perm_3)
    return test_stat_permR, sample_distribution_permR


@app.cell
def _(sample_distribution_permR, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_permR, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Related permutation null: 3-condition mean gap", xlabel="Statistic under the null", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(test_stat_permR, sample_distribution_permR, graph_hyp_test):
    graph_hyp_test(test_stat_permR, sample_distribution_permR, alternative="larger", null_hypothesis="H0: condition labels are interchangeable within subjects", alternative_hypothesis="Ha: the mean-gap statistic departs from interchangeable condition labels")
    return


@app.cell
def _(test_stat_permR, sample_distribution_permR, get_p_value, mo):
    _p = get_p_value(sample_distribution_permR, test_stat_permR, alternative="larger")
    mo.md("3 related conditions, permutation: there is sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "3 related conditions, permutation: there is insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4 related conditions
    Generate rearrangements within students and calculate the mean absolute gap for each permutation. No centering or separate resampling of conditions is used.
    """)
    return


@app.cell
def _(test1, test2, test3, test4, shuffle_k_related_samples, mean_gap):
    rel_perm_4 = shuffle_k_related_samples(test1, test2, test3, test4, seed=2049)
    test_stat_permR4 = mean_gap(test1, test2, test3, test4)
    sample_distribution_permR4 = mean_gap(*rel_perm_4)
    return test_stat_permR4, sample_distribution_permR4


@app.cell
def _(sample_distribution_permR4, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_permR4, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Related permutation null: 4-condition mean gap", xlabel="Statistic under the null", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(test_stat_permR4, sample_distribution_permR4, graph_hyp_test):
    graph_hyp_test(test_stat_permR4, sample_distribution_permR4, alternative="larger", null_hypothesis="H0: condition labels are interchangeable within subjects", alternative_hypothesis="Ha: the mean-gap statistic departs from interchangeable condition labels")
    return


@app.cell
def _(test_stat_permR4, sample_distribution_permR4, get_p_value, mo):
    _p = get_p_value(sample_distribution_permR4, test_stat_permR4, alternative="larger")
    mo.md("4 related conditions, permutation: there is sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "4 related conditions, permutation: there is insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell
def _(layout_A, layout_B, layout_C, shuffle_k_related_samples, mean_gap):
    layout_perm_null = mean_gap(*shuffle_k_related_samples(layout_A, layout_B, layout_C, seed=2050))
    return (layout_perm_null,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Assume layout labels are interchangeable within each employee under the no-condition-effect null. Calculate the permutation p-value for the prepared mean gap and interpret it at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_layout_perm_p = None
    return (student_layout_perm_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_layout_perm_p = get_p_value(layout_perm_null, layout_observed, alternative="larger")
    ```
    Here p ≈ 0.000250. Reject the no-condition-effect null at 0.05. The mean-gap statistic supplies evidence of a layout effect under the stated within-employee interchangeability assumption; it does not identify every differing pair.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Bootstrap and permutation procedures compare observed mean separation with statistics generated under a specified null. The observed and resampled statistics must use the same calculation.
    - Independent bootstrap tests shift and resample each group separately. Related bootstrap tests shift each condition and resample whole subjects, preserving their measurement links.
    - Independent permutations reassign pooled observations while preserving group sizes. Related permutations rearrange conditions within each subject.
    - Mean dispersion and the average absolute mean gap become large when means spread apart. Their evidence is assessed in the upper tail, even though the population alternative allows means to differ in any direction.
    - Bootstrap equality-of-means nulls and permutation interchangeability nulls are distinct. Resampling does not remove the need to justify the design and null construction.
    - Rejection provides evidence against the stated null, without identifying every differing pair. Non-rejection means insufficient evidence, rather than confirmation of equality. These simulated examples illustrate procedures rather than establish real teaching effects.

    ## Check your understanding
    1. Three classes contain different students without matching across classes. Which design should the resampling procedure preserve: independent groups or repeated measurements?
    2. The same employees complete a task under three layouts. What should remain together when bootstrap samples are generated?
    3. A resampling comparison gives p = 0.02 at significance level 0.05. What decision should you report about its stated null?
    4. The same procedure instead gives p = 0.30. Does that establish equality of the population means?
    5. The overall equal-mean bootstrap test rejects. Does it identify which pairs of population means differ?
    6. During a related permutation, can Employee 1's time move to Employee 2's row?
    7. Why can randomly generated sample means differ even when the resampling construction imposes equal means?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    1. Independent groups: preserve separate group samples and their original sizes.
    2. Each employee's measurements across all layouts. Select whole employee rows, with replacement.
    3. Reject the stated null because 0.02 ≤ 0.05. Describe the evidence in terms of the null and the original population question.
    4. No. There is insufficient evidence to reject the stated null at 0.05; equality has not been established.
    5. No. It supplies evidence of an overall difference. Pairwise questions need suitable follow-up procedures.
    6. No. Values remain within the same employee's row; only condition labels are rearranged.
    7. Resampled sample means fluctuate because of sampling variation. Equal population means do not force every realized sample mean to be identical.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer.
    - Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    - [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html): permutation nulls and finite-simulation p-values. This lesson implements custom resampling procedures.
    - [NumPy: independent shuffling with Generator.permuted](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permuted.html).
    """)
    return


if __name__ == "__main__":
    app.run()
