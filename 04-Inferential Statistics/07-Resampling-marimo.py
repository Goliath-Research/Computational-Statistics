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
    # Resampling Hypothesis Tests for Two Samples
    ## Learning goals
    By the end of this lesson, you should be able to:
    - Apply bootstrap and permutation procedures to compare two independent samples.
    - Apply resampling procedures to analyze paired measurements.
    - Formulate hypotheses and choose a statistic that answers the population question.
    - Interpret resampling distributions, p-values, and conclusions in context.

    We will compare scores from separate classes, then scores from the same students before and after a course. A resampling test generates statistics under a specified null hypothesis and compares them with the observed statistic. Bootstrap procedures draw with replacement; permutation procedures rearrange observations or labels.

    We use 4,000 resamples for each calculation. Seeded random generators make the examples reproducible. Resampling p-values are approximations; more resamples improve their numerical precision without adding observations to the original sample.

    ## 1. Bootstrap for two independent samples
    We simulate 100 scores for Class C from a normal population with mean 85 and standard deviation 3, and 95 scores for Class D from a normal population with mean 90 and standard deviation 3. The classes contain different students without matched pairs. These teaching data do not establish that a teaching method causes better scores.
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(123)
    classC = _rng.normal(85, 3, size=100)
    classD = _rng.normal(90, 3, size=95)
    print("First 10 Class C scores:", classC[:10].round(2))
    print("First 10 Class D scores:", classD[:10].round(2))
    return classC, classD


@app.cell
def _(classC, classD, mo, pd):
    class_summary = pd.DataFrame({"Class": ["C", "D"], "Students": [len(classC), len(classD)], "Mean": [classC.mean(), classD.mean()], "Median": [pd.Series(classC).median(), pd.Series(classD).median()], "Sample SD": [classC.std(ddof=1), classD.std(ddof=1)]})
    mo.Html(class_summary.round(3).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the original score distributions
    The kernel density curves are smooth summaries of each sample. Their horizontal locations show where scores concentrate; their widths show spread. Each curve has area one, so density is not a student count. Dashed lines mark the sample means. The graph describes the samples; the test evaluates evidence about a population difference.
    """)
    return


@app.cell
def _(classC, classD, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=classC, fill=True, color="limegreen", label="Class C", ax=_ax)
    sns.kdeplot(x=classD, fill=True, color="orange", label="Class D", ax=_ax)
    _ax.axvline(classC.mean(), color="green", linestyle="--", label="C sample mean")
    _ax.axvline(classD.mean(), color="darkorange", linestyle="--", label="D sample mean")
    _ax.set(title="Original class scores", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Constructing the null hypothesis for a mean comparison
    We want to test whether the population means are equal. Resampling the original groups would retain their observed mean gap. Instead, shift each sample to a common mean before resampling it:

    $$x_i^*=x_i-\bar x+\bar x_{\mathrm{combined}}.$$

    Subtracting and adding constants changes location while preserving each group's spread and shape. The shifted samples have equal means, but need not have equal variances. We then resample each group separately, keeping its original sample size. This is an approximate bootstrap test for equal means, rather than a relabeling test.
    """)
    return


@app.cell
def _(classC, classD, mo, np, pd):
    overall_mean = np.concatenate((classC, classD)).mean()
    classC_shifted = classC - classC.mean() + overall_mean
    classD_shifted = classD - classD.mean() + overall_mean
    shift_summary = pd.DataFrame({"Sample": ["Shifted C", "Shifted D"], "Mean": [classC_shifted.mean(), classD_shifted.mean()], "Sample SD": [classC_shifted.std(ddof=1), classD_shifted.std(ddof=1)]})
    mo.Html(shift_summary.round(3).to_html(index=False, border=0))
    return classC_shifted, classD_shifted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next graph shows the shifted samples. Their mean lines coincide; their separate spreads remain. This is the data construction used to simulate equal population means.
    """)
    return


@app.cell
def _(classC_shifted, classD_shifted, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=classC_shifted, fill=True, color="limegreen", label="Class C", ax=_ax)
    sns.kdeplot(x=classD_shifted, fill=True, color="orange", label="Class D", ax=_ax)
    _ax.axvline(classC_shifted.mean(), color="green", linestyle="--", label="C sample mean")
    _ax.axvline(classD_shifted.mean(), color="darkorange", linestyle="--", label="D sample mean")
    _ax.set(title="Shifted class scores: common mean", xlabel="Score", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generating bootstrap samples
    `generate_samples` returns a DataFrame with one resample per column. Every column contains the same number of observations as the supplied sample. Drawing with replacement allows an observation to appear more than once in a column. NumPy generates all draws together, without a Python loop over resamples.

    The default seed makes repeated calls reproducible. When generating resamples for independent groups, use different seeds so their random draws are generated separately. The examples below specify these seeds.
    """)
    return


@app.cell
def _(np, pd):
    def generate_samples(sample_data, num_samples=4_000, seed=2026):
        """Generate bootstrap samples; each DataFrame column is one resample."""
        sample_size = len(sample_data)
        rng = np.random.default_rng(seed)
        draws = rng.choice(sample_data, replace=True, size=sample_size * num_samples)
        return pd.DataFrame(draws.reshape(sample_size, num_samples), columns=["S" + str(k) for k in range(num_samples)])
    return (generate_samples,)


@app.cell
def _(classC_shifted, classD_shifted, generate_samples, mo):
    df_C = generate_samples(classC_shifted, seed=2026)
    df_D = generate_samples(classD_shifted, seed=2027)
    mo.vstack([mo.md("**First five rows and first five bootstrap samples for C and D:**"), mo.Html(df_C.iloc[:5, :5].round(2).to_html(border=0)), mo.Html(df_D.iloc[:5, :5].round(2).to_html(border=0))])
    return df_C, df_D


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistic: difference in means
    The observed statistic is the Class C sample mean minus the Class D sample mean. Calculate the same difference for every pair of bootstrap columns. Since the resampled data were shifted to equal means, these simulated differences fluctuate around zero.
    """)
    return


@app.cell
def _(classC, classD, df_C, df_D):
    dMeans = classC.mean() - classD.mean()
    sample_distribution_dMeans = (df_C.mean() - df_D.mean()).to_numpy()
    print("Observed difference in means:", round(dMeans, 3))
    return dMeans, sample_distribution_dMeans


@app.cell
def _(sample_distribution_dMeans, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_dMeans, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Bootstrap null distribution: difference in means", xlabel="Statistic under the null hypothesis", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculating a p-value
    `get_p_value` compares the observed statistic with the simulated null statistics. `smaller` uses the lower tail, `larger` the upper tail, and `two-sided` doubles the smaller tail, capped at one. Tail comparisons include equally extreme values.

    For B simulated statistics, each tail is estimated as (extreme resamples + 1)/(B + 1). This finite-simulation convention avoids reporting p = 0 simply because no simulated statistic reached the observed value. It does not make the bootstrap exact. With 4,000 resamples, the smallest two-sided value this convention reports is about 0.0005.
    """)
    return


@app.cell
def _(np):
    def get_p_value(sample_distribution, obs_value, alternative="two-sided"):
        """Estimate a resampling p-value using inclusive tails and a plus-one correction."""
        values = np.asarray(sample_distribution)
        left = (np.count_nonzero(values <= obs_value) + 1) / (len(values) + 1)
        right = (np.count_nonzero(values >= obs_value) + 1) / (len(values) + 1)
        if alternative == "smaller":
            return float(left)
        if alternative == "larger":
            return float(right)
        return float(min(1, 2 * min(left, right)))
    return (get_p_value,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reporting and visualizing the test
    `graph_hyp_test` uses the same resampling p-value function throughout the lesson. It prints the hypotheses, statistic, p-value, significance level, and decision, and returns the figure. Pass the hypothesis text for the question being tested. The red quantile boundaries are visual approximations; the printed decision uses the calculated p-value.
    """)
    return


@app.cell
def _(get_p_value, np, plt, sns):
    def graph_hyp_test(sample_value, sample_distribution, alpha=0.05, alternative="two-sided", null_hypothesis="H0: population difference = 0", alternative_hypothesis="Ha: population difference ≠ 0"):
        """Report a resampling test and plot its null distribution and critical tails."""
        p_val = get_p_value(sample_distribution, sample_value, alternative)
        print("--- Resampling hypothesis test ---")
        print(null_hypothesis)
        print(alternative_hypothesis)
        print(f"Observed statistic = {sample_value:.4g}; p-value = {p_val:.4g}")
        print(f"Significance level = {alpha:g}")
        print("There is sufficient evidence to reject the null hypothesis." if p_val <= alpha else "There is insufficient evidence to reject the null hypothesis.")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(x=sample_distribution, color="skyblue", ax=ax)
        x, y = ax.lines[0].get_data()
        ax.fill_between(x, y, color="skyblue", alpha=0.35)
        if alternative == "two-sided":
            boundaries = np.percentile(sample_distribution, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        elif alternative == "smaller":
            boundaries = [np.percentile(sample_distribution, 100 * alpha)]
        else:
            boundaries = [np.percentile(sample_distribution, 100 * (1 - alpha))]
        for boundary in boundaries:
            ax.axvline(boundary, color="orangered", linestyle="--", label="Approximate critical boundary")
        x, y = ax.lines[0].get_data()
        if alternative in ("two-sided", "smaller"):
            ax.fill_between(x, y, where=x <= boundaries[0], color="orangered", alpha=0.4)
        if alternative in ("two-sided", "larger"):
            ax.fill_between(x, y, where=x >= boundaries[-1], color="orangered", alpha=0.4)
        ax.axvline(sample_value, color="black", linewidth=2, label="Observed statistic")
        ax.set(title="Resampling null distribution and critical regions", xlabel="Statistic", ylabel="Density")
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        fig.tight_layout()
        plt.close(fig)
        return fig
    return (graph_hyp_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hypotheses:** H0: μC = μD; Ha: μC ≠ μD. Use significance level 0.05.

    The blue density curve summarizes the simulated null statistics. Red boundaries mark approximate critical regions, and red shading indicates their tails. The black line marks the observed statistic. Its position shows how unusual the observed result is under this null construction. The p-value uses the resampled values directly, not the smoothed curve.
    """)
    return


@app.cell
def _(dMeans, sample_distribution_dMeans, graph_hyp_test):
    graph_hyp_test(dMeans, sample_distribution_dMeans, alternative="two-sided", null_hypothesis="H0: μC = μD", alternative_hypothesis="Ha: μC ≠ μD")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The observed mean difference is well outside the central null distribution. At significance level 0.05, the sample provides evidence that the population mean scores differ. The observed Class C mean is lower. The small simulated p-value expresses strong evidence here; it is not an exact measure of an extremely small tail probability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistic: difference in medians
    A median comparison asks a different population question. Shifting to equal **means** need not produce equal **medians**. For this example, center each group on its own sample median before drawing bootstrap samples. Both centered medians are zero, while the group shapes and spreads remain.

    This is an approximate median bootstrap test for independent continuous populations with well-behaved medians. It illustrates changing both the statistic and the null construction to match the question.
    """)
    return


@app.cell
def _(classC, classD, generate_samples, np):
    classC_median_null = classC - np.median(classC)
    classD_median_null = classD - np.median(classD)
    df_C_median = generate_samples(classC_median_null, seed=2028)
    df_D_median = generate_samples(classD_median_null, seed=2029)
    dMedians = np.median(classC) - np.median(classD)
    sample_distribution_dMedians = (df_C_median.median() - df_D_median.median()).to_numpy()
    print("Observed difference in medians:", round(dMedians, 3))
    return dMedians, sample_distribution_dMedians


@app.cell
def _(sample_distribution_dMedians, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_dMedians, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Bootstrap null distribution: difference in medians", xlabel="Statistic under the null hypothesis", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hypotheses:** H0: population median C = population median D; Ha: population median C ≠ population median D. Use significance level 0.05.

    The blue density curve summarizes the simulated null statistics. Red boundaries mark approximate critical regions, and red shading indicates their tails. The black line marks the observed statistic. Its position shows how unusual the observed result is under this null construction. The p-value uses the resampled values directly, not the smoothed curve.
    """)
    return


@app.cell
def _(dMedians, sample_distribution_dMedians, graph_hyp_test):
    graph_hyp_test(dMedians, sample_distribution_dMedians, alternative="two-sided", null_hypothesis="H0: population median C = population median D", alternative_hypothesis="Ha: population median C ≠ population median D")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At significance level 0.05, these data provide evidence of different population medians. This conclusion concerns medians, rather than population means.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistic: Welch t
    We can also scale the mean difference by its estimated standard error:

    $$t=\frac{\bar x_C-\bar x_D}{\sqrt{s_C^2/n_C+s_D^2/n_D}}.$$

    Use sample variances with divisor n−1 for both the observed statistic and every resample. The null remains equal population means, so reuse the **mean-shifted** bootstrap columns. Here the resampling distribution supplies the reference distribution; we do not calculate a theoretical t-distribution p-value.
    """)
    return


@app.cell
def _(classC, classD, df_C, df_D, np):
    t = (classC.mean() - classD.mean()) / np.sqrt(classC.var(ddof=1) / len(classC) + classD.var(ddof=1) / len(classD))
    sample_distribution_t = ((df_C.mean() - df_D.mean()) / np.sqrt(df_C.var(ddof=1) / df_C.shape[0] + df_D.var(ddof=1) / df_D.shape[0])).to_numpy()
    print("Observed Welch statistic:", round(t, 3))
    return t, sample_distribution_t


@app.cell
def _(sample_distribution_t, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_t, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Bootstrap null distribution: Welch statistic", xlabel="Statistic under the null hypothesis", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hypotheses:** H0: μC = μD; Ha: μC ≠ μD. Use significance level 0.05.

    The blue density curve summarizes the simulated null statistics. Red boundaries mark approximate critical regions, and red shading indicates their tails. The black line marks the observed statistic. Its position shows how unusual the observed result is under this null construction. The p-value uses the resampled values directly, not the smoothed curve.
    """)
    return


@app.cell
def _(t, sample_distribution_t, graph_hyp_test):
    graph_hyp_test(t, sample_distribution_t, alternative="two-sided", null_hypothesis="H0: μC = μD", alternative_hypothesis="Ha: μC ≠ μD")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The studentized mean comparison also rejects equal population means at 0.05. The raw mean difference and the Welch statistic address the same mean hypothesis; they scale the evidence differently. Agreement in this example is not a reason to choose whichever procedure gives a smaller p-value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A new problem: delivery times
    Two independent delivery services have the following delivery times in minutes. We want to know whether their population mean delivery times differ. We prepare separate mean-centered bootstrap samples, using the procedures already taught.
    """)
    return


@app.cell
def _(generate_samples, mo, np, pd):
    delivery_A = np.array([28.4, 31.2, 29.7, 32.5, 30.1, 27.8, 33.3, 29.0])
    delivery_B = np.array([34.1, 32.8, 35.6, 33.5, 36.2, 31.9, 34.8, 35.0])
    delivery_table = pd.DataFrame({"Service A (min)": delivery_A, "Service B (min)": delivery_B})
    mo.Html(delivery_table.to_html(index=False, border=0))
    delivery_observed = delivery_A.mean() - delivery_B.mean()
    delivery_boot_A = generate_samples(delivery_A - delivery_A.mean(), seed=2030)
    delivery_boot_B = generate_samples(delivery_B - delivery_B.mean(), seed=2031)
    delivery_boot_null = (delivery_boot_A.mean() - delivery_boot_B.mean()).to_numpy()
    return delivery_A, delivery_B, delivery_observed, delivery_boot_null


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the two-sided bootstrap p-value for the difference in population mean delivery times using the prepared resampling distribution. State the hypotheses and interpret the result at significance level 0.05.
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
    student_delivery_boot_p = get_p_value(delivery_boot_null, delivery_observed)
    ```
    H0: μA = μB; Ha: μA ≠ μB. The seeded calculation gives p ≈ 0.000500. Reject equal population means at 0.05. The data provide evidence of different population mean delivery times; Service A has the lower observed mean.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bootstrap for paired samples
    We simulate before and after scores for the same 90 students. Before-scores come from a normal population with mean 80 and standard deviation 4. Each after-score equals that student's before-score plus a change drawn from a normal population with mean 6 and standard deviation 3.

    Define each difference as **after minus before**. Positive values represent improved scores. The paired mean question is H0: μd = 0 versus Ha: μd ≠ 0. Equality of population means does not mean every student's difference is zero.
    """)
    return


@app.cell
def _(np, mo, pd):
    _rng = np.random.default_rng(12)
    grade_before = _rng.normal(80, 4, size=90)
    grade_after = grade_before + _rng.normal(6, 3, size=90)
    diff = grade_after - grade_before
    stat = diff.mean()
    paired_table = pd.DataFrame({"Before": grade_before, "After": grade_after, "After − before": diff})
    mo.Html(paired_table.head(10).round(2).to_html(index=False, border=0))
    print("Observed mean improvement:", round(stat, 3))
    return grade_before, grade_after, diff, stat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Keep the two measurements from each student together when forming a difference. Then center the differences at zero and resample them with replacement. The centered data represent the mean null; their resampled means describe sampling variation under that construction. Resampling the uncentered differences would retain the observed mean improvement.
    """)
    return


@app.cell
def _(diff, generate_samples, mo):
    diff_Ho = diff - diff.mean()
    df_pair = generate_samples(diff_Ho, seed=2032)
    sample_distribution_pair = df_pair.mean().to_numpy()
    mo.Html(df_pair.iloc[:5, :5].round(2).to_html(border=0))
    return sample_distribution_pair, df_pair


@app.cell
def _(sample_distribution_pair, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_pair, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Paired bootstrap null distribution: mean change", xlabel="Statistic under the null hypothesis", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hypotheses:** H0: μd = 0; Ha: μd ≠ 0. Use significance level 0.05.

    The blue density curve summarizes the simulated null statistics. Red boundaries mark approximate critical regions, and red shading indicates their tails. The black line marks the observed statistic. Its position shows how unusual the observed result is under this null construction. The p-value uses the resampled values directly, not the smoothed curve.
    """)
    return


@app.cell
def _(stat, sample_distribution_pair, graph_hyp_test):
    graph_hyp_test(stat, sample_distribution_pair, alternative="two-sided", null_hypothesis="H0: μd = 0", alternative_hypothesis="Ha: μd ≠ 0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data provide evidence that the population mean change differs from zero. The observed change is positive. These are simulated measurements; the test illustrates the procedure rather than establishes a causal effect of a course.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A new problem: task completion times
    Ten workers perform the same task before and after practice. Times are in seconds. Each row belongs to one worker. Define changes as after minus before, so negative values indicate faster completion. The prepared bootstrap distribution below represents a population mean change of zero.
    """)
    return


@app.cell
def _(generate_samples, mo, np, pd):
    task_before = np.array([52, 48, 61, 55, 46, 58, 50, 63, 54, 49])
    task_after = np.array([47, 46, 59, 50, 48, 54, 49, 60, 56, 47])
    task_diff = task_after - task_before
    task_observed = task_diff.mean()
    task_table = pd.DataFrame({"Worker": np.arange(1, 11), "Before (s)": task_before, "After (s)": task_after})
    mo.Html(task_table.to_html(index=False, border=0))
    task_boot_null = generate_samples(task_diff - task_diff.mean(), seed=2033).mean().to_numpy()
    return task_diff, task_observed, task_boot_null


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the bootstrap p-value for evidence that practice reduces population mean completion time. State the hypotheses and interpret your result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_task_boot_p = None
    return (student_task_boot_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_task_boot_p = get_p_value(task_boot_null, task_observed, alternative="smaller")
    ```
    H0: μd = 0; Ha: μd < 0. A negative change means a shorter completion time. Here p ≈ 0.00150. Reject the mean-zero null at 0.05: these data provide evidence of reduced population mean completion time.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Permutation test for two independent samples
    Return to the original Class C and Class D scores. Pool the scores and randomly divide them into groups of 100 and 95, without replacement within each rearrangement. Each rearrangement retains every observed score exactly once; the same allocation can occur again in another repetition.

    Under the permutation null, group labels are interchangeable: the populations have the same distribution, or a suitable randomized design justifies the reassignments. Equal means alone do not guarantee this condition if the distributions or variances differ. We use the difference in means to detect a location difference under this null.

    The simulated group means will vary. The null does **not** say every rearranged mean difference is zero; it says the observed labels should not make the statistic unusually extreme.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shuffling the samples
    `shuffle_2_samples` keeps your column-per-rearrangement DataFrame structure. NumPy independently shuffles each column of the pooled data before splitting it at the original first-group size. This avoids calling a Python function separately for every DataFrame column.
    """)
    return


@app.cell
def _(np, pd):
    def shuffle_2_samples(sample1, sample2, num_samples=4_000, seed=2034):
        """Reassign pooled observations without replacement; return two DataFrames."""
        pool = np.concatenate([sample1, sample2])
        rng = np.random.default_rng(seed)
        repeated = np.broadcast_to(pool[:, None], (len(pool), num_samples))
        shuffled = rng.permuted(repeated, axis=0)
        columns = ["S" + str(k) for k in range(num_samples)]
        return (pd.DataFrame(shuffled[:len(sample1)], columns=columns), pd.DataFrame(shuffled[len(sample1):], columns=columns))
    return (shuffle_2_samples,)


@app.cell
def _(classC, classD, mo, shuffle_2_samples):
    df_C_p, df_D_p = shuffle_2_samples(classC, classD)
    mo.vstack([mo.md("**First five rows and five permutations for each class:**"), mo.Html(df_C_p.iloc[:5, :5].round(2).to_html(border=0)), mo.Html(df_D_p.iloc[:5, :5].round(2).to_html(border=0))])
    return df_C_p, df_D_p


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Statistic: difference in means
    Calculate the observed difference on the original scores, then the same statistic for each rearrangement. The histogram shows differences generated by changing labels, rather than by drawing new bootstrap observations.
    """)
    return


@app.cell
def _(classC, classD, df_C_p, df_D_p):
    dMeans_p = classC.mean() - classD.mean()
    sample_distribution_dMeans_p = (df_C_p.mean() - df_D_p.mean()).to_numpy()
    print("Observed difference in means:", round(dMeans_p, 3))
    return dMeans_p, sample_distribution_dMeans_p


@app.cell
def _(sample_distribution_dMeans_p, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_dMeans_p, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Permutation null distribution: difference in means", xlabel="Statistic under the null hypothesis", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hypotheses:** H0: the class population distributions are identical; Ha: the mean-difference statistic departs from the label-interchangeable null. Use significance level 0.05.

    The blue density curve summarizes the simulated null statistics. Red boundaries mark approximate critical regions, and red shading indicates their tails. The black line marks the observed statistic. Its position shows how unusual the observed result is under this null construction. The p-value uses the resampled values directly, not the smoothed curve.
    """)
    return


@app.cell
def _(dMeans_p, sample_distribution_dMeans_p, graph_hyp_test):
    graph_hyp_test(dMeans_p, sample_distribution_dMeans_p, alternative="two-sided", null_hypothesis="H0: the class population distributions are identical", alternative_hypothesis="Ha: the mean-difference statistic departs from the label-interchangeable null")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The observed difference is unusual under interchangeable class labels, so reject that null at 0.05. The difference-in-means statistic points to a location difference in these simulated equal-spread populations. This permutation null is stronger than the bootstrap equal-mean null.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For another application, reuse the delivery times. For this permutation exercise, assume service labels are interchangeable under the null of identical delivery-time distributions.
    """)
    return


@app.cell
def _(delivery_A, delivery_B, shuffle_2_samples):
    delivery_perm_A, delivery_perm_B = shuffle_2_samples(delivery_A, delivery_B, seed=2035)
    delivery_perm_null = (delivery_perm_A.mean() - delivery_perm_B.mean()).to_numpy()
    return (delivery_perm_null,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate a two-sided permutation p-value using the difference in mean delivery times. State the null hypothesis and interpret the result at significance level 0.05.
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
    student_delivery_perm_p = get_p_value(delivery_perm_null, delivery_observed)
    ```
    The null is identical population delivery-time distributions, with interchangeable service labels. Here p ≈ 0.000500. Reject that null at 0.05. The observed mean delivery time is lower for Service A.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Permutation test for paired samples
    A paired permutation swaps the two measurements **within** each pair, never between different students. With a difference defined as after minus before, swapping the measurements changes d to −d. We implement this by multiplying each original difference by a randomly chosen +1 or −1.

    Unlike the paired bootstrap, do not center the differences first. Under this sign-flip null, the signs are interchangeable: in a sampling model, the difference distribution is symmetric about zero. A randomized paired design can also justify within-pair swaps under its no-effect null. A population mean of zero alone does not justify sign flips for asymmetric differences. Our simulated normally distributed changes satisfy symmetry under the zero-change null.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Randomly changing signs
    `shuffle_1_sample` returns one sign-flipped sample per DataFrame column. Each original difference stays in its row; only its sign can change. NumPy generates all random signs together, avoiding a Python function call for each table entry.
    """)
    return


@app.cell
def _(np, pd):
    def shuffle_1_sample(sample, num_samples=4_000, seed=2036):
        """Generate paired sign-flip samples; each DataFrame column is one rearrangement."""
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(len(sample), num_samples))
        shuffled = np.asarray(sample)[:, None] * signs
        return pd.DataFrame(shuffled, columns=["S" + str(k) for k in range(num_samples)])
    return (shuffle_1_sample,)


@app.cell
def _(diff, mo, shuffle_1_sample):
    df_pair_p = shuffle_1_sample(diff)
    sample_distribution_pair_p = df_pair_p.mean().to_numpy()
    mo.Html(df_pair_p.iloc[:5, :5].round(2).to_html(border=0))
    return df_pair_p, sample_distribution_pair_p


@app.cell
def _(sample_distribution_pair_p, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.histplot(x=sample_distribution_pair_p, bins=30, color="steelblue", ax=_ax)
    _ax.set(title="Paired permutation null distribution: mean change", xlabel="Statistic under the null hypothesis", ylabel="Number of resamples")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hypotheses:** H0: differences are symmetric about zero; Ha: the mean-change statistic departs from the sign-interchangeable null. Use significance level 0.05.

    The blue density curve summarizes the simulated null statistics. Red boundaries mark approximate critical regions, and red shading indicates their tails. The black line marks the observed statistic. Its position shows how unusual the observed result is under this null construction. The p-value uses the resampled values directly, not the smoothed curve.
    """)
    return


@app.cell
def _(stat, sample_distribution_pair_p, graph_hyp_test):
    graph_hyp_test(stat, sample_distribution_pair_p, alternative="two-sided", null_hypothesis="H0: differences are symmetric about zero", alternative_hypothesis="Ha: the mean-change statistic departs from the sign-interchangeable null")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The observed mean improvement is unusual under the sign-flip null. Reject it at 0.05. Under the symmetric location model used in this simulation, the evidence supports a positive population location change. A sign-flip test is not a general mean-zero test for arbitrary asymmetric data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reuse the workers’ task times for a paired permutation exercise. Assume a symmetric difference model under the zero-change null. The sign-flip distribution is prepared from the original, uncentered differences.
    """)
    return


@app.cell
def _(shuffle_1_sample, task_diff):
    task_perm_null = shuffle_1_sample(task_diff, seed=2037).mean().to_numpy()
    return (task_perm_null,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the one-sided paired permutation p-value for a reduction in completion time. Interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_task_perm_p = None
    return (student_task_perm_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_task_perm_p = get_p_value(task_perm_null, task_observed, alternative="smaller")
    ```
    After minus before is negative for faster completion, so use the lower tail. Under the stated symmetric model, the null is a zero-centered difference distribution. Here p ≈ 0.02274. Reject that null at 0.05: under the stated symmetric location model, these data provide evidence of a reduction in completion time.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Bootstrap and permutation tests compare an observed statistic with values generated under a specified null hypothesis. The resampling construction must match the population question.
    - Independent bootstrap comparisons resample the groups separately after imposing the relevant equal-location null. Permutation comparisons reassign pooled observations under interchangeable labels.
    - Paired procedures preserve each individual's two measurements. The paired bootstrap resamples centered differences; paired permutations swap measurements within pairs, implemented as sign flips.
    - The mean and median answer different population questions. A scaled mean statistic still addresses means, but accounts for estimated sampling variability.
    - A p-value at or below the chosen significance level supplies evidence against the null. A larger value means insufficient evidence to reject it, rather than proof of equality.
    - Resampling replaces a theoretical reference distribution with a simulated one; it does not remove the need for a suitable sampling design or null assumptions. Conclusions must remain within the scope of the data.

    ## Check your understanding
    1. Two classes contain different students with no matching between classes. Is this an independent or paired comparison?
    2. The same workers are measured before and after practice. Each before-time is matched to that worker's after-time. Is this an independent or paired comparison?
    3. A researcher wants to compare typical delivery times using population medians. Which statistic should be calculated for the original data and every resample?
    4. A two-sided resampling test of equal population means gives p = 0.02. At significance level 0.05, what decision and conclusion should you report?
    5. The same test instead gives p = 0.30. Does that establish equal population means?
    6. In the student-score examples, changes are defined as after minus before. What does a positive observed mean change mean?
    7. In a paired permutation, can one worker's before-time be swapped with a different worker's after-time? Explain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    1. Independent: the measurements belong to separate groups without matched pairs.
    2. Paired: each worker contributes both measurements.
    3. The first service's sample median minus the second service's sample median, in a consistent order. The bootstrap null construction must also impose equal medians.
    4. Reject equal population means because 0.02 ≤ 0.05. The data provide evidence that the population means differ.
    5. No. There is insufficient evidence to reject equal population means at 0.05; equality has not been established.
    6. The sample's average after-score is higher than its average before-score. The test assesses whether this provides evidence about a population change.
    7. No. Swaps stay within each worker's pair; mixing workers would destroy the pairing.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapter 16. Chapman & Hall/CRC.
    - Davison, A. C., and Hinkley, D. V. (1997). *Bootstrap Methods and their Applications*. Chapter 4. Cambridge University Press.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    - [SciPy documentation: permutation-test null hypotheses and simulated p-values](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html). The lesson implements its own resampling functions.
    """)
    return


if __name__ == "__main__":
    app.run()
