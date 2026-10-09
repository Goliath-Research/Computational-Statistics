# /// script
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "scipy", "statsmodels", "seaborn"]
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
    from scipy.stats import skew
    from statsmodels.distributions.empirical_distribution import ECDF
    plt.style.use("seaborn-v0_8-whitegrid")
    return ECDF, mo, np, pd, plt, skew, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootstrap One-Sample Hypothesis Tests

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Formulate hypotheses about a population mean, variance, or skewness.
    - Apply bootstrap methods to perform one-sample hypothesis tests.
    - Interpret bootstrap p-values and communicate conclusions in the context of the problem.
    - Recognize the assumptions and limitations of bootstrap inference.

    ## 1. The sample and the bootstrap
    A one-sample hypothesis test compares a population parameter with a specified value. The sample statistic provides the evidence for that comparison.

    The bootstrap estimates how a statistic varies across samples by repeatedly sampling **with replacement** from the observed data. Each resample has the same size as the original sample. Some observations can appear more than once; others may not appear at all.

    Our data represent the heights, in centimetres, of 50 fictitious community-centre visitors. They are generated from the integers 158 through 174, with equal probabilities. This generating distribution is discrete uniform, rather than normal. The seed makes the lesson reproducible.
    """)
    return


@app.cell
def _(np):
    heights = np.random.default_rng(1234).integers(158, 175, size=50)
    sample_mean = heights.mean()
    print("First 10 heights (cm):", heights[:10])
    print("Sample size:", len(heights))
    print("Sample mean (cm):", round(sample_mean, 2))
    return heights, sample_mean


@app.cell
def _(heights, np, plt, sample_mean):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(heights, bins=np.arange(157.5, 175.5, 1), color="#4C78A8", edgecolor="white")
    _ax.axvline(sample_mean, color="black", label=f"Sample mean {sample_mean:.2f}")
    _ax.set(title="Community-centre visitors", xlabel="Height (cm)", ylabel="Number of visitors")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generating bootstrap samples
    `generate_samples_b` returns a DataFrame with one bootstrap sample in each column. A seeded generator makes repeated calls with the same inputs reproducible. Its default seed means we do not need to specify another seed for every calculation.

    We use 4,000 resamples. This is the number of simulated samples, not the number of observations in each sample. More resamples reduce simulation noise; they do not add information to the original sample.
    """)
    return


@app.cell
def _(np, pd):
    def generate_samples_b(sample_data, num_samples=4_000, seed=2026):
        """Return bootstrap samples as columns, each the size of the original sample."""
        rng = np.random.default_rng(seed)
        samples = rng.choice(sample_data, size=(len(sample_data), num_samples), replace=True)
        return pd.DataFrame(samples, columns=[f"S{k + 1}" for k in range(num_samples)])

    return (generate_samples_b,)


@app.cell
def _(generate_samples_b, heights, mo):
    height_samples = generate_samples_b(heights)
    print("Bootstrap DataFrame shape:", height_samples.shape)
    mo.Html(height_samples.iloc[:5, :5].to_html(index=False, border=0))
    return (height_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The table shows only five rows and five columns. All 50 observations in each of the 4,000 resamples are retained. Taking the mean of each column gives the bootstrap distribution of the mean.
    """)
    return


@app.cell
def _(height_samples):
    mean_distribution = height_samples.mean().to_numpy()
    print("Number of bootstrap means:", len(mean_distribution))
    return (mean_distribution,)


@app.cell
def _(mean_distribution, plt, sample_mean):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(mean_distribution, bins=30, color="#54A24B", edgecolor="white")
    _ax.axvline(sample_mean, color="black", label=f"Observed mean {sample_mean:.2f}")
    _ax.set(title="Bootstrap distribution of the sample mean", xlabel="Mean height (cm)", ylabel="Number of resamples")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The histogram shows the bootstrap means directly. A kernel density estimate (KDE) gives a smooth view of the same distribution. The smoothing is for visualization; the p-values below are calculated from the simulated values.
    """)
    return


@app.cell
def _(mean_distribution, plt, sample_mean, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=mean_distribution, color="mediumseagreen", fill=True, ax=_ax)
    _ax.axvline(sample_mean, color="black", label=f"Observed mean {sample_mean:.2f}")
    _ax.set(title="Bootstrap distribution of the sample mean", xlabel="Mean height (cm)", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Connecting the bootstrap to the null hypothesis
    Suppose we test \(H_0:\mu=170\). Resampling the original heights estimates variation around the sample mean, not around 170. We must account for that difference before calculating a p-value.

    Let \(\hat\theta\) be the observed statistic, \(\theta_0\) its hypothesized population value, and \(\hat\theta^*\) a bootstrap statistic. The bootstrap errors

    $$e^*=\hat\theta^*-\hat\theta$$

    approximate the sampling errors \(\hat\theta-\theta\). Under the null hypothesis, the observed error is \(\hat\theta-\theta_0\). We compare that observed error with the bootstrap errors.

    For plotting, the equivalent reference values are

    $$T_0^*=\hat\theta^*-\hat\theta+\theta_0.$$

    Compare \(\hat\theta\) with these reference values. We subtract the **observed statistic**, rather than the average of the bootstrap statistics, to preserve the bootstrap estimate of bias. The average reference value therefore need not equal \(\theta_0\) exactly.

    For the mean, this is also equivalent to resampling heights shifted by \(170-\bar x\). For other statistics, it is an approximation on the statistic scale, rather than a claim that shifting the observations enforces the null hypothesis.
    """)
    return


@app.cell
def _(mean_distribution, sample_mean):
    mean_null_170 = mean_distribution - sample_mean + 170
    print("Observed mean:", round(sample_mean, 2))
    print("Hypothesized mean:", 170)
    return (mean_null_170,)


@app.cell
def _(mean_null_170, plt, sample_mean, sns):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=mean_null_170, color="dodgerblue", fill=True, ax=_ax)
    _ax.axvline(170, color="orangered", linestyle="--", label="Hypothesized mean 170")
    _ax.axvline(sample_mean, color="black", label=f"Observed mean {sample_mean:.2f}")
    _ax.set(title="Bootstrap reference for a mean of 170 cm", xlabel="Mean height (cm)", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Critical regions
    At significance level \(\alpha=0.05\), a left-tailed test rejects for unusually low statistics; a right-tailed test rejects for unusually high statistics. An equal-tailed two-sided test allocates 0.025 to each tail, with cutoffs at the 2.5th and **97.5th** percentiles.

    The three-panel comparison below shows all three critical regions for the same reference distribution. Orange shading marks the rejection regions. Individual test graphs then add the observed statistic as a black line. The plotting function is reused for mean, variance, and skewness.
    """)
    return


@app.cell
def _(np, plt, sns):
    def graph_bootstrap_test(null_distribution, observed, alternative="two-sided", alpha=0.05, measure="Statistic"):
        """Plot the reference distribution, observed statistic, and critical cutoffs."""
        if alternative == "smaller":
            percentiles = [100 * alpha]
        elif alternative == "larger":
            percentiles = [100 * (1 - alpha)]
        else:
            percentiles = [100 * alpha / 2, 100 * (1 - alpha / 2)]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(x=null_distribution, color="lightskyblue", fill=True, ax=ax)
        # Draw an unfilled KDE to obtain coordinates for the shaded tail regions.
        sns.kdeplot(x=null_distribution, color="steelblue", ax=ax)
        density_x, density_y = ax.lines[-1].get_data()
        cutoffs = np.percentile(null_distribution, percentiles)
        if alternative == "smaller":
            tail = density_x <= cutoffs[0]
        elif alternative == "larger":
            tail = density_x >= cutoffs[0]
        else:
            tail = (density_x <= cutoffs[0]) | (density_x >= cutoffs[1])
        ax.fill_between(density_x, 0, density_y, where=tail, color="orangered", alpha=0.5, label="Critical region")
        for cutoff in np.percentile(null_distribution, percentiles):
            ax.axvline(cutoff, color="#F58518", linestyle="--", label=f"Critical value {cutoff:.3f}")
        ax.axvline(observed, color="black", linewidth=2, label=f"Observed {observed:.3f}")
        ax.set(title=f"Bootstrap reference: {alternative} alternative", xlabel=measure, ylabel="Density")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plt.close(fig)
        return fig

    return (graph_bootstrap_test,)


@app.cell
def _(mean_null_170, np, plt, sns):
    alpha = 0.05
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for _ax, _alternative, _title in zip(_axes, ["smaller", "larger", "two-sided"], ["Critical region: left", "Critical region: right", "Critical region: two-sided"]):
        sns.kdeplot(x=mean_null_170, color="lightskyblue", fill=True, ax=_ax)
        sns.kdeplot(x=mean_null_170, color="steelblue", ax=_ax)
        _x, _y = _ax.lines[-1].get_data()
        if _alternative == "smaller":
            _cutoffs = np.percentile(mean_null_170, [100 * alpha])
            _tail = _x <= _cutoffs[0]
            _label = "α = 0.05"
        elif _alternative == "larger":
            _cutoffs = np.percentile(mean_null_170, [100 * (1 - alpha)])
            _tail = _x >= _cutoffs[0]
            _label = "α = 0.05"
        else:
            _cutoffs = np.percentile(mean_null_170, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            _tail = (_x <= _cutoffs[0]) | (_x >= _cutoffs[1])
            _label = "α/2 = 0.025 in each tail"
        _ax.fill_between(_x, 0, _y, where=_tail, color="orangered", alpha=0.6)
        for _cutoff in _cutoffs:
            _ax.axvline(_cutoff, color="orangered", linestyle="--")
        _ax.set_ylim(0, max(_y) * 1.2)
        _ax.text(0.5, 0.92, _label, transform=_ax.transAxes, ha="center", color="darkred")
        _ax.set(title=_title, xlabel="Mean height (cm)", ylabel="Density")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A sample mean is 166 cm. We want reference values for a hypothesized mean of 170 cm. What constant is added to each bootstrap mean? Why should we not simply compare 170 with the unadjusted bootstrap means?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    Add 4 cm. The adjustment makes the comparison represent the hypothesized population mean. Unadjusted bootstrap means describe variation around the observed sample estimate, rather than directly representing the null hypothesis.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Calculating the p-value
    The p-value estimates how often the null reference produces a statistic at least as extreme as the observed statistic, in the direction specified by the alternative.

    ### A smaller alternative
    For \(H_0:\mu=170\) against \(H_a:\mu<170\), count reference means **at or below the observed mean** and divide by the number of resamples. The comparison uses the observed mean, not the hypothesized value 170.
    """)
    return


@app.cell
def _(graph_bootstrap_test, mean_null_170, sample_mean):
    graph_bootstrap_test(mean_null_170, sample_mean, alternative="smaller", measure="Mean height (cm)")
    return


@app.cell
def _(mean_null_170, np, sample_mean):
    mean_left_p = np.mean(mean_null_170 <= sample_mean)
    print(f"Left-tailed p-value: {mean_left_p:.4g}")
    return (mean_left_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The empirical cumulative distribution function (ECDF) gives the proportion of reference values at or below any chosen value. Evaluating it at the observed mean gives the same left-tail calculation.
    """)
    return


@app.cell
def _(ECDF, mean_null_170, sample_mean):
    mean_null_ecdf = ECDF(mean_null_170)
    mean_left_ecdf = float(mean_null_ecdf(sample_mean))
    print(f"ECDF left-tailed p-value: {mean_left_ecdf:.4g}")
    return (mean_null_ecdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A larger alternative
    For \(H_a:\mu>170\), count reference means **at or above the observed mean**. We include equality in both one-sided tails because the statistics can be discrete.

    The ECDF includes equality in its lower tail, so \(1-F(t)\) gives the proportion strictly above \(t\). Add the proportion equal to \(t\) to obtain the inclusive upper tail. The two inclusive tails can sum to more than 1 when there are ties.
    """)
    return


@app.cell
def _(graph_bootstrap_test, mean_null_170, sample_mean):
    graph_bootstrap_test(mean_null_170, sample_mean, alternative="larger", measure="Mean height (cm)")
    return


@app.cell
def _(mean_null_170, np, sample_mean):
    mean_right_p = np.mean(mean_null_170 >= sample_mean)
    print(f"Right-tailed p-value: {mean_right_p:.4g}")
    return (mean_right_p,)


@app.cell
def _(mean_null_170, mean_null_ecdf, np, sample_mean):
    mean_right_ecdf = 1 - float(mean_null_ecdf(sample_mean)) + np.mean(mean_null_170 == sample_mean)
    print(f"ECDF right-tailed p-value: {mean_right_ecdf:.4g}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A two-sided alternative
    For \(H_a:\mu\neq 170\), this lesson uses an **equal-tailed** p-value: double the smaller one-sided tail probability and cap the result at 1. It corresponds to allocating significance equally between the two tails. Other two-sided definitions are possible, especially for asymmetric distributions.
    """)
    return


@app.cell
def _(graph_bootstrap_test, mean_null_170, sample_mean):
    graph_bootstrap_test(mean_null_170, sample_mean, alternative="two-sided", measure="Mean height (cm)")
    return


@app.cell
def _(mean_left_p, mean_right_p):
    mean_two_p = min(1.0, 2 * min(mean_left_p, mean_right_p))
    print(f"Two-sided p-value: {mean_two_p:.4g}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A reusable p-value function
    `get_p_value` collects these calculations. Supply the reference distribution, the observed statistic, and the alternative. This function is reused by the reporting procedure below.

    A bootstrap p-value is a simulation estimate. If no resamples reach the observed tail, the empirical estimate is zero; this does not mean the underlying probability is exactly zero. Our reports state this explicitly. With 4,000 resamples, each one-sided tail count changes the estimate by 1/4,000.
    """)
    return


@app.cell
def _(np):
    def get_p_value(sample_distribution, obs_value, alternative="two-sided"):
        """Return an empirical p-value with inclusive one-sided tails."""
        values = np.asarray(sample_distribution)
        left = np.mean(values <= obs_value)
        right = np.mean(values >= obs_value)
        if alternative == "smaller":
            return float(left)
        if alternative == "larger":
            return float(right)
        return float(min(1.0, 2 * min(left, right)))

    return (get_p_value,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following fixed data record delivery times, in hours, for 20 independently selected deliveries. We will use this dataset in the exercises, separately from the visitor heights.
    """)
    return


@app.cell
def _(np):
    delivery_times = np.array([22, 25, 23, 28, 26, 24, 27, 23, 29, 25, 24, 26, 30, 22, 27, 25, 28, 24, 26, 23])
    print("Delivery times (hours):", delivery_times)
    return (delivery_times,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generate bootstrap samples for these delivery times. We will reuse the same samples for the mean and variance exercises.
    """)
    return


@app.cell
def _(delivery_times, generate_samples_b):
    delivery_samples = generate_samples_b(delivery_times)
    print("Delivery bootstrap shape:", delivery_samples.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Do the delivery times provide evidence that the population mean exceeds 24 hours? State the hypotheses, calculate the p-value, and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_delivery_mean_p = None
    print("p-value:", student_delivery_mean_p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_delivery_mean_p = get_p_value(delivery_samples.mean().to_numpy() - delivery_times.mean() + 24, delivery_times.mean(), alternative="larger")
    ```
    Test \(H_0:\mu=24\) against \(H_a:\mu>24\). The estimated p-value is 0.0045, so reject the null hypothesis at 0.05. The sample provides evidence that the population mean delivery time exceeds 24 hours.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. A reusable bootstrap hypothesis-test function
    `boot_1sample_HT` receives the observed statistic, its bootstrap distribution, and the hypothesized population value. We calculate the bootstrap distribution once, then reuse it for different hypotheses.

    The function forms the reference values, calculates the p-value, and prints the test name, hypotheses, observed statistic, significance level, and decision. It returns the p-value. `measure` supplies a meaningful label, so a variance or skewness is not incorrectly called a sample mean.

    These are **approximate** tests. They assume independent, representative observations and a statistic whose bootstrap errors adequately approximate its sampling errors. Normal observations are not required. Small samples, extreme outliers, or heavy tails can nevertheless make the approximation unreliable. Increasing the number of resamples reduces simulation noise, not these limitations.
    """)
    return


@app.cell
def _(get_p_value, np):
    def boot_1sample_HT(sample_value, sample_distribution, population_value, alpha=0.05, alternative="two-sided", measure="Statistic"):
        """Report a centred-error bootstrap test and return its estimated p-value."""
        signs = {"two-sided": "≠", "smaller": "<", "larger": ">"}
        null_distribution = np.asarray(sample_distribution) - sample_value + population_value
        p_value = get_p_value(null_distribution, sample_value, alternative)
        print("--- Bootstrap one-sample hypothesis test ---")
        print(f"H0: {measure} = {population_value:g}")
        print(f"Ha: {measure} {signs[alternative]} {population_value:g}")
        print(f"Observed {measure}: {sample_value:.4g}")
        print(f"Significance level: {alpha:g}")
        if p_value == 0:
            print(f"Estimated p-value: 0 (no simulated values in the relevant tail; {len(null_distribution)} resamples)")
        else:
            print(f"Estimated p-value: {p_value:.4g}")
        if p_value <= alpha:
            print("There is sufficient evidence to reject the null hypothesis.")
        else:
            print("There is insufficient evidence to reject the null hypothesis.")
        return p_value

    return (boot_1sample_HT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does the population mean height differ from 170 cm?
    Test \(H_0:\mu=170\) against \(H_a:\mu\neq 170\), at significance level 0.05.
    """)
    return


@app.cell
def _(boot_1sample_HT, mean_distribution, sample_mean):
    height_mean_170_p = boot_1sample_HT(sample_mean, mean_distribution, 170, alternative="two-sided", measure="Mean height (cm)")
    return (height_mean_170_p,)


@app.cell
def _(graph_bootstrap_test, mean_distribution, sample_mean):
    graph_bootstrap_test(mean_distribution - sample_mean + 170, sample_mean, alternative="two-sided", measure="Mean height (cm)")
    return


@app.cell(hide_code=True)
def _(height_mean_170_p, mo):
    _text = "The sample provides evidence that the population mean height differs from 170 cm." if height_mean_170_p <= 0.05 else "The sample does not provide sufficient evidence that the population mean height differs from 170 cm."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does the population mean height differ from 168 cm?
    This is a new hypothesized value: \(H_0:\mu=168\), \(H_a:\mu\neq 168\). We reuse the bootstrap means, but adjust the reference values to this hypothesis.
    """)
    return


@app.cell
def _(boot_1sample_HT, mean_distribution, sample_mean):
    height_mean_168_p = boot_1sample_HT(sample_mean, mean_distribution, 168, alternative="two-sided", measure="Mean height (cm)")
    return (height_mean_168_p,)


@app.cell
def _(graph_bootstrap_test, mean_distribution, sample_mean):
    graph_bootstrap_test(mean_distribution - sample_mean + 168, sample_mean, alternative="two-sided", measure="Mean height (cm)")
    return


@app.cell(hide_code=True)
def _(height_mean_168_p, mo):
    _text = "The sample provides evidence that the population mean height differs from 168 cm." if height_mean_168_p <= 0.05 else "The sample does not provide sufficient evidence that the population mean height differs from 168 cm."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the population mean height above 166 cm?
    Test \(H_0:\mu=166\) against \(H_a:\mu>166\). This right-tailed question uses the upper tail. Choose the alternative from the research question before inspecting the sample; the different alternatives here illustrate how the procedure works.
    """)
    return


@app.cell
def _(boot_1sample_HT, mean_distribution, sample_mean):
    height_mean_166_p = boot_1sample_HT(sample_mean, mean_distribution, 166, alternative="larger", measure="Mean height (cm)")
    return (height_mean_166_p,)


@app.cell
def _(graph_bootstrap_test, mean_distribution, sample_mean):
    graph_bootstrap_test(mean_distribution - sample_mean + 166, sample_mean, alternative="larger", measure="Mean height (cm)")
    return


@app.cell(hide_code=True)
def _(height_mean_166_p, mo):
    _text = "The sample provides evidence that the population mean height exceeds 166 cm." if height_mean_166_p <= 0.05 else "The sample does not provide sufficient evidence that the population mean height exceeds 166 cm."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Testing the population variance
    The variance measures how much heights vary around their population mean. We use the sample variance \(s^2\) with divisor \(n-1\), consistently for the observed sample and every bootstrap resample.

    The centred-error method now compares \(s^2-\sigma_0^2\) with bootstrap errors \(s^{2*}-s^2\). This is an approximate variance test, not a procedure that shifts heights to change their variance. Shifted reference values are on the statistic scale; they are not necessarily variances calculated from a population satisfying the null.

    Variance inference is sensitive to extreme observations and needs a finite fourth population moment for the usual bootstrap approximation. The bounded height distribution used here meets this condition. A small p-value concerns the population variance, not simply whether the observed variance is above or below the reference value.
    """)
    return


@app.cell
def _(height_samples, heights, np):
    sample_variance = np.var(heights, ddof=1)
    variance_distribution = height_samples.var(ddof=1).to_numpy()
    print("Observed sample variance (cm²):", round(sample_variance, 3))
    return sample_variance, variance_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does the population variance differ from 20 cm²?
    Test \(H_0:\sigma^2=20\) against \(H_a:\sigma^2\neq 20\).
    """)
    return


@app.cell
def _(boot_1sample_HT, sample_variance, variance_distribution):
    height_variance_20_p = boot_1sample_HT(sample_variance, variance_distribution, 20, alternative="two-sided", measure="Variance (cm²)")
    return (height_variance_20_p,)


@app.cell
def _(graph_bootstrap_test, sample_variance, variance_distribution):
    graph_bootstrap_test(variance_distribution - sample_variance + 20, sample_variance, alternative="two-sided", measure="Variance (cm²)")
    return


@app.cell(hide_code=True)
def _(height_variance_20_p, mo):
    _text = "The sample provides evidence that the population variance differs from 20 cm²." if height_variance_20_p <= 0.05 else "The sample does not provide sufficient evidence that the population variance differs from 20 cm²."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the population variance below 30 cm²?
    Test \(H_0:\sigma^2=30\) against \(H_a:\sigma^2<30\).
    """)
    return


@app.cell
def _(boot_1sample_HT, sample_variance, variance_distribution):
    height_variance_30_p = boot_1sample_HT(sample_variance, variance_distribution, 30, alternative="smaller", measure="Variance (cm²)")
    return (height_variance_30_p,)


@app.cell
def _(graph_bootstrap_test, sample_variance, variance_distribution):
    graph_bootstrap_test(variance_distribution - sample_variance + 30, sample_variance, alternative="smaller", measure="Variance (cm²)")
    return


@app.cell(hide_code=True)
def _(height_variance_30_p, mo):
    _text = "The sample provides evidence that the population variance is below 30 cm²." if height_variance_30_p <= 0.05 else "The sample does not provide sufficient evidence that the population variance is below 30 cm²."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Do the delivery times provide evidence that the population variance differs from 6 hours²? State the hypotheses, calculate the p-value, and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_delivery_variance_p = None
    print("p-value:", student_delivery_variance_p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_delivery_variance_p = get_p_value(delivery_samples.var(ddof=1).to_numpy() - np.var(delivery_times, ddof=1) + 6, np.var(delivery_times, ddof=1))
    ```
    Test \(H_0:\sigma^2=6\) against \(H_a:\sigma^2\neq 6\). The estimated p-value is 0.777, so there is insufficient evidence to reject the null hypothesis. The sample does not provide sufficient evidence that the population variance differs from 6 hours².
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Testing population skewness
    Skewness describes asymmetry. A negative population skewness indicates a longer tail toward lower values; positive skewness indicates a longer tail toward higher values. Zero skewness alone does not establish symmetry or normality.

    We use `skew(..., bias=False)` for both the observed sample and every resample. The bias correction makes this definition consistent with Pandas' `skew` method. Calculating the observed value with one definition and bootstrap values with another would distort the comparison.

    Skewness is especially sensitive to outliers and heavy tails. Its usual bootstrap approximation needs a finite sixth population moment and nonzero variance. The bounded height example meets these conditions.

    We test \(H_0:\gamma_1=0\) against \(H_a:\gamma_1<0\), where \(\gamma_1\) denotes population skewness. The question concerns negative skewness, rather than normality.
    """)
    return


@app.cell
def _(height_samples, heights, skew):
    sample_skewness = skew(heights, bias=False)
    skewness_distribution = skew(height_samples.to_numpy(), axis=0, bias=False)
    print("Observed sample skewness:", round(sample_skewness, 3))
    return sample_skewness, skewness_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is population skewness negative?
    Use the lower tail of the centred-error reference distribution.
    """)
    return


@app.cell
def _(boot_1sample_HT, sample_skewness, skewness_distribution):
    height_skewness_p = boot_1sample_HT(sample_skewness, skewness_distribution, 0, alternative="smaller", measure="Skewness")
    return (height_skewness_p,)


@app.cell
def _(graph_bootstrap_test, sample_skewness, skewness_distribution):
    graph_bootstrap_test(skewness_distribution - sample_skewness + 0, sample_skewness, alternative="smaller", measure="Skewness")
    return


@app.cell(hide_code=True)
def _(height_skewness_p, mo):
    _text = "The sample provides evidence that the population skewness is negative." if height_skewness_p <= 0.05 else "The sample does not provide sufficient evidence that the population skewness is negative."
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Do the delivery times provide evidence of positive population skewness? State the hypotheses, calculate the p-value, and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_delivery_skewness_p = None
    print("p-value:", student_delivery_skewness_p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_delivery_skewness_p = get_p_value(skew(delivery_samples.to_numpy(), axis=0, bias=False) - skew(delivery_times, bias=False), skew(delivery_times, bias=False), alternative="larger")
    ```
    Test \(H_0:\gamma_1=0\) against \(H_a:\gamma_1>0\). The estimated p-value is 0.128, so there is insufficient evidence to reject the null hypothesis. The sample does not provide sufficient evidence of positive population skewness.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Bootstrap tests compare sample evidence with a hypothesized population parameter.
    - Resampling with replacement estimates sampling variation; centred bootstrap errors connect that variation to the null hypothesis.
    - The alternative determines whether we use the lower tail, upper tail, or both tails.
    - The p-value and significance level determine whether to reject the null hypothesis. Failing to reject does not establish that it is true.
    - Mean, variance, and skewness tests require consistent definitions of the observed and bootstrap statistics.
    - Bootstrap inference is approximate. Independent, representative observations and adequate estimation of sampling variation remain necessary, even without assuming normally distributed data.

    ## Check your understanding
    1. If the original sample contains 50 observations, how many are in each bootstrap resample?
    2. Why do we subtract the observed statistic before comparing bootstrap errors with a null hypothesis?
    3. Which percentiles define an equal-tailed two-sided 5% critical region?
    4. Can inclusive left- and right-tail probabilities sum to more than 1? Explain.
    5. Does a bootstrap p-value of zero establish that the underlying probability is zero?
    6. Does zero skewness establish that a population is normal?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    1. Each resample contains 50 observations, drawn with replacement.
    2. The subtraction estimates sampling error around the observed estimate. We compare it with the observed departure from the hypothesized value.
    3. The 2.5th and 97.5th percentiles.
    4. Yes. Values equal to the observed statistic are included in both tails.
    5. No. It means none of the simulated values reached the relevant tail. The simulation has finite resolution.
    6. No. Many nonnormal populations have zero skewness; even symmetry does not imply normality.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Davison, A. C., and Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press, Chapter 4.
    - Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC, Chapter 16.
    - Hall, P., and Wilson, S. R. (1991). [Two Guidelines for Bootstrap Hypothesis Testing](https://doi.org/10.2307/2532163). *Biometrics*, 47(2), 757–762.
    """)
    return


if __name__ == "__main__":
    app.run()
