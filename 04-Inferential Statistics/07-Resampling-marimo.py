# /// script
# dependencies = [
#     "marimo",
#     "seaborn",
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

    sns.set_style("whitegrid")
    N_REPLICATES = 2_000
    return N_REPLICATES, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Resampling Tests for Two Samples

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Separate a bootstrap null for two means from a permutation null.
    - Shift two independent samples to a common mean before resampling them with replacement.
    - Build a paired bootstrap null by centering the differences at 0.
    - Reassign pooled observations, without replacement, for an independent permutation test.
    - Flip the signs of paired differences for a paired permutation test.
    - Use an upper or a two-sided tail that matches the statistic.

    ## 1. Two nulls
    The sentence "any score could have landed in either group" is the **permutation** idea. The groups are labels. If the labels are arbitrary, shuffling them does not change the measurements.

    A **bootstrap** null for a difference of means is different. Each group is resampled with replacement after both groups have been shifted onto the same mean. The shift removes the observed mean gap. The resampling estimates the remaining noise.

    This lesson uses 2,000 replicates. The source notebook used 10,000. The procedures are the same, and the Monte Carlo error in a p-value is a little larger.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Group A is [1, 2] and Group B is [10, 12].
    1. What is the combined mean?
    2. After shifting each group to that combined mean, what is the difference of the shifted means?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Combined mean:** (1 + 2 + 10 + 12) / 4 = 6.25.
2. **Shifted difference:** 0. Each shifted group has mean 6.25, so the difference of means is 0.
""")})
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(123)
    class_c = _rng.normal(85, 3, size=100)
    class_d = _rng.normal(90, 3, size=95)
    return class_c, class_d


@app.cell
def _(N_REPLICATES, np):
    def tail_p(null_stats, observed, alternative="two-sided"):
        null_stats = np.asarray(null_stats, dtype=float)
        left = np.mean(null_stats <= observed)
        right = np.mean(null_stats > observed)
        if alternative == "smaller":
            return float(left)
        if alternative == "larger":
            return float(right)
        return float(min(1.0, 2 * min(left, right)))

    def bootstrap_shifted_means(left, right, n_replicates=N_REPLICATES, seed=2026):
        """Bootstrap each sample after shifting both onto the pooled mean."""
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        pooled_mean = np.concatenate([left, right]).mean()
        left = left - left.mean() + pooled_mean
        right = right - right.mean() + pooled_mean
        rng = np.random.default_rng(seed)
        left_draws = rng.choice(left, size=(n_replicates, len(left)), replace=True)
        right_draws = rng.choice(right, size=(n_replicates, len(right)), replace=True)
        return left_draws, right_draws

    def permutation_means(left, right, n_replicates=N_REPLICATES, seed=2026):
        """Shuffle pooled observations into the original group sizes."""
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        pooled = np.concatenate([left, right])
        rng = np.random.default_rng(seed)
        order = np.argsort(rng.random((n_replicates, len(pooled))), axis=1)
        shuffled = pooled[order]
        return shuffled[:, : len(left)], shuffled[:, len(left) :]

    return bootstrap_shifted_means, permutation_means, tail_p


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bootstrap for independent samples
    Class C is drawn from a normal distribution with mean 85. Class D uses mean 90. The sample sizes differ.
    The observed statistic is the difference of means, medians, or Welch \(t\) statistics, computed on the original samples.
    The null replicates use the shifted samples.
    """)
    return


@app.cell
def _(bootstrap_shifted_means, class_c, class_d, mo, np, pd, tail_p):
    left_draws, right_draws = bootstrap_shifted_means(class_c, class_d)
    observed_mean_diff = class_c.mean() - class_d.mean()
    observed_median_diff = np.median(class_c) - np.median(class_d)

    def _welch(left, right):
        return (left.mean() - right.mean()) / np.sqrt(left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right))

    observed_t = _welch(class_c, class_d)
    null_mean = left_draws.mean(axis=1) - right_draws.mean(axis=1)
    null_median = np.median(left_draws, axis=1) - np.median(right_draws, axis=1)
    null_t = np.array([_welch(left, right) for left, right in zip(left_draws, right_draws)])
    bootstrap_results = pd.DataFrame([
        {"Statistic": "Difference of means", "Observed": observed_mean_diff, "p_value": tail_p(null_mean, observed_mean_diff)},
        {"Statistic": "Difference of medians", "Observed": observed_median_diff, "p_value": tail_p(null_median, observed_median_diff)},
        {"Statistic": "Welch t", "Observed": observed_t, "p_value": tail_p(null_t, observed_t)},
    ])
    bootstrap_results["Decision_at_0.05"] = np.where(
        bootstrap_results["p_value"] <= 0.05, "Reject H0", "Do not reject H0"
    )
    mo.Html(
        bootstrap_results.round(4).to_html(index=False, border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return bootstrap_results, null_mean, observed_mean_diff


@app.cell
def _(class_c, class_d, null_mean, observed_mean_diff, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    sns.kdeplot(class_c, ax=_axes[0], fill=True, color="#54A24B", label="Class C")
    sns.kdeplot(class_d, ax=_axes[0], fill=True, color="#F58518", label="Class D")
    _axes[0].legend(frameon=False, fontsize=8)
    _axes[0].set(title="Original samples", xlabel="Score")
    _axes[1].hist(null_mean, bins=30, color="#4C78A8", edgecolor="white")
    _axes[1].axvline(observed_mean_diff, color="black", linewidth=2, label="Observed difference")
    _axes[1].legend(frameon=False, fontsize=8)
    _axes[1].set(title="Bootstrap null for the mean difference", xlabel="Mean difference")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Why are the mean, median, and \(t\) rows allowed to disagree?
    2. The Welch statistic uses divisor \(n-1\). Would a divisor of \(n\) be the same statistic?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Different statistics:** Each row tests its own statistic. A mean gap, a median gap, and a \(t\) ratio are three questions that happen to use the same samples.
2. **Divisor:** No. Changing the divisor changes the statistic, so both the observed value and the null replicates have to be recomputed with the same formula.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Bootstrap for paired samples
    The link between a before-score and an after-score stays inside the difference.
    Under a mean difference of 0, the bootstrap draws come from the centered differences, `diff - diff.mean()`.
    Comparing the observed mean with a bootstrap distribution that was not centered tests nothing about 0: that cloud sits on the observed mean by construction.
    """)
    return


@app.cell
def _(N_REPLICATES, mo, np, pd, tail_p):
    _rng = np.random.default_rng(12)
    grade_before = _rng.normal(80, 4, size=90)
    grade_after = _rng.normal(86, 3, size=90)
    differences = grade_after - grade_before
    centered = differences - differences.mean()
    null_paired = _rng.choice(centered, size=(N_REPLICATES, len(centered)), replace=True).mean(axis=1)
    paired_bootstrap = pd.DataFrame({
        "Observed_mean_difference": [differences.mean()],
        "p_value": [tail_p(null_paired, differences.mean())],
        "Decision_at_0.05": ["Reject H0" if tail_p(null_paired, differences.mean()) <= 0.05 else "Do not reject H0"],
    })
    mo.Html(
        paired_bootstrap.round(4).to_html(index=False, border=0, col_space=160)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return differences, null_paired, paired_bootstrap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Permutation tests
    For independent samples, pool the scores and deal them back into the original group sizes. That is sampling without replacement from the pooled list.
    For paired samples, a swap of the two measurements in a pair changes the sign of that pair's difference. The null replicates are random sign flips. The differences are not centered first. The signs are the part the null says is arbitrary.
    """)
    return


@app.cell
def _(class_c, class_d, differences, mo, np, pd, permutation_means, tail_p):
    left_perm, right_perm = permutation_means(class_c, class_d, seed=99)
    null_perm = left_perm.mean(axis=1) - right_perm.mean(axis=1)
    observed = class_c.mean() - class_d.mean()
    signs = np.random.default_rng(99).choice([-1.0, 1.0], size=(len(null_perm), len(differences)))
    null_signs = (signs * differences).mean(axis=1)
    permutation_results = pd.DataFrame([
        {
            "Design": "Independent classes",
            "Observed": observed,
            "p_value": tail_p(null_perm, observed),
        },
        {
            "Design": "Paired grades",
            "Observed": differences.mean(),
            "p_value": tail_p(null_signs, differences.mean()),
        },
    ])
    permutation_results["Decision_at_0.05"] = np.where(
        permutation_results["p_value"] <= 0.05, "Reject H0", "Do not reject H0"
    )
    mo.Html(
        permutation_results.round(4).to_html(index=False, border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return null_perm, permutation_results


@app.cell
def _(differences, np, null_perm, observed_mean_diff, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    _axes[0].hist(null_perm, bins=30, color="#54A24B", edgecolor="white")
    _axes[0].axvline(observed_mean_diff, color="black", linewidth=2)
    _axes[0].set(title="Permutation null, classes", xlabel="Mean difference")
    _signs = np.random.default_rng(3).choice([-1.0, 1.0], size=(2_000, len(differences)))
    _axes[1].hist((_signs * differences).mean(axis=1), bins=30, color="#F58518", edgecolor="white")
    _axes[1].axvline(differences.mean(), color="black", linewidth=2)
    _axes[1].set(title="Sign-flip null, paired grades", xlabel="Mean difference")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    One paired difference is \(8 - 5 = 3\).
    1. What is the difference if those two measurements are swapped?
    2. Is that swap the same as multiplying 3 by \(-1\)?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Swapped difference:** \(5 - 8 = -3\).
2. **Sign:** Yes. Swapping the two members of a pair multiplies the difference by \(-1\).
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A permutation test reassigns labels. An independent two-sample permutation pools the observations and reshuffles them.
    - A bootstrap test for a mean difference resamples with replacement after the groups have been shifted onto one common mean.
    - A paired bootstrap centers the differences at 0 and then resamples those centered differences.
    - A paired permutation flips signs. It does not center the differences before the flip.
    - The mean, the median, and the Welch \(t\) are different statistics. Each p-value belongs to the statistic that was resampled.
    - A two-sided p-value doubles the smaller tail and does not exceed 1.
    - These classes and grades are simulated. A rejection here is a check of the procedure, not a finding about a course.

    ## Check your understanding
    1. Which procedure draws with replacement?
    2. What is the difference of means after both groups are shifted to the pooled mean?
    3. What does a sign flip represent for a pair?
    4. Why must a paired analysis keep the before-score attached to its after-score until the difference is formed?
    5. The observed mean sits in the center of an unshifted bootstrap distribution. Is that evidence for a mean of 0?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **With replacement:** The bootstrap. The permutation reshuffles the existing observations.
2. **Shifted means:** 0.
3. **Sign flip:** Swapping the two measurements inside the pair.
4. **Attachment:** The scientific unit is the pair. Detaching the scores would mix one person's before-score with another person's after-score.
5. **Unshifted cloud:** No. An unshifted bootstrap distribution of the mean is centered on the observed mean, so the observed mean is typical of that cloud whether or not the mean is 0.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC, Chapter 16.
    - Davison, A. C., and Hinkley, D. V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press, Chapter 4.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    """)
    return


if __name__ == "__main__":
    app.run()
