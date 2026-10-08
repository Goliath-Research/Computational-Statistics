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
    N_REPLICATES = 1_000
    return N_REPLICATES, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Resampling Tests for Several Samples

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Measure a gap among several means with one non-negative statistic.
    - Read that test with an upper-tail p-value.
    - Build an independent bootstrap null by shifting every group to the grand mean and resampling it.
    - Keep related measurements together by resampling subjects as blocks.
    - Permute independent labels by shuffling the pool, and permute related measurements within each subject.

    ## 1. A statistic for several means
    With more than two groups, a single difference of means is no longer enough.
    The statistic below is the square root of a weighted average of squared gaps between the group means and the grand mean.
    It is 0 when all the means are equal, and it grows when they spread out.
    Because the statistic cannot be negative, the matching p-value is the **upper tail**: the share of null replicates at least as large as the observed value.
    A two-sided p-value treats a small statistic as evidence against the null as well. That does not match this distance.

    The source notebook reused the first group's bootstrap draws for the other groups, and it used a two-sided p-value. Each group is resampled from itself here, and the p-value is the upper tail.
    This lesson uses 1,000 replicates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Three group means are equal.
    1. What is the dispersion statistic?
    2. Why is a very small statistic not evidence against equal means?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Dispersion:** 0. Every squared gap is 0.
2. **Small values:** The null says the means are equal, which is exactly when the statistic is small. Evidence against that null is a statistic that is large compared with the null replicates.
""")})
    return


@app.cell
def _(N_REPLICATES, np):
    def dispersion(groups):
        groups = [np.asarray(group, dtype=float) for group in groups]
        sizes = np.array([len(group) for group in groups])
        means = np.array([group.mean() for group in groups])
        grand = np.average(means, weights=sizes)
        return float(np.sqrt(np.sum(sizes / sizes.sum() * (means - grand) ** 2)))

    def mean_gap(groups):
        means = [np.asarray(group, dtype=float).mean() for group in groups]
        total = 0.0
        pairs = 0
        for i, left in enumerate(means):
            for right in means[i + 1 :]:
                total += abs(left - right)
                pairs += 1
        return total / pairs

    def upper_p(null_stats, observed):
        return float(np.mean(np.asarray(null_stats) >= observed))

    def bootstrap_independent(groups, n_replicates=N_REPLICATES, seed=2026):
        groups = [np.asarray(group, dtype=float) for group in groups]
        grand = np.concatenate(groups).mean()
        shifted = [group - group.mean() + grand for group in groups]
        rng = np.random.default_rng(seed)
        null_stats = np.empty(n_replicates)
        for i in range(n_replicates):
            draws = [rng.choice(group, size=len(group), replace=True) for group in shifted]
            null_stats[i] = dispersion(draws)
        return null_stats

    def permute_independent(groups, n_replicates=N_REPLICATES, seed=2026):
        groups = [np.asarray(group, dtype=float) for group in groups]
        pooled = np.concatenate(groups)
        sizes = [len(group) for group in groups]
        cuts = np.cumsum(sizes)[:-1]
        rng = np.random.default_rng(seed)
        null_stats = np.empty(n_replicates)
        for i in range(n_replicates):
            shuffled = rng.permutation(pooled)
            null_stats[i] = dispersion(np.split(shuffled, cuts))
        return null_stats

    return bootstrap_independent, dispersion, mean_gap, permute_independent, upper_p


@app.cell
def _(np):
    _rng = np.random.default_rng(50)
    groups = [
        _rng.normal(80, 6, size=50),
        _rng.normal(82, 4, size=48),
        _rng.normal(83, 5, size=52),
        _rng.normal(60, 5, size=44),
    ]
    return (groups,)


@app.cell
def _(groups, mo, pd, plt, sns):
    summary = pd.DataFrame({
        "Group": [f"Group {i}" for i in range(1, 5)],
        "n": [len(group) for group in groups],
        "Mean": [group.mean() for group in groups],
        "SD": [group.std(ddof=1) for group in groups],
    })
    _fig, _ax = plt.subplots(figsize=(6.5, 3.3))
    for _group, _name, _color in zip(groups, summary["Group"], ["#4C78A8", "#F58518", "#54A24B", "#E45756"]):
        sns.kdeplot(_group, ax=_ax, fill=True, alpha=0.25, color=_color, label=_name)
    _ax.legend(frameon=False, fontsize=8)
    _ax.set(title="Four independent groups", xlabel="Grade")
    _fig.tight_layout()
    plt.close(_fig)
    mo.Html(
        summary.round(2).to_html(index=False, border=0, col_space=90)
        .replace("<table ", '<table style="width: auto;" ')
    )
    _fig
    return (summary,)


@app.cell
def _(bootstrap_independent, dispersion, groups, mo, pd, permute_independent, upper_p):
    _rows = []
    for _label, _selected in [("Groups 1–3", groups[:3]), ("Groups 1–4", groups)]:
        _observed = dispersion(_selected)
        _boot = upper_p(bootstrap_independent(_selected, seed=1), _observed)
        _perm = upper_p(permute_independent(_selected, seed=2), _observed)
        _rows.append({
            "Groups": _label,
            "Dispersion": _observed,
            "Bootstrap_p": _boot,
            "Permutation_p": _perm,
            "Bootstrap_decision": "Reject H0" if _boot <= 0.05 else "Do not reject H0",
            "Permutation_decision": "Reject H0" if _perm <= 0.05 else "Do not reject H0",
        })
    independent_results = pd.DataFrame(_rows)
    mo.Html(
        independent_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (independent_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. When Group 4 is added, does the observed dispersion get smaller or larger?
    2. The source code built the second and third bootstrap samples from the first shifted group. What null does that implement?
    """)
    return


@app.cell(hide_code=True)
def _(independent_results, mo):
    _three = independent_results.loc[independent_results["Groups"] == "Groups 1–3", "Dispersion"].iloc[0]
    _four = independent_results.loc[independent_results["Groups"] == "Groups 1–4", "Dispersion"].iloc[0]
    _answers = f"""
1. **Dispersion:** It changes from {_three:.3f} to {_four:.3f}. Group 4 pulls the means apart, so the statistic increases.
2. **Copied group:** All three null samples would be draws from Group 1's shifted values. The null would ignore the other groups' spreads and sample sizes.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Related samples
    Related measurements on the same subject have to stay together.
    The statistic is the average absolute difference among the condition means.
    It is also non-negative, so the p-value is again the upper tail.

    The bootstrap null first shifts every condition onto the grand mean. It then draws subjects with replacement and keeps all of that subject's conditions. That is block resampling.
    The permutation null shuffles the condition labels inside each subject.

    The source notebook later replaced the shifted first condition with the shifted fourth condition, so two of the four bootstrap series were the same series. Each condition keeps its own shifted values here.
    """)
    return


@app.cell
def _(N_REPLICATES, mean_gap, np, upper_p):
    def bootstrap_related(columns, n_replicates=N_REPLICATES, seed=2026):
        columns = [np.asarray(column, dtype=float) for column in columns]
        grand = np.concatenate(columns).mean()
        shifted = np.column_stack([column - column.mean() + grand for column in columns])
        rng = np.random.default_rng(seed)
        index = rng.integers(0, len(shifted), size=(n_replicates, len(shifted)))
        resampled = shifted[index]
        return np.array([mean_gap(resampled[i].T) for i in range(n_replicates)])

    def permute_related(columns, n_replicates=N_REPLICATES, seed=2026):
        data = np.column_stack([np.asarray(column, dtype=float) for column in columns])
        rng = np.random.default_rng(seed)
        null_stats = np.empty(n_replicates)
        for i in range(n_replicates):
            shuffled = rng.permuted(data, axis=1)
            null_stats[i] = mean_gap(shuffled.T)
        return null_stats

    _rng = np.random.default_rng(50)
    related = [
        _rng.normal(60, 5, size=50),
        _rng.normal(61, 4, size=50),
        _rng.normal(61, 5, size=50),
        _rng.normal(110, 5, size=50),
    ]
    return bootstrap_related, permute_related, related


@app.cell
def _(bootstrap_related, mean_gap, mo, pd, permute_related, related, upper_p):
    _rows = []
    for _label, _columns in [("Tests 1–3", related[:3]), ("Tests 1–4", related)]:
        _observed = mean_gap(_columns)
        _boot = upper_p(bootstrap_related(_columns, seed=3), _observed)
        _perm = upper_p(permute_related(_columns, seed=4), _observed)
        _rows.append({
            "Conditions": _label,
            "Mean_absolute_gap": _observed,
            "Bootstrap_p": _boot,
            "Permutation_p": _perm,
            "Bootstrap_decision": "Reject H0" if _boot <= 0.05 else "Do not reject H0",
            "Permutation_decision": "Reject H0" if _perm <= 0.05 else "Do not reject H0",
        })
    related_results = pd.DataFrame(_rows)
    mo.Html(
        related_results.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (related_results,)


@app.cell
def _(plt, related, sns):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.3))
    for _column, _name, _color in zip(related, ["Test 1", "Test 2", "Test 3", "Test 4"], ["#4C78A8", "#F58518", "#54A24B", "#E45756"]):
        sns.kdeplot(_column, ax=_ax, fill=True, alpha=0.25, color=_color, label=_name)
    _ax.legend(frameon=False, fontsize=8)
    _ax.set(title="Related conditions", xlabel="Score")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A subject has scores 5, 5, and 9 on three conditions.
    1. If the condition labels are permuted, can the multiset of scores for that subject change?
    2. Why would resampling the three conditions separately break the related-sample design?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Multiset:** No. A within-subject permutation reorders 5, 5, and 9. It does not replace them with new scores.
2. **Separate resampling:** The three scores would no longer belong to the same subject. The block draw keeps them together.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A multi-group mean test needs one statistic that is large when the means spread out.
    - For a non-negative distance, the p-value is the upper tail of the null replicates.
    - The independent bootstrap shifts every group to the grand mean and resamples that group with replacement.
    - The independent permutation shuffles the pooled observations into the original group sizes.
    - Related measurements stay in subject blocks. The bootstrap resamples subjects. The permutation reorders conditions within each subject.
    - Copying one group's draws into the other groups, or overwriting one shifted condition with another, changes the null that the p-value refers to.
    - Nearby simulated groups can fail to reject. Adding a distant group makes the distance large.
    - The scores are simulated.

    ## Check your understanding
    1. What is the dispersion when every group mean equals the grand mean?
    2. Which tail is used for that dispersion?
    3. What is held fixed in an independent permutation: the scores, or the sample sizes?
    4. What is resampled in the related bootstrap: individual scores, or subjects?
    5. Why is a two-sided p-value a poor match to a statistic that is always at least 0?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Dispersion:** 0.
2. **Tail:** The upper tail.
3. **Permutation:** Both. The observed scores are reused, and they are dealt into the original sample sizes.
4. **Blocks:** Subjects. All conditions for a chosen subject are kept together.
5. **Two-sided:** The lower tail is the direction of equal means, which is the null rather than the alternative.
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
