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
    return mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Permutation Tests for Two Samples

    ## Learning goals
    By the end of this lesson, you should be able to:
    - State a permutation null as a claim about labels, not about a normal curve.
    - Shuffle pooled observations into the original group sizes.
    - Read a two-sided or one-sided permutation p-value for a difference of means.
    - Turn a paired swap into a sign flip of the difference.
    - Apply the same functions to a small spending example and a paired area example.

    ## 1. Independent samples
    If the group label is irrelevant, an observation from Class C could have been recorded in Class D.
    The permutation distribution deals the pooled scores back into groups of the original sizes, without replacement.
    This lesson uses 2,000 shuffles.

    Class C is drawn from a normal distribution with mean 85, and Class D from a normal distribution with mean 90.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The pooled list is [1, 2, 3], and the first group has size 1.
    1. How many distinct assignments give a different member to the first group?
    2. Does a permutation create a new number, such as 2.5?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Assignments:** 3. The first group can receive 1, 2, or 3.
2. **New numbers:** No. A permutation reuses the observed values.
""")})
    return


@app.cell
def _(np):
    def tail_p(null_stats, observed, alternative="two-sided"):
        null_stats = np.asarray(null_stats, dtype=float)
        left = float(np.mean(null_stats <= observed))
        right = float(np.mean(null_stats > observed))
        if alternative == "smaller":
            return left
        if alternative == "larger":
            return right
        return float(min(1.0, 2 * min(left, right)))

    def independent_permutation(left, right, n_replicates=2_000, seed=2026):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        pooled = np.concatenate([left, right])
        rng = np.random.default_rng(seed)
        order = np.argsort(rng.random((n_replicates, len(pooled))), axis=1)
        shuffled = pooled[order]
        null_diff = shuffled[:, : len(left)].mean(axis=1) - shuffled[:, len(left) :].mean(axis=1)
        observed = left.mean() - right.mean()
        return observed, null_diff

    def paired_sign_flips(before, after, n_replicates=2_000, seed=2026):
        differences = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_replicates, len(differences)))
        return differences.mean(), (signs * differences).mean(axis=1), differences

    return independent_permutation, paired_sign_flips, tail_p


@app.cell
def _(independent_permutation, mo, np, pd, plt, sns, tail_p):
    _rng = np.random.default_rng(123)
    class_c = _rng.normal(85, 3, size=100)
    class_d = _rng.normal(90, 3, size=95)
    observed_classes, null_classes = independent_permutation(class_c, class_d)
    class_p = tail_p(null_classes, observed_classes)
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    sns.kdeplot(class_c, ax=_axes[0], fill=True, color="#54A24B", label="Class C")
    sns.kdeplot(class_d, ax=_axes[0], fill=True, color="#F58518", label="Class D")
    _axes[0].legend(frameon=False, fontsize=8)
    _axes[0].set(title="Classes", xlabel="Score")
    _axes[1].hist(null_classes, bins=30, color="#4C78A8", edgecolor="white")
    _axes[1].axvline(observed_classes, color="black", linewidth=2)
    _axes[1].set(title="Permutation distribution", xlabel="Mean difference")
    _fig.tight_layout()
    plt.close(_fig)
    class_row = {
        "Comparison": "Class C − Class D",
        "Observed": observed_classes,
        "Alternative": "two-sided",
        "p_value": class_p,
        "Decision_at_0.05": "Reject H0" if class_p <= 0.05 else "Do not reject H0",
    }
    _fig
    return class_c, class_d, class_row


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Book spending
    The question in the source notebook is whether science students spend more, on average, than art students.
    \(H_0\): the two mean amounts are equal.
    \(H_a\): the science mean is larger.
    The alternative is one-sided, so the p-value is the right tail of science minus art.
    A large p-value means do not reject equal means. It does not establish that the spending is the same.
    """)
    return


@app.cell
def _(independent_permutation, mo, np, pd, tail_p, class_row):
    science = np.array([190, 280, 290, 250, 300, 286, 298, 243, 220, 310], dtype=float)
    art = np.array([280, 260, 250, 220, 240, 260, 270, 260, 250, 300], dtype=float)
    observed_spend, null_spend = independent_permutation(science, art, seed=8)
    spend_p = tail_p(null_spend, observed_spend, alternative="larger")
    spending_results = pd.DataFrame([
        class_row,
        {
            "Comparison": "Science − art",
            "Observed": observed_spend,
            "Alternative": "larger",
            "p_value": spend_p,
            "Decision_at_0.05": "Reject H0" if spend_p <= 0.05 else "Do not reject H0",
        },
    ])
    mo.Html(
        spending_results.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return art, science, spending_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. How many science amounts and how many art amounts are in the lists?
    2. Why is "the spending is similar" too strong a sentence for a p-value above 0.05?
    """)
    return


@app.cell(hide_code=True)
def _(art, mo, science, spending_results):
    _decision = spending_results.loc[spending_results["Comparison"] == "Science − art", "Decision_at_0.05"].iloc[0]
    _answers = f"""
1. **Counts:** {len(science)} science amounts and {len(art)} art amounts.
2. **Wording:** The decision is "{_decision}". With ten amounts in each list, a modest difference can produce a large p-value. That is a limit of the sample, not a demonstration that the means are equal.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Paired sign flips
    Swapping the two measurements in a pair multiplies their difference by \(-1\).
    The null distribution is the mean of the differences after random sign flips.
    The first simulated after-sample is close to the before-sample. The second is higher.
    The area measurements are the paired example from the source notebook. The figure shows those area differences. The source notebook drew the previous grade differences in that spot.
    """)
    return


@app.cell
def _(mo, np, paired_sign_flips, pd, plt, tail_p):
    _rng = np.random.default_rng(12)
    grade_before = _rng.normal(85.5, 3, size=80)
    grade_after_near = _rng.normal(86, 4, size=80)
    grade_after_high = _rng.normal(90, 3, size=80)
    area_a = np.array([2.92, 1.88, 5.35, 3.81, 4.69, 4.86, 5.81, 5.55])
    area_b = np.array([1.84, 0.95, 4.26, 3.18, 3.44, 3.69, 4.95, 4.47])
    _rows = []
    _nulls = {}
    for _name, _before, _after, _alternative in [
        ("Grades, near", grade_before, grade_after_near, "two-sided"),
        ("Grades, high", grade_before, grade_after_high, "two-sided"),
        ("Grades, high", grade_before, grade_after_high, "larger"),
        ("Area B − area A", area_a, area_b, "two-sided"),
    ]:
        _observed, _null, _differences = paired_sign_flips(_before, _after, seed=30 + len(_rows))
        _pvalue = tail_p(_null, _observed, _alternative)
        _nulls[_name + _alternative] = (_null, _observed, _differences)
        _rows.append({
            "Comparison": _name,
            "Alternative": _alternative,
            "Observed_mean_difference": _observed,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    paired_results = pd.DataFrame(_rows)
    _null, _observed, _differences = _nulls["Area B − area Atwo-sided"]
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    _axes[0].hist(_differences, bins=8, color="#F58518", edgecolor="white")
    _axes[0].axvline(0, color="#E45756")
    _axes[0].set(title="Area differences", xlabel="B − A", ylabel="Count")
    _axes[1].hist(_null, bins=30, color="#4C78A8", edgecolor="white")
    _axes[1].axvline(_observed, color="black", linewidth=2)
    _axes[1].set(title="Sign-flip null", xlabel="Mean difference")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return (paired_results,)


@app.cell
def _(mo, paired_results):
    mo.Html(
        paired_results.round(4).to_html(index=False, border=0, col_space=140)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The pair is (4, 1), written as after minus before? Use before = 4 and after = 1.
    1. What is the difference after − before?
    2. What is the difference after the pair is swapped?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Difference:** \(1 - 4 = -3\).
2. **Swapped:** \(4 - 1 = 3\), which is \(-1\) times the original difference.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A permutation test reuses the observed numbers and changes their labels.
    - Independent groups are rebuilt by shuffling the pooled sample into the original sizes.
    - The p-value's tail follows the alternative. "Science spends more" is a right tail of science minus art.
    - A p-value above 0.05 means do not reject the null. With ten observations per group, that is a weak basis for saying two means are equal.
    - A paired permutation flips signs. Swapping the two measurements in a pair is that sign flip.
    - The same observed difference can be tested two-sided or one-sided. Those are different alternatives.
    - The classes, grades, spending lists, and areas are teaching numbers.

    ## Check your understanding
    1. Are permutation samples drawn with replacement?
    2. What is held fixed when the class labels are shuffled?
    3. A one-sided alternative says the first mean is larger. Which tail is the p-value?
    4. Why does the paired test flip signs instead of pooling every before-score and after-score?
    5. Two permutation p-values, 0.03 and 0.20, come from different statistics. Must the decisions agree?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Replacement:** No.
2. **Fixed values:** The observed scores. Only the group assignment changes.
3. **Tail:** The right tail of first mean minus second mean.
4. **Pairs:** Pooling would break the match inside each pair. The sign flip keeps the two measurements together and exchanges their order.
5. **Decisions:** No. Each p-value answers its own statistic and alternative.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    - Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC, Chapter 16.
    """)
    return


if __name__ == "__main__":
    app.run()
