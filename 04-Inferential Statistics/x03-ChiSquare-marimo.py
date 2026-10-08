# /// script
# dependencies = [
#     "marimo",
#     "seaborn",
#     "scipy",
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

    sns.set_style("whitegrid")
    return mo, np, pd, plt, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chi-square Tests

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Separate a bar chart of counts from a bar chart of relative frequencies.
    - Test goodness of fit with observed counts and expected counts.
    - Explain why multiplying proportions by 100 is not a sample size.
    - Test independence from a contingency table that excludes the margins.
    - Treat a large p-value as "do not reject," not as proof of the null hypothesis.

    ## 1. Counts and relative frequencies
    A **count** is how many observations fall in a category.
    A **relative frequency** is that count divided by the sample size.
    The two bar charts have the same shape and different vertical scales.
    A chi-square test needs the counts, because the same proportions are more surprising in a larger sample.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Counts are 20, 30, and 50.
    1. What is the relative frequency of the third category?
    2. If every count is doubled, which chart changes its bar heights: the count chart or the relative-frequency chart?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Relative frequency:** 50/100 = 0.5.
2. **Doubling:** The count chart doubles. The relative frequencies stay 0.2, 0.3, and 0.5.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A fictitious reference distribution
    The labels below are invented so the arithmetic is visible. They are not a census.
    The reference has 220,000 labels. Town X has 123. Town Y has 363 and is much more concentrated in one category.
    """)
    return


@app.cell
def _(pd):
    categories = ["asian", "black", "hispanic", "white"]
    reference_counts = pd.Series({"asian": 10_000, "black": 50_000, "hispanic": 60_000, "white": 100_000}).loc[categories]
    town_x_counts = pd.Series({"asian": 8, "black": 25, "hispanic": 30, "white": 60}).loc[categories]
    town_y_counts = pd.Series({"asian": 8, "black": 25, "hispanic": 30, "white": 300}).loc[categories]
    reference_proportions = reference_counts / reference_counts.sum()
    return reference_counts, reference_proportions, town_x_counts, town_y_counts


@app.cell
def _(plt, reference_counts, reference_proportions):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    reference_counts.plot(kind="bar", ax=_axes[0], rot=0, color="#4C78A8")
    _axes[0].set(title="Reference counts", ylabel="Count")
    reference_proportions.plot(kind="bar", ax=_axes[1], rot=0, color="#4C78A8")
    _axes[1].set(title="Reference proportions", ylabel="Proportion")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mo, pd, reference_proportions, town_x_counts, town_y_counts):
    comparison = pd.DataFrame({
        "Reference": reference_proportions,
        "Town_X": town_x_counts / town_x_counts.sum(),
        "Town_Y": town_y_counts / town_y_counts.sum(),
    })
    mo.Html(
        comparison.round(3).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (comparison,)


@app.cell
def _(comparison, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.5))
    comparison.plot(kind="bar", ax=_ax, rot=0, color=["#4C78A8", "#F58518", "#E45756"])
    _ax.set(title="Relative frequencies", ylabel="Proportion", xlabel="")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Goodness of fit
    \(H_0\): the town counts follow the reference proportions.
    \(H_a\): they do not.

    Expected count for a category is \(n\) times the reference proportion.
    The original notebook called relative frequencies times 100 "observed counts." Those values are not the town's counts, and 100 is not Town X's sample size. This lesson passes the actual counts to `scipy.stats.chisquare`.
    """)
    return


@app.cell
def _(mo, pd, reference_proportions, st, town_x_counts, town_y_counts):
    _rows = []
    for _name, _counts in [("Town X", town_x_counts), ("Town Y", town_y_counts)]:
        _expected = reference_proportions.to_numpy() * _counts.sum()
        _stat, _pvalue = st.chisquare(_counts.to_numpy(), _expected)
        _rows.append({
            "Town": _name,
            "n": int(_counts.sum()),
            "Chi_square": _stat,
            "p_value": _pvalue,
            "Decision_at_0.05": "Reject H0" if _pvalue <= 0.05 else "Do not reject H0",
        })
    gof_results = pd.DataFrame(_rows)
    mo.Html(
        gof_results.round(4).to_html(index=False, border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (gof_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Reference proportions are 0.5 and 0.5. Observed counts are 30 and 10.
    1. What is \(n\)?
    2. What are the expected counts?
    3. Which town in the table is the one whose proportions stay close to the reference?
    """)
    return


@app.cell(hide_code=True)
def _(gof_results, mo):
    _town_x = gof_results.loc[gof_results["Town"] == "Town X", "Decision_at_0.05"].iloc[0]
    _answers = f"""
1. **Sample size:** 40.
2. **Expected counts:** 20 and 20.
3. **Close proportions:** Town X. Its decision at the 0.05 level is "{_town_x}". Town Y is the concentrated sample.
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Independence
    \(H_0\): the row category and the column category are independent.
    \(H_a\): they are associated.

    The voter table draws race and party separately, with seed 10, so the simulation itself is an independence model.
    `chi2_contingency` expects the cell counts. Do not include the margin row or margin column.
    """)
    return


@app.cell
def _(np, pd):
    _rng = np.random.default_rng(10)
    voters = pd.DataFrame({
        "race": _rng.choice(["black", "hispanic", "white"], size=1_000, p=[0.20, 0.30, 0.50]),
        "party": _rng.choice(["democrat", "independent", "republican"], size=1_000, p=[0.40, 0.20, 0.40]),
    })
    voter_counts = pd.crosstab(voters["race"], voters["party"])
    return (voter_counts,)


@app.cell
def _(mo, pd, st, voter_counts):
    _stat, _pvalue, _df, _expected = st.chi2_contingency(voter_counts)
    expected_counts = pd.DataFrame(_expected, index=voter_counts.index, columns=voter_counts.columns)
    voter_result = pd.DataFrame({
        "Chi_square": [_stat],
        "df": [_df],
        "p_value": [_pvalue],
        "Decision_at_0.05": ["Reject H0" if _pvalue <= 0.05 else "Do not reject H0"],
    })
    mo.Html(
        voter_counts.to_html(border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return expected_counts, voter_result


@app.cell
def _(expected_counts, mo, voter_result):
    mo.Html(
        voter_result.round(4).to_html(index=False, border=0, col_space=130)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(expected_counts, mo):
    mo.Html(
        expected_counts.round(1).to_html(border=0, col_space=120)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(plt, voter_counts):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    voter_counts.plot(kind="bar", ax=_ax, rot=0, color=["#4C78A8", "#F58518", "#54A24B"])
    _ax.set(title="Simulated voters", ylabel="Count", xlabel="Race")
    _ax.legend(frameon=False, title="Party")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The degree counts are a small fixed table. "Bachelors" corrects the spelling in the source notebook.
    Several expected counts are modest, so the chi-square approximation is only a rough guide.
    """)
    return


@app.cell
def _(mo, pd, plt, st):
    degree_counts = pd.DataFrame(
        {"Bachelors": [10, 6], "Masters": [20, 9], "Doctorate": [30, 17]},
        index=["Male", "Female"],
    )
    _stat, _pvalue, _df, _expected = st.chi2_contingency(degree_counts)
    degree_result = pd.DataFrame({
        "Chi_square": [_stat],
        "df": [_df],
        "p_value": [_pvalue],
        "Decision_at_0.05": ["Reject H0" if _pvalue <= 0.05 else "Do not reject H0"],
    })
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    degree_counts.plot(kind="bar", ax=_ax, rot=0, color=["#4C78A8", "#F58518", "#54A24B"])
    _ax.set(title="Degree counts", ylabel="Count", xlabel="")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return degree_counts, degree_result


@app.cell
def _(degree_counts, degree_result, mo):
    mo.vstack([
        mo.Html(
            degree_counts.to_html(border=0, col_space=110)
            .replace("<table ", '<table style="width: auto;" ')
        ),
        mo.Html(
            degree_result.round(4).to_html(index=False, border=0, col_space=130)
            .replace("<table ", '<table style="width: auto;" ')
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. The expected-count table is printed under the voter result. Do those entries have to be integers?
    2. Why is a large p-value the anticipated outcome for the simulated voters?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Expected counts:** No. They are \(n\) times fitted probabilities, so they can be fractional. The observed counts are integers.
2. **Simulation:** Race and party were drawn independently. The null hypothesis is the model that generated the table.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Count charts and relative-frequency charts can share a shape and use different scales.
    - Goodness of fit compares observed counts with expected counts. The expected count is the sample size times the reference proportion.
    - Multiplying a proportion by 100 does not create the sample size of the town.
    - An independence test uses the inside of the contingency table. The margins are inputs to the expected counts, not extra cells.
    - A p-value above 0.05 means do not reject the null hypothesis.
    - Small expected counts weaken the chi-square approximation.

    ## Check your understanding
    1. Observed counts equal the expected counts. What is the chi-square statistic?
    2. Two samples have the same proportions, and one sample is much larger. Can their goodness-of-fit p-values differ?
    3. What does `chi2_contingency` return besides the statistic and the p-value?
    4. Should a "total" column be included in the test?
    5. Does "do not reject independence" mean the variables are independent?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Statistic:** 0.
2. **Sample size:** Yes. The larger sample produces larger count gaps when the proportions differ from the reference, so its p-value can be smaller.
3. **Other output:** The degrees of freedom and the expected counts.
4. **Total column:** No.
5. **Decision:** No. It means this table did not supply enough evidence against independence.
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
