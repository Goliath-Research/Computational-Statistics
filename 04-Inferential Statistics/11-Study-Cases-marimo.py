# /// script
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "seaborn", "statsmodels"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from goliath_research import resampling as rs
    lesson_dir = Path(__file__).resolve().parent
    sns.set_style("whitegrid")
    return lesson_dir, mo, np, pd, plt, rs, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Case Studies with Resampling Tests
    ## Learning goals
    By the end of this lesson, you should be able to:
    - Apply resampling methods to investigate statistical questions using real data.
    - Interpret hypothesis tests and confidence intervals in the context of a problem.
    - Draw conclusions supported by the data and the study design.

    In this lesson, we apply the bootstrap and permutation procedures from the previous lessons to restaurant bills, glass composition, and student grades. The `resampling` module in `goliath_research` provides the procedures we will use.

    Use your `goliath_research` package with the corrected `resampling.py` module inside it. Keep `tips.csv`, `glass.csv`, and `student-mat.csv` beside this notebook. Each test generates **4,000 resamples**. Fixed seeds reproduce the examples when they are rerun.

    ## 1. Applying resampling tests
    Begin with a question about the populations represented by the data. Identify the measurements being compared and whether they come from separate groups or from the same subjects. State the hypotheses before calculating the result.

    A resampling test compares the observed difference among groups with results generated under its null hypothesis. A small p-value provides evidence against that hypothesis. We then explain what the result means for the original question.

    The bootstrap examples ask whether population means are equal. Permutation examples require labels to be interchangeable under their null. Each case explains how that assumption applies to its data.

    ### Preparing data for a comparison
    `get_dict` collects observations into named groups. For restaurant bills by day, it creates one array of bills for each observed day. The day names identify the arrays when we calculate and report the comparison. For repeated measurements, we instead supply the matched grade columns without changing their row order.
    """)
    return


@app.cell
def _(np):
    def get_dict(dfr, values, labels):
        """Collect a measured variable into arrays for separate groups.

        Parameters
        ----------
        dfr : pandas DataFrame
            Table containing the measurements and group labels.
        values : str
            Name of the numeric measurement column, such as "total_bill".
        labels : str
            Name of the grouping column, such as "day".

        Returns
        -------
        dict
            Each key is an observed group label. Its value is a NumPy array
            of that group's measurements. This function prepares independent
            groups; repeated measurements must retain their subject matching.
        """
        return {key: part[values].to_numpy(dtype=float) for key, part in dfr.groupby(labels, observed=True)}
    return (get_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Selecting the procedure
    `make_model` creates the test object used in the worked examples. Choose `Bootstrap` or `Permutation` for the method, and `Independent` or `Related` for the study design. The returned object contains the observed statistic and its resampled null distribution. We reuse that object to calculate its p-value, draw the graphs, and obtain follow-up results.
    """)
    return


@app.cell
def _(rs):
    def make_model(data, kind_test="Bootstrap", kind_sample="Independent", num_samples=4_000, seed=2026):
        """Create a bootstrap or permutation test for the specified design.

        Parameters
        ----------
        data : dict
            Named arrays of measurements. For related samples, position i
            must refer to the same subject in every array.
        kind_test : str
            "Bootstrap" or "Permutation".
        kind_sample : str
            "Independent" or "Related".
        num_samples : int
            Number of resamples; defaults to 4,000.
        seed : int
            Seed for reproducing the generated samples.

        Returns
        -------
        resampling test object
            One of the four goliath_research.resampling test classes.
            Use its methods for the p-value, intervals and pairwise results.
        """
        models = {("Bootstrap", "Independent"): rs.BootstrapIndependentHT,
                  ("Permutation", "Independent"): rs.PermutationIndependentHT,
                  ("Bootstrap", "Related"): rs.BootstrapRelatedHT,
                  ("Permutation", "Related"): rs.PermutationRelatedHT}
        return models[kind_test, kind_sample](*data.values(), num_samples=num_samples, seed=seed)
    return (make_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Mean bars and a complete test report
    `plot_bars` displays each group's observed mean and prints the test report. The bars show the measured quantity, such as dollars or grades. They describe the samples; confidence intervals shown later describe uncertainty about the population means.

    With two groups, the statistic is the first sample mean minus the second. A two-sided question considers differences in either direction. With several groups, the statistic summarizes how far their means spread apart. Larger distances indicate more separation, so those tests compare the observed distance with the upper tail of the null distribution.

    The report includes the hypotheses, sample summaries, statistic, p-value, significance level, and decision. At level 0.05, a p-value at or below 0.05 leads to rejection. A larger value means insufficient evidence to reject, not proof of equality. Interpret the decision in terms of the case's population question.

    The p-value is estimated from 4,000 generated statistics. The calculation includes a finite-simulation adjustment so it does not report zero merely because no generated statistic reached the observed value.
    """)
    return


@app.cell
def _(make_model, np, plt, rs):
    def plot_bars(data, kind_test="Bootstrap", kind_sample="Independent", title="", ylabel="Observed mean", model=None, seed=2026, alpha=0.05):
        """Display observed means and print a complete resampling-test report.

        Parameters
        ----------
        data : dict
            Group or condition labels mapped to observation arrays.
        kind_test, kind_sample : str
            Method and design used when model is not supplied.
        title, ylabel : str
            Plot title and measurement label, including units where relevant.
        model : resampling test object or None
            Previously fitted test for these same data. Reusing it keeps the
            report and plot tied to the same resampling calculation.
        seed : int
            Seed used only if this function creates the test object.
        alpha : float
            Significance level for the printed decision; defaults to 0.05.

        Returns
        -------
        matplotlib Figure
            Bars showing sample means. They are not counts or confidence
            intervals. The printed report supplies the test result.
        """
        if model is None:
            model = make_model(data, kind_test, kind_sample, seed=seed)
        is_bootstrap = isinstance(model, (rs.BootstrapIndependentHT, rs.BootstrapRelatedHT))
        is_related = isinstance(model, (rs.BootstrapRelatedHT, rs.PermutationRelatedHT))
        print("--- " + ("Bootstrap" if is_bootstrap else "Permutation") + " resampling test ---")
        print("Design:", "Related measurements" if is_related else "Independent groups")
        if is_bootstrap:
            print("H0: all population means are equal")
            print("Ha: at least one population mean differs")
        elif is_related:
            print("H0: condition labels are interchangeable within subjects")
            print("Ha: the mean-comparison statistic departs from that null")
        else:
            print("H0: group labels are interchangeable under identical population distributions")
            print("Ha: the mean-comparison statistic departs from that null")
        # Keep the same sample order as the fitted test.
        for label, a in data.items():
            print(f"{label}: n = {len(a)}, mean = {np.mean(a):.3f}, sample SD = {np.std(a, ddof=1):.3f}")
        statistic, p_value = model.get_observed_stat(), model.get_p_value()
        print(f"Observed statistic = {statistic:.4g}; p-value = {p_value:.4g}")
        print("Tail:", "two-sided" if len(data) == 2 else "upper")
        print(f"Significance level = {alpha:g}")
        print("There is sufficient evidence to reject the null hypothesis." if p_value <= alpha else "There is insufficient evidence to reject the null hypothesis.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([str(k) for k in data], [np.mean(a) for a in data.values()], color="steelblue")
        ax.set(title=title, xlabel="", ylabel=ylabel)
        ax.set_title(title, pad=36)
        ax.text(0.5, 1.02, f"Resampling p-value = {p_value:.4g}", ha="center", transform=ax.transAxes)
        fig.tight_layout()
        plt.close(fig)
        return fig
    return (plot_bars,)


@app.cell
def _(np, plt):
    def plot_confidence_intervals(data, labels, means=None, title="95% bootstrap confidence intervals", xlabel="Population mean"):
        """Plot confidence limits for group or condition population means.

        Parameters
        ----------
        data : array-like, shape (number of groups, 2)
            Lower and upper confidence limits, one row per group.
        labels : sequence
            Group labels in the same order as the confidence-limit rows.
        means : sequence or None
            Observed sample means to mark with points. If omitted, points
            mark interval midpoints and are labeled accordingly.
        title, xlabel : str
            Plot title and population quantity being estimated, with units.

        Returns
        -------
        matplotlib Figure
            Horizontal confidence intervals. The worked examples use 95%
            percentile bootstrap intervals. Their overlap is not a pairwise
            test decision.
        """
        # Confidence-limit rows and labels must refer to the same groups in the same order.
        intervals = np.asarray(data)
        points = intervals.mean(axis=1) if means is None else np.asarray(means)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hlines(labels, intervals[:, 0], intervals[:, 1], color="navy")
        ax.scatter(points, labels, color="navy", label="Observed sample mean" if means is not None else "Interval midpoint")
        ax.set(title=title, xlabel=xlabel, ylabel="Group or condition")
        ax.legend()
        fig.tight_layout()
        plt.close(fig)
        return fig
    return (plot_confidence_intervals,)


@app.cell
def _(make_model, plot_bars, plt):
    def plot_bonferroni(data, kind_test="Bootstrap", kind_sample="Independent", title="", ylabel="Observed mean", model=None, seed=2026, alpha=0.01):
        """Summarize Bonferroni pairwise decisions with letters above mean bars.

        Parameters
        ----------
        data : dict
            Named observation arrays, ordered consistently with the model.
        kind_test, kind_sample : str
            Method and design used when model is not supplied.
        title, ylabel : str
            Plot title and measured quantity, including units.
        model : resampling test object or None
            Previously fitted test for these same data.
        seed : int
            Seed used only if this function creates the test object.
        alpha : float
            Family significance level for the adjusted pairwise comparisons;
            defaults to 0.01.

        Returns
        -------
        matplotlib Figure
            Sample mean bars annotated with non-rejection subset letters.
            A shared letter means the pair was not rejected after correction;
            it does not establish equal population means.
        """
        if model is None:
            model = make_model(data, kind_test, kind_sample, seed=seed)
        subsets = model.get_homogeneous_subsets(alpha)
        letters = ["" for _ in data]
        # A condition may share different non-rejected comparisons, so letters can overlap.
        for number, subset in enumerate(subsets):
            for index in subset:
                letters[index] += chr(ord("A") + number)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar([str(k) for k in data], [a.mean() for a in data.values()], color="steelblue")
        ax.bar_label(bars, labels=letters, padding=4)
        ax.margins(y=0.15)
        ax.set(title=title, xlabel="", ylabel=ylabel)
        ax.set_title(title, pad=36)
        ax.text(0.5, 1.02, f"Pairwise Bonferroni family level = {alpha:g}; shared letters indicate non-rejection", ha="center", fontsize=9, transform=ax.transAxes)
        fig.tight_layout()
        plt.close(fig)
        return fig
    return (plot_bonferroni,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Restaurant bills
    The tips dataset records 244 restaurant observations, including total bill and tip in dollars, recorded sex and smoking-status categories, and day of the week. We study the **total bill**, not tip amounts, in the worked examples.

    We treat restaurant observations as independent for these examples. The dataset is observational and comes from one restaurant; it does not support a causal claim about the grouping variables or an unrestricted claim about all restaurants. If repeated visits or other dependence are present, these independent procedures would need reconsideration.
    """)
    return


@app.cell
def _(lesson_dir, pd, mo):
    tips = sns.load_dataset("tips")
    mo.Html(tips.head(10).to_html(index=False, border=0))    
    return (tips,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Total bill by recorded sex
    Form groups from `total_bill` and `sex`. The bootstrap asks whether their population mean bills are equal. The permutation comparison requires interchangeable group labels under identical bill distributions. The bars show mean bill amounts in dollars, not counts of diners.
    """)
    return


@app.cell
def _(tips, get_dict):
    data_sex = get_dict(tips, "total_bill", "sex")
    return (data_sex,)


@app.cell
def _(data_sex, make_model):
    tips_sex_bootstrap = make_model(data_sex, "Bootstrap", "Independent", seed=100)
    return (tips_sex_bootstrap,)


@app.cell
def _(data_sex, tips_sex_bootstrap, plot_bars):
    plot_bars(data_sex, model=tips_sex_bootstrap, title="Total bill by recorded sex: bootstrap", ylabel="Mean total bill ($)")
    return


@app.cell
def _(tips_sex_bootstrap, mo):
    _p = tips_sex_bootstrap.get_p_value()
    mo.md("Total bill by recorded sex, equal-mean bootstrap: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Total bill by recorded sex, equal-mean bootstrap: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell
def _(data_sex, make_model):
    tips_sex_permutation = make_model(data_sex, "Permutation", "Independent", seed=200)
    return (tips_sex_permutation,)


@app.cell
def _(data_sex, tips_sex_permutation, plot_bars):
    plot_bars(data_sex, model=tips_sex_permutation, title="Total bill by recorded sex: permutation", ylabel="Mean total bill ($)")
    return


@app.cell
def _(tips_sex_permutation, mo):
    _p = tips_sex_permutation.get_p_value()
    mo.md("Total bill by recorded sex, interchangeable-label permutation: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Total bill by recorded sex, interchangeable-label permutation: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confidence intervals for the population means
    These 95% percentile intervals use **unshifted bootstrap means** from the original samples. They estimate each population mean, rather than represent the equal-mean null. The points mark the observed sample means, not the interval midpoints. These are separate marginal intervals, not simultaneous pairwise intervals. Their overlap is not a substitute for a test of a difference.
    """)
    return


@app.cell
def _(data_sex, tips_sex_bootstrap, pd, mo):
    tips_sex_ci = tips_sex_bootstrap.confidence_intervals()
    tips_sex_ci_table = pd.DataFrame({"Group": [str(k) for k in data_sex], "Observed mean": [a.mean() for a in data_sex.values()], "95% lower": tips_sex_ci[:, 0], "95% upper": tips_sex_ci[:, 1]})
    mo.Html(tips_sex_ci_table.round(3).to_html(index=False, border=0))
    return (tips_sex_ci,)


@app.cell
def _(data_sex, tips_sex_ci, plot_confidence_intervals):
    plot_confidence_intervals(tips_sex_ci, [str(k) for k in data_sex], means=[a.mean() for a in data_sex.values()], title="95% bootstrap intervals for population means", xlabel="Population mean bill ($)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Total bill by smoking status
    Form groups from `total_bill` and `smoker`. The bootstrap asks whether their population mean bills are equal. The permutation comparison requires interchangeable group labels under identical bill distributions. The bars show mean bill amounts in dollars, not counts of diners.
    """)
    return


@app.cell
def _(tips, get_dict):
    data_smoker = get_dict(tips, "total_bill", "smoker")
    return (data_smoker,)


@app.cell
def _(data_smoker, make_model):
    tips_smoker_bootstrap = make_model(data_smoker, "Bootstrap", "Independent", seed=101)
    return (tips_smoker_bootstrap,)


@app.cell
def _(data_smoker, tips_smoker_bootstrap, plot_bars):
    plot_bars(data_smoker, model=tips_smoker_bootstrap, title="Total bill by smoking status: bootstrap", ylabel="Mean total bill ($)")
    return


@app.cell
def _(tips_smoker_bootstrap, mo):
    _p = tips_smoker_bootstrap.get_p_value()
    mo.md("Total bill by smoking status, equal-mean bootstrap: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Total bill by smoking status, equal-mean bootstrap: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell
def _(data_smoker, make_model):
    tips_smoker_permutation = make_model(data_smoker, "Permutation", "Independent", seed=201)
    return (tips_smoker_permutation,)


@app.cell
def _(data_smoker, tips_smoker_permutation, plot_bars):
    plot_bars(data_smoker, model=tips_smoker_permutation, title="Total bill by smoking status: permutation", ylabel="Mean total bill ($)")
    return


@app.cell
def _(tips_smoker_permutation, mo):
    _p = tips_smoker_permutation.get_p_value()
    mo.md("Total bill by smoking status, interchangeable-label permutation: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Total bill by smoking status, interchangeable-label permutation: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confidence intervals for the population means
    These 95% percentile intervals use **unshifted bootstrap means** from the original samples. They estimate each population mean, rather than represent the equal-mean null. The points mark the observed sample means, not the interval midpoints. These are separate marginal intervals, not simultaneous pairwise intervals. Their overlap is not a substitute for a test of a difference.
    """)
    return


@app.cell
def _(data_smoker, tips_smoker_bootstrap, pd, mo):
    tips_smoker_ci = tips_smoker_bootstrap.confidence_intervals()
    tips_smoker_ci_table = pd.DataFrame({"Group": [str(k) for k in data_smoker], "Observed mean": [a.mean() for a in data_smoker.values()], "95% lower": tips_smoker_ci[:, 0], "95% upper": tips_smoker_ci[:, 1]})
    mo.Html(tips_smoker_ci_table.round(3).to_html(index=False, border=0))
    return (tips_smoker_ci,)


@app.cell
def _(data_smoker, tips_smoker_ci, plot_confidence_intervals):
    plot_confidence_intervals(tips_smoker_ci, [str(k) for k in data_smoker], means=[a.mean() for a in data_smoker.values()], title="95% bootstrap intervals for population means", xlabel="Population mean bill ($)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Total bill by day
    Form groups from `total_bill` and `day`. The bootstrap asks whether their population mean bills are equal. The permutation comparison requires interchangeable group labels under identical bill distributions. The bars show mean bill amounts in dollars, not counts of diners.
    """)
    return


@app.cell
def _(tips, get_dict):
    data_day = get_dict(tips, "total_bill", "day")
    return (data_day,)


@app.cell
def _(data_day, make_model):
    tips_day_bootstrap = make_model(data_day, "Bootstrap", "Independent", seed=102)
    return (tips_day_bootstrap,)


@app.cell
def _(data_day, tips_day_bootstrap, plot_bars):
    plot_bars(data_day, model=tips_day_bootstrap, title="Total bill by day: bootstrap", ylabel="Mean total bill ($)")
    return


@app.cell
def _(tips_day_bootstrap, mo):
    _p = tips_day_bootstrap.get_p_value()
    mo.md("Total bill by day, equal-mean bootstrap: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Total bill by day, equal-mean bootstrap: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell
def _(data_day, make_model):
    tips_day_permutation = make_model(data_day, "Permutation", "Independent", seed=202)
    return (tips_day_permutation,)


@app.cell
def _(data_day, tips_day_permutation, plot_bars):
    plot_bars(data_day, model=tips_day_permutation, title="Total bill by day: permutation", ylabel="Mean total bill ($)")
    return


@app.cell
def _(tips_day_permutation, mo):
    _p = tips_day_permutation.get_p_value()
    mo.md("Total bill by day, interchangeable-label permutation: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Total bill by day, interchangeable-label permutation: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confidence intervals for the population means
    These 95% percentile intervals use **unshifted bootstrap means** from the original samples. They estimate each population mean, rather than represent the equal-mean null. The points mark the observed sample means, not the interval midpoints. These are separate marginal intervals, not simultaneous pairwise intervals. Their overlap is not a substitute for a test of a difference.
    """)
    return


@app.cell
def _(data_day, tips_day_bootstrap, pd, mo):
    tips_day_ci = tips_day_bootstrap.confidence_intervals()
    tips_day_ci_table = pd.DataFrame({"Group": [str(k) for k in data_day], "Observed mean": [a.mean() for a in data_day.values()], "95% lower": tips_day_ci[:, 0], "95% upper": tips_day_ci[:, 1]})
    mo.Html(tips_day_ci_table.round(3).to_html(index=False, border=0))
    return (tips_day_ci,)


@app.cell
def _(data_day, tips_day_ci, plot_confidence_intervals):
    plot_confidence_intervals(tips_day_ci, [str(k) for k in data_day], means=[a.mean() for a in data_day.values()], title="95% bootstrap intervals for population means", xlabel="Population mean bill ($)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A new question about tip amounts
    Now use the `tip` column, measured in dollars, to compare smoking-status groups. This changes the outcome from total bill to tip amount. The prepared bootstrap model below imposes equal population mean tip amounts.
    """)
    return


@app.cell
def _(tips, get_dict, make_model):
    tip_groups = get_dict(tips, "tip", "smoker")
    tip_amount_model = make_model(tip_groups, "Bootstrap", "Independent", seed=300)
    return (tip_amount_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the bootstrap p-value for comparing population mean tip amounts between smoking-status groups. State the hypotheses and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_tip_p = None
    return (student_tip_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_tip_p = tip_amount_model.get_p_value()
    ```
    H0: equal population mean tip amounts; Ha: different population mean tip amounts. Here p ≈ 0.9203. There is insufficient evidence to reject equal population mean tip amounts at 0.05. This observational comparison does not establish a causal smoking-status effect.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Glass identification
    The glass dataset contains 214 specimens. `Na` records sodium oxide content as a percentage by weight. `Type` identifies the glass category; the file contains types 1, 2, 3, 5, 6, and 7. Type 4 is absent, so we do not create an empty group for it.

    We compare sodium oxide content across the observed types, treating specimens as independent. The overall bootstrap asks whether the population mean concentrations are equal. The permutation comparison requires interchangeable type labels under identical sodium-content distributions. Neither procedure classifies an individual specimen.
    """)
    return


@app.cell
def _(lesson_dir, pd, mo, get_dict):
    glass = pd.read_csv(lesson_dir / "glass.csv")
    data_na = get_dict(glass, "Na", "Type")
    mo.Html(glass.head(10).to_html(index=False, border=0))
    return glass, data_na


@app.cell
def _(glass, mo):
    sodium_summary = glass.groupby("Type")["Na"].agg(n="size", Mean="mean", SD="std").reset_index()
    mo.Html(sodium_summary.round(3).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bootstrap comparison of sodium content
    """)
    return


@app.cell
def _(data_na, make_model):
    glass_bootstrap = make_model(data_na, "Bootstrap", "Independent", seed=11)
    return (glass_bootstrap,)


@app.cell
def _(data_na, glass_bootstrap, plot_bars):
    plot_bars(data_na, model=glass_bootstrap, title="Sodium oxide by glass type: bootstrap", ylabel="Mean sodium oxide (% by weight)")
    return


@app.cell
def _(glass_bootstrap, mo):
    _p = glass_bootstrap.get_p_value()
    mo.md("Sodium content across glass types, bootstrap: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Sodium content across glass types, bootstrap: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pairwise comparisons with Bonferroni correction
    The overall result does not identify the differing pairs. We now test every pair and multiply each two-sided p-value by the number of pairs, capped at one. The family significance level here is **0.01**.

    Each pair uses its own resampling null. A shared letter means that pair was **not rejected** after correction; it does not prove equal means. A group can carry multiple letters because non-rejection is not transitive. The table gives the pairwise decisions explicitly, and the graph summarizes them with letters above the observed means.
    """)
    return


@app.cell
def _(data_na, glass_bootstrap, pd, mo):
    glass_bootstrap_adjusted = glass_bootstrap.bonferroni_correction()
    _labels = list(data_na)
    glass_bootstrap_pair_table = pd.DataFrame([{"Group 1": str(_labels[i]), "Group 2": str(_labels[j]), "Adjusted p-value": glass_bootstrap_adjusted[i, j], "Reject at 0.01": bool(glass_bootstrap_adjusted[i, j] <= 0.01)} for i in range(len(_labels)) for j in range(i + 1, len(_labels))])
    mo.Html(glass_bootstrap_pair_table.to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}"))
    return


@app.cell
def _(data_na, glass_bootstrap, plot_bonferroni):
    plot_bonferroni(data_na, model=glass_bootstrap, alpha=0.01, title="Bonferroni pairwise summary", ylabel="Mean sodium oxide (% by weight)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Permutation comparison of sodium content
    """)
    return


@app.cell
def _(data_na, make_model):
    glass_permutation = make_model(data_na, "Permutation", "Independent", seed=12)
    return (glass_permutation,)


@app.cell
def _(data_na, glass_permutation, plot_bars):
    plot_bars(data_na, model=glass_permutation, title="Sodium oxide by glass type: permutation", ylabel="Mean sodium oxide (% by weight)")
    return


@app.cell
def _(glass_permutation, mo):
    _p = glass_permutation.get_p_value()
    mo.md("Sodium content across glass types, permutation: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Sodium content across glass types, permutation: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pairwise comparisons with Bonferroni correction
    The overall result does not identify the differing pairs. We now test every pair and multiply each two-sided p-value by the number of pairs, capped at one. The family significance level here is **0.01**.

    Each pair uses its own resampling null. A shared letter means that pair was **not rejected** after correction; it does not prove equal means. A group can carry multiple letters because non-rejection is not transitive. The table gives the pairwise decisions explicitly, and the graph summarizes them with letters above the observed means.
    """)
    return


@app.cell
def _(data_na, glass_permutation, pd, mo):
    glass_permutation_adjusted = glass_permutation.bonferroni_correction()
    _labels = list(data_na)
    glass_permutation_pair_table = pd.DataFrame([{"Group 1": str(_labels[i]), "Group 2": str(_labels[j]), "Adjusted p-value": glass_permutation_adjusted[i, j], "Reject at 0.01": bool(glass_permutation_adjusted[i, j] <= 0.01)} for i in range(len(_labels)) for j in range(i + 1, len(_labels))])
    mo.Html(glass_permutation_pair_table.to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}"))
    return


@app.cell
def _(data_na, glass_permutation, plot_bonferroni):
    plot_bonferroni(data_na, model=glass_permutation, alpha=0.01, title="Bonferroni pairwise summary", ylabel="Mean sodium oxide (% by weight)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confidence intervals for the population means
    These 95% percentile intervals use **unshifted bootstrap means** from the original samples. They estimate each population mean, rather than represent the equal-mean null. The points mark the observed sample means, not the interval midpoints. These are separate marginal intervals, not simultaneous pairwise intervals. Their overlap is not a substitute for a test of a difference.
    """)
    return


@app.cell
def _(data_na, glass_bootstrap, pd, mo):
    sodium_ci = glass_bootstrap.confidence_intervals()
    sodium_ci_table = pd.DataFrame({"Group": [str(k) for k in data_na], "Observed mean": [a.mean() for a in data_na.values()], "95% lower": sodium_ci[:, 0], "95% upper": sodium_ci[:, 1]})
    mo.Html(sodium_ci_table.round(3).to_html(index=False, border=0))
    return (sodium_ci,)


@app.cell
def _(data_na, sodium_ci, plot_confidence_intervals):
    plot_confidence_intervals(sodium_ci, [str(k) for k in data_na], means=[a.mean() for a in data_na.values()], title="95% bootstrap intervals for population means", xlabel="Population mean sodium oxide (% by weight)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A new question about calcium content
    The `Ca` column records calcium oxide content, also as a percentage by weight. We prepare a separate bootstrap model to compare its population means across glass types. Do not reuse sodium's null distribution for a different measured variable.
    """)
    return


@app.cell
def _(glass, get_dict, make_model):
    calcium_groups = get_dict(glass, "Ca", "Type")
    calcium_model = make_model(calcium_groups, "Bootstrap", "Independent", seed=301)
    return (calcium_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the bootstrap p-value for equal population mean calcium oxide content across glass types. Interpret it at significance level 0.05. Does the overall result identify every differing pair?
    """)
    return


@app.cell
def _():
    student_calcium_p = None
    return (student_calcium_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_calcium_p = calcium_model.get_p_value()
    ```
    The null is equality of all population mean calcium oxide concentrations. Here p ≈ 0.02999. Reject equal population mean calcium oxide content at 0.05. An overall rejection does not identify every differing pair; follow-up comparisons require their own correction.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Student performance
    The student CSV contains 395 students' mathematics records. `G1` is the first-period grade, `G2` the second-period grade, and `G3` the final grade. Grades range from 0 to 20. The three observations in each row belong to the **same student**, so the design is related, not three independent classes.

    The block bootstrap resamples complete student rows after centering condition means. It asks whether the population mean grades are equal across periods. The permutation comparison rearranges grades only within a student and additionally requires period labels to be interchangeable under its null. Because these are ordered school periods rather than randomized treatments, that assumption needs substantive justification; the permutation result is conditional on it.

    We treat different students as independent for the demonstration. Shared classes or schools may introduce dependence that this simple model does not account for. Neither comparison establishes why grades changed.
    """)
    return


@app.cell
def _(lesson_dir, pd, mo):
    students = pd.read_csv(lesson_dir / "student-mat.csv", sep=";")
    data_st = {"G1": students["G1"].to_numpy(dtype=float), "G2": students["G2"].to_numpy(dtype=float), "G3": students["G3"].to_numpy(dtype=float)}
    mo.Html(students[["G1", "G2", "G3"]].head(10).to_html(index=False, border=0))
    return students, data_st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bootstrap comparison across grade periods
    """)
    return


@app.cell
def _(data_st, make_model):
    grades_bootstrap = make_model(data_st, "Bootstrap", "Related", seed=21)
    return (grades_bootstrap,)


@app.cell
def _(data_st, grades_bootstrap, plot_bars):
    plot_bars(data_st, model=grades_bootstrap, title="Grade means across periods: bootstrap", ylabel="Mean grade (0–20)")
    return


@app.cell
def _(grades_bootstrap, mo):
    _p = grades_bootstrap.get_p_value()
    mo.md("Grade periods, bootstrap: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Grade periods, bootstrap: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pairwise comparisons with Bonferroni correction
    The overall result does not identify the differing pairs. We now test every pair and multiply each two-sided p-value by the number of pairs, capped at one. The family significance level here is **0.01**.

    Each pair uses its own resampling null. A shared letter means that pair was **not rejected** after correction; it does not prove equal means. A group can carry multiple letters because non-rejection is not transitive. The table gives the pairwise decisions explicitly, and the graph summarizes them with letters above the observed means.
    """)
    return


@app.cell
def _(data_st, grades_bootstrap, pd, mo):
    grades_bootstrap_adjusted = grades_bootstrap.bonferroni_correction()
    _labels = list(data_st)
    grades_bootstrap_pair_table = pd.DataFrame([{"Group 1": str(_labels[i]), "Group 2": str(_labels[j]), "Adjusted p-value": grades_bootstrap_adjusted[i, j], "Reject at 0.01": bool(grades_bootstrap_adjusted[i, j] <= 0.01)} for i in range(len(_labels)) for j in range(i + 1, len(_labels))])
    mo.Html(grades_bootstrap_pair_table.to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}"))
    return


@app.cell
def _(data_st, grades_bootstrap, plot_bonferroni):
    plot_bonferroni(data_st, model=grades_bootstrap, alpha=0.01, title="Bonferroni pairwise summary", ylabel="Mean grade (0–20)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Permutation comparison across grade periods
    """)
    return


@app.cell
def _(data_st, make_model):
    grades_permutation = make_model(data_st, "Permutation", "Related", seed=22)
    return (grades_permutation,)


@app.cell
def _(data_st, grades_permutation, plot_bars):
    plot_bars(data_st, model=grades_permutation, title="Grade means across periods: permutation", ylabel="Mean grade (0–20)")
    return


@app.cell
def _(grades_permutation, mo):
    _p = grades_permutation.get_p_value()
    mo.md("Grade periods, permutation: sufficient evidence to reject the stated null at 0.05." if _p <= 0.05 else "Grade periods, permutation: insufficient evidence to reject the stated null at 0.05. This does not establish equality.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pairwise comparisons with Bonferroni correction
    The overall result does not identify the differing pairs. We now test every pair and multiply each two-sided p-value by the number of pairs, capped at one. The family significance level here is **0.01**.

    Each pair uses its own resampling null. A shared letter means that pair was **not rejected** after correction; it does not prove equal means. A group can carry multiple letters because non-rejection is not transitive. The table gives the pairwise decisions explicitly, and the graph summarizes them with letters above the observed means.
    """)
    return


@app.cell
def _(data_st, grades_permutation, pd, mo):
    grades_permutation_adjusted = grades_permutation.bonferroni_correction()
    _labels = list(data_st)
    grades_permutation_pair_table = pd.DataFrame([{"Group 1": str(_labels[i]), "Group 2": str(_labels[j]), "Adjusted p-value": grades_permutation_adjusted[i, j], "Reject at 0.01": bool(grades_permutation_adjusted[i, j] <= 0.01)} for i in range(len(_labels)) for j in range(i + 1, len(_labels))])
    mo.Html(grades_permutation_pair_table.to_html(index=False, border=0, float_format=lambda x: f"{x:.4g}"))
    return


@app.cell
def _(data_st, grades_permutation, plot_bonferroni):
    plot_bonferroni(data_st, model=grades_permutation, alpha=0.01, title="Bonferroni pairwise summary", ylabel="Mean grade (0–20)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confidence intervals for the population means
    These 95% percentile intervals use **unshifted bootstrap means** from the original samples. They estimate each population mean, rather than represent the equal-mean null. The points mark the observed sample means, not the interval midpoints. These are separate marginal intervals, not simultaneous pairwise intervals. Their overlap is not a substitute for a test of a difference.
    """)
    return


@app.cell
def _(data_st, grades_bootstrap, pd, mo):
    grades_ci = grades_bootstrap.confidence_intervals()
    grades_ci_table = pd.DataFrame({"Group": [str(k) for k in data_st], "Observed mean": [a.mean() for a in data_st.values()], "95% lower": grades_ci[:, 0], "95% upper": grades_ci[:, 1]})
    mo.Html(grades_ci_table.round(3).to_html(index=False, border=0))
    return (grades_ci,)


@app.cell
def _(data_st, grades_ci, plot_confidence_intervals):
    plot_confidence_intervals(grades_ci, [str(k) for k in data_st], means=[a.mean() for a in data_st.values()], title="95% bootstrap intervals for population means", xlabel="Population mean grade (0–20)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A focused first-to-final comparison
    For another question, compare only each student's first-period and final grades. The prepared paired bootstrap model keeps the student matching. The two-sided statistic is the mean of G1 minus the mean of G3.
    """)
    return


@app.cell
def _(data_st, make_model):
    first_final_model = make_model({"G1": data_st["G1"], "G3": data_st["G3"]}, "Bootstrap", "Related", seed=302)
    return (first_final_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate the paired bootstrap p-value for a difference between population mean first-period and final grades. State the hypotheses and interpret the result at significance level 0.05.
    """)
    return


@app.cell
def _():
    student_first_final_p = None
    return (student_first_final_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    ```python
    student_first_final_p = first_final_model.get_p_value()
    ```
    H0: μG1 = μG3; Ha: μG1 ≠ μG3. Here p ≈ 0.0004999. Reject equal population mean first-period and final grades at 0.05. The matched measurements provide evidence about a population mean difference, without establishing the cause of any change.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Grouping a real dataset starts with a population question, a measured variable, and a suitable sampling design. Separate groups and repeated measurements need different resampling procedures.
    - Bootstrap equal-mean tests and permutation interchangeable-label tests use different null constructions. Their p-values need not agree, and their assumptions must be justified for the application.
    - Confidence intervals estimate population means using unshifted bootstrap samples. Separate mean intervals do not replace tests of differences or corrected pairwise intervals.
    - An overall rejection does not identify each differing pair. Bonferroni comparisons account for the family of pairwise tests, while shared letters summarize non-rejection rather than prove equality.
    - Report each case's actual result in its context. A larger p-value means insufficient evidence to reject the stated null, and an observational association does not establish a causal effect.

    ## Check your understanding
    1. A researcher compares bills from smoking-status groups, with different restaurant observations in each group. Which design is being assumed?
    2. A researcher compares G1, G2, and G3 for the same students. What must stay together during bootstrap resampling?
    3. A bootstrap test of equal population mean sodium content across glass types gives p = 0.01. At significance level 0.05, what can you conclude? Does it identify the differing type pairs?
    4. Two glass types have a Bonferroni-adjusted p-value of 0.03 when the chosen family significance level is 0.01. What is the pairwise decision? Does it establish equal means?
    5. What do the endpoints in the restaurant confidence-interval table estimate?
    6. Does a difference in bills between the recorded groups establish that the grouping variable caused the difference?
    7. Why does a permutation comparison of grade periods require an assumption beyond equality of their means?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
    1. Independent groups, conditional on treating the restaurant observations as independent.
    2. Each student's grades across periods. Resample whole student rows, not each grade column separately.
    3. Reject equal population means at 0.05. There is evidence of an overall mean difference, without identifying every differing pair.
    4. Do not reject that pair's null at family level 0.01 because 0.03 > 0.01. This does not establish equal population means.
    5. Lower and upper percentile-bootstrap confidence limits for each group's population mean total bill.
    6. No. The observational data do not establish causation.
    7. The within-student grade labels must be interchangeable under the permutation null. Equal means alone do not ensure identical period distributions or an invariant joint distribution.
    """)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - [Seaborn example datasets: restaurant tips](https://github.com/mwaskom/seaborn-data). The local CSV contains the standard tips data.
    - [UCI Machine Learning Repository: Glass Identification](https://archive.ics.uci.edu/dataset/42/glass+identification).
    - [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).
    - Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
    - Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
    """)
    return


if __name__ == "__main__":
    app.run()
