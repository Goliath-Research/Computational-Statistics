import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conditional Probability

    ## Learning goals
    - Calculate probabilities from counts and cross-tabulations.
    - Explain how a condition changes the group we consider and the denominator.
    - Distinguish P(A | B) from P(B | A).
    - Recover an overall probability using the law of total probability.
    - Calculate a reversed conditional probability using Bayes' rule and verify it with counts.
    - Interpret associations without assuming causation.

    ## 1. Start with a small example
    The following table is **invented for teaching**. Imagine choosing one of these 40 students uniformly at random.

    | Study-time group | Passed | Did not pass | Total |
    |---|---:|---:|---:|
    | Higher | 12 | 8 | 20 |
    | Lower | 9 | 11 | 20 |
    | Total | 21 | 19 | 40 |

    An **event** is a set of possible results, such as selecting a student who passed.
    Without a condition, all 40 students are eligible: P(Passed) = 21/40 = 0.525.
    Given higher study time, only those 20 students are eligible: P(Passed | Higher) = 12/20 = 0.600.
    Given passing, only the 21 passing students are eligible: P(Higher | Passed) = 12/21 ≈ 0.571.
    The same 12 students appear in both numerators, but the denominators differ.

    ## Conditional probability
    “Given B” means that we restrict attention to outcomes satisfying B. B need not occur earlier in time.

    $$P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B)>0.$$

    Here A ∩ B means that both A and B hold. For equally likely selections from a table:

    $$P(A\mid B)=\frac{\text{number satisfying both A and B}}{\text{number satisfying B}}.$$

    ### Try it yourself
    Before continuing, calculate P(Passed | Lower), P(Lower | Passed), and P(Higher and Passed).
    Explain which students form the denominator each time.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({"Check your answers": mo.md("P(Passed | Lower) = 9/20 = 0.450. P(Lower | Passed) = 9/21 ≈ 0.429. P(Higher and Passed) = 12/40 = 0.300.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Work with the student dataset
    We now use the mathematics file `student-mat.csv` from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).
    Place the CSV in the course's `data` folder (one directory above this lesson), or in a `data` folder beside this lesson.

    Imagine selecting one recorded student uniformly at random. Calculated probabilities describe that selection from this file. They do not automatically describe all students.

    We retain only five columns needed for these questions:
    - `studytime`: categories 1 (<2 hours), 2 (2–5 hours), 3 (5–10 hours), 4 (>10 hours).
    - `internet`: internet access at home, `yes` or `no`.
    - `G1`, `G2`, `G3`: first-period, second-period, and final grades, on a 0–20 scale.

    Study time is recorded in categories, not as exact hours. “Higher study time” below means categories 3 and 4.
    The passing threshold is **our definition for this lesson**, not a claim about the original school's grading policy.
    """)
    return


@app.cell
def _(Path, pd):
    lesson_directory = Path(__file__).resolve().parent
    candidate_paths = [lesson_directory.parent / "data" / "student-mat.csv",
                       lesson_directory / "data" / "student-mat.csv",
                       lesson_directory / "student-mat.csv"]
    dataset_path = next((path for path in candidate_paths if path.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError("student-mat.csv is missing. Put it in the course data folder, a data folder beside this lesson, or beside this lesson.")
    student_file = pd.read_csv(dataset_path, sep=";")
    required_columns = ["studytime", "internet", "G1", "G2", "G3"]
    if not set(required_columns).issubset(student_file.columns):
        raise ValueError("The CSV must contain studytime, internet, G1, G2, and G3.")
    source_data = student_file[required_columns].copy()
    if source_data.empty or source_data.isna().any().any():
        raise ValueError("The selected columns must contain students and no missing values.")
    if not source_data["studytime"].isin([1, 2, 3, 4]).all():
        raise ValueError("studytime must contain categories 1–4.")
    if not source_data["internet"].isin(["yes", "no"]).all():
        raise ValueError("internet must contain yes or no.")
    for grade_column in ["G1", "G2", "G3"]:
        if not pd.api.types.is_numeric_dtype(source_data[grade_column]) or not source_data[grade_column].between(0, 20).all():
            raise ValueError("Grades must be numeric values between 0 and 20.")
    print(f"Loaded {len(source_data)} student records.")
    source_data.head()
    return (source_data,)


@app.cell
def _(mo):
    passing_control = mo.ui.slider(start=0, stop=100, step=5, value=60, label="Passing threshold (%)")
    passing_control
    return (passing_control,)


@app.cell
def _(passing_control, source_data):
    passing_percent = passing_control.value
    data = source_data.copy()
    for grade_name in ["G1", "G2", "G3"]:
        data[f"{grade_name}pass"] = data[grade_name] * 5 >= passing_percent
    data["HigherStudyTime"] = data["studytime"].isin([3, 4])
    return data, passing_percent


@app.cell
def _(mo, passing_percent):
    mo.md(f"""
    **Current rule:** a grade passes when its percentage is at least {passing_percent}%, equivalent to at least {passing_percent / 5:g} on the 0–20 scale. All calculations below use this rule.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A Boolean passing indicator is `True` for passing and `False` otherwise.
    Python counts `True` as 1 and `False` as 0. Therefore its sum counts passes, and its mean is the passing proportion.
    We checked missing values first so an unrecorded grade cannot silently become a failure.

    The helper below returns a count ratio. If the condition selects nobody, the probability is undefined—not zero.
    """)
    return


@app.cell
def _(np):
    def probability_from_counts(event, condition):
        denominator = int(condition.sum())
        numerator = int((event & condition).sum())
        probability = numerator / denominator if denominator else np.nan
        return numerator, denominator, probability


    def format_probability(value):
        return "undefined (empty conditioning group)" if np.isnan(value) else f"{value:.3f}"

    return format_probability, probability_from_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Passing given higher study time
    Start with G1. The cross-tabulation below shows every combination; `All` contains totals.
    Identify the intersection count and the size of the higher-study group before looking at the calculation.
    """)
    return


@app.cell
def _(data, pd):
    study_crosstab = pd.crosstab(data["G1pass"], data["HigherStudyTime"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    study_crosstab.index = ["Did not pass G1", "Passed G1"]
    study_crosstab.columns = ["Lower study time", "Higher study time"]
    study_crosstab["All"] = study_crosstab.sum(axis=1)
    study_crosstab.loc["All"] = study_crosstab.sum(axis=0)
    study_crosstab
    return


@app.cell
def _(data, format_probability, probability_from_counts):
    joint_count, subgroup_count, conditional_g1 = probability_from_counts(data["G1pass"], data["HigherStudyTime"])
    print(f"P(G1pass and HigherStudyTime) = {joint_count}/{len(data)} = {joint_count / len(data):.3f}")
    print(f"P(HigherStudyTime) = {subgroup_count}/{len(data)} = {subgroup_count / len(data):.3f}")
    print(f"P(G1pass | HigherStudyTime) = {joint_count}/{subgroup_count} = {format_probability(conditional_g1)}")
    print("Check using the filtered indicator's mean:", format_probability(data.loc[data["HigherStudyTime"], "G1pass"].mean()))
    return


@app.cell
def _(data):
    study_comparison = data.groupby("HigherStudyTime").agg(Students=("G1pass", "size"), G1_rate=("G1pass", "mean"), G2_rate=("G2pass", "mean"), G3_rate=("G3pass", "mean")).reindex([False, True])
    study_comparison["Students"] = study_comparison["Students"].fillna(0).astype(int)
    study_comparison.index = ["Lower study time", "Higher study time"]
    study_comparison.round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Calculate the G1 conditional probability directly from the cross-tabulation.
    2. Compare both study groups for G1, G2, and G3. Is the direction of the comparison identical for every grade?
    3. Predict what happens to a fixed group's passing rate if the threshold decreases. Move the threshold from 60% to 50% and check.

    **Discussion:** Lowering the threshold cannot decrease the passing rate within a fixed group. It can leave it unchanged. A difference between study groups is an association; this calculation alone does not show that studying longer caused it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Passing given internet access
    To compare passing rates, we need both students with internet access and students without it.
    Read the student counts as well as the proportions. A proportion based on a small group has less observational support.
    """)
    return


@app.cell
def _(data):
    internet_comparison = data.groupby("internet").agg(Students=("G1pass", "size"), G1_rate=("G1pass", "mean"), G2_rate=("G2pass", "mean"), G3_rate=("G3pass", "mean")).reindex(["no", "yes"])
    internet_comparison["Students"] = internet_comparison["Students"].fillna(0).astype(int)
    internet_comparison.round(3)
    return (internet_comparison,)


@app.cell
def _(internet_comparison, passing_percent, plt):
    _figure, _axes = plt.subplots(figsize=(8, 4))
    internet_comparison[["G1_rate", "G2_rate", "G3_rate"]].rename(columns={"G1_rate": "G1", "G2_rate": "G2", "G3_rate": "G3"}).plot.bar(ax=_axes, rot=0)
    _axes.set(xlabel="Internet access at home", ylabel="Passing proportion", ylim=(0, 1), title=f"Passing rates by internet access: cutoff {passing_percent}%")
    _axes.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3)
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Identify the denominator for P(G3pass | internet = yes).
    2. Compare this probability with P(G3pass | internet = no).
    3. Could differences in other characteristics contribute to the observed comparison?

    **Discussion:** These are descriptive associations in the recorded data. The comparisons do not establish causation or automatically generalize to a broader student population.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The law of total probability
    The four study categories form a **partition**: each student belongs to exactly one category, and together the categories include every student.
    Let A mean passing G3 and Bᵢ mean belonging to study category i.

    $$P(A)=\sum_{i=1}^{4}P(A\mid B_i)P(B_i).$$

    This is a weighted average. Each group's passing rate is multiplied by its share of all students.
    A group with twice as many students gets twice as much weight.
    For a category with no students, its contribution is zero; its within-group passing rate is undefined.
    """)
    return


@app.cell
def _(data):
    study_summary = data.groupby("studytime")["G3pass"].agg(Students="size", Passed="sum").reindex([1, 2, 3, 4], fill_value=0)
    study_summary["Group_probability"] = study_summary["Students"] / len(data)
    study_summary["Pass_given_group"] = study_summary["Passed"] / study_summary["Students"].where(study_summary["Students"] > 0)
    # Passed / total is the same contribution as group rate × group share,
    # and correctly gives zero for an empty group.
    study_summary["Weighted_contribution"] = study_summary["Passed"] / len(data)
    study_summary.round(4)
    return (study_summary,)


@app.cell
def _(data, np, study_summary):
    overall_weighted = study_summary["Weighted_contribution"].sum()
    overall_direct = data["G3pass"].mean()
    print(f"Weighted sum: P(G3pass) = {overall_weighted:.4f}")
    print(f"Direct count: {int(data['G3pass'].sum())}/{len(data)} = {overall_direct:.4f}")
    print("Do the two methods agree?", np.isclose(overall_weighted, overall_direct))
    return (overall_direct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Multiply each group's conditional passing rate by its group probability. Add the contributions.
    2. Why would an unweighted average of the four passing rates generally be wrong?
    3. Change the cutoff and check that the weighted and direct calculations still agree.

    **Discussion:** An unweighted average gives equal importance to groups of unequal sizes. Use full precision for calculations; round only the displayed results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Bayes' rule: reverse the condition
    Now we are told that the selected student passed G3. We ask which study group the student belongs to.

    $$P(B_i\mid A)=\frac{P(A\mid B_i)P(B_i)}{P(A)},\qquad P(A)>0.$$

    The numerator describes the intersection. Dividing by P(A) changes the reference group to passing students.
    We can verify the answer directly: count passing students in category i and divide by all passing students.
    """)
    return


@app.cell
def _(mo):
    chosen_group = mo.ui.dropdown(options={"1: less than 2 hours": 1, "2: 2–5 hours": 2, "3: 5–10 hours": 3, "4: more than 10 hours": 4}, value="2: 2–5 hours", label="Study category")
    chosen_group
    return (chosen_group,)


@app.cell
def _(chosen_group, data, overall_direct, study_summary):
    selected_category = chosen_group.value
    selected_row = study_summary.loc[selected_category]
    passing_total = int(data["G3pass"].sum())
    if passing_total == 0:
        print("P(study category | G3pass) is undefined: no students passed at this cutoff.")
    elif selected_row["Students"] == 0:
        print("This category has no students. Its probability among passing students is 0.")
        print("The group's own passing rate is undefined, so do not multiply it as a numeric rate.")
    else:
        bayes_result = selected_row["Pass_given_group"] * selected_row["Group_probability"] / overall_direct
        direct_result = selected_row["Passed"] / passing_total
        print(f"P(G3pass | category {selected_category}) = {selected_row['Pass_given_group']:.4f}")
        print(f"P(category {selected_category}) = {selected_row['Group_probability']:.4f}")
        print(f"Bayes: P(category {selected_category} | G3pass) = {bayes_result:.4f}")
        print(f"Direct count: {int(selected_row['Passed'])}/{passing_total} = {direct_result:.4f}")
    return


@app.cell
def _(data, study_summary):
    passing_count = int(data["G3pass"].sum())
    posterior_table = study_summary[["Students", "Passed", "Group_probability", "Pass_given_group"]].copy()
    posterior_table["Group_given_pass"] = posterior_table["Passed"] / passing_count if passing_count else float("nan")
    posterior_table.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Choose categories 2 and 3. Calculate the reversed probabilities using both methods.
    2. Explain why P(G3pass | category 2) and P(category 2 | G3pass) have different denominators.
    3. Add the four probabilities of study category given passing. Why must they sum to 1 when at least one student passed?
    4. Lower the cutoff. Must every category's share among passing students increase?

    **Discussion:** No. Lowering the cutoff can add passing students in different proportions across categories. Although each fixed group's passing rate cannot decrease, its share among passing students may increase, decrease, or remain unchanged.

    ## Conclusions
    - Conditioning restricts our reference group and changes the denominator.
    - P(A | B) and P(B | A) generally differ.
    - A probability conditioned on an empty group is undefined, not zero.
    - The law of total probability combines group rates using group-size weights.
    - Bayes' rule reverses conditioning; direct counts provide a useful verification.
    - Changing our passing definition changes the calculated events and results.
    - Results describe this file's students; associations do not establish causation.

    ## Check your understanding
    1. In the introductory table, what is P(Did not pass | Higher)?
    2. Which denominator belongs to P(Higher | Passed): 20, 21, or 40?
    3. Should an overall rate give equal weights to groups of different sizes?
    4. If no student meets a condition, is its conditional passing probability zero?
    5. Does a higher observed passing rate establish a causal effect?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({"Check your answers": mo.md("1. 8/20 = 0.40. 2. 21, the number who passed. 3. No; weight by group proportions. 4. No; it is undefined. 5. No; an association alone does not establish causation.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), for the dataset and variable definitions.
    - Unpingco, J. (2019). *Python for Probability, Statistics, and Machine Learning*. Springer, Chapter 2.

    This lesson contains no random sampling, so it needs no random seed. The same CSV and passing threshold reproduce the same calculations.
    """)
    return


if __name__ == "__main__":
    app.run()
