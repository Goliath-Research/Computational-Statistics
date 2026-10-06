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
    - Recognize when a conditional probability is undefined.
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

    - Without a condition, all 40 students are eligible: P(Passed) = 21/40 = 0.525.
    - Given higher study time, only those 20 students are eligible: P(Passed | Higher) = 12/20 = 0.600.
    - Given passing, only the 21 passing students are eligible: P(Higher | Passed) = 12/21 ≈ 0.571.
    - The same 12 students appear in both numerators, but the denominators differ.

    ## Conditional probability
    “Given B” means that we restrict attention to outcomes satisfying B. B need not occur earlier in time.

    $$P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B)>0.$$

    Here A ∩ B means that both A and B hold. For equally likely selections from a table:

    $$P(A\mid B)=\frac{\text{number satisfying both A and B}}{\text{number satisfying B}}.$$

    In each **Try it yourself** section with a Python task, write your script in
    the code cell immediately below the instructions. Replace the `None` placeholders
    with your calculations and run the cell. Open this notebook in the **marimo editor**
    to edit and run these cells; a read-only page does not allow code editing.

    Then expand **Show answers**. The explanations, complete solution script, and
    discussion are together in that one answer cell.

    ### Try it yourself
    Before continuing, calculate: 
    - P(Passed | Lower) 
    - P(Lower | Passed)
    - P(Higher and Passed)
    
    Explain which students form the denominator each time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show answers": mo.md("""
1. P(Passed | Lower) = 9/20 = 0.450. The denominator is the 20 lower-study students.
2. P(Lower | Passed) = 9/21 ≈ 0.429. The denominator is the 21 passing students.
3. P(Higher and Passed) = 12/40 = 0.300. The denominator is all 40 students.
""")
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Work with the student dataset
    We now use the mathematics file `student-mat.csv` from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).
    Place the CSV in the course's `data` folder (one directory above this lesson), or in a `data` folder beside this lesson.

    Imagine selecting one recorded student uniformly at random. Calculated probabilities describe that selection from this file. They do not automatically describe all students.

    We retain only five columns needed for these questions:
    - `studytime`: weekly study-time categories 1 (<2 hours), 2 (2–5 hours), 3 (5–10 hours), 4 (>10 hours).
    - `internet`: internet access at home, `yes` or `no`.
    - `G1`, `G2`, `G3`: first-period, second-period, and final grades, on a 0–20 scale.

    Study time is recorded in categories, not as exact hours. “Higher study time” below means categories 3 and 4.
    The passing threshold is **our definition for this lesson**, not a claim about the original school's grading policy.
    """)
    return


@app.cell
def _(mo, pd):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")

    source_data = student_file[
        ["studytime", "internet", "G1", "G2", "G3"]
    ].copy()

    print(f"Loaded {len(source_data)} student records.")
    mo.Html(
        source_data.head().to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (source_data,)


@app.cell
def _(mo):
    passing_control = mo.ui.slider(start=0, stop=100, step=5, value=60, show_value=True, label="Passing threshold (%)")
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
def _(mo, passing_percent, hide_code=True):
    _text = (
        f"**Current rule:** a grade passes when its percentage is "
        f"at least {passing_percent}%, equivalent to at least "
        f"{passing_percent / 5:g} on the 0–20 scale. "
        "All calculations below use this rule."
    )
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A Boolean passing indicator is `True` for passing and `False` otherwise.
    Python counts `True` as 1 and `False` as 0. Therefore its sum counts passes, and its mean is the passing proportion.    

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
def _(data, mo, pd):
    study_crosstab = pd.crosstab(
        data["G1pass"], data["HigherStudyTime"]
    ).reindex(index=[False, True], columns=[False, True], fill_value=0)

    study_crosstab.index = ["Did not pass G1", "Passed G1"]
    study_crosstab.columns = ["Lower study time", "Higher study time"]
    study_crosstab["All"] = study_crosstab.sum(axis=1)
    study_crosstab.loc["All"] = study_crosstab.sum(axis=0)

    mo.Html(
        study_crosstab.to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell
def _(data, format_probability, probability_from_counts):
    joint_count, subgroup_count, conditional_g1 = probability_from_counts(data["G1pass"], data["HigherStudyTime"])
    print(f"P(G1pass and HigherStudyTime) = {joint_count}/{len(data)} = {joint_count / len(data):.3f}")
    print(f"P(HigherStudyTime) = {subgroup_count}/{len(data)} = {subgroup_count / len(data):.3f}")
    print(f"P(G1pass | HigherStudyTime) = {joint_count}/{subgroup_count} = {format_probability(conditional_g1)}")
    print("\nCheck using the filtered indicator's mean:", format_probability(data.loc[data["HigherStudyTime"], "G1pass"].mean()))
    return


@app.cell
def _(data, mo):
    study_comparison = data.groupby("HigherStudyTime").agg(Students=("G1pass", "size"), G1_rate=("G1pass", "mean"), G2_rate=("G2pass", "mean"), G3_rate=("G3pass", "mean")).reindex([False, True])
    study_comparison["Students"] = study_comparison["Students"].fillna(0).astype(int)
    study_comparison.index = ["Lower study time", "Higher study time"]
    mo.Html(
        study_comparison.round(3).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Try it yourself
    1. Calculate the G1 conditional probability directly from the cross-tabulation.
    2. Compare both study groups for G1, G2, and G3. Is the direction of the comparison identical for every grade?
    3. Predict what happens to a fixed group's passing rate if the threshold decreases. Move the threshold from 60% to 50% and check.

**Write your script in the next cell:**
    - Select students with `data["HigherStudyTime"]`.
    - Count the selected students and their G1 passes using `.sum()`.
    - Divide passes by students. Use `np.nan` if the selected group is empty.
    - Print your result, then move the passing-threshold slider.
    """)
    return


@app.cell
def _(data):
    student_study_group = data["HigherStudyTime"]
    student_study_count = None
    student_study_passes = None
    student_study_probability = None

    print("Students in the group:", student_study_count)
    print("G1 passes in the group:", student_study_passes)
    print("P(G1pass | HigherStudyTime):", student_study_probability)
    return student_study_count, student_study_passes, student_study_probability


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. Divide the G1 passes in the higher-study group by that group's total.
2. Read the three grade columns for both groups; the direction can differ by grade and cutoff.
3. Lowering the threshold cannot decrease a fixed group's passing rate; it may leave it unchanged.

```python
student_study_group = data["HigherStudyTime"]
student_study_count = student_study_group.sum()
student_study_passes = (data["G1pass"] & student_study_group).sum()
student_study_probability = student_study_passes / student_study_count if student_study_count else np.nan

print("Students in the group:", student_study_count)
print("G1 passes in the group:", student_study_passes)
print("P(G1pass | HigherStudyTime):", student_study_probability)
```

**Discussion:** Lowering the threshold cannot decrease the passing rate within a fixed group. It can leave it unchanged. A difference between study groups is an association; this calculation alone does not show that studying longer caused it.
    """)})
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
def _(data, mo):
    internet_comparison = data.groupby("internet").agg(Students=("G1pass", "size"), G1_rate=("G1pass", "mean"), G2_rate=("G2pass", "mean"), G3_rate=("G3pass", "mean")).reindex(["no", "yes"])
    internet_comparison["Students"] = internet_comparison["Students"].fillna(0).astype(int)
    mo.Html(
        internet_comparison.round(3).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
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

**Write your script in the next cell:**
    - Select each internet group with `data["internet"] == "yes"` or `"no"`.
    - Calculate each group's G3 passing rate using the filtered indicator's `.mean()`.
    - Print both results and compare with the table. An empty group's mean is undefined.
    """)
    return


@app.cell
def _(data):
    # Replace None with the two conditional passing rates.
    student_internet_yes = None
    student_internet_no = None

    print("P(G3pass | internet = yes):", student_internet_yes)
    print("P(G3pass | internet = no):", student_internet_no)
    return student_internet_yes, student_internet_no


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. The denominator is the number of students with internet access at home.
2. Compare the G3 rates in the table; each uses its own internet group's total.
3. Yes. Other characteristics may contribute to the observed association.

```python
student_internet_yes = data.loc[data["internet"] == "yes", "G3pass"].mean()
student_internet_no = data.loc[data["internet"] == "no", "G3pass"].mean()

print("P(G3pass | internet = yes):", student_internet_yes)
print("P(G3pass | internet = no):", student_internet_no)
```

**Discussion:** These are descriptive associations in the recorded data. The comparisons do not establish causation or automatically generalize to a broader student population.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The law of total probability
    The four study categories form a **partition**: each student belongs to exactly one category, and together the categories include every student.
    Let A mean passing G3 and Bᵢ mean belonging to study category i.

    $$P(A)=\sum_{i:\,P(B_i)>0}P(A\mid B_i)P(B_i).$$

    This is a weighted average. Each group's passing rate is multiplied by its share of all students.
    A group with twice as many students gets twice as much weight.
    The sum includes only nonempty categories. An empty category has intersection probability P(A ∩ Bᵢ) = 0, so its contribution is zero. Its conditional passing probability is undefined; we do not multiply an undefined value by zero.
    """)
    return


@app.cell
def _(data, mo):
    study_summary = data.groupby("studytime")["G3pass"].agg(Students="size", Passed="sum").reindex([1, 2, 3, 4], fill_value=0)
    study_summary["Group_probability"] = study_summary["Students"] / len(data)
    study_summary["Pass_given_group"] = study_summary["Passed"] / study_summary["Students"].where(study_summary["Students"] > 0)
    # Passed / total is the same contribution as group rate × group share,
    # and correctly gives zero for an empty group.
    study_summary["Weighted_contribution"] = study_summary["Passed"] / len(data)
    mo.Html(
        study_summary.round(4).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
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

**Write your script in the next cell:**
    - Use the `Pass_given_group` and `Group_probability` columns of `study_summary`.
    - Multiply the columns and add the contributions with `.sum()`.
    - For empty groups, use `.fillna(0)` on the contributions.
    - Calculate the direct rate from `data["G3pass"].mean()` and print both results.
    """)
    return


@app.cell
def _():
    # Replace None with your weighted and direct calculations.
    student_total_weighted = None
    student_total_direct = None

    print("Weighted probability:", student_total_weighted)
    print("Direct probability:", student_total_direct)
    return student_total_weighted, student_total_direct


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. Multiply each nonempty group's rate by its share of all students, then add.
2. An unweighted average gives equal importance to groups of unequal sizes.
3. The weighted and direct methods agree at every cutoff when calculated from the same data.

```python
student_total_weighted = (study_summary["Pass_given_group"] * study_summary["Group_probability"]).fillna(0).sum()
student_total_direct = data["G3pass"].mean()

print("Weighted probability:", student_total_weighted)
print("Direct probability:", student_total_direct)
```

**Discussion:** An unweighted average gives equal importance to groups of unequal sizes. Use full precision for calculations; round only the displayed results.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Bayes' rule: reverse the condition
    Now we are told that the selected student passed G3. We ask which study group the student belongs to.

    $$P(B_i\mid A)=\frac{P(A\mid B_i)P(B_i)}{P(A)},\qquad P(A)>0,\;P(B_i)>0.$$

    For an empty category, P(Bᵢ | A) = 0 when P(A) > 0, obtained directly from its zero intersection count. The displayed product formula does not apply because P(A | Bᵢ) is undefined.

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
def _(data, mo, study_summary):
    passing_count = int(data["G3pass"].sum())
    posterior_table = study_summary[["Students", "Passed", "Group_probability", "Pass_given_group"]].copy()
    posterior_table["Group_given_pass"] = posterior_table["Passed"] / passing_count if passing_count else float("nan")
    mo.Html(
        posterior_table.round(4).to_html(border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
### Try it yourself
    1. Choose categories 2 and 3. Calculate the reversed probabilities using both methods.
    2. Explain why P(G3pass | category 2) and P(category 2 | G3pass) have different denominators.
    3. Add the four probabilities of study category given passing. Why must they sum to 1 when at least one student passed?
    4. Lower the cutoff. Must every category's share among passing students increase?

**Write your script in the next cell:**
    - Use the category selected by `chosen_group.value`.
    - Select its row with `study_summary.loc[chosen_group.value]`.
    - Calculate P(category | G3pass) using Bayes' rule and direct counts.
    - If nobody passed, both answers are undefined (`np.nan`).
    - If the category is empty but someone passed, both answers are 0.
    - Print both results. Choose categories 2 and 3 and change the cutoff.
    """)
    return


@app.cell
def _(chosen_group, data, study_summary):
    student_bayes_row = study_summary.loc[chosen_group.value]
    student_bayes_total = data["G3pass"].sum()

    # Replace None with your calculations, including the empty-group cases.
    student_bayes_probability = None
    student_bayes_direct = None

    print("Bayes probability:", student_bayes_probability)
    print("Direct probability:", student_bayes_direct)
    return student_bayes_probability, student_bayes_direct


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. For each category, multiply its passing rate by its group probability and divide by the overall passing probability. Check by dividing its passes by all passes.
2. P(G3pass | category 2) uses all category-2 students; P(category 2 | G3pass) uses all passing students.
3. The four categories partition the passing students, so their shares sum to 1 when anyone passed.
4. No. Each category's share among passing students may increase, decrease, or remain unchanged.

```python
student_bayes_row = study_summary.loc[chosen_group.value]
student_bayes_total = data["G3pass"].sum()

if student_bayes_total == 0:
    student_bayes_probability = np.nan
    student_bayes_direct = np.nan
elif student_bayes_row["Students"] == 0:
    student_bayes_probability = 0
    student_bayes_direct = 0
else:
    student_bayes_probability = (
        student_bayes_row["Pass_given_group"]
        * student_bayes_row["Group_probability"]
        / data["G3pass"].mean()
    )
    student_bayes_direct = student_bayes_row["Passed"] / student_bayes_total

print("Bayes probability:", student_bayes_probability)
print("Direct probability:", student_bayes_direct)
```

**Discussion:** No. Lowering the cutoff can add passing students in different proportions across categories. Although each fixed group's passing rate cannot decrease, its share among passing students may increase, decrease, or remain unchanged.
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - Conditioning restricts our reference group and uses that group as the denominator. The numerical probability may change or remain the same.
    - P(A | B) and P(B | A) generally differ.
    - A probability conditioned on an empty group is undefined, not zero.
    - The law of total probability combines group rates using group-size weights.
    - Bayes' rule reverses conditioning; direct counts provide a useful verification.
    - Changing the passing threshold changes the criterion defining a pass; event membership and numerical results may change or remain the same.
    - Results describe this file's students; associations do not establish causation.

    ## Check your understanding
    1. In the introductory table, what is P(Did not pass | Higher)?
    2. Which denominator belongs to P(Higher | Passed): 20, 21, or 40?
    3. Should an overall rate give equal weights to groups of different sizes?
    4. If no student meets a condition, is its conditional passing probability zero?
    5. Does a higher observed passing rate establish a causal effect?
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md("""
1. 8/20 = 0.40.
2. 21, the number who passed.
3. No; weight by group proportions.
4. No; it is undefined.
5. No; an association alone does not establish causation.
""")})
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
