import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 10)
    plt.rcParams["figure.max_open_warning"] = 0
    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conditional Probability

    ## Objectives

    - Compute a conditional probability from a cross-tabulation and from a filtered column.
    - Compare passing rates for students who study at least 5 hours a week with passing rates for students who have internet at home.
    - Use the law of total probability to recover P(passing the final grade) from weekly study time.
    - Use Bayes' rule to find the probability of a study-time group, given that the student passed.

    ## Background

    Conditional probability is the probability of one event given that another event has already occurred. This lesson computes those probabilities from the Student Performance dataset. The questions compare grades with weekly study time and with internet access at home.

    ## Datasets Used

    The Student Performance file `student-mat.csv` records math students. Each row is one student. This lesson uses the school, sex, address, weekly study time, extra school support, internet access, and the three grades G1, G2, and G3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conditional probability

    `P(A|B)` is the probability of A given that B has occurred.

    The formula used in this lesson is

    \[
    P(A|B) = \frac{P(A \cap B)}{P(B)}
    \]
    """)
    return


@app.cell
def _(pd):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    print(student_file.shape)
    student_file.head()
    return (student_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The file has 33 columns. The working table keeps nine of them:

    - **school**: student's school (`GP` or `MS`)
    - **sex**: student's sex (`F` or `M`)
    - **address**: home address (`U` urban or `R` rural)
    - **studytime**: weekly study time (`1` is less than 2 hours, `2` is 2 to 5 hours, `3` is 5 to 10 hours, `4` is more than 10 hours)
    - **schoolsup**: extra educational support (`yes` or `no`)
    - **internet**: internet access at home (`yes` or `no`)
    - **G1**: first-period grade, from 0 to 20
    - **G2**: second-period grade, from 0 to 20
    - **G3**: final grade, from 0 to 20

    The grades are on a 0–20 scale, so a grade times 5 is a percent. `G1pass` is 1 when that percent is at least `passing_percent`, and 0 otherwise. The same rule builds `G2pass` and `G3pass`.

    `StudyHard` is 1 when `studytime` is at least 3, which means 5 hours a week or more.
    """)
    return


@app.cell
def _(np, student_file):
    passing_percent = 60

    data = student_file[
        [
            "school",
            "sex",
            "address",
            "studytime",
            "schoolsup",
            "internet",
            "G1",
            "G2",
            "G3",
        ]
    ].copy()
    data["G1pass"] = np.where(data.G1 * 5 >= passing_percent, 1, 0)
    data["G2pass"] = np.where(data.G2 * 5 >= passing_percent, 1, 0)
    data["G3pass"] = np.where(data.G3 * 5 >= passing_percent, 1, 0)
    data["StudyHard"] = np.where(data.studytime >= 3, 1, 0)
    print(data.shape)
    data.head()
    return data, passing_percent


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    `passing_percent` is set in the cell above. Change it from 60 to 50 and run that cell. The charts and probabilities below follow the new cutoff.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 1. Passing, given that the student studies hard

    The question is: what is the probability that a student scores at least 60% on G1, G2, or G3, given that the student studies at least 5 hours a week?

    Each bar chart counts failures (0) and passes (1).
    """)
    return


@app.cell
def _(data, plt):
    _figure, _axes = plt.subplots()
    data.G1pass.value_counts().plot(kind="bar", rot=True, title="G1pass", ax=_axes)
    _figure
    return


@app.cell
def _(data, plt):
    _figure, _axes = plt.subplots()
    data.G2pass.value_counts().plot(kind="bar", rot=True, title="G2pass", ax=_axes)
    _figure
    return


@app.cell
def _(data, plt):
    _figure, _axes = plt.subplots()
    data.G3pass.value_counts().plot(kind="bar", rot=True, title="G3pass", ax=_axes)
    _figure
    return


@app.cell
def _(data, plt):
    _figure, _axes = plt.subplots()
    data.StudyHard.value_counts().plot(
        kind="bar",
        rot=True,
        title="Study time greater than or equal to 5 hours a week",
        ax=_axes,
    )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More students fail than pass each of the three grades. The last chart shows how many students study at least 5 hours a week (`StudyHard` = 1).

    ### Computing P(G1pass | StudyHard)

    The cross-tabulation counts every combination of `G1pass` and `StudyHard`. The margins are the row and column totals.
    """)
    return


@app.cell
def _(data, pd):
    c1 = pd.crosstab(data.G1pass, data.StudyHard, margins=True)
    c1
    return (c1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the table before using the formula. With the original 60% cutoff, 395 students are in the file, 92 of them study hard, and 48 of those 92 also pass G1.

    \[
    P(G1pass \mid StudyHard) = \frac{P(G1pass \cap StudyHard)}{P(StudyHard)}
    \]
    """)
    return


@app.cell
def _():
    print("P(G1pass and StudyHard) = %.2f" % (48 / 395))
    print("P(StudyHard) = %.2f" % (92 / 395))
    print("P(G1pass | StudyHard) = %.2f" % (48 / 92))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same ratio can be read from the table: the cell where `G1pass` is 1 and `StudyHard` is 1, divided by the `StudyHard` = 1 column total.
    """)
    return


@app.cell
def _(c1):
    # Reading the probability from the cross-tabulation
    print("P(G1pass | StudyHard) = %.2f" % (c1[1][1] / c1[1]["All"]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Filtering the column gives the same conditional distribution. Among students with `StudyHard` equal to 1, `value_counts(normalize=True)` reports the share who passed and the share who did not.
    """)
    return


@app.cell
def _(data):
    data.G1pass[data.StudyHard == 1].value_counts(normalize=True)
    return


@app.cell
def _(data):
    print(
        "P(G1pass | StudyHard) = %.2f"
        % (data.G1pass[data.StudyHard == 1].value_counts(normalize=True)[1])
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same filter produces the conditional passing rates for G2 and G3.
    """)
    return


@app.cell
def _(data):
    print(
        "P(G2pass | StudyHard) = %.2f"
        % (data.G2pass[data.StudyHard == 1].value_counts(normalize=True)[1])
    )
    print(
        "P(G3pass | StudyHard) = %.2f"
        % (data.G3pass[data.StudyHard == 1].value_counts(normalize=True)[1])
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 2. Passing, given internet access at home

    The question is now: what is the probability of a grade of at least 60%, given that the student has internet at home?

    The chart counts `yes` and `no` in the `internet` column.
    """)
    return


@app.cell
def _(data, plt):
    _figure, _axes = plt.subplots()
    data.internet.value_counts().plot(
        kind="bar", rot=True, title="Internet access at home", ax=_axes
    )
    _figure
    return


@app.cell
def _(data):
    print(
        "P(G1pass | internet) = %.2f"
        % (data.G1pass[data.internet == "yes"].value_counts(normalize=True)[1])
    )
    print(
        "P(G2pass | internet) = %.2f"
        % (data.G2pass[data.internet == "yes"].value_counts(normalize=True)[1])
    )
    print(
        "P(G3pass | internet) = %.2f"
        % (data.G3pass[data.internet == "yes"].value_counts(normalize=True)[1])
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 3. The law of total probability

    `studytime` has four values:

    - 1: less than 2 hours
    - 2: 2 to 5 hours
    - 3: 5 to 10 hours
    - 4: more than 10 hours

    The law of total probability splits P(G3pass) across those four groups:

    \[
    \begin{align*}
    P(G3pass) &= P(G3pass \mid studytime=1)\,P(studytime=1) \\
    &+ P(G3pass \mid studytime=2)\,P(studytime=2) \\
    &+ P(G3pass \mid studytime=3)\,P(studytime=3) \\
    &+ P(G3pass \mid studytime=4)\,P(studytime=4)
    \end{align*}
    \]
    """)
    return


@app.cell
def _(data, plt):
    _figure, _axes = plt.subplots()
    data.studytime.value_counts().plot(
        kind="bar", rot=True, title="Weekly study time", ax=_axes
    )
    _figure
    return


@app.cell
def _(data):
    # Counts of each study-time value
    data.studytime.value_counts()
    return


@app.cell
def _(data):
    # Share of students in each study-time value
    data.studytime.value_counts(normalize=True)
    return


@app.cell
def _(data):
    st1 = data.studytime.value_counts(normalize=True)[1]
    st2 = data.studytime.value_counts(normalize=True)[2]
    st3 = data.studytime.value_counts(normalize=True)[3]
    st4 = data.studytime.value_counts(normalize=True)[4]
    print("P(studytime = 1) = %.2f" % (st1))
    print("P(studytime = 2) = %.2f" % (st2))
    print("P(studytime = 3) = %.2f" % (st3))
    print("P(studytime = 4) = %.2f" % (st4))
    return st1, st2, st3, st4


@app.cell
def _(data):
    p3givenst1 = data.G3pass[data.studytime == 1].value_counts(normalize=True)[1]
    p3givenst2 = data.G3pass[data.studytime == 2].value_counts(normalize=True)[1]
    p3givenst3 = data.G3pass[data.studytime == 3].value_counts(normalize=True)[1]
    p3givenst4 = data.G3pass[data.studytime == 4].value_counts(normalize=True)[1]
    print("P(G3pass | studytime = 1) = %.2f" % (p3givenst1))
    print("P(G3pass | studytime = 2) = %.2f" % (p3givenst2))
    print("P(G3pass | studytime = 3) = %.2f" % (p3givenst3))
    print("P(G3pass | studytime = 4) = %.2f" % (p3givenst4))
    return p3givenst1, p3givenst2, p3givenst3, p3givenst4


@app.cell
def _(p3givenst1, p3givenst2, p3givenst3, p3givenst4, st1, st2, st3, st4):
    pG3pass = (
        p3givenst1 * st1
        + p3givenst2 * st2
        + p3givenst3 * st3
        + p3givenst4 * st4
    )
    print(
        "P(G3pass) = (%.2f)*(%.2f) + (%.2f)*(%.2f) + (%.2f)*(%.2f) + (%.2f)*(%.2f) = %.2f"
        % (
            p3givenst1,
            st1,
            p3givenst2,
            st2,
            p3givenst3,
            st3,
            p3givenst4,
            st4,
            pG3pass,
        )
    )
    return (pG3pass,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 4. Bayes' rule

    A student is chosen at random and has passed G3. Two follow-up questions use Bayes' rule:

    - What is the probability that this student studied 2 to 5 hours a week (`studytime` = 2)?
    - What is the probability that this student studied 5 to 10 hours a week (`studytime` = 3)?

    \[
    P(studytime=2 \mid G3pass)
    = \frac{P(G3pass \mid studytime=2)\,P(studytime=2)}{P(G3pass)}
    \]

    The denominator is the total probability computed above.
    """)
    return


@app.cell
def _(p3givenst2, pG3pass, st2):
    print(
        "P(studytime=2 | G3pass) = P(G3pass | studytime=2) * P(studytime=2) / P(G3pass) = %.2f"
        % (p3givenst2 * st2 / pG3pass)
    )
    return


@app.cell
def _(p3givenst3, pG3pass, st3):
    print(
        "P(studytime=3 | G3pass) = P(G3pass | studytime=3) * P(studytime=3) / P(G3pass) = %.2f"
        % (p3givenst3 * st3 / pG3pass)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `chosen_studytime` to 1, 2, 3, or 4 and run the cell. The result is the probability of that study-time group, given that the student passed G3.
    """)
    return


@app.cell
def _(data, pG3pass):
    chosen_studytime = 4
    p_pass_given_group = data.G3pass[data.studytime == chosen_studytime].value_counts(
        normalize=True
    )[1]
    p_group = data.studytime.value_counts(normalize=True)[chosen_studytime]
    print(
        "P(studytime=%i | G3pass) = %.2f"
        % (chosen_studytime, p_pass_given_group * p_group / pG3pass)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - A conditional probability is a count in a subgroup divided by the size of that subgroup. The cross-tabulation, the column filter, and the formula agree.
    - Students who study at least 5 hours a week pass G1 more often than the class as a whole. Internet access at home is a second condition worth comparing.
    - The law of total probability rebuilds P(G3pass) by weighting each study-time group's passing rate by the size of that group.
    - Bayes' rule reverses the question: start from the students who passed, and find how likely each study-time group is.

    ## References

    - Unpingco, J. (2019) *Python for Probability, Statistics, and Machine Learning*, USA: Springer, chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
