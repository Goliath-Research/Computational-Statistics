# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "scipy",
# ]
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
    import scipy.stats as st

    sns.set_style("whitegrid")
    return mo, np, pd, plt, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bootstrap Confidence Intervals

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Explain why a bootstrap sample can repeat some observations and omit others.
    - Distinguish the original observations from bootstrap values of a statistic.
    - Interpret a percentile bootstrap confidence interval.
    - Calculate bootstrap intervals for center, spread, and shape using Python.
    - Explain why bootstrapping cannot correct an unrepresentative sample.
    - Apply percentile bootstrap intervals to the population mean of student grades.

    ## 1. Resampling the sample you have
    Suppose we collect one sample and calculate its mean. A different sample would usually give a different mean, but collecting many new samples may be expensive or impractical.

    The **bootstrap** uses the observations we already have to approximate how a statistic varies from sample to sample.

    A **bootstrap sample** is drawn from the original observations **with replacement**. After selecting an observation, we leave it available to be selected again. This means some observations may appear more than once and others may not appear at all.

    Each bootstrap sample has the same number of observations as the original sample. We calculate the statistic from each resample. For example, 4,000 bootstrap samples give 4,000 means, called **bootstrap replicates** of the mean.

    ### From bootstrap replicates to an interval
    A **percentile bootstrap interval** uses two percentiles of those calculated statistics. For a 95% interval, the endpoints are the 2.5th and 97.5th percentiles of the bootstrap replicates. The interval therefore includes the middle 95% of their values.

    These percentiles come from the **bootstrap statistics**, not directly from the original observations. An interval for the mean estimates the population mean; it is not a range intended to contain 95% of individual observations.

    As with other confidence intervals, 95% confidence describes the method across repeated original samples. Bootstrap coverage is approximate, and the actual percentage can differ from 95%.

    ### What does the method assume?
    We do not have to choose a normal or uniform population model to perform this bootstrap. However, the ordinary method used here assumes independent observations from the same population and a sample that represents that population reasonably well.

    A very small sample may miss important population features. Resampling it cannot create those missing features. More resamples reduce randomness in the computed endpoints; they do not add new information about the population.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    The observed sample is [3, 5, 8].
    1. How many observations are in each bootstrap sample?
    2. Can 5 appear more than once in one bootstrap sample?
    3. Can 4 appear in a bootstrap sample?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Size:** 3, the same size as the observed sample.
2. **Repeats:** Yes. Sampling with replacement can draw the same observation more than once.
3. **New values:** No. A bootstrap sample uses only values that were observed.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Ages of community visitors
    We will simulate the ages of 1,000 adult visitors to a community center. For this teaching example, ages are generated between 18 and 85 using a uniform distribution.

    The variable `ages` stores the observations. The seed `2026` reproduces the same sample whenever the cell is rerun. We display only the first 10 ages; calculations use all 1,000 unrounded values.

    The uniform distribution is used only to generate the example. The bootstrap calculations will resample the observed ages without using the population formula.

    In the histogram, each bar counts ages in a range. The two lines mark the sample mean and median.
    """)
    return


@app.cell
def _(np):
    ages = np.random.default_rng(2026).uniform(18, 85, size=1_000)
    print('First 10 ages:', ages[:10].round(0))
    return (ages,)


@app.cell
def _(ages, np, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.hist(ages, bins=15, color="#4C78A8", edgecolor="white")
    _ax.axvline(ages.mean(), color="#54A24B", linewidth=2, label=f"Mean {ages.mean():.1f}")
    _ax.axvline(np.median(ages), color="#F58518", linewidth=2, label=f"Median {np.median(ages):.1f}")
    _ax.set(title="Simulated ages of community visitors", xlabel="Age", ylabel="Count")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generating bootstrap samples
    We will define a general function, `generate_samples_b`, and reuse it throughout the lesson.

    - `sample_data` contains the original observations.
    - `num_samples` is the number of bootstrap samples; we use 4,000 by default.
    - `seed` makes the resampling reproducible.

    The function returns a DataFrame where **each column is one bootstrap sample**. Each column has as many observations as the original sample. `replace=True` allows an observation to be selected more than once.
    """)
    return


@app.cell
def _(np, pd):
    def generate_samples_b(sample_data, num_samples=4_000, seed=2026):
        """Return bootstrap samples as columns of a DataFrame."""
        sample_size = len(sample_data)
        rng = np.random.default_rng(seed)
        samples = rng.choice(sample_data, size=(sample_size, num_samples), replace=True)
        columns = [f"S{k}" for k in range(num_samples)]
        return pd.DataFrame(samples, columns=columns)
    return (generate_samples_b,)


@app.cell
def _(ages, generate_samples_b, mo):
    age_samples = generate_samples_b(ages, seed=2026)
    print("Bootstrap sample table shape:", age_samples.shape)
    mo.Html(age_samples.iloc[:5, :5].round(2).to_html(index=False, border=0))
    return (age_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The complete table has **1,000 rows and 4,000 columns**. Each column is a resample of the 1,000 observed ages. The HTML preview shows only the first five rows of the first five samples.

    ### Calculating a percentile interval
    We now define `confidence_interval`, which can be used for any of the statistics in this lesson.

    Its input is a collection of **bootstrap statistics**, such as the 4,000 sample means, rather than the original observations. The argument `confidence` is a percentage: use `95` for a 95% interval.

    For 95% confidence, the function selects the 2.5th and 97.5th percentiles. It returns the lower and upper endpoints. For a different confidence level, the same function selects the corresponding percentiles.
    """)
    return


@app.cell
def _(np):
    def confidence_interval(sample_distribution, confidence=95):
        """Return a percentile interval from bootstrap statistics."""
        alpha = 100 - confidence
        low, high = np.percentile(sample_distribution, [alpha / 2, 100 - alpha / 2])
        return low, high
    return (confidence_interval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bootstrap interval for mean age
    `age_samples.mean()` calculates the mean of each column, giving 4,000 bootstrap means. We pass those means to `confidence_interval` to calculate a 95% interval for the population mean age.

    The standard deviation of the bootstrap means estimates the **standard error** of the sample mean. We calculate it with `ddof=1`.
    """)
    return


@app.cell
def _(age_samples, ages, confidence_interval):
    mean_replicates = age_samples.mean().to_numpy()
    mean_interval = confidence_interval(mean_replicates, 95)
    print("Sample mean age (years):", ages.mean().round(3))
    print(f"95% interval for mean age (years): {mean_interval[0]:.3f} to {mean_interval[1]:.3f}")
    print("Bootstrap standard error (years):", mean_replicates.std(ddof=1).round(3))
    return mean_interval, mean_replicates


@app.cell(hide_code=True)
def _(mean_interval, mo):
    mo.md(f"""
    The 95% percentile interval runs from **{mean_interval[0]:.3f} to {mean_interval[1]:.3f} years**. 
    
    It estimates the mean age of the population of adult visitors represented by this sample. 
    
    It does not describe the ages of 95% of individual visitors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same type of interval with SciPy
    We have calculated a percentile bootstrap interval using our own functions. SciPy offers a library alternative, `st.bootstrap`. Here is one example using the same visitor ages and the same confidence level.

    - `(ages,)` supplies the data in a one-item tuple.
    - `np.mean` specifies the statistic.
    - `confidence_level=0.95` requests 95% confidence.
    - `method="percentile"` selects the method we have learned.
    - `n_resamples=4_000` requests 4,000 resamples.
    - `batch=100` limits how many resamples are processed at once.
    - `rng` supplies a seeded random generator.

    The returned `confidence_interval` contains the lower and upper endpoints. We will continue using our general functions in the remaining examples and exercises.
    """)
    return


@app.cell
def _(ages, np, st):
    scipy_mean_result = st.bootstrap(
        (ages,), np.mean, confidence_level=0.95, method="percentile",
        n_resamples=4_000, batch=100,
        rng=np.random.default_rng(2026)
    )
    print(f"SciPy 95% mean interval (years): {scipy_mean_result.confidence_interval.low:.3f} to {scipy_mean_result.confidence_interval.high:.3f}")
    return (scipy_mean_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both approaches estimate the population mean age using a percentile bootstrap interval. Their endpoints need not match exactly: the implementations arrange the random draws differently, even with the same seed. Small differences from finite resampling are expected.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### New data: delivery times
    An online store records delivery times for a random sample of 80 orders. Times are measured in days. Deliveries can occasionally take much longer than usual, so we simulate positive data with a longer right tail.

    The next cell stores the times in `delivery_times`, using seed `2028`, and prints only the first 10. Assume orders are independent and the sample represents the store's deliveries.
    """)
    return


@app.cell
def _(np):
    delivery_times = np.random.default_rng(2028).lognormal(mean=1.2, sigma=0.5, size=80)
    print("First 10 delivery times (days):", delivery_times[:10].round(2))
    return (delivery_times,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate a 95% percentile bootstrap confidence interval for the population mean delivery time.
    """)
    return


@app.cell
def _(confidence_interval, delivery_times, generate_samples_b):
    student_delivery_mean_interval = None
    print("Mean interval (days):", student_delivery_mean_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_delivery_samples = generate_samples_b(delivery_times, seed=2029)
student_delivery_mean_interval = confidence_interval(student_delivery_samples.mean().to_numpy(), 95)
print("Mean interval (days):", student_delivery_mean_interval)
```

This interval estimates the population's average delivery time, in days. It is not a range containing 95% of individual delivery times.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mean and median
    The **mean** is the arithmetic average. The **median** is the middle value after sorting the observations. Both describe center, but they respond differently to the data.

    We already calculated 4,000 bootstrap means in Section 2. The next cell calculates the median of each column in `age_samples`, giving 4,000 bootstrap medians from the same resamples used for the means. The graph shows how these statistics vary across resamples. Its horizontal axis shows mean or median age, rather than individual ages.

    A wider bootstrap distribution indicates greater variability of that estimate. For this uniform population, the median generally varies more than the mean. This is not a rule for every population.

    The table reports 90%, 95%, and 99% percentile intervals for each statistic. For the same bootstrap distribution, higher confidence gives a wider interval.
    """)
    return


@app.cell
def _(age_samples, confidence_interval):
    median_replicates = age_samples.median().to_numpy()
    median_interval = confidence_interval(median_replicates, 95)
    print(f"95% interval for median age (years): {median_interval[0]:.3f} to {median_interval[1]:.3f}")
    print("Bootstrap standard error (years):", median_replicates.std(ddof=1).round(3))
    return median_interval, median_replicates


@app.cell
def _(mean_replicates, median_replicates, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.hist(mean_replicates, bins=40, color="#F58518", alpha=0.55, label="Bootstrap means")
    _ax.hist(median_replicates, bins=40, color="#54A24B", alpha=0.45, label="Bootstrap medians")
    _ax.set(title="Bootstrap distributions", xlabel="Statistic", ylabel="Count")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(confidence_interval, mean_replicates, median_replicates, mo, pd):
    _rows = []
    for _name, _replicates in (("Mean", mean_replicates), ("Median", median_replicates)):
        for _level in (90, 95, 99):
            _low, _high = confidence_interval(_replicates, _level)
            _rows.append({"Statistic": _name, "Confidence": f"{_level}%", "Low": _low, "High": _high})
    center_intervals = pd.DataFrame(_rows)
    mo.Html(center_intervals.round(2).to_html(index=False, border=0, col_space=110))
    return (center_intervals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the mean intervals on a graph
    Each panel below shows the **same bootstrap means**. The red lines mark the endpoints for the confidence level shown above that panel.

    Compare the distance between the red lines: the 99% interval is wider than the 95% and 90% intervals. Higher confidence includes a larger part of the same bootstrap distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `graph_confidence_interval` plots any collection of bootstrap statistics and marks the percentile interval. It reuses `confidence_interval` for the endpoints.

    `statistic_name` labels the graph. The optional `ax` lets us place several graphs in one figure; when it is omitted, the function creates its own figure. It returns the figure so marimo can display it.
    """)
    return


@app.cell
def _(confidence_interval, plt):
    def graph_confidence_interval(sample_distribution, confidence=95, statistic_name="Mean", ax=None):
        """Plot bootstrap statistics and their percentile interval."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.4))
        else:
            fig = ax.figure
        low, high = confidence_interval(sample_distribution, confidence)
        ax.hist(sample_distribution, bins=30, color="#4C78A8", edgecolor="white")
        ax.axvline(low, color="#E45756", linewidth=1.5)
        ax.axvline(high, color="#E45756", linewidth=1.5)
        ax.set(title=f"{statistic_name}: {confidence}% CI ({low:.2f}, {high:.2f})",
               xlabel=statistic_name, ylabel="Count")
        fig.tight_layout()
        plt.close(fig)
        return fig
    return (graph_confidence_interval,)


@app.cell
def _(graph_confidence_interval, mean_replicates, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3))
    for _ax, _level in zip(_axes, (90, 95, 99)):
        graph_confidence_interval(mean_replicates, _level, "Mean age (years)", ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate a 95% percentile bootstrap confidence interval for the population median delivery time. What does this interval estimate?
    """)
    return


@app.cell
def _(confidence_interval, delivery_times, generate_samples_b):
    student_delivery_median_interval = None
    print("Median interval (days):", student_delivery_median_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_delivery_samples = generate_samples_b(delivery_times, seed=2030)
student_delivery_median_interval = confidence_interval(student_delivery_samples.median().to_numpy(), 95)
print("Median interval (days):", student_delivery_median_interval)
```

The interval estimates the population median: the time separating the shorter half of deliveries from the longer half. It is not a range for the times of 95% of individual deliveries.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Spread and shape
    We return to the ages of community visitors. We will calculate one statistic at a time, explain what it tells us, and obtain its 95% bootstrap interval.

    Each column of `age_samples` is one bootstrap sample. We calculate each statistic column by column, then pass the resulting values to our general `confidence_interval` function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variance
    Variance measures the spread of ages around their mean. The sample variance is written $s^2$ and is measured in **years squared**.

    We use `ddof=1`, so the sample variance divides by $n-1$. `age_samples.var(ddof=1)` calculates one variance per column. Our `confidence_interval` function then gives the interval for the population variance.
    """)
    return


@app.cell
def _(age_samples, ages, confidence_interval, np, pd):
    variance_replicates = age_samples.var(ddof=1).to_numpy()
    _interval = confidence_interval(variance_replicates, 95)
    print("Sample variance (years squared):", round(ages.var(ddof=1), 3))
    print(f"95% interval (years squared): {_interval[0]:.3f} to {_interval[1]:.3f}")
    return (variance_replicates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standard deviation
    Standard deviation is the square root of variance. Unlike variance, it has the same units as the observations: **years**. A larger standard deviation means the ages are more spread out.

    `age_samples.std(ddof=1)` calculates one standard deviation per column. We reuse the same age resamples and our interval function.
    """)
    return


@app.cell
def _(age_samples, ages, confidence_interval, np, pd):
    sd_replicates = age_samples.std(ddof=1).to_numpy()
    _interval = confidence_interval(sd_replicates, 95)
    print("Sample standard deviation (years):", round(ages.std(ddof=1), 3))
    print(f"95% interval (years): {_interval[0]:.3f} to {_interval[1]:.3f}")
    return (sd_replicates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interquartile range
    The **interquartile range (IQR)** is the 75th percentile minus the 25th percentile. It measures the spread of the middle half of the ages and is expressed in **years**.

    We subtract the 25th percentile from the 75th percentile in each column of `age_samples`. The interval below estimates the population IQR, rather than giving the lower and upper quartiles themselves.
    """)
    return


@app.cell
def _(age_samples, ages, confidence_interval, np, pd):
    iqr_replicates = (age_samples.quantile(0.75) - age_samples.quantile(0.25)).to_numpy()
    _interval = confidence_interval(iqr_replicates, 95)
    print("Sample interquartile range (years):", round(np.percentile(ages, 75) - np.percentile(ages, 25), 3))
    print(f"95% interval (years): {_interval[0]:.3f} to {_interval[1]:.3f}")
    return (iqr_replicates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Skewness
    Skewness describes asymmetry. A positive value indicates a longer right tail, and a negative value indicates a longer left tail. A symmetric population has skewness zero, although its sample skewness may differ from zero.

    Pandas `skew()` calculates the adjusted sample skewness for each column. Skewness has **no measurement units**. The bootstrap interval expresses uncertainty about population skewness. Including zero does not establish that the population is symmetric.
    """)
    return


@app.cell
def _(age_samples, ages, confidence_interval, np, pd):
    skewness_replicates = age_samples.skew().to_numpy()
    _interval = confidence_interval(skewness_replicates, 95)
    print("Sample skewness (unitless):", round(pd.Series(ages).skew(), 3))
    print(f"95% interval (unitless): {_interval[0]:.3f} to {_interval[1]:.3f}")
    return (skewness_replicates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Excess kurtosis
    Excess kurtosis measures the standardized fourth moment minus 3. It helps describe tail weight; it is not simply a measure of how tall the histogram's peak is.

    A normal population has excess kurtosis zero. This simulated uniform population has excess kurtosis $-1.2$. Pandas `kurt()` calculates adjusted sample excess kurtosis for each column. The result has **no measurement units**.

    The interval below estimates population excess kurtosis without using the known uniform-population value in its calculation.
    """)
    return


@app.cell
def _(age_samples, ages, confidence_interval, np, pd):
    kurtosis_replicates = age_samples.kurt().to_numpy()
    _interval = confidence_interval(kurtosis_replicates, 95)
    print("Sample excess kurtosis (unitless):", round(pd.Series(ages).kurt(), 3))
    print(f"95% interval (unitless): {_interval[0]:.3f} to {_interval[1]:.3f}")
    return (kurtosis_replicates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing the results
    We have now calculated each statistic separately. The following cells collect the results into an HTML table and graphs for comparison. The table includes 90%, 95%, and 99% intervals, reusing the resamples already calculated. The red lines on each graph mark the 95% endpoints.
    """)
    return


@app.cell
def _(iqr_replicates, kurtosis_replicates, sd_replicates, skewness_replicates, variance_replicates):
    spread_replicates = {
        "Variance": variance_replicates, "Standard deviation": sd_replicates,
        "Interquartile range": iqr_replicates, "Skewness": skewness_replicates,
        "Excess kurtosis": kurtosis_replicates,
    }
    return (spread_replicates,)


@app.cell
def _(confidence_interval, mo, pd, spread_replicates):
    _rows = []
    for _name, _values in spread_replicates.items():
        for _level in (90, 95, 99):
            _low, _high = confidence_interval(_values, _level)
            _rows.append({"Statistic": _name, "Confidence": f"{_level}%", "Low": _low, "High": _high})
    spread_intervals = pd.DataFrame(_rows)
    mo.Html(spread_intervals.round(3).to_html(index=False, border=0, col_space=140))
    return (spread_intervals,)


@app.cell
def _(graph_confidence_interval, plt, spread_replicates):
    _fig, _axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for _ax, (_name, _values) in zip(_axes.ravel(), spread_replicates.items()):
        graph_confidence_interval(_values, 95, _name, ax=_ax)
        _ax.set_title(_ax.get_title(), fontsize=9)
    _axes.ravel()[-1].axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which two percentiles form a 90% percentile interval?
    2. Does a bootstrap interval for the median use a $t$ critical value?
    3. The age population in this simulation is uniform. Did the bootstrap calculation use that fact?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Percentiles:** The 5th and the 95th.
2. **Median:** No. The interval reads percentiles of the bootstrap medians.
3. **Uniform formula:** No. The calculation resamples the observed ages and takes percentiles of the resulting statistics. The uniform model was used only to create the sample.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Using `delivery_times`, calculate 95% percentile bootstrap confidence intervals for the population variance and standard deviation. Report the units of each interval.
    """)
    return


@app.cell
def _(confidence_interval, delivery_times, generate_samples_b):
    student_delivery_variance_interval = None
    student_delivery_sd_interval = None
    print("Variance interval (days squared):", student_delivery_variance_interval)
    print("Standard deviation interval (days):", student_delivery_sd_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_delivery_samples = generate_samples_b(delivery_times, seed=2033)
student_delivery_variance_interval = confidence_interval(student_delivery_samples.var(ddof=1).to_numpy(), 95)
student_delivery_sd_interval = confidence_interval(student_delivery_samples.std(ddof=1).to_numpy(), 95)
print("Variance interval (days squared):", student_delivery_variance_interval)
print("Standard deviation interval (days):", student_delivery_sd_interval)
```

The same delivery resamples are used for both statistics. Pandas calculates the variance and standard deviation of each column with `ddof=1`; our `confidence_interval` function supplies the endpoints. Variance is measured in days squared; standard deviation is measured in days.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Bootstrap intervals for student grades
    We will now apply the bootstrap to recorded student grades. Each row in `student-mat.csv` describes one student. The three columns used here contain grades on a 0–20 scale:

    - **G1:** first-period grade.
    - **G2:** second-period grade.
    - **G3:** final grade.

    The file uses semicolons to separate columns. Place it in the `../data` folder relative to the working directory. The next cell loads the data and displays the first five rows of grades as an HTML table.

    We treat students as independent observations. Our intervals estimate mean grades for the population represented by this sample, rather than automatically applying to all students.
    """)
    return


@app.cell
def _(mo, pd):
    student_file = pd.read_csv("../data/student-mat.csv", sep=";")
    grades = student_file[["G1", "G2", "G3"]].copy()
    print("Number of students:", len(grades))
    mo.Html(grades.head().to_html(index=False, border=0))
    return grades, student_file


@app.cell
def _(grades, plt, sns):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    for _name, _color in zip(["G1", "G2", "G3"], ["#E45756", "#4C78A8", "#54A24B"]):
        sns.kdeplot(grades[_name], ax=_ax, fill=True, alpha=0.25, color=_color, label=_name)
    _ax.set(title="Grade distributions", xlabel="Grade")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bootstrap interval for the first-period mean grade
    We will use `G1` as our worked example. The observed mean is our point estimate of the population's mean first-period grade.

    The next cell uses `generate_samples_b` to draw 4,000 samples with replacement from the observed `G1` grades. It calculates a mean for each sample, then uses `confidence_interval` to find the 95% percentile interval. Each resample has as many grades as there are students in the original sample.

    The calculation does not draw new grades from a normal distribution. It resamples the grades already observed in the file.
    """)
    return


@app.cell
def _(confidence_interval, generate_samples_b, grades):
    g1_grades = grades["G1"].to_numpy()
    g1_samples = generate_samples_b(g1_grades, seed=2034)
    g1_replicates = g1_samples.mean().to_numpy()
    g1_interval = confidence_interval(g1_replicates, 95)
    print(f"Sample mean G1 grade: {g1_grades.mean():.3f}")
    print(f"95% bootstrap interval: {g1_interval[0]:.3f} to {g1_interval[1]:.3f}")
    return g1_grades, g1_interval, g1_replicates


@app.cell(hide_code=True)
def _(g1_interval, mo):
    mo.md(f"""
    The 95% percentile bootstrap interval runs from **{g1_interval[0]:.3f} to {g1_interval[1]:.3f} grade points**.

    This interval estimates the **population mean first-period grade**. It is not a range containing 95% of individual students' grades.

    In the graph below, each value in the histogram is a **mean from one bootstrap sample**. The red lines mark the interval's endpoints. Compare this with the earlier grade-distribution graph, which shows individual grades.
    """)
    return


@app.cell
def _(g1_replicates, graph_confidence_interval):
    graph_confidence_interval(g1_replicates, 95, "Mean G1 grade")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Calculate a 95% percentile bootstrap confidence interval for the population mean second-period grade (`G2`).
    2. Calculate a 95% percentile bootstrap confidence interval for the population mean final grade (`G3`).
    3. What does each interval estimate?
    """)
    return


@app.cell
def _(confidence_interval, generate_samples_b, grades):
    student_g2_interval = None
    student_g3_interval = None
    print("G2 mean interval:", student_g2_interval)
    print("G3 mean interval:", student_g3_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_g2_samples = generate_samples_b(grades["G2"], seed=2035)
student_g2_interval = confidence_interval(student_g2_samples.mean().to_numpy(), 95)
student_g3_samples = generate_samples_b(grades["G3"], seed=2036)
student_g3_interval = confidence_interval(student_g3_samples.mean().to_numpy(), 95)
print(f"G2 mean interval: {student_g2_interval[0]:.3f} to {student_g2_interval[1]:.3f}")
print(f"G3 mean interval: {student_g3_interval[0]:.3f} to {student_g3_interval[1]:.3f}")
```

The first interval estimates the population mean second-period grade. The second estimates the population mean final grade. Both are measured in grade points, and neither is an interval for individual grades.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A bootstrap sample draws from the original observations **with replacement** and has the same sample size.
    - We recalculate the statistic for every resample. The resulting bootstrap statistics approximate its variation across samples.
    - A 95% percentile interval uses the 2.5th and 97.5th percentiles of the **bootstrap statistics**, not the original observations.
    - The method can estimate uncertainty for means, medians, and many measures of spread and shape without specifying a normal population model.
    - The ordinary bootstrap still depends on independent observations and a reasonably representative sample. It cannot recover population features missing from the sample.
    - Our general functions generate bootstrap samples, calculate percentile intervals, and plot the bootstrap statistics. The same functions work with ages, delivery times, and grades.
    - More resamples make the endpoints more stable. More original observations provide more information about the population; these are different improvements.

    ## Check your understanding
    1. A sample has 40 rows. How many rows does one bootstrap sample have?
    2. What does “with replacement” mean?
    3. Which percentiles form a 99% percentile bootstrap interval?
    4. Does a 95% bootstrap interval for the mean contain 95% of individual observations?
    5. Can increasing the number of resamples correct a sample that excludes an important part of the population?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Rows:** 40, the same number as in the original sample.
2. **Replacement:** After selecting a row, it remains available to be selected again. A resample may repeat rows and omit others.
3. **99% endpoints:** The 0.5th and 99.5th percentiles of the bootstrap statistics.
4. **Individual observations:** No. The interval estimates the population mean; it is not an interval for individual observations.
5. **Representativeness:** No. Resampling cannot supply information missing from the original sample.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer, Chapter 18.
    - [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), for the grade file.
    - Kim, A. *Comprehensive Confidence Intervals for Python Developers*. https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers
    """)
    return


if __name__ == "__main__":
    app.run()
