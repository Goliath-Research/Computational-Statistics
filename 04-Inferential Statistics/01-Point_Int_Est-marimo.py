# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "statsmodels",
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
    import statsmodels.stats.api as sm

    sns.set_style("whitegrid")
    return mo, np, pd, plt, sm, sns, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point and Interval Estimation

    ## Learning goals
    By the end of this lesson, you should be able to:
    - Use a sample mean or proportion to estimate a population value.
    - Explain the difference between a point estimate and a confidence interval.
    - Distinguish standard deviation, standard error, and margin of error.
    - Calculate confidence intervals for a mean, a paired mean difference, and a variance.
    - Explain what a confidence level means using repeated samples.

    ## 1. Two kinds of estimate
    Suppose we want to know the average age of all registered voters. Surveying every voter would be difficult, so we collect a **sample** and calculate its average age.

    The average age of all voters is the **population mean**. The average age of the people in our sample is the **sample mean**. We use the sample mean to estimate the population mean.

    We can report an estimate in two ways:

    - **Point estimate:** one number. For example, “Our estimate of the average age is 45 years.”
    - **Interval estimate:** a range. For example, “Our 95% confidence interval for the average age is from 44 to 46 years.” These numbers are only an illustration.

    A range helps us express uncertainty because a different sample would usually give a different estimate. We will explain the meaning of “95% confidence” later in the lesson.

    An **estimator** is the calculation we use, such as taking the sample mean. An **estimate** is the result of that calculation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A sample of ages is [40, 42, 45, 48, 55].
    1. What is the point estimate of the mean?
    2. Is that number guaranteed to equal the population mean?
    3. What would an interval estimate add?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Point estimate:** (40 + 42 + 45 + 48 + 55) / 5 = 46.
2. **Equality:** No. A sample mean estimates the population mean. It does not have to equal it.
3. **Interval:** It adds a range to express uncertainty about the average age of the population, rather than reporting only 46. We will learn how to interpret its confidence level in Section 3.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A point estimate of mean age
    We will generate 1,000 example ages using a normal distribution with mean 45 and standard deviation 8. These are simulated data for learning, rather than ages collected from actual voters.

    In this example, we know the population mean is 45 because we chose it when generating the data. In a real survey, that mean would be unknown.

    The next cells generate the sample, draw a histogram, and summarize the data. The line on the histogram marks the sample mean. Compare this estimate with the known population mean of 45.

    The seed `2026` makes the random example reproducible: rerunning the cell gives the same ages. The displayed first 10 ages are rounded, but the calculations use the original values.
    """)
    return


@app.cell
def _(np):
    # A local seeded generator reproduces the sample when this cell is rerun.
    # These are illustrative normal measurements, not a realistic model of all voter ages.
    ages = np.random.default_rng(2026).normal(45, 8, size=1_000)
    print('First 10 ages:', ages[:10].round(0))
    return (ages,)


@app.cell
def _(ages, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.hist(ages, bins=20, color="#4C78A8", edgecolor="white")
    _ax.axvline(ages.mean(), color="#E45756", linewidth=2, label=f"Sample mean {ages.mean():.2f}")
    _ax.set(title="Simulated voter ages", xlabel="Age", ylabel="Count")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(ages, mo, pd):
    age_summary = pd.DataFrame({
        "Quantity": [
            "Sample size",
            "Sample mean",
            "Variance with divisor n",
            "Variance with divisor n - 1",
            "Standard deviation with divisor n",
            "Standard deviation with divisor n - 1",
        ],
        "Value": [
            len(ages),
            ages.mean(),
            ages.var(ddof=0),
            ages.var(ddof=1),
            ages.std(ddof=0),
            ages.std(ddof=1),
        ],
    })
    mo.Html(
        age_summary.round(2).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (age_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Understanding the sample summary
    The **mean** describes the center of the ages. The **variance** and **standard deviation** describe how spread out they are.

    To calculate variance, we subtract the sample mean from each age, square the differences, and add them. We then divide that total:

    - By $n$, the number of ages, to describe the variability in this sample.
    - By $n-1$, to obtain the usual sample variance used to estimate the population variance.

    NumPy selects the divisor with `ddof`:

    - `ages.var(ddof=0)` divides by $n$.
    - `ages.var(ddof=1)` divides by $n-1$.

    Why subtract 1? We used the same data to estimate the mean. This leaves $n-1$ independent deviations, called **degrees of freedom**. The correction makes the variance estimate unbiased: across many independent random samples, its average equals the population variance, provided that variance is finite. This property does not require a normal distribution.

    Here, the known population standard deviation is 8, so its variance is $8^2=64$. The variances calculated from our sample will usually differ from 64.

    Standard deviation is the square root of variance. It is expressed in years, while variance is expressed in years squared. Taking the square root of an unbiased variance estimate does not make the standard deviation estimate unbiased.

    ### Standard deviation, standard error, and sampling error
    These quantities answer different questions:

    - **Standard deviation:** How much do individual ages vary?
    - **Standard error of the mean:** How much would the sample mean vary if we collected new samples of the same size?
    - **Sampling error:** How far is this particular sample mean from the true population mean?

    We estimate the standard error using

    $$\text{Standard error}=\frac{s}{\sqrt n},$$

    where $s$ is the sample standard deviation calculated with `ddof=1`. The function `st.sem(ages)` performs this calculation.

    In our simulation, the sampling error is `ages.mean() - 45`. We can calculate it because we know the true mean. In a real survey, we usually cannot calculate the actual sampling error.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a code task, replace each `None` in the next cell and run it in the marimo editor.

    ### Try it yourself
    Using `ages`:
    1. Store the sample mean in `age_mean`.
    2. Store the standard deviation with divisor $n-1$ in `age_sd`.
    3. Store the proportion of ages below 50 in `proportion_under_50`.

    `(ages < 50)` creates Boolean values. Their mean is the fraction that are `True`, because `True` counts as 1 and `False` as 0.
    """)
    return


@app.cell
def _(ages):
    age_mean = None
    age_sd = None
    proportion_under_50 = None
    print("Mean:", age_mean)
    print("SD (n-1):", age_sd)
    print("Proportion under 50:", proportion_under_50)
    return age_mean, age_sd, proportion_under_50


@app.cell(hide_code=True)
def _(ages, mo):
    _mean = ages.mean()
    _sd = ages.std(ddof=1)
    _prop = float((ages < 50).mean())
    _answers = f"""
1. **Mean:** {_mean:.3f}.
2. **Standard deviation:** {_sd:.3f}, using `ddof=1`.
3. **Proportion:** {_prop:.3f}. For a normal population with mean 45 and standard deviation 8, the probability below 50 is about 0.73. The sample proportion estimates that probability.

```python
age_mean = ages.mean()
age_sd = ages.std(ddof=1)
proportion_under_50 = (ages < 50).mean()
```
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Confidence interval for a mean
    The sample mean is our point estimate. Now we will add a **margin of error** to form a confidence interval:

    $$\text{Sample mean}\;\pm\;\text{margin of error}.$$

    The margin of error is the distance from the sample mean to either endpoint. For example, an estimate of 45 with a margin of error of 1 gives an interval from 44 to 46. It is not a guarantee that the actual estimation error is at most 1.

    ### What does 95% confidence mean?
    Imagine collecting many new samples in the same way and calculating a 95% confidence interval from each one. When the method's assumptions hold, about 95% of those intervals will contain the true population mean.

    The mean stays fixed, but the samples and their intervals change. The 95% describes how often the **method** succeeds across repeated samples. It does not mean there is a 95% probability that the fixed mean lies inside the particular interval we have already calculated.

    ### Calculating the interval
    We use a Student's $t$ interval because the population standard deviation is unknown:

    $$\bar x\pm t_{n-1,\,1-\alpha/2}\frac{s}{\sqrt n}.$$

    In this formula, $\bar x$ is the sample mean, $s/\sqrt n$ is its standard error, and $t$ is the **critical value**, a multiplier determined by the confidence level and the degrees of freedom. For a mean interval, the degrees of freedom are $n-1$.

    SciPy's `st.t.interval` and statsmodels' `tconfint_mean` calculate the endpoints for us. We use the formula to understand the interval; we do not need to calculate its critical value or assemble its endpoints ourselves.

    ### Reading the Python call and the table
    In `st.t.interval`, the first argument is the confidence level. The remaining arguments supply:

    - `len(ages) - 1`: the degrees of freedom.
    - `loc`: the sample mean, which is the interval's center.
    - `scale`: the standard error, calculated with `st.sem(ages)`.

    `Low` and `High` are the endpoints. `Contains_45` tells us whether the interval includes the known mean of 45. We can check this only because these data are simulated.

    For the same sample, 99% confidence requires a wider interval than 95% or 90% confidence. A wider range gives the method a greater chance of including the true mean.

    ### When can we use this method?
    The formula gives exact confidence levels for independent observations from a normal population. Independent means that one observation does not determine another.

    For large independent samples, the mean interval can also work approximately for other distributions with finite variance. Strong skewness or outliers may require more data. A large sample does not fix a biased survey or observations that depend on each other.
    """)
    return


@app.cell
def _(ages, mo, pd, st):
    _mean = ages.mean()
    # sem uses ddof=1 by default: sample SD divided by sqrt(sample size).
    _sem = st.sem(ages)
    _rows = []
    for _level in (0.90, 0.95, 0.99):
        # interval accepts the central confidence level, such as 0.95.
        _low, _high = st.t.interval(_level, len(ages) - 1, loc=_mean, scale=_sem)
        _rows.append({
            "Confidence": f"{_level:.0%}",
            "Low": _low,
            "High": _high,
            "Contains_45": _low <= 45 <= _high,
        })
    mean_intervals = pd.DataFrame(_rows)
    print(f"Sample mean = {_mean:.3f}; standard error = {_sem:.3f}.")
    mo.Html(
        mean_intervals.round(3).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (mean_intervals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Which of the three intervals is widest?
    2. What does `Contains_45` check, and why can this lesson compute it?
    3. Does "95% confidence" mean that the population mean has a 95% chance of sitting in this particular interval?
    """)
    return


@app.cell(hide_code=True)
def _(mean_intervals, mo):
    _widths = mean_intervals["High"] - mean_intervals["Low"]
    _widest = mean_intervals.loc[_widths.idxmax(), "Confidence"]
    _answers = rf"""
1. **Widest interval:** {_widest}. For the same data, greater confidence requires a wider range. In the formula, the critical value becomes larger.
2. **Known mean:** `Contains_45` asks whether the interval covers 45. The lesson can compute it because the simulation set the population mean to 45. The interval is centered on the sample mean.
3. **One interval:** No. If we collected many new samples and calculated a 95% interval each time, about 95% of the intervals would include the true mean. The particular interval we have already calculated either includes it or misses it.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same mean intervals using statsmodels
    We can calculate the same mean intervals with another library: **statsmodels**.

    `sm.DescrStatsW(ages)` creates a summary of our sample. We supply no weights, so each age contributes equally. Its method `tconfint_mean` calculates the interval.

    The two libraries ask for different inputs:

    - SciPy asks for the confidence level: `0.95` for 95% confidence.
    - statsmodels asks for `alpha`: `1 - 0.95 = 0.05` for the same confidence level.

    The table below compares the results. `Matches SciPy` is `True` when both libraries give the same endpoints, allowing for tiny differences from computer arithmetic.
    """)
    return


@app.cell
def _(ages, mo, np, pd, sm, st):
    _summary = sm.DescrStatsW(ages)
    _rows = []
    for _level in (0.90, 0.95, 0.99):
        _low, _high = _summary.tconfint_mean(alpha=1 - _level)
        _scipy = st.t.interval(_level, len(ages) - 1, loc=ages.mean(), scale=st.sem(ages))
        _rows.append({"Confidence": f"{_level:.0%}", "Low": _low, "High": _high,
                      "Matches SciPy": np.allclose((_low, _high), _scipy)})
    mo.Html(pd.DataFrame(_rows).round(3).to_html(index=False, border=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### New data: package weights
    A factory checks package weights to assess its filling process. We will simulate a random sample of 40 package weights, measured in grams, using a normal distribution with mean 500 grams and standard deviation 12 grams.

    The data are stored in `package_weights`. The seed `2027` reproduces the sample. The next cell displays only the first 10 weights, rounded to two decimals; all 40 unrounded values are used in the calculations.
    """)
    return


@app.cell
def _(np):
    package_weights = np.random.default_rng(2027).normal(500, 12, size=40)
    print("First 10 package weights (grams):", package_weights[:10].round(2))
    return (package_weights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    A factory checks the weights of a random sample of 40 packages. The weights, in grams, are stored in `package_weights`. Assume the package weights follow a normal distribution and the observations are independent.

    Calculate a 95% confidence interval for the mean weight of all packages produced by the factory.
    """)
    return


@app.cell
def _(package_weights, sm, st):
    student_mean_interval = None
    print("95% mean interval:", student_mean_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
**Using SciPy:**

```python
student_mean_interval = st.t.interval(
    0.95, len(package_weights) - 1,
    loc=package_weights.mean(), scale=st.sem(package_weights)
)
print("95% mean interval:", student_mean_interval)
```

**Using statsmodels:**

```python
student_mean_interval = sm.DescrStatsW(package_weights).tconfint_mean(alpha=0.05)
print("95% mean interval:", student_mean_interval)
```

Both give approximately (495.624, 503.850) grams.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Paired difference of means
    Sometimes two measurements belong together: for example, a measurement before and after a treatment on the same person. These are **paired observations**.

    For this example, assume that `x1` and `x2` contain measurements on the same 10 people, in the same order. The first value in `x1` goes with the first value in `x2`, and so on. Pairing comes from how data are collected; the arrays alone do not tell us whether measurements are paired.

    We first calculate one difference for each person:

    $$d_i=x_{2,i}-x_{1,i}.$$

    The first pair gives $151-148=3$. A positive difference means the second measurement is larger; a negative difference means it is smaller.

    We now have **10 differences**. We estimate the population mean difference using their average, $\bar d$, and calculate its interval in the same way as a single mean:

    $$\bar d\pm t_{n-1,1-\alpha/2}\frac{s_d}{\sqrt n}.$$

    Here, $s_d$ is the sample standard deviation of the differences and $n=10$ is the number of pairs. The degrees of freedom are $10-1=9$.

    ### Reading the result
    Zero means “no average difference.”

    - If the whole interval is positive, it supports a positive population mean difference.
    - If the whole interval is negative, it supports a negative population mean difference.
    - If the interval includes zero, the data do not rule out a zero mean difference at this confidence level. This does not prove that the mean difference is zero.

    For the matching two-sided $t$ test, an interval that excludes zero corresponds to rejecting a zero mean difference at $\alpha=1-\text{confidence level}$.

    ### Assumptions and plots
    Different people's pairs should be independent. With this small sample, we also assume the population of **differences** is approximately normal and has no strong outliers. We do not need to assume that both sets of measurements separately have normal distributions.

    The plots show smoothed views of the measurements and differences. They help us explore the data, but only 10 differences are not enough to establish normality from a plot. If the two samples were independent rather than paired, we would need a different interval.
    """)
    return


@app.cell
def _(mo, np, pd, st):
    x1 = np.array([148, 128, 69, 34, 155, 123, 101, 150, 139, 98])
    x2 = np.array([151, 146, 32, 70, 155, 142, 134, 157, 150, 130])
    # Subtract within each pair to preserve the matching.
    differences = x2 - x1
    _n = len(differences)
    _df = _n - 1
    _mean = differences.mean()
    _sd = differences.std(ddof=1)
    _rows = []
    for _level in (0.90, 0.95, 0.99):
        _low, _high = st.t.interval(_level, _df, loc=_mean, scale=st.sem(differences))
        _rows.append({
            "Confidence": f"{_level:.0%}",
            "Low": _low,
            "High": _high,
            "Contains_0": _low <= 0 <= _high,
        })
    difference_intervals = pd.DataFrame(_rows)
    mo.Html(
        difference_intervals.round(2).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return difference_intervals, differences, x1, x2


@app.cell
def _(differences, plt, sns, x1, x2):
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.3))
    sns.kdeplot(x=x1, ax=_axes[0], fill=True, color="#4C78A8", label="x1")
    sns.kdeplot(x=x2, ax=_axes[0], fill=True, color="#54A24B", label="x2")
    _axes[0].legend(frameon=False)
    _axes[0].set(title="Paired samples", xlabel="Value")
    sns.kdeplot(x=differences, ax=_axes[1], fill=True, color="#F58518")
    _axes[1].axvline(0, color="#E45756", linewidth=1.5)
    _axes[1].set(title="Paired differences", xlabel="x2 − x1")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Does the 95% interval for the mean difference contain 0?
    2. Why is the sample size 10 rather than 20?
    """)
    return


@app.cell(hide_code=True)
def _(difference_intervals, differences, mo):
    _row = difference_intervals.loc[difference_intervals["Confidence"] == "95%"].iloc[0]
    _contains = "Yes" if _row.Contains_0 else "No"
    _answers = rf"""
1. **Zero:** {_contains}. The 95% interval is ({_row.Low:.2f}, {_row.High:.2f}). The mean difference is {differences.mean():.2f}.
2. **Sample size:** There are 10 pairs. The interval uses the 10 differences, so $n = 10$ and the degrees of freedom are 9.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### New data: task completion times
    A software company measures how long the same 12 employees take to complete a task before and after training. Times are recorded in minutes. Each value in `times_before` is paired with the value in the same position in `times_after`.

    Assume the differences between employees are independent and the population of paired time differences is approximately normal. These are fixed example data, so no random seed is needed.
    """)
    return


@app.cell
def _(np):
    times_before = np.array([42, 38, 45, 40, 36, 48, 44, 39, 41, 46, 37, 43])
    times_after = np.array([37, 36, 39, 37, 35, 41, 40, 34, 39, 40, 34, 39])
    print("Before training (minutes):", times_before)
    print("After training (minutes):", times_after)
    return times_after, times_before


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Calculate a 95% confidence interval for the population mean change in task completion time, defined as after minus before. What does the interval tell you about the average change?
    """)
    return


@app.cell
def _(sm, st, times_after, times_before):
    student_time_differences = None
    student_difference_interval = None
    print("95% interval for the mean change (minutes):", student_difference_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
**Using SciPy:**

```python
student_time_differences = times_after - times_before
student_difference_interval = st.t.interval(
    0.95, len(student_time_differences) - 1,
    loc=student_time_differences.mean(), scale=st.sem(student_time_differences)
)
print("95% interval for the mean change (minutes):", student_difference_interval)
```

**Using statsmodels:**

```python
student_time_differences = times_after - times_before
student_difference_interval = sm.DescrStatsW(student_time_differences).tconfint_mean(alpha=0.05)
print("95% interval for the mean change (minutes):", student_difference_interval)
```

Both give approximately (-5.181, -2.819) minutes.

The whole interval is negative. The data support a decrease in average task completion time after training. This paired comparison alone does not establish that training caused the decrease.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Confidence interval for a normal variance
    A mean interval estimates the population's center. A **variance interval** estimates how spread out the population is.

    We will use the nine measurements below to estimate the population variance. The point estimate is the sample variance, $s^2$, calculated with `ddof=1`.

    For independent observations from a normal population, we use the **chi-square distribution** to calculate the interval:

    $$\left(\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}},\;\frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}}\right).$$

    In the Python example, `st.chi2.interval` supplies both endpoints for the scaled chi-square distribution. Taking their reciprocals in reverse order gives the variance interval. This follows the formula above without calculating tail probabilities or critical values separately.

    The scale is $1 / ((n-1)s^2)$. The lower variance endpoint is the reciprocal of the upper chi-square endpoint, and vice versa. `[::-1]` reverses the endpoint order in Python.

    The chi-square distribution is not symmetric, so the interval need not extend equally on each side of the sample variance. This method works because $(n-1)s^2/\sigma^2$ has a chi-square distribution under normal sampling; $\sigma^2$ denotes the population variance.

    Variance has squared measurement units. To obtain an interval for the population **standard deviation**, take the square root of each endpoint.

    **Important assumption:** this method requires independent measurements from a normal population. Departures from normality can substantially affect the confidence level of the variance interval.
    """)
    return


@app.cell
def _(mo, np, pd, st):
    measurements = np.array([8.01, 8.95, 9.65, 9.15, 8.06, 8.95, 8.03, 8.19, 8.03])
    _n = len(measurements)
    _df = _n - 1
    _s2 = measurements.var(ddof=1)
    _rows = []
    for _level in (0.90, 0.95, 0.99):
        # SciPy supplies the scaled chi-square interval; reciprocals give the variance interval.
        _chi_interval = st.chi2.interval(_level, _df, scale=1 / (_df * _s2))
        _low, _high = 1 / np.array(_chi_interval)[::-1]
        _rows.append({"Confidence": f"{_level:.0%}", "Low": _low, "High": _high})
    variance_intervals = pd.DataFrame(_rows)
    print(f"Sample variance s² = {_s2:.3f}, with divisor n - 1.")
    mo.Html(
        variance_intervals.round(3).to_html(index=False, border=0, col_space=110)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return measurements, variance_intervals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Is the variance interval centered on the sample variance?
    2. What distributional assumption does this interval use?
    """)
    return


@app.cell(hide_code=True)
def _(measurements, mo, variance_intervals):
    _s2 = measurements.var(ddof=1)
    _row = variance_intervals.loc[variance_intervals["Confidence"] == "95%"].iloc[0]
    _mid = (_row.Low + _row.High) / 2
    _answers = rf"""
1. **Center:** No. The 95% interval is ({_row.Low:.3f}, {_row.High:.3f}). Its midpoint is {_mid:.3f}, while $s^2$ is {_s2:.3f}. 
Unlike the mean interval, the variance interval does not have the form “estimate ± margin of error.” Its endpoints come from different chi-square 
critical values, producing an asymmetric interval.
2. **Assumption:** The observations are treated as a normal sample. The chi-square interval is not a general-purpose interval for every distribution.
"""
    mo.accordion({"Show answers": mo.md(_answers)}, lazy=True)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Using the 40 package weights in `package_weights`, calculate a 95% confidence interval for the population variance and a 95% confidence interval for the population standard deviation. Report the units for each interval.
    """)
    return


@app.cell
def _(np, package_weights, st):
    student_variance_interval = None
    student_sd_interval = None
    print("Variance interval:", student_variance_interval)
    print("Standard deviation interval:", student_sd_interval)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
```python
student_variance_interval = 1 / np.array(st.chi2.interval(
    0.95, len(package_weights) - 1,
    scale=1 / ((len(package_weights) - 1) * package_weights.var(ddof=1))
))[::-1]
student_sd_interval = np.sqrt(student_variance_interval)
print("Variance interval (grams squared):", student_variance_interval)
print("Standard deviation interval (grams):", student_sd_interval)
```

The variance interval is approximately (110.985, 272.696) grams squared. The standard deviation interval is approximately (10.535, 16.514) grams.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Seeing the meaning of a confidence level
    We will repeat the sampling process to see why confidence refers to the method rather than one result.

    We now return to the **simulated ages** from Section 2. We use the same normal age model, with mean 45 years and standard deviation 8 years, and draw fresh samples rather than reusing the original `ages` array. The next cell repeats these steps 20 times:

    1. Draw a new sample of 40 ages.
    2. Calculate its sample mean and a 90% confidence interval.
    3. Check whether the interval contains the true mean of 45.

    In the graph, each dot is a sample mean and its vertical line is the confidence interval. The horizontal red line marks the true population mean. An interval contains the mean if its vertical line touches or crosses that horizontal line. Red dots identify intervals that miss it.

    We expect about $0.90\times20=18$ intervals to contain 45. The actual count may be different because sampling is random. “90% confidence” does not require exactly 18 successes in every group of 20.

    We draw directly from the normal distribution, so its population mean is exactly 45. The seed lets us reproduce this particular experiment.

    If we increase the sample size, the intervals usually become narrower because the sample means vary less. The confidence level remains 90% under this normal model.
    """)
    return


@app.cell
def _(mo, np, pd, plt, st):
    _rng = np.random.default_rng(2027)
    _rows = []
    for _i in range(20):
        # Draw directly from the normal model whose true mean is exactly 45.
        # A finite simulated population would have its own mean, not exactly 45.
        _sample = _rng.normal(45, 8, size=40)
        _low, _high = st.t.interval(0.90, len(_sample) - 1, loc=_sample.mean(), scale=st.sem(_sample))
        _rows.append({
            "Sample": _i + 1,
            "Mean": _sample.mean(),
            "Low": _low,
            "High": _high,
            "Covers_45": _low <= 45 <= _high,
        })
    coverage_table = pd.DataFrame(_rows)
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.errorbar(
        coverage_table["Sample"],
        coverage_table["Mean"],
        yerr=[
            coverage_table["Mean"] - coverage_table["Low"],
            coverage_table["High"] - coverage_table["Mean"],
        ],
        fmt="o",
        color="#4C78A8",
        ecolor="#9ECAE1",
        elinewidth=1,
    )
    _misses = coverage_table.loc[~coverage_table["Covers_45"]]
    _ax.scatter(_misses["Sample"], _misses["Mean"], color="#E45756", zorder=3, label="Interval misses 45")
    _ax.axhline(45, color="#E45756", linewidth=1.5, label="Population mean 45")
    _ax.set(title="Twenty 90% intervals for mean age", xlabel="Sample", ylabel="Sample mean age (years)")
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _covered = int(coverage_table["Covers_45"].sum())
    print(f"{_covered} of 20 intervals contain 45.")
    _fig
    return (coverage_table,)


@app.cell
def _(coverage_table, mo):
    mo.Html(
        coverage_table.round(2).to_html(index=False, border=0, col_space=90)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. If none of these 20 intervals had missed 45, would the procedure be invalid?
    2. If the sample size increased from 40 to 1,000, what would happen to interval width and the percentage of intervals expected to contain the true mean?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **One experiment:** No. All 20 intervals could contain 45 in one run. We expect about 18 on average across many runs; we do not require exactly 18 each time.
2. **Width and coverage:** Larger samples generally give narrower intervals. The theoretical coverage remains 90% under this normal model: narrower intervals are accompanied by less variable sample means. The expected number covering 45 is still 18 out of 20.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions
    - A **point estimate** uses one number from a sample to estimate a population value. Examples include the sample mean and sample proportion.
    - A **confidence interval** gives a range to express uncertainty about that value.
    - **Standard deviation** describes the spread of observations. **Standard error** describes how much a sample mean would vary across repeated samples.
    - For a mean interval, the **margin of error** is the critical value multiplied by the standard error. The endpoints are the mean minus and plus that margin.
    - A 95% confidence method produces intervals containing the true value in about 95% of repeated samples when its assumptions hold.
    - For the same sample, higher confidence gives a wider interval. At the same confidence level, larger samples generally give narrower mean intervals.
    - For paired data, first calculate one difference per pair. Then calculate a mean interval using those differences.
    - A chi-square variance interval requires independent observations from a normal population. Taking square roots of its endpoints gives a standard deviation interval.
    - SciPy and statsmodels give the same mean intervals here. Remember that SciPy takes the confidence level and statsmodels takes `alpha`.

    ### Check your understanding
    1. A sample mean is 12. Is 12 necessarily the population mean?
    2. Which interval is wider, 90% or 99%, for the same sample?
    3. We calculate 200 intervals using a valid 90% confidence method. About how many should contain the population value?
    4. An interval for a mean difference is $(-1.2,4.0)$. Does it include zero?
    5. Why do we divide by the smaller chi-square critical value when calculating the upper variance endpoint?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Point estimate:** Not necessarily. It is the estimate produced by this sample.
2. **Width:** The 99% interval is wider.
3. **Coverage count:** About $0.90 \times 200 = 180$. The count in one set of 200 varies around that value.
4. **Zero:** Yes. Values from $-1.2$ through $4.0$ are inside the interval.
5. **Chi-square:** The lower tail quantile is the smaller number. Dividing by a smaller positive number produces the upper endpoint.
""")}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    - Dekking, F. M., Kraaikamp, C., Lopuhaä, H. P., and Meester, L. E. (2005). *A Modern Introduction to Probability and Statistics*. Springer, Chapter 23.
    """)
    return


if __name__ == "__main__":
    app.run()
