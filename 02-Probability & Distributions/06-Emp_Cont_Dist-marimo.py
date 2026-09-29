import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams["figure.max_open_warning"] = 0
    return mo, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Continuous Empirical Distributions

    ## Objectives

    - Build a bimodal sample and fit an empirical cumulative distribution function to it.
    - Use that function to compute probabilities.
    - Draw a new sample from the fitted function and compare it with the original sample.

    ## Background

    An empirical distribution is built from a sample rather than from a named family such as the normal distribution. The empirical cumulative distribution function, or ECDF, at a number `x` is the proportion of the sample that is less than or equal to `x`. This lesson fits an ECDF to a two-peaked sample and then uses the fitted curve to simulate new data.

    ## Datasets Used

    This lesson does not use an external dataset. The sample is a mixture of two normal samples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A bimodal sample

    `sample1` is 400 draws from a normal distribution with mean 30 and standard deviation 5. `sample2` is 700 draws from a normal distribution with mean 50 and standard deviation 5. `np.hstack` joins them end to end into one sample with two peaks.
    """)
    return


@app.cell
def _():
    from scipy.stats import norm

    return (norm,)


@app.cell
def _(norm):
    sample1 = norm.rvs(size=400, loc=30, scale=5)
    return (sample1,)


@app.cell
def _(plt, sample1):
    _figure, _axes = plt.subplots()
    _axes.set_title("sample1")
    _axes.hist(sample1, bins=30)
    _figure
    return


@app.cell
def _(norm):
    sample2 = norm.rvs(size=700, loc=50, scale=5)
    return (sample2,)


@app.cell
def _(plt, sample2):
    _figure, _axes = plt.subplots()
    _axes.set_title("sample2")
    _axes.hist(sample2, bins=30)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `np.hstack` places the second array to the right of the first. The small example uses lists. The same call then joins the two normal samples.
    """)
    return


@app.cell
def _(np):
    a = [1, 2, 3]
    b = [4, 5, 6]
    np.hstack((a, b))
    return


@app.cell
def _(np, sample1, sample2):
    sample = np.hstack((sample1, sample2))
    return (sample,)


@app.cell
def _(plt, sample):
    _figure, _axes = plt.subplots()
    _axes.set_title("Bimodal Distribution")
    _axes.hist(sample, bins=30)
    _figure
    return


@app.cell
def _(sample, sns, plt):
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=sample, bins=30, kde=True, color="mediumblue", ax=_axes)
    _axes.set_title("Bimodal Distribution")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A kernel density curve estimates the probability density of the sample. `fill=True` shades the area under the curve. The two peaks are the two normal samples that were joined.
    """)
    return


@app.cell
def _(sample, sns, plt):
    _figure, _axes = plt.subplots()
    _axes = sns.kdeplot(x=sample, fill=True, color="mediumblue", ax=_axes)
    _axes.set_title("Bimodal Distribution")
    _figure
    return


@app.cell
def _(sample, sns, plt):
    _figure, _axes = plt.subplots()
    _axes = sns.kdeplot(x=sample, color="mediumblue", ax=_axes)
    _axes.set_title("Bimodal Distribution")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The empirical distribution function

    `ECDF` from statsmodels fits the empirical cumulative distribution function of a sample. Calling the result at a number `x` returns the proportion of the sample at or below `x`.
    """)
    return


@app.cell
def _(sample):
    from statsmodels.distributions.empirical_distribution import ECDF

    ecdf = ECDF(sample)
    print(type(ecdf))
    return ECDF, ecdf


@app.cell
def _(ecdf, plt):
    _figure, _axes = plt.subplots()
    _axes.plot(ecdf.x, ecdf.y, color="mediumblue")
    _axes.set_title("Cumulative Bimodal Distribution")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ecdf.x` holds the sorted sample values, with an extra point at the left so the step function starts at height 0. The domain printed below is the smallest and largest of those x values. The curve climbs in two stages, one for each peak of the sample.
    """)
    return


@app.cell
def _(ecdf):
    print("Domain of ecdf: (%.2f, %.2f)" % (min(ecdf.x), max(ecdf.x)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Probabilities from the ECDF

    `ecdf(x)` estimates P(X ≤ x). The complement `1 - ecdf(x)` estimates P(X > x).
    """)
    return


@app.cell
def _(ecdf):
    print("P(x < 10) = %.3f" % ecdf(10))
    print("P(x < 20) = %.3f" % ecdf(20))
    print("P(x > 20) = %.3f" % (1 - ecdf(20)))
    print("P(x < 30) = %.3f" % ecdf(30))
    print("P(x > 30) = %.3f" % (1 - ecdf(30)))
    print("P(x < 60) = %.3f" % ecdf(60))
    print("P(x > 60) = %.3f" % (1 - ecdf(60)))
    print("P(x < 70) = %.3f" % ecdf(70))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `query_x` and run the cell.
    """)
    return


@app.cell
def _(ecdf):
    query_x = 45
    print("P(x < %i) = %.3f" % (query_x, ecdf(query_x)))
    print("P(x > %i) = %.3f" % (query_x, 1 - ecdf(query_x)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Drawing a new sample from the ECDF

    One way to simulate from an ECDF is the inverse-transform method. Draw a uniform number `u` between 0 and 1, then find the smallest sample value whose cumulative probability is at least `u`. That value is one draw from the fitted distribution.

    `np.argmax(ecdf.y >= u)` returns the first position where the cumulative curve reaches `u`.
    """)
    return


@app.cell
def _():
    from scipy.stats import uniform

    return (uniform,)


@app.cell
def _(uniform):
    # 1,000 uniform values on [0, 1]
    unif = uniform.rvs(size=1000, loc=0, scale=1)
    print("Uniform values: (%.4f, %.4f)" % (min(unif), max(unif)))
    return (unif,)


@app.cell
def _(ecdf, np, unif):
    idx = [np.argmax(ecdf.y >= unif[i]) for i in range(len(unif))]
    new_sample = [ecdf.x[idx[i]] for i in range(len(unif))]
    print(
        "New generated values: (%.4f, %.4f)"
        % (min(new_sample), max(new_sample))
    )
    return (new_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulated sample should show the same two peaks. The overlay charts compare it with the original sample. Changing `bins` changes the width of the bars, not the data.
    """)
    return


@app.cell
def _(new_sample, sns, plt):
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=new_sample, bins=30, kde=True, color="crimson", ax=_axes)
    _axes.set_title("Bimodal Distribution (simulated data)")
    _figure
    return


@app.cell
def _(new_sample, plt, sample, sns):
    _figure, _axes = plt.subplots(figsize=(8, 5))
    sns.histplot(
        x=sample, bins=30, kde=True, color="mediumblue", label="Original Data", ax=_axes
    )
    sns.histplot(
        x=new_sample, bins=30, kde=True, color="crimson", label="Simulated Data", ax=_axes
    )
    _axes.legend()
    _axes.set_title("Bimodal Distributions")
    _figure
    return


@app.cell
def _(new_sample, plt, sample, sns):
    _figure, _axes = plt.subplots(figsize=(8, 5))
    sns.histplot(
        x=sample, bins=50, kde=True, color="mediumblue", label="Original Data", ax=_axes
    )
    sns.histplot(
        x=new_sample, bins=50, kde=True, color="crimson", label="Simulated Data", ax=_axes
    )
    _axes.legend()
    _axes.set_title("Bimodal Distributions")
    _figure
    return


@app.cell
def _(new_sample, plt, sample, sns):
    _figure, _axes = plt.subplots(figsize=(8, 5))
    sns.kdeplot(x=sample, fill=True, color="mediumblue", label="Original Data", ax=_axes)
    sns.kdeplot(x=new_sample, fill=True, color="crimson", label="Simulated Data", ax=_axes)
    _axes.legend()
    _axes.set_title("Bimodal Distributions")
    _figure
    return


@app.cell
def _(new_sample, plt, sample, sns):
    _figure, _axes = plt.subplots(figsize=(8, 5))
    sns.kdeplot(x=sample, color="mediumblue", label="Original Data", ax=_axes)
    sns.kdeplot(x=new_sample, color="crimson", label="Simulated Data", ax=_axes)
    _axes.legend()
    _axes.set_title("Bimodal Distributions")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fitting an ECDF to the simulated sample produces a second step function. It should track the original step function, with small differences because the new sample is finite.
    """)
    return


@app.cell
def _(ECDF, ecdf, new_sample, plt):
    new_ecdf = ECDF(new_sample)
    _figure, _axes = plt.subplots(figsize=(8, 5))
    _axes.plot(ecdf.x, ecdf.y, color="mediumblue", label="Original Data")
    _axes.plot(new_ecdf.x, new_ecdf.y, color="crimson", label="Simulated Data")
    _axes.legend()
    _axes.set_title("Cumulative Bimodal Distributions")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - An ECDF is a step function built from a sample. Its height at `x` is the share of observations at or below `x`.
    - Joining two normal samples with different means produces a bimodal sample. The ECDF of that sample rises in two stages.
    - Uniform random numbers, read through the ECDF, produce a new sample with the same two peaks.
    - The histogram, the density curve, and the ECDF of the new sample all stay close to those of the original sample.

    ## References

    - Unpingco, J. (2019) *Python for Probability, Statistics, and Machine Learning*, USA: Springer, chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
