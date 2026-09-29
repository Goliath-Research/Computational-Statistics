import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import uniform

    sns.set_style("whitegrid")
    plt.rcParams["figure.max_open_warning"] = 0
    return mo, plt, sns, uniform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Continuous Distributions

    ## Objectives

    - Generate uniform, normal, and exponential samples and read their shapes from histograms and density curves.
    - Compute waiting-time probabilities from a uniform distribution on an interval.
    - Compute test-score probabilities from a normal distribution.

    ## Background

    A continuous distribution assigns probability to intervals rather than to separate points. This lesson uses three families. A uniform distribution spreads probability evenly across an interval. A normal distribution is symmetric about its mean. An exponential distribution describes the waiting time until the next event when events happen at a constant average rate.

    ## Datasets Used

    This lesson does not use an external dataset. The samples are drawn from SciPy distributions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uniform distribution

    `uniform.rvs` draws from a continuous uniform distribution.

    - `loc` is the start of the interval.
    - `scale` is the width of the interval. The values fall between `loc` and `loc + scale`.
    - `size` is the number of draws.

    A sample of 10 is only a list of numbers. A sample of 10,000 shows the flat shape.
    """)
    return


@app.cell
def _(uniform):
    # 10 values on the interval from 0 to 1
    uniform.rvs(size=10, loc=0, scale=1)
    return


@app.cell
def _(uniform):
    # 10 values on the interval from 0 to 10
    unf = uniform.rvs(size=10, loc=0, scale=10)
    print(unf)
    print(type(unf))
    return (unf,)


@app.cell
def _(unf):
    print("The min value is  = %.2f" % (unf.min()))
    print("The mean value is = %.2f" % (unf.mean()))
    print("The max value is  = %.2f" % (unf.max()))
    return


@app.cell
def _(unf):
    print("The variance is           = %.2f" % (unf.var()))
    print("The standard deviation is = %.2f" % (unf.std()))
    print("The range is              = %.2f" % (unf.max() - unf.min()))
    return


@app.cell
def _(uniform):
    # 10,000 values on the interval from 0 to 10
    data_uniform1 = uniform.rvs(loc=0, scale=10, size=10000)
    return (data_uniform1,)


@app.cell
def _(data_uniform1, sns, plt):
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_uniform1, kde=False, color="darkblue", ax=_axes)
    _axes.set(xlabel="Uniform", ylabel="Frequency")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `bins` sets how many bars the histogram uses. A density curve (`kde=True`) sketches the shape on top of the bars. For a uniform sample the curve is roughly flat, with some wobble at the ends.
    """)
    return


@app.cell
def _(data_uniform1, sns, plt):
    # 10 bins
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_uniform1, bins=10, kde=False, color="darkblue", ax=_axes)
    _axes.set(xlabel="Uniform", ylabel="Frequency")
    _figure
    return


@app.cell
def _(data_uniform1, sns, plt):
    # Histogram with a density curve
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(data_uniform1, kde=True, color="darkblue", ax=_axes)
    _axes.set(title="Uniform Distribution")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next two samples use different intervals: 20 to 40 (`loc` 20, `scale` 20) and 10 to 60 (`loc` 10, `scale` 50). The density charts put all three samples on one pair of axes. A wider interval spreads the same probability over more numbers, so its curve is lower.
    """)
    return


@app.cell
def _(sns, uniform, plt):
    data_uniform2 = uniform.rvs(loc=20, scale=20, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_uniform2, kde=True, color="darkgreen", ax=_axes)
    _axes.set(title="Uniform Distribution")
    _figure
    return (data_uniform2,)


@app.cell
def _(sns, uniform, plt):
    data_uniform3 = uniform.rvs(loc=10, scale=50, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_uniform3, kde=True, color="red", ax=_axes)
    _axes.set(title="Uniform Distribution")
    _figure
    return (data_uniform3,)


@app.cell
def _(data_uniform1, data_uniform2, data_uniform3, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x=data_uniform3, fill=True, color="red", label="U(10,60)", ax=_axes)
    sns.kdeplot(
        x=data_uniform2, fill=True, color="darkgreen", label="U(20,40)", ax=_axes
    )
    sns.kdeplot(x=data_uniform1, fill=True, color="blue", label="U(0,10)", ax=_axes)
    _axes.set(title="Uniform Distributions")
    _axes.legend()
    _figure
    return


@app.cell
def _(data_uniform1, data_uniform2, data_uniform3, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x=data_uniform3, color="red", label="U(10,60)", ax=_axes)
    sns.kdeplot(x=data_uniform2, color="darkgreen", label="U(20,40)", ax=_axes)
    sns.kdeplot(x=data_uniform1, color="blue", label="U(0,10)", ax=_axes)
    _axes.set(title="Uniform Distributions")
    _axes.legend()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1. Waiting for an elevator

    After the button is pressed, the elevator arrives at a time spread uniformly between 0 and 40 seconds. `uniform(0, 40)` is that distribution: it starts at 0 and has width 40.

    `cdf(t)` is the probability that the waiting time is `t` seconds or less.

    - The probability of waiting less than 15 seconds is `cdf(15)`.
    - The probability of waiting more than 15 seconds is `1 - cdf(15)`.
    - The probability of waiting between 10 and 30 seconds is `cdf(30) - cdf(10)`.
    """)
    return


@app.cell
def _(uniform):
    elev = uniform(0, 40)
    print(
        "P(the elevator takes less than 15 seconds) = %.3f" % elev.cdf(15)
    )
    print(
        "P(the elevator takes more than 15 seconds) = %.3f" % (1 - elev.cdf(15))
    )
    print(
        "P(the elevator takes between 10 and 30 seconds) = %.3f"
        % (elev.cdf(30) - elev.cdf(10))
    )
    return (elev,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `wait_limit` and run the cell. The two probabilities add to 1.
    """)
    return


@app.cell
def _(elev):
    wait_limit = 25
    print("P(wait < %i seconds) = %.3f" % (wait_limit, elev.cdf(wait_limit)))
    print(
        "P(wait > %i seconds) = %.3f" % (wait_limit, 1 - elev.cdf(wait_limit))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Normal distribution

    The normal distribution is symmetric about its mean. Drawn as a density, it is a bell curve.

    `norm.rvs(loc, scale, size)` draws a sample. `loc` is the mean. `scale` is the standard deviation, not the variance.
    """)
    return


@app.cell
def _():
    from scipy.stats import norm

    return (norm,)


@app.cell
def _(norm):
    # 10 values from N(0, 1): mean 0, standard deviation 1
    n1 = norm.rvs(loc=0, scale=1, size=10)
    print(n1)
    print("\n[Min, Max] = [%.2f, %.2f]" % (n1.min(), n1.max()))
    print(type(n1))
    return (n1,)


@app.cell
def _(norm):
    # 10 values from N(0, 10): mean 0, standard deviation 10
    n2 = norm.rvs(loc=0, scale=10, size=10)
    print(n2)
    print("\n[Min, Max] = [%.2f, %.2f]" % (n2.min(), n2.max()))
    print(type(n2))
    return (n2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ten values do not show the bell. The histograms below use 10,000 values. The labels give the mean and the standard deviation: N(0, 1), N(5, 2), and N(10, 5). A larger standard deviation makes a wider bell.
    """)
    return


@app.cell
def _(norm, sns, plt):
    data_normal1 = norm.rvs(loc=0, scale=1, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_normal1, bins=100, kde=True, color="blue", ax=_axes)
    _axes.set(title="Normal Distribution")
    _figure
    return (data_normal1,)


@app.cell
def _(norm, sns, plt):
    # N(5, 2): mean 5, standard deviation 2
    data_normal2 = norm.rvs(loc=5, scale=2, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_normal2, bins=100, kde=True, color="darkgreen", ax=_axes)
    _axes.set(title="Normal Distribution")
    _figure
    return (data_normal2,)


@app.cell
def _(norm, sns, plt):
    # N(10, 5): mean 10, standard deviation 5
    data_normal3 = norm.rvs(loc=10, scale=5, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_normal3, bins=100, kde=True, color="red", ax=_axes)
    _axes.set(title="Normal Distribution")
    _figure
    return (data_normal3,)


@app.cell
def _(data_normal1, data_normal2, data_normal3, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x=data_normal1, fill=True, color="blue", label="N(0, 1)", ax=_axes)
    sns.kdeplot(
        x=data_normal2, fill=True, color="darkgreen", label="N(5, 2)", ax=_axes
    )
    sns.kdeplot(x=data_normal3, fill=True, color="red", label="N(10, 5)", ax=_axes)
    _axes.set(title="Normal Distributions")
    _axes.legend()
    _figure
    return


@app.cell
def _(data_normal1, data_normal2, data_normal3, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x=data_normal1, color="blue", label="N(0, 1)", ax=_axes)
    sns.kdeplot(x=data_normal2, color="darkgreen", label="N(5, 2)", ax=_axes)
    sns.kdeplot(x=data_normal3, color="red", label="N(10, 5)", ax=_axes)
    _axes.set(title="Normal Distributions")
    _axes.legend()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2. An admission test

    Scores on an admission test are approximately normal with mean 600 and standard deviation 100. `cdf(s)` is the probability of scoring `s` or less.

    - The probability of scoring above 550 is `1 - cdf(550)`.
    - The probability of scoring below 800 is `cdf(800)`.
    - The probability of scoring between 500 and 800 is `cdf(800) - cdf(500)`.
    """)
    return


@app.cell
def _(norm):
    test = norm(600, 100)
    print("P(test score > 550) = %.3f" % (1 - test.cdf(550)))
    print("P(test score < 800) = %.3f" % (test.cdf(800)))
    print("P(500 <= test score <= 800) = %.3f" % (test.cdf(800) - test.cdf(500)))
    return (test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `score_cutoff` and run the cell.
    """)
    return


@app.cell
def _(test):
    score_cutoff = 700
    print("P(test score > %i) = %.3f" % (score_cutoff, 1 - test.cdf(score_cutoff)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exponential distribution

    The exponential distribution describes the time until the next event when events occur continuously and independently at a constant average rate. Its values fall in the interval from `loc` upward.

    `expon.rvs(loc, scale, size)` draws a sample. `loc` shifts the distribution. `scale` stretches it. The mean of a standard exponential random variable (`loc` 0, `scale` 1) is 1. After a shift and a stretch, the mean is `loc + scale`.
    """)
    return


@app.cell
def _():
    from scipy.stats import expon

    return (expon,)


@app.cell
def _(expon):
    # 10 values from Exp with loc 0 and scale 1
    e1 = expon.rvs(loc=0, scale=1, size=10)
    print(e1)
    print("\n[Min, Max] = [%.2f, %.2f]" % (e1.min(), e1.max()))
    print(type(e1))
    return (e1,)


@app.cell
def _(e1):
    print("The mean value is         = %.2f" % (e1.mean()))
    print("The variance is           = %.2f" % (e1.var()))
    print("The standard deviation is = %.2f" % (e1.std()))
    return


@app.cell
def _(expon):
    # 10 values shifted to start at 10; scale stays 1
    e2 = expon.rvs(loc=10, scale=1, size=10)
    print(e2)
    print("\n[Min, Max] = [%.2f, %.2f]" % (e2.min(), e2.max()))
    print(type(e2))
    return (e2,)


@app.cell
def _(e2):
    print("The mean value is         = %.2f" % (e2.mean()))
    print("The variance is           = %.2f" % (e2.var()))
    print("The standard deviation is = %.2f" % (e2.std()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The histograms use 10,000 values. The first starts at 0 with scale 1. The second starts at 10 with scale 1, so it has the same shape, moved to the right. The third starts at 10 with scale 10, so it is much more spread out. Every curve is high near its left end and then falls.
    """)
    return


@app.cell
def _(expon, sns, plt):
    data_exp1 = expon.rvs(loc=0, scale=1, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_exp1, kde=True, color="blue", ax=_axes)
    _axes.set(title="Exponential Distribution")
    _figure
    return (data_exp1,)


@app.cell
def _(expon, sns, plt):
    data_exp2 = expon.rvs(loc=10, scale=1, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_exp2, kde=True, color="darkgreen", ax=_axes)
    _axes.set(title="Exponential Distribution")
    _figure
    return (data_exp2,)


@app.cell
def _(expon, sns, plt):
    data_exp3 = expon.rvs(loc=10, scale=10, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=data_exp3, kde=True, color="red", ax=_axes)
    _axes.set(title="Exponential Distribution")
    _figure
    return (data_exp3,)


@app.cell
def _(data_exp1, data_exp2, data_exp3, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(
        x=data_exp1, fill=True, color="blue", label="loc=0, scale=1", ax=_axes
    )
    sns.kdeplot(
        x=data_exp2, fill=True, color="darkgreen", label="loc=10, scale=1", ax=_axes
    )
    sns.kdeplot(
        x=data_exp3, fill=True, color="red", label="loc=10, scale=10", ax=_axes
    )
    _axes.set(title="Exponential Distributions")
    _axes.legend()
    _figure
    return


@app.cell
def _(data_exp1, data_exp2, data_exp3, plt, sns):
    _figure, _axes = plt.subplots(figsize=(10, 5))
    sns.kdeplot(x=data_exp1, color="blue", label="loc=0, scale=1", ax=_axes)
    sns.kdeplot(x=data_exp2, color="darkgreen", label="loc=10, scale=1", ax=_axes)
    sns.kdeplot(x=data_exp3, color="red", label="loc=10, scale=10", ax=_axes)
    _axes.set(title="Exponential Distributions")
    _axes.legend()
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `exp_scale` and run the cell. Larger scales stretch the sample to the right.
    """)
    return


@app.cell
def _(expon, sns, plt):
    exp_scale = 5
    your_exp = expon.rvs(loc=0, scale=exp_scale, size=10000)
    _figure, _axes = plt.subplots()
    _axes = sns.histplot(x=your_exp, kde=True, color="purple", ax=_axes)
    _axes.set(title="Exponential sample, scale = %s" % exp_scale)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - A uniform sample spreads across its interval. The density is flat, and a wider interval produces a lower curve.
    - A normal sample forms a bell centered at `loc`. `scale` is the standard deviation: larger values make a wider bell.
    - `cdf` answers interval questions. For the elevator, it gives the probability of arriving before a chosen time. For the test, it gives the probability of scoring below a chosen score.
    - An exponential sample piles up at its left end and has a long right tail. Its mean is `loc + scale`.

    ## References

    - Unpingco, J. (2019) *Python for Probability, Statistics, and Machine Learning*, USA: Springer, chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
