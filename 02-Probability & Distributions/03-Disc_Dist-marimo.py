import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import bernoulli

    sns.set_style("whitegrid")
    plt.rcParams["figure.max_open_warning"] = 0
    return bernoulli, mo, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Discrete Distributions

    ## Objectives

    - Generate Bernoulli trials and see how the balance of 0s and 1s changes with the success probability.
    - Use the binomial distribution to compute the probability of a given number of successes in a fixed number of trials.
    - Generate Poisson counts at different rates and compare the shapes of the distributions.

    ## Background

    A discrete distribution describes a variable that takes separate values, together with how often each value appears. This lesson builds three of them in SciPy and graphs them with Seaborn: Bernoulli for one yes-or-no trial, binomial for the number of successes in several such trials, and Poisson for the number of events in a fixed interval.

    ## Datasets Used

    This lesson does not use an external dataset. Every sample is drawn from a distribution in the code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bernoulli distribution

    A Bernoulli trial has two outcomes: `1` (success) and `0` (failure). A coin toss is the usual example. The parameter `p` is the probability of success.
    """)
    return


@app.cell
def _(bernoulli):
    # Tossing a coin 10 times
    coins = bernoulli.rvs(size=10, p=0.5)
    coins
    return (coins,)


@app.cell
def _(coins, plt):
    _figure, _axes = plt.subplots()
    _axes.plot(coins, "ob")
    _axes.set_title("Bernoulli Trials: Coins")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_bernoulli_trials` draws one Bernoulli sample. `size` is the number of trials and `prob` is the probability of 1.
    """)
    return


@app.cell
def _(bernoulli, plt):
    def plot_bernoulli_trials(size=10, prob=0.5):
        values = bernoulli.rvs(size=size, p=prob)
        _figure, _axes = plt.subplots()
        _axes.plot(values, "ob")
        _axes.set_title("Bernoulli Trials - Probability = " + str(prob))
        plt.show()

    return (plot_bernoulli_trials,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The default call uses 10 trials and `prob` 0.5, so 0 and 1 are equally likely.
    """)
    return


@app.cell
def _(plot_bernoulli_trials):
    # The default values are 10 experiments with probability 0.5
    plot_bernoulli_trials()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `prob` equal to 0.8, most of the points are 1. With `prob` equal to 1, every point is 1. With `prob` equal to 0.2, most of the points are 0.
    """)
    return


@app.cell
def _(plot_bernoulli_trials):
    plot_bernoulli_trials(prob=0.8)
    return


@app.cell
def _(plot_bernoulli_trials):
    plot_bernoulli_trials(prob=1)
    return


@app.cell
def _(plot_bernoulli_trials):
    plot_bernoulli_trials(prob=0.2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `trial_probability` and run the cell. Try values close to 0 and close to 1.
    """)
    return


@app.cell
def _(plot_bernoulli_trials):
    trial_probability = 0.65
    plot_bernoulli_trials(size=30, prob=trial_probability)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `bernoulli.rvs` returns a NumPy array, so the sample has a minimum, a mean, a maximum, a variance, and a standard deviation.
    """)
    return


@app.cell
def _(bernoulli):
    bern = bernoulli.rvs(size=10, p=0.8)
    print(bern)
    print(type(bern))
    return (bern,)


@app.cell
def _(bern):
    print("The min value is  = %i" % (bern.min()))
    print("The mean value is = %.2f" % (bern.mean()))
    print("The max value is  = %i" % (bern.max()))
    return


@app.cell
def _(bern):
    print("The variance is           = %.2f" % (bern.var()))
    print("The standard deviation is = %.2f" % (bern.std()))
    print("The range is              = %.2f" % (bern.max() - bern.min()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A sample of 10 is too small to see the shape. The next two charts use 10,000 draws. The first uses `p` = 0.8. The second is a fair coin, `p` = 0.5.
    """)
    return


@app.cell
def _(bernoulli):
    data_bern = bernoulli.rvs(size=10000, p=0.8)
    return (data_bern,)


@app.cell
def _(data_bern, sns, plt):
    _figure, _axes = plt.subplots()
    _axes = sns.countplot(x=data_bern, ax=_axes)
    _axes.set_title("Bernoulli Distribution")
    _axes.set(xlabel="Values", ylabel="Frequency")
    _figure
    return


@app.cell
def _(bernoulli, sns, plt):
    fair_coins = bernoulli.rvs(size=10000, p=0.5)
    _figure, _axes = plt.subplots()
    _axes = sns.countplot(x=fair_coins, ax=_axes)
    _axes.set_title("Coins")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binomial distribution

    A binomial random variable counts the successes in `n` independent Bernoulli trials that share the same success probability `p`. `binom.rvs(n, p, size)` repeats that whole experiment `size` times. Each returned number is a count of successes, from 0 to `n`.
    """)
    return


@app.cell
def _():
    from scipy.stats import binom

    return (binom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One experiment: toss a fair coin 10 times and count the heads. Then repeat that experiment 5 times, and then 10 times.
    """)
    return


@app.cell
def _(binom):
    binom.rvs(n=10, p=0.5, size=1)
    return


@app.cell
def _(binom):
    binom.rvs(n=10, p=0.5, size=5)
    return


@app.cell
def _(binom):
    bnm = binom.rvs(n=10, p=0.5, size=10)
    print(bnm)
    print(type(bnm))
    return (bnm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `bnm` is a NumPy array of counts, so it has the same summary statistics as the Bernoulli sample. Ten experiments is still a small sample, so the mean may not be close to `n * p` = 5.
    """)
    return


@app.cell
def _(bnm):
    print("The min value is  = %i" % (bnm.min()))
    print("The mean value is = %.2f" % (bnm.mean()))
    print("The max value is  = %i" % (bnm.max()))
    return


@app.cell
def _(bnm):
    print("The variance is           = %.2f" % (bnm.var()))
    print("The standard deviation is = %.2f" % (bnm.std()))
    print("The range is              = %.2f" % (bnm.max() - bnm.min()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_binomial` repeats the experiment many times and draws the counts. The default is a fair coin tossed 10 times, repeated 1,000 times. That histogram is roughly symmetric around 5.
    """)
    return


@app.cell
def _(binom, plt, sns):
    def plot_binomial(n=10, prob=0.5, size=1000):
        data_binom = binom.rvs(n=n, p=prob, size=size)
        grid = sns.displot(data_binom, kde=False, color="darkgreen")
        grid.set(title="Binomial Distribution")
        plt.show()

    return (plot_binomial,)


@app.cell
def _(plot_binomial):
    plot_binomial()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When `prob` is 0.8, successes are common, so the histogram piles up toward the right. When `prob` is 0.2, it piles up toward the left.
    """)
    return


@app.cell
def _(plot_binomial):
    plot_binomial(prob=0.8)
    return


@app.cell
def _(plot_binomial):
    plot_binomial(prob=0.2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `success_probability` and run the cell. The histogram uses 10 trials, repeated 1,000 times.
    """)
    return


@app.cell
def _(plot_binomial):
    success_probability = 0.9
    plot_binomial(prob=success_probability)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example. Side effects of a medication

    Five percent of adults who take a medication have a side effect. A sample contains 100 patients. The calculations below use `p` = 0.05 and `n` = 100.

    - a) Exactly 5 patients have a side effect. This is a point probability, so it uses `binom.pmf`.
    - b) 5 patients or fewer have a side effect. This is a cumulative probability, so it uses `binom.cdf`.
    - c) More than 5 patients have a side effect. That is the complement of (b).
    - d) Between 1 and 10 patients have a side effect.
    """)
    return


@app.cell
def _(binom):
    print(
        "P(exactly 5 patients with side effects) = %.4f"
        % binom.pmf(k=5, n=100, p=0.05)
    )
    print(
        "P(5 patients or fewer with side effects) = %.4f"
        % binom.cdf(k=5, n=100, p=0.05)
    )
    print(
        "P(more than 5 patients with side effects) = %.4f"
        % (1 - binom.cdf(k=5, n=100, p=0.05))
    )
    print(
        "P(between 1 and 10 patients with side effects) = %.4f"
        % (binom.cdf(k=10, n=100, p=0.05) - binom.cdf(k=0, n=100, p=0.05))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Part (d) can also be built by adding the point probabilities from 1 through 10.
    """)
    return


@app.cell
def _(binom):
    pr = 0
    for k in range(1, 11):
        pr = pr + binom.pmf(k=k, n=100, p=0.05)
    print("P(between 1 and 10 patients with side effects) = %.4f" % pr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poisson distribution

    A Poisson random variable counts how many times an event happens in a fixed interval of time or space. The distribution has one parameter, `mu` (μ): the average number of events in that interval. The mean and the variance are both equal to `mu`.
    """)
    return


@app.cell
def _():
    from scipy.stats import poisson

    return (poisson,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next three cells each draw 10 Poisson values. The rate is 1, then 3, then 10. Larger rates produce larger counts.
    """)
    return


@app.cell
def _(poisson):
    poisson.rvs(mu=1, size=10)
    return


@app.cell
def _(poisson):
    poisson.rvs(mu=3, size=10)
    return


@app.cell
def _(poisson):
    poisson.rvs(mu=10, size=10)
    return


@app.cell
def _(poisson):
    pss = poisson.rvs(mu=2, size=10)
    print(pss)
    print(type(pss))
    return (pss,)


@app.cell
def _(pss):
    print("The min value is  = %i" % (pss.min()))
    print("The mean value is = %.2f" % (pss.mean()))
    print("The max value is  = %i" % (pss.max()))
    return


@app.cell
def _(pss):
    print("The variance is           = %.2f" % (pss.var()))
    print("The standard deviation is = %.2f" % (pss.std()))
    print("The range is              = %.2f" % (pss.max() - pss.min()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_Poisson` draws a large sample at a chosen rate. The three charts use 10,000 values with `mu` equal to 1, 2, and 4. The pile of counts shifts to the right as the rate grows.
    """)
    return


@app.cell
def _(plt, poisson, sns):
    def plot_Poisson(mu=2, size=10):
        data_poisson = poisson.rvs(mu=mu, size=size)
        grid = sns.displot(data_poisson, kde=False, color="darkred")
        grid.set(title="Poisson Distribution")
        plt.show()

    return (plot_Poisson,)


@app.cell
def _(plot_Poisson):
    plot_Poisson(mu=1, size=10000)
    return


@app.cell
def _(plot_Poisson):
    plot_Poisson(mu=2, size=10000)
    return


@app.cell
def _(plot_Poisson):
    plot_Poisson(mu=4, size=10000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `event_rate` and run the cell.
    """)
    return


@app.cell
def _(plot_Poisson):
    event_rate = 8
    plot_Poisson(mu=event_rate, size=10000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - A Bernoulli sample contains only 0s and 1s. Raising `p` produces more 1s.
    - A binomial sample counts successes across `n` trials. The histogram is nearly symmetric when `p` is 1/2, and it skews when `p` moves away from 1/2.
    - `binom.pmf` gives the probability of one exact count. `binom.cdf` gives the probability of that count or anything smaller.
    - A Poisson sample counts events. Its center sits near `mu`, and both the mean and the variance are `mu`.

    ## References

    - Unpingco, J. (2019) *Python for Probability, Statistics, and Machine Learning*, USA: Springer, chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
