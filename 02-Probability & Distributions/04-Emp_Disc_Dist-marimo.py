import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.stats import rv_discrete

    sns.set_style("whitegrid")
    plt.rcParams["figure.max_open_warning"] = 0
    return mo, np, plt, rv_discrete, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Empirical Discrete Distributions

    ## Objectives

    - Build a discrete distribution for an unfair die and compare its theoretical probabilities with the frequencies in a simulated sample.
    - Increase the sample size and watch the simulated frequencies move toward the theoretical probabilities.
    - Build a discrete distribution from an observed sample, then draw a new sample from that distribution.

    ## Background

    The empirical distribution describes one sample. At each value, it is the proportion of observations less than or equal to that value. This lesson stays with discrete variables: an unfair die, and a distribution built from a mixture of two simulated samples. Green points are the probabilities used to draw the data. Bars are the frequencies in the sample that was actually drawn.

    ## Datasets Used

    This lesson does not use an external dataset. The samples are simulated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## An unfair die

    The faces are 1 through 6. The probability of 5 is 1/2. Each of the other faces has probability 1/10. Those six probabilities add to 1, which `rv_discrete` requires.
    """)
    return


@app.cell
def _(np):
    dk = np.arange(1, 7)
    dk
    return (dk,)


@app.cell
def _():
    pk = (1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2, 1 / 10)
    pk
    return (pk,)


@app.cell
def _(np, pk):
    # The probabilities of a discrete distribution must add to 1.
    print(np.array(pk).sum())
    return


@app.cell
def _(dk, pk, rv_discrete):
    unfair_die = rv_discrete(values=(dk, pk))
    return (unfair_die,)


@app.cell
def _(dk, plt, unfair_die):
    _figure, _axes = plt.subplots()
    _axes.plot(dk, unfair_die.pmf(dk), "go", ms=10)
    _axes.vlines(dk, 0, unfair_die.pmf(dk), colors="g", lw=2)
    _axes.set_title("Original Probabilities")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ten draws are enough to see a sample, and too few for the sample frequencies to match the green points. `np.unique` reports the faces that appeared and how often.
    """)
    return


@app.cell
def _(unfair_die):
    gen_10_values = unfair_die.rvs(size=10)
    print(gen_10_values)
    print(type(gen_10_values))
    return (gen_10_values,)


@app.cell
def _(gen_10_values, np):
    elem, freq = np.unique(gen_10_values, return_counts=True)
    freq = freq / len(gen_10_values)
    print(dict(zip(elem, freq)))
    return elem, freq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Green points and stems are the theoretical probabilities. The bars are the frequencies in this sample of 10.
    """)
    return


@app.cell
def _(dk, elem, freq, plt, unfair_die):
    _figure, _axes = plt.subplots()
    _axes.plot(dk, unfair_die.pmf(dk), "go", ms=10)
    _axes.vlines(dk, 0, unfair_die.pmf(dk), colors="g", lw=2)
    _axes.bar(elem, freq, color="peachpuff")
    _axes.set_title("Expected and Observed Probabilities")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `generate_unfair_die(n)` draws `n` values, prints the sample frequencies, and draws the same comparison chart. Two calls with the same `n` do not match, because each call draws a new sample.
    """)
    return


@app.cell
def _(dk, np, plt, unfair_die):
    def generate_unfair_die(n=10):
        gen_values = unfair_die.rvs(size=n)
        faces, sample_freq = np.unique(gen_values, return_counts=True)
        sample_freq = sample_freq / n
        print(dict(zip(faces, sample_freq)))
        _figure, _axes = plt.subplots()
        _axes.plot(dk, unfair_die.pmf(dk), "go", ms=10)
        _axes.vlines(dk, 0, unfair_die.pmf(dk), colors="g", lw=2)
        _axes.bar(faces, sample_freq, color="peachpuff")
        _axes.set_title("Expected and Observed Probabilities")
        plt.show()

    return (generate_unfair_die,)


@app.cell
def _(generate_unfair_die):
    generate_unfair_die(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At 100 draws the bars move toward the green points. At 1,000, and then at 10,000, the match is closer still.
    """)
    return


@app.cell
def _(generate_unfair_die):
    generate_unfair_die(100)
    return


@app.cell
def _(generate_unfair_die):
    generate_unfair_die(1000)
    return


@app.cell
def _(generate_unfair_die):
    generate_unfair_die(10000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `n_die` and run the cell. A small sample wanders. A large sample stays close to the green points.
    """)
    return


@app.cell
def _(generate_unfair_die):
    n_die = 50
    generate_unfair_die(n_die)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A distribution built from a sample

    The next distribution is not written down in advance. It is estimated from data.

    `seq1` is 500 draws from the integers 1 through 6, each equally likely. `seq2` is 500 Poisson draws with `mu` = 1. The two samples are concatenated, and the frequencies in the combined sample become the probabilities of a new discrete distribution.
    """)
    return


@app.cell
def _(np):
    # 500 uniform integers from 1 to 6, and their frequencies
    seq1 = np.random.randint(1, 7, size=500)
    elem1, freq1 = np.unique(seq1, return_counts=True)
    freq1 = freq1 / len(seq1)
    print(dict(zip(elem1, freq1)))
    return (seq1,)


@app.cell
def _():
    from scipy.stats import poisson

    return (poisson,)


@app.cell
def _(np, poisson):
    # 500 Poisson values with mean 1, and their frequencies
    seq2 = poisson.rvs(mu=1, size=500)
    elem2, freq2 = np.unique(seq2, return_counts=True)
    freq2 = freq2 / len(seq2)
    print(dict(zip(elem2, freq2)))
    return (seq2,)


@app.cell
def _(np, seq1, seq2):
    original_values = np.concatenate([seq1, seq2])
    xk, fk = np.unique(original_values, return_counts=True)
    fk = fk / len(original_values)
    print(dict(zip(xk, fk)))
    return fk, original_values, xk


@app.cell
def _(fk, rv_discrete, xk):
    new_disc_f = rv_discrete(values=(xk, fk))
    return (new_disc_f,)


@app.cell
def _(new_disc_f, plt, xk):
    _figure, _axes = plt.subplots()
    _axes.plot(xk, new_disc_f.pmf(xk), "go", ms=10)
    _axes.vlines(xk, 0, new_disc_f.pmf(xk), colors="g", lw=2)
    _axes.set_title("Original Probabilities")
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `generate_discrete_dist` draws `n` new values from this distribution and compares them with the probabilities it was built from. Green points are those probabilities. Gray bars are the new sample.
    """)
    return


@app.cell
def _(new_disc_f):
    new_10_values = new_disc_f.rvs(size=10)
    new_10_values
    return (new_10_values,)


@app.cell
def _(np, plt, xk):
    def generate_discrete_dist(disc_f, n=10):
        gen_values = disc_f.rvs(size=n)
        faces, sample_freq = np.unique(gen_values, return_counts=True)
        sample_freq = sample_freq / n
        print(dict(zip(faces, sample_freq)))
        _figure, _axes = plt.subplots()
        _axes.plot(xk, disc_f.pmf(xk), "go", ms=10)
        _axes.vlines(xk, 0, disc_f.pmf(xk), colors="g", lw=2)
        _axes.bar(faces, sample_freq, color="lightgray")
        _axes.set_title("Expected and Observed Probabilities")
        plt.show()

    return (generate_discrete_dist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ten new values are not enough: the gray bars sit far from the green points. Each larger sample below is a better match. By 100,000 draws, the bars and the points nearly agree.
    """)
    return


@app.cell
def _(generate_discrete_dist, new_disc_f):
    # The default sample size is 10
    generate_discrete_dist(new_disc_f)
    return


@app.cell
def _(generate_discrete_dist, new_disc_f):
    generate_discrete_dist(new_disc_f, 100)
    return


@app.cell
def _(generate_discrete_dist, new_disc_f):
    generate_discrete_dist(new_disc_f, 1000)
    return


@app.cell
def _(generate_discrete_dist, new_disc_f):
    generate_discrete_dist(new_disc_f, 10000)
    return


@app.cell
def _(generate_discrete_dist, new_disc_f):
    generate_discrete_dist(new_disc_f, 100000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `n_new` and run the cell.
    """)
    return


@app.cell
def _(generate_discrete_dist, new_disc_f):
    n_new = 250
    generate_discrete_dist(new_disc_f, n_new)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - A discrete distribution can be defined by listing each value and its probability, as with the unfair die.
    - The frequencies in a small sample can miss some values and over-represent others.
    - Larger samples pull the observed frequencies toward the probabilities that generated them.
    - The same comparison works when the probabilities themselves were estimated from an earlier sample.

    ## References

    - Unpingco, J. (2019) *Python for Probability, Statistics, and Machine Learning*, USA: Springer, chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
