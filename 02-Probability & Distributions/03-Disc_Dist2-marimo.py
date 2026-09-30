import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import bernoulli, binom, poisson
    return mo, np, pd, plt, bernoulli, binom, poisson


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Discrete Probability Distributions
    
    ## Learning goals
    - Identify a discrete random variable and its possible values.
    - Choose between Bernoulli, binomial, and Poisson models and state their assumptions.
    - Distinguish theoretical probabilities from simulated relative frequencies.
    - Connect a binomial count to a sum of independent Bernoulli trials.
    - Calculate exact, cumulative, upper-tail, and interval probabilities.
    - Compare theoretical means and variances with observed summaries.
    
    ## 1. What is a discrete random variable?
    A **random variable** assigns a number to an experiment's outcome.
    
    A **discrete** random variable has a finite or countably infinite set of possible values.
    
    For example, let X be the face of a fair die: X can be 1, 2, 3, 4, 5, or 6. 
    Its probability distribution gives P(X = k) = 1/6 for each of these values.
    Every probability is nonnegative, and all probabilities add to one.
    
    A **probability mass function (PMF)** gives P(X = k) for each possible value k.
    A simulated sample gives observed counts and relative frequencies. Those frequencies need not match the PMF exactly.
    
    | Model | Meaning of X | Possible values | Parameters |
    |---|---|---|---|
    | Bernoulli | Success indicator for one trial | 0, 1 | p |
    | Binomial | Success count in n trials | 0 through n | n, p |
    | Poisson | Event count in a specified interval | 0, 1, 2, … | μ |
    
    “Success” simply means the outcome being counted. It need not be desirable.
    All datasets below are simulated; no external files are needed.
    
    ## Reproducible experiments
    A seed is a number that lets us repeat a simulation and obtain the same results. 
    Each simulation uses its own seed, so running other cells does not affect its results. 
    Changing the seed produces different observations, but the distribution’s probabilities 
    remain the same.

    To compare sample sizes, we generate one large sample and examine its first 10 observations, 
    then its first 100, and then its first 1,000. Each larger sample includes the observations 
    already examined, showing how the results change as we include more observations.
    """)
    return


@app.cell
def _(mo):
    seed_control = mo.ui.slider(start=1, stop=100, step=1, value=42, show_label=True, label="Simulation seed")
    seed_control
    return (seed_control,)


@app.cell
def _(np, pd, plt):
    
    def draw_sample(model, size, seed):
        """Generate a reproducible sample."""
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer.")
        return model.rvs(size=size, random_state=seed)


    def compare_summaries(values, model):
        """Compare model properties with descriptive summaries of the sample."""
        return pd.DataFrame({
            "Quantity": ["Mean", "Variance", "Standard deviation"],
            "Theoretical": [model.mean(), model.var(), model.std()],
            "Observed": [values.mean(), values.var(ddof=0), values.std(ddof=0)]
        }).round(4)


    def plot_distribution(values, model, support, title):
        """Compare relative frequencies with probability masses at integers."""
        support = np.asarray(support)
        counts = np.bincount(np.asarray(values, dtype=int), minlength=int(support[-1]) + 1)
        observed = counts[support] / len(values)
        theoretical = model.pmf(support)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(support, observed, width=0.75, color="tab:blue", alpha=0.6, label="Observed relative frequency")
        ax.plot(support, theoretical, "o", color="tab:orange", label="Theoretical probability")
        ax.set(xlabel="Value of X", ylabel="Probability / relative frequency", title=title)
        ax.set_xticks(support[::max(1, len(support) // 16)])
        ax.set_xlim(-0.6, support[-1] + 0.6)
        ax.set_ylim(0, max(0.05, observed.max(), theoretical.max()) * 1.20)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        plt.close(fig)
        return fig
    return compare_summaries, draw_sample, plot_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bernoulli: one yes-or-no outcome
    Let X = 1 for success and X = 0 otherwise. Then:
    
    $$P(X=1)=p,\qquad P(X=0)=1-p.$$
    
    For a fair coin, define heads as success and set p = 0.5. A weighted coin can use another p.
    The theoretical mean is p; the theoretical variance is p(1−p).
    For a sample of 0s and 1s, its mean is exactly its observed proportion of successes.
    
    **Predict first:** With p = 0.8, how many successes do you expect in 100 trials? Must the observed count equal that prediction?
    """)
    return


@app.cell
def _(mo):
    bern_p = mo.ui.slider(start=0, stop=1, step=0.05, value=0.8, show_label=True, label="Bernoulli probability p")
    bern_size = mo.ui.slider(steps=[10, 30, 100, 1000, 10000], value=100, show_label=True, label="Bernoulli observations")
    mo.hstack([bern_p, bern_size])
    return bern_p, bern_size


@app.cell
def _(bern_p, bernoulli, draw_sample, seed_control):
    bern_model = bernoulli(p=bern_p.value)
    bern_sequence = draw_sample(bern_model, 10000, seed_control.value)
    return bern_model, bern_sequence


@app.cell
def _(bern_sequence, bern_size):
    bern_values = bern_sequence[:bern_size.value]
    print("First 20 outcomes:", bern_values[:20])
    print(f"Successes: {int(bern_values.sum())} out of {len(bern_values)}")
    print(f"Observed success proportion: {bern_values.mean():.4f}")
    return (bern_values,)


@app.cell
def _(bern_model, bern_values, plot_distribution):
    plot_distribution(bern_values, bern_model, [0, 1], "Bernoulli: probability model and observations")
    return


@app.cell
def _(bern_model, bern_values, compare_summaries):    
    print(compare_summaries(bern_values, bern_model).to_string(index=False))
    return


@app.cell
def _(bern_values, np, plt):
    _shown = bern_values[:30]
    _figure, _axes = plt.subplots(figsize=(8, 2.5))
    _axes.plot(np.arange(1, len(_shown) + 1), _shown, "o", color="tab:blue")
    _axes.set(xlabel="Trial number", ylabel="Outcome", yticks=[0, 1], ylim=(-0.2, 1.2), title="First trials: this is a sequence, not a distribution plot")
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Compare 10, 100, and 10,000 observations with p fixed at 0.8. Is agreement guaranteed to improve at every increase?
    2. Try p = 0 and p = 1. What happens to the outcomes and variance?
    3. Hold the parameters fixed and change the seed. Which quantities change: observed summaries, theoretical summaries, or both?
    
    **Discussion:** The expected success count in 100 trials at p = 0.8 is 80, but it is not guaranteed. Larger samples generally approximate model probabilities more closely, with random fluctuations. At p = 0 or 1 the outcome is constant and the variance is zero.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. From Bernoulli trials to a binomial count
    Let B₁, …, Bₙ be independent Bernoulli trials with the same success probability p.
    Their sum X = B₁ + ⋯ + Bₙ counts successes and follows a binomial distribution.
    
    The next table shows **five experiments**, each containing **ten trials**. Each row sum is one binomial observation.
    """)
    return


@app.cell
def _(np, pd, seed_control):
    _generator = np.random.default_rng(seed_control.value + 1)
    trial_matrix = _generator.binomial(n=1, p=0.5, size=(5, 10))
    trial_table = pd.DataFrame(trial_matrix, columns=[f"Trial {i}" for i in range(1, 11)])
    trial_table["Success count X"] = trial_matrix.sum(axis=1)
    trial_table.index = [f"Experiment {i}" for i in range(1, 6)]
    print(trial_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Binomial: successes in a fixed number of trials
    The model requires a fixed trial count n, two outcomes per trial, independence, and the same p for all trials.
    Its possible values are 0 through n.
    
    $$P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}.$$
    
    The coefficient counts the ways to position k successes among n trials.
    The theoretical mean is np and the variance is np(1−p).
    
    In `binom.rvs(n=10, p=0.5, size=1000)`, **n** is trials per experiment and **size** is the number of repeated experiments. Each returned number is a count, not a proportion.
    """)
    return


@app.cell
def _(mo):
    bin_n = mo.ui.slider(start=1, stop=30, step=1, value=10, show_label=True, label="Trials per experiment n")
    bin_p = mo.ui.slider(start=0, stop=1, step=0.05, value=0.5, show_label=True, label="Success probability p")
    bin_size = mo.ui.slider(steps=[10, 100, 1000, 10000], value=1000, show_label=True, label="Repeated experiments displayed")
    mo.vstack([bin_n, bin_p, bin_size])
    return bin_n, bin_p, bin_size


@app.cell
def _(bin_n, bin_p, binom, draw_sample, seed_control):
    bin_model = binom(n=bin_n.value, p=bin_p.value)
    bin_sequence = draw_sample(bin_model, 10000, seed_control.value + 2)
    return bin_model, bin_sequence


@app.cell
def _(bin_sequence, bin_size):
    bin_values = bin_sequence[:bin_size.value]
    print("First 10 experiment counts:", bin_values[:10])
    return (bin_values,)


@app.cell
def _(bin_model, bin_n, bin_values, np, plot_distribution):
    plot_distribution(bin_values, bin_model, np.arange(bin_n.value + 1), "Binomial: success counts across repeated experiments")
    return


@app.cell
def _(bin_model, bin_values, compare_summaries):
    print(compare_summaries(bin_values, bin_model).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shape and skewness
    For 0 < p < 1, the theoretical binomial distribution is symmetric when p = 0.5.
    For p < 0.5 it is **right-skewed**: most mass is at smaller counts, with a tail toward larger counts.
    For p > 0.5 it is **left-skewed**: most mass is at larger counts, with a tail toward smaller counts.
    Skewness is named for the tail direction, not where the tallest bars stand.
    At p = 0 or 1, the distribution is concentrated at one value.
    A simulated chart need not reproduce theoretical symmetry exactly.
    
    ### Try it yourself
    1. Set n = 10 and compare p = 0.2, 0.5, and 0.8. Predict each theoretical mean and tail direction first.
    2. With n and p fixed, increase the repeated experiment count. Does the theoretical distribution change?
    3. With p = 0.5, increase n from 10 to 20. What happens to the theoretical mean and variance?
    4. Explain why changing n and changing size represent different changes to the experiment.
    
    **Discussion:** Increasing size changes how much data we simulate, not the model. Increasing n changes 
    the number of trials within each experiment and therefore the distribution of X.
    """)
    return


@app.cell(hide_code=True)
def _(binom, mo):
    mo.md(rf"""
    ## 5. Calculate probabilities: a binomial example
    **Example:** It is known that 5% of adults who take a certain medication experience negative side effects. 
    We have a random sample of 100 patients, and we want to calculate the probability that:

    - a) Exactly 5 patients experience side effects.
    - b) 5 patients or fewer experience side effects.
    - c) More than 5 patients experience side effects.
    - d) Between 1 and 10 patients, inclusive, experience side effects.
    
    Assume patients’ outcomes are independent and each patient has the same probability of experiencing 
    side effects.

    Let \(X\) be the number of patients experiencing side effects. We use a binomial distribution with 
    \(n=100\) and \(p=0.05\).

    | Question | Mathematical event | SciPy calculation | Result |
    |---|---|---|--- |
    | Exactly 5 | \(X\) = 5 | `binom.pmf(5, n=100, p=0.05)` | {binom.pmf(5, n=100, p=0.05):.4f} |
    | At most 5 | \(X\) ≤ 5 | `binom.cdf(5, n=100, p=0.05)` | {binom.cdf(5, n=100, p=0.05):.4f} |
    | More than 5 | \(X\) > 5 | `1 - binom.cdf(5, n=100, p=0.05)` | {1 - binom.cdf(5, n=100, p=0.05):.4f} |
    | Between 1 and 10, inclusive | 1 ≤ \(X\) ≤ 10 | `binom.cdf(10, n=100, p=0.05) - binom.cdf(0, n=100, p=0.05)` | {binom.cdf(10, n=100, p=0.05) - binom.cdf(0, n=100, p=0.05):.4f} |
    
    The **probability mass function**, **pmf**, gives the probability of an exact count. 
    
    The **cumulative distribution function**, **cdf**, gives the probability of that count or fewer.

    For **more than 5**, subtract the probability of **5 or fewer** from 1. 
    
    For **1 through 10**, subtract the probability of zero from the probability of **10 or fewer**.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Try it yourself

    In the same example, 5% of adults taking a certain medication
    experience negative side effects. We consider 100 patients.

    In each code cell below, replace `None` with your calculation.
    Use `binom.pmf()` or `binom.cdf()`, with `n=100` and `p=0.05`.

    Your answer will be checked automatically.

    **Remember:** “at least five” includes five.
    """)
    return


@app.cell
def _(binom):
    # 1. What is the probability that no patients experience side effects?
    answer_1 = None
    return (answer_1,)


@app.cell(hide_code=True)
def _(answer_1, binom, mo):
    if answer_1 is None:
        _feedback = "Enter your calculation above."
    elif abs(answer_1 - binom.pmf(0, n=100, p=0.05)) < 0.0001:
        _feedback = "✅ Correct!"
    else:
        _feedback = "❌ Try again. Use the probability mass function for exactly zero."

    mo.md(_feedback)
    return


@app.cell
def _(binom):
    # 2. What is the probability that at least one patient experiences side effects?
    answer_2 = None
    return (answer_2,)


@app.cell(hide_code=True)
def _(answer_2, binom, mo):
    if answer_2 is None:
        _feedback = "Enter your calculation above."
    elif abs(answer_2 - (1 - binom.cdf(0, n=100, p=0.05))) < 0.0001:
        _feedback = "✅ Correct!"
    else:
        _feedback = "❌ Try again. Subtract the probability of zero patients from 1."

    mo.md(_feedback)
    return


@app.cell
def _(binom):
    # 3. What is the probability that at least five patients experience side effects?
    answer_3 = None
    return (answer_3,)


@app.cell(hide_code=True)
def _(answer_3, binom, mo):
    if answer_3 is None:
        _feedback = "Enter your calculation above."
    elif abs(answer_3 - (1 - binom.cdf(4, n=100, p=0.05))) < 0.0001:
        _feedback = "✅ Correct!"
    else:
        _feedback = "❌ Try again. At least five excludes zero through four."

    mo.md(_feedback)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Poisson: events in an interval
    A Poisson random variable counts events in a specified time or space interval. Its possible values are all nonnegative integers, with no fixed upper limit.
    
    $$P(X=k)=e^{-\mu}\frac{\mu^k}{k!}.$$
    
    Its theoretical mean and variance both equal μ. Observed sample mean and variance need not equal μ or each other.
    
    For a **homogeneous Poisson process**, assume a constant event rate, independent counts in disjoint intervals, and that two or more events in a sufficiently short interval are very unlikely compared with one event.
    These assumptions need justification for real data: clustering or changing rates can make the model unsuitable.
    
    **Rate versus expected count:** If arrivals occur at rate r per hour over t hours, μ = r × t. SciPy's `mu` is the expected count for the specified interval, not a rate without units.
    For r = 3 per hour, a half-hour interval has μ = 1.5; two hours have μ = 6.
    """)
    return


@app.cell
def _(mo):
    arrival_rate = mo.ui.slider(start=0, stop=10, step=0.5, value=3, show_label=True, label="Arrivals per hour r")
    interval_hours = mo.ui.slider(start=0.5, stop=3, step=0.5, value=1, show_label=True, label="Interval length (hours)")
    poi_size = mo.ui.slider(steps=[10, 100, 1000, 10000], value=1000, show_label=True, label="Intervals observed")
    mo.vstack([arrival_rate, interval_hours, poi_size])
    return arrival_rate, interval_hours, poi_size


@app.cell
def _(arrival_rate, draw_sample, interval_hours, poisson, seed_control):
    poisson_mu = arrival_rate.value * interval_hours.value
    poi_model = poisson(mu=poisson_mu)
    poi_sequence = draw_sample(poi_model, 10000, seed_control.value + 3)
    return poi_model, poi_sequence, poisson_mu


@app.cell
def _(poi_sequence, poi_size):
    poi_values = poi_sequence[:poi_size.value]
    print("First 10 interval counts:", poi_values[:10])
    return (poi_values,)


@app.cell
def _(np, poi_model, poi_values, poisson_mu):
    # Display at least 99.9% of theoretical mass and every observed value.
    poisson_max = max(int(poi_model.ppf(0.999)), int(poi_values.max()), 1)
    poisson_support = np.arange(poisson_max + 1)
    print(f"Expected count μ = {poisson_mu:g}")
    print(f"Display: 0 through {poisson_max}; theoretical probability above display = {poi_model.sf(poisson_max):.6g}")
    print("The Poisson support is unbounded when μ > 0; this chart uses a finite display range.")
    return (poisson_support,)


@app.cell
def _(poi_model, poi_values, plot_distribution, poisson_support):
    plot_distribution(poi_values, poi_model, poisson_support, "Poisson: event counts per interval")
    return


@app.cell
def _(compare_summaries, poi_model, poi_values):
    print(compare_summaries(poi_values, poi_model).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Keep r = 3 arrivals per hour. Predict μ for 0.5, 1, and 2 hours, then check the controls.
    2. For a fixed interval, compare 10 and 10,000 observations. Are sample mean and variance exactly equal?
    3. Increase μ. Describe changes to the center and spread.
    4. Would a constant-rate model be suitable for a service with a strong lunchtime arrival surge?
    
    **Discussion:** Both theoretical mean and variance increase with μ. Increasing sample size leaves these theoretical properties unchanged. A strong time-dependent rate conflicts with a homogeneous Poisson-process assumption.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Calculate Poisson probabilities
    For a hypothetical constant rate of three arrivals per hour, the count in one hour has μ = 3.
    Use the same PMF, CDF, and survival-function ideas as for the binomial model.
    """)
    return


@app.cell
def _(pd, poisson):
    arrival_model = poisson(mu=3)
    poisson_probabilities = pd.DataFrame({
        "Event": ["Exactly 2", "At most 2", "More than 2", "1 through 4, inclusive"],
        "Probability": [arrival_model.pmf(2), arrival_model.cdf(2), arrival_model.sf(2), arrival_model.cdf(4) - arrival_model.cdf(0)]
    })
    poisson_probabilities.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Descriptive variance versus estimation
    The displayed observed variance uses `ddof=0`: it describes the simulated observations, dividing the sum of squared deviations by their count.
    An unbiased estimator of a population variance uses `ddof=1` for independent, identically distributed observations (and needs at least two observations).
    Neither convention makes a finite sample variance exactly equal to the theoretical variance.
    
    ## Conclusions
    | Model | Mean | Variance |
    |---|---|---|
    | Bernoulli(p) | p | p(1−p) |
    | Binomial(n,p) | np | np(1−p) |
    | Poisson(μ) | μ | μ |
    
    - Choose a model by what X counts and whether its assumptions fit the experiment.
    - Bernoulli describes one indicator; binomial describes a sum of independent indicators with the same p.
    - Poisson describes event counts under an appropriate count model; specify the interval and expected count.
    - Theoretical probabilities and moments are model properties. Simulated frequencies and summaries vary across experiments.
    - More observations generally improve the approximation, without guaranteeing exact or steadily improving agreement.
    - Use the PMF for an exact value, the CDF for “at most,” and the survival function for “more than.”
    
    ## Check your understanding
    1. One inspected item is defective or not defective. Which model fits its indicator?
    2. Count defective items among 20 independent items with the same defect probability. Which model fits?
    3. Count arrivals in an hour under a homogeneous Poisson process. Which model fits?
    4. For Binomial(20, 0.3), what are the mean and variance?
    5. A sample from Poisson(4) has variance 4.3. Is this alone a contradiction?
    6. Which function gives P(X ≥ 5)?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({"Answers": mo.md("1. Bernoulli. 2. Binomial with n = 20. 3. Poisson with μ equal to the hourly expected count. 4. Mean = 6; variance = 4.2. 5. No; sample statistics fluctuate. 6. sf(4), since X is integer-valued.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference
    Unpingco, J. (2019). *Python for Probability, Statistics, and Machine Learning*. Springer, Chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
