import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import uniform, norm, expon
    return mo, np, pd, plt, uniform, norm, expon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Continuous Probability Distributions
    
    ## Learning goals
    - Distinguish a density height from a probability and interpret probability as area.
    - Describe uniform, normal, and exponential models and their parameters.
    - Compare simulated histograms and summaries with theoretical distributions.
    - Calculate lower-tail, upper-tail, and interval probabilities using the cumulative distribution function (CDF) and survival function (SF).
    - Convert an exponential event rate to its mean waiting time and explain location shifts.
    
    ## 1. From discrete masses to continuous densities
    For the continuous models in this lesson, an exact point has probability zero: P(X = x) = 0.
    Probabilities belong to intervals and equal areas under the **probability density function (PDF)**:
    
    $$P(a<X<b)=\int_a^b f(x)\,dx.$$
    
    The density is nonnegative and its total area is one. Its height is not a probability and may exceed one.
    For example, Uniform(0, 0.5) has density 2, but P(0.1 < X < 0.2) = 2 × 0.1 = 0.2.
    The density's units are reciprocal measurement units; probability has no units.
    
    The **cumulative distribution function (CDF)** is F(t) = P(X ≤ t).
    The **survival function (SF)** is SF(t) = P(X > t) = 1 − F(t).
    For these continuous distributions:
    - **cdf(t)** gives P(X < t) as well as P(X ≤ t).
    - **sf(t)**, the **survival function**, gives P(X > t).
    - **cdf(b) - cdf(a)** gives the probability between a and b.
    Including or excluding an endpoint makes no difference because its probability is zero.
    Unlike integer-valued distributions, these continuous models need no integer threshold adjustment: P(X ≥ t) = P(X > t) = SF(t).
    Measured or rounded values can have positive recorded frequencies; that does not contradict an underlying continuous model.
    
    ## Simulations and plots
    No external data are needed. Each experiment uses a fresh seeded generator.
    With fixed parameters and seed, changing sample size displays prefixes of one stored sequence.
    Changing histogram bins changes the visual summary, not the observations or model.
    We use **density-normalized histograms**: each bar's area equals the proportion of observations in that bin, and the total histogram area is one.
    The line is the theoretical PDF, not an estimated smoothing curve.
    Kernel density estimates can blur sharp boundaries and extend into impossible regions, so they are not our main teaching plots.
    """)
    return


@app.cell
def _(mo):
    simulation_seed = mo.ui.slider(start=1, stop=100, step=1, value=42, show_value=True, label="Simulation seed")
    sample_size = mo.ui.slider(steps=[10, 100, 1000, 10000], value=1000, show_value=True, label="Observations displayed")
    histogram_bins = mo.ui.slider(start=5, stop=60, step=5, value=20, show_value=True, label="Histogram bins")
    mo.vstack([simulation_seed, sample_size, histogram_bins])
    return histogram_bins, sample_size, simulation_seed


@app.cell
def _(np, pd, plt):
    def draw_continuous(model, seed):
        return model.rvs(size=10000, random_state=np.random.default_rng(seed))


    def summary_table(values, model):
        return pd.DataFrame({"Quantity": ["Mean", "Variance (descriptive, ddof=0)", "Standard deviation"],
                             "Theoretical": [model.mean(), model.var(), model.std()],
                             "Observed": [values.mean(), values.var(ddof=0), values.std(ddof=0)]}).round(4)


    def plot_density(values, model, bins, title, bounded=False):
        # Include every observed value and the central 99.8% model interval.
        if bounded:
            lower, upper = model.support()
        else:
            lower = min(float(model.ppf(0.001)), float(values.min()))
            upper = max(float(model.ppf(0.999)), float(values.max()))
            if np.isfinite(model.support()[0]):
                lower = model.support()[0]
        x = np.linspace(lower, upper, 1000)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(values, bins=np.linspace(lower, upper, bins + 1), density=True,
                color="tab:blue", alpha=0.45, label="Observed density histogram")
        ax.plot(x, model.pdf(x), color="tab:orange", label="Theoretical PDF")
        ax.set(xlabel="Value of X", ylabel="Density", title=title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        plt.close(fig)
        return fig


    def shaded_interval(model, a, b, title):
        lower = min(a, float(model.ppf(0.001)))
        upper = max(b, float(model.ppf(0.999)))
        x = np.linspace(lower, upper, 1000)
        shade = np.linspace(a, b, 300)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, model.pdf(x), color="tab:blue")
        ax.fill_between(shade, model.pdf(shade), alpha=0.35, color="tab:orange")
        ax.set(xlabel="Value of X", ylabel="Density", title=f"{title}: area = {model.cdf(b) - model.cdf(a):.4f}")
        fig.tight_layout()
        plt.close(fig)
        return fig
    return draw_continuous, plot_density, shaded_interval, summary_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Uniform: constant density on an interval
    For a < b, Uniform(a,b) assigns the same probability to intervals of the same length lying inside [a,b]. It does not guarantee equally spaced observations.
    
    $$f(x)=\frac{1}{b-a}\quad(a\le x\le b),\qquad E[X]=\frac{a+b}{2},\qquad \operatorname{Var}(X)=\frac{(b-a)^2}{12}.$$
    
    The uniform PDF is zero outside [a,b].

    SciPy uses **loc=a** and **scale=b-a**. Scale is the width, not the upper endpoint.
    A wider interval has a lower density because the total area must remain one.
    """)
    return


@app.cell
def _(mo):
    uniform_start = mo.ui.slider(start=0, stop=20, step=1, value=0, show_value=True, label="Uniform lower endpoint a")
    uniform_width = mo.ui.slider(start=0.5, stop=40, step=0.5, value=10, show_value=True, label="Uniform width b−a")
    mo.hstack([uniform_start, uniform_width])
    return uniform_start, uniform_width


@app.cell
def _(draw_continuous, simulation_seed, uniform, uniform_start, uniform_width):
    uniform_model = uniform(loc=uniform_start.value, scale=uniform_width.value)
    uniform_sequence = draw_continuous(uniform_model, simulation_seed.value)
    return uniform_model, uniform_sequence


@app.cell
def _(sample_size, uniform_sequence):
    uniform_values = uniform_sequence[:sample_size.value]
    print("First 10 values:", uniform_values[:10])
    return (uniform_values,)


@app.cell
def _(histogram_bins, plot_density, uniform_model, uniform_values):
    plot_density(uniform_values, uniform_model, histogram_bins.value, "Uniform: sample versus model", bounded=True)
    return


@app.cell
def _(summary_table, uniform_model, uniform_values, mo):
    mo.Html(
        summary_table(uniform_values, uniform_model).to_html(border=0, col_space=110, index=False)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked example: an elevator
    **Assume** the waiting time T is Uniform(0,40) seconds. This is a teaching model, not a universal description of elevator arrivals.
    Predict probabilities by interval lengths before using SciPy:
    P(T < 15) = 15/40; P(T > 15) = 25/40; P(10 < T < 30) = 20/40.
    """)
    return


@app.cell
def _(pd, uniform, mo):
    elevator_model = uniform(loc=0, scale=40)
    elevator_results = pd.DataFrame({"Event": ["T < 15", "T > 15", "10 < T < 30"], "Probability": [elevator_model.cdf(15), elevator_model.sf(15), elevator_model.cdf(30) - elevator_model.cdf(10)]})
    mo.Html(
        elevator_results.round(4).to_html(border=0, col_space=110, index=False)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (elevator_model,)


@app.cell
def _(elevator_model, shaded_interval):
    shaded_interval(elevator_model, 10, 30, "Elevator waiting time, seconds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Double the uniform width while holding a fixed. Predict the density height and variance.
    2. Change only the bins. Do the observations or theoretical probabilities change?
    3. What are P(T < 0), P(T > 40), and P(T = 15) in the elevator model?
    
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Double the width:** The density height halves and the variance quadruples.
2. **Change only the bins:** Neither the observations nor the theoretical probabilities change. Only the histogram summary changes.
3. **Elevator probabilities:** P(T < 0) = P(T > 40) = P(T = 15) = 0. The support is [0,40], and an exact point has probability zero.

**Discussion:** Doubling width halves density height and quadruples variance. The three elevator probabilities are all zero. Histogram appearance is affected by sample size and binning; sample agreement is not guaranteed to improve at each increase.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself: calculate probabilities
    Use the same elevator model: T is Uniform(0,40) seconds.

    1. Calculate P(T < 20).
    2. Calculate P(T > 30).
    3. Calculate P(5 < T < 25).

    In the next cell, replace **None** with your calculation. Use **elevator_model.cdf()**, **elevator_model.sf()**, or a difference of CDF values.
    Run your cell, then expand **Show answers** to compare with the explanations, complete solution code, and numerical answers.
    """)
    return


@app.cell
def _():
    # 1. Calculate P(T < 20).
    uniform_answer_1 = None
    # 2. Calculate P(T > 30).
    uniform_answer_2 = None
    # 3. Calculate P(5 < T < 25).
    uniform_answer_3 = None
    print("Answer 1:", uniform_answer_1)
    print("Answer 2:", uniform_answer_2)
    print("Answer 3:", uniform_answer_3)
    return


@app.cell(hide_code=True)
def _(elevator_model, mo):
    _answers = f"""
1. **P(T < 20)** = {elevator_model.cdf(20):.4f}. The interval length is 20 seconds, so 20/40 = 0.5.
2. **P(T > 30)** = {elevator_model.sf(30):.4f}. The upper interval has length 10 seconds, so 10/40 = 0.25.
3. **P(5 < T < 25)** = {elevator_model.cdf(25) - elevator_model.cdf(5):.4f}. The interval has length 20 seconds, so (25−5)/40 = 0.5.

```python
uniform_answer_1 = elevator_model.cdf(20)
uniform_answer_2 = elevator_model.sf(30)
uniform_answer_3 = elevator_model.cdf(25) - elevator_model.cdf(5)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Normal: location and spread
    We use the conventional notation **N(μ, σ²)**: the second argument is the variance.
    SciPy's **norm(loc=mu, scale=sigma)** instead takes the **standard deviation** as scale.
    For example, **norm(loc=5, scale=2)** corresponds to N(5,4).
    
    The PDF is symmetric and bell-shaped, with mean μ and variance σ². Its support is the entire real line.
    Changing μ shifts the curve. Increasing σ broadens it and lowers its peak.
    A normal model can approximate measurements, but it is not automatically suitable for every variable; bounded scores and nonnegative quantities require attention to the implied tails.
    """)
    return


@app.cell
def _(mo):
    normal_mean = mo.ui.slider(start=-10, stop=20, step=1, value=5, show_value=True, label="Normal mean μ")
    normal_sd = mo.ui.slider(start=0.5, stop=10, step=0.5, value=2, show_value=True, label="Normal standard deviation σ")
    mo.hstack([normal_mean, normal_sd])
    return normal_mean, normal_sd


@app.cell
def _(draw_continuous, norm, normal_mean, normal_sd, simulation_seed):
    normal_model = norm(loc=normal_mean.value, scale=normal_sd.value)
    normal_sequence = draw_continuous(normal_model, simulation_seed.value + 1)
    return normal_model, normal_sequence


@app.cell
def _(normal_sequence, sample_size):
    normal_values = normal_sequence[:sample_size.value]
    return (normal_values,)


@app.cell
def _(histogram_bins, normal_model, normal_values, plot_density):
    plot_density(normal_values, normal_model, histogram_bins.value, "Normal: sample versus model")
    return


@app.cell
def _(normal_model, normal_values, summary_table, mo):
    mo.Html(
        summary_table(normal_values, normal_model).to_html(border=0, col_space=110, index=False)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked example: admission scores
    Assume scores are approximately normal with mean 600 and standard deviation 100: N(600,10000).
    Let S be a score. Calculate P(S > 550), P(S < 800), and P(500 < S < 800).
    These are approximate probabilities under a continuous model of scores.
    """)
    return


@app.cell
def _(norm, pd, mo):
    score_model = norm(loc=600, scale=100)
    score_results = pd.DataFrame({"Event": ["S > 550", "S < 800", "500 < S < 800"], "Probability": [score_model.sf(550), score_model.cdf(800), score_model.cdf(800) - score_model.cdf(500)]})
    mo.Html(
        score_results.round(4).to_html(border=0, col_space=110, index=False)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (score_model,)


@app.cell
def _(score_model, shaded_interval):
    shaded_interval(score_model, 500, 800, "Admission score interval")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. For the score model, predict P(S > 600) using symmetry.
    2. What happens to variance when σ doubles?
    3. Compare 10 and 10,000 observations. Must the sample mean equal μ?
    4. Does a finite sample's minimum or maximum define the model's support?
    
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Symmetry:** P(S > 600) = 0.5, because 600 is the normal model's mean.
2. **Double σ:** The variance σ² quadruples.
3. **Sample mean:** Neither sample size forces the sample mean to equal μ. Larger independent samples tend to approximate μ more closely, but improvement need not be monotonic.
4. **Support:** No. Finite sample extremes do not bound the normal model's support, which is the entire real line.

**Discussion:** P(S > 600) = 0.5. Doubling σ quadruples variance. Sample summaries fluctuate. The theoretical normal support remains unbounded even though every simulated sample has finite extremes.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself: calculate probabilities
    Use the same admission-score model: S is normal with mean 600 and standard deviation 100.

    1. Calculate P(S < 600).
    2. Calculate P(S > 700).
    3. Calculate P(500 < S < 700).

    In the next cell, replace **None** with your calculation. Use **score_model.cdf()**, **score_model.sf()**, or a difference of CDF values.
    Run your cell, then expand **Show answers** to compare with the explanations, complete solution code, and numerical answers.
    """)
    return


@app.cell
def _():
    # 1. Calculate P(S < 600).
    normal_answer_1 = None
    # 2. Calculate P(S > 700).
    normal_answer_2 = None
    # 3. Calculate P(500 < S < 700).
    normal_answer_3 = None
    print("Answer 1:", normal_answer_1)
    print("Answer 2:", normal_answer_2)
    print("Answer 3:", normal_answer_3)
    return


@app.cell(hide_code=True)
def _(score_model, mo):
    _answers = f"""
1. **P(S < 600)** = {score_model.cdf(600):.4f}. By symmetry, half the distribution is below the mean.
2. **P(S > 700)** = {score_model.sf(700):.4f}. The SF gives the probability above 700.
3. **P(500 < S < 700)** = {score_model.cdf(700) - score_model.cdf(500):.4f}. Subtract the cumulative probability at 500 from that at 700.

```python
normal_answer_1 = score_model.cdf(600)
normal_answer_2 = score_model.sf(700)
normal_answer_3 = score_model.cdf(700) - score_model.cdf(500)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Exponential: waiting for the next event
    For a homogeneous Poisson process with constant rate λ > 0, the waiting time W until the next event is exponential:
    
    $$f(w)=\lambda e^{-\lambda w}\quad(w\ge0),\qquad E[W]=\frac1\lambda,\qquad \operatorname{Var}(W)=\frac1{\lambda^2}.$$
    
    The process assumes independent counts in disjoint intervals and an appropriate constant-rate event mechanism. Merely knowing an average rate is not enough to establish an exponential model.
    SciPy uses **expon(loc=0, scale=1/rate)**. Scale is the mean waiting time, not the rate.
    At three arrivals per hour, scale = 1/3 hour = 20 minutes.
    The exponential density is highest at zero, with a right tail.
    
    A shifted variable T = c + W uses **loc=c**. Its support starts at c, mean is c + scale, and variance is scale².
    This adds a fixed delay; it is not the ordinary interarrival time itself. The unshifted exponential waiting time is memoryless; adding a fixed delay does not preserve that property for T.
    """)
    return


@app.cell
def _(mo):
    exp_rate = mo.ui.slider(start=0.5, stop=10, step=0.5, value=3, show_value=True, label="Event rate λ per hour")
    exp_shift = mo.ui.slider(start=0, stop=2, step=0.25, value=0, show_value=True, label="Fixed delay c (hours)")
    mo.hstack([exp_rate, exp_shift])
    return exp_rate, exp_shift


@app.cell
def _(draw_continuous, exp_rate, exp_shift, expon, simulation_seed):
    exponential_model = expon(loc=exp_shift.value, scale=1 / exp_rate.value)
    exponential_sequence = draw_continuous(exponential_model, simulation_seed.value + 2)
    return exponential_model, exponential_sequence


@app.cell
def _(exponential_sequence, sample_size):
    exponential_values = exponential_sequence[:sample_size.value]
    return (exponential_values,)


@app.cell
def _(exponential_model, exponential_values, histogram_bins, plot_density):
    plot_density(exponential_values, exponential_model, histogram_bins.value, "Exponential with optional fixed delay: sample versus model")
    return


@app.cell
def _(exponential_model, exponential_values, summary_table, mo):
    mo.Html(
        summary_table(exponential_values, exponential_model).to_html(border=0, col_space=110, index=False)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Worked example: waiting for an arrival
    For a hypothetical homogeneous Poisson process at three arrivals per hour, W has mean 20 minutes.
    Keep units consistent: ten minutes is 1/6 hour, and thirty minutes is 1/2 hour.
    Calculate P(W < 10 minutes), P(W > 30 minutes), and P(10 < W < 30 minutes).
    """)
    return


@app.cell
def _(expon, pd, mo):
    waiting_model = expon(loc=0, scale=1 / 3)
    waiting_results = pd.DataFrame({"Event": ["W < 10 minutes", "W > 30 minutes", "10 < W < 30 minutes"], "Probability": [waiting_model.cdf(1 / 6), waiting_model.sf(1 / 2), waiting_model.cdf(1 / 2) - waiting_model.cdf(1 / 6)]})
    mo.Html(
        waiting_results.round(4).to_html(border=0, col_space=110, index=False)
        .replace("<table ", '<table style="width: auto;" ')
    )
    return (waiting_model,)


@app.cell
def _(shaded_interval, waiting_model):
    shaded_interval(waiting_model, 1 / 6, 1 / 2, "Waiting time interval, hours")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Double λ with c fixed at zero. Predict the mean and variance.
    2. Add a one-hour fixed delay. What changes: mean, variance, or both?
    3. Under the unshifted model, compare P(W > 30 minutes) with the probability of waiting another 30 minutes given that 20 minutes have already passed: P(W > 50 minutes | W > 20 minutes).
    
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Double λ:** With c = 0, the mean 1/λ halves and the variance 1/λ² becomes one quarter of its original value.
2. **One-hour fixed delay:** The mean increases by one hour; the variance stays the same.
3. **Memorylessness:** P(W > 30 minutes) = P(W > 50 minutes | W > 20 minutes) = exp(−1.5) ≈ 0.2231. The additional wait is 30 minutes in both cases.

**Discussion:** Doubling λ halves the mean and quarters the variance. A fixed delay changes the mean but not variance.
The unshifted model is **memoryless**:

$$P(W>s+t\mid W>s)=P(W>t),\qquad s,t\ge0.$$

At λ = 3 per hour, both thirty-minute probabilities equal exp(−1.5), approximately 0.2231. This is a model property, not a rule for every waiting-time situation.
""")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself: calculate probabilities
    Use the same unshifted waiting-time model at three arrivals per hour. Its mean is 1/3 hour. Convert minutes to hours before calculating.

    1. Calculate P(W < 20 minutes).
    2. Calculate P(W > 40 minutes).
    3. Calculate P(20 < W < 40 minutes).

    In the next cell, replace **None** with your calculation. Use **waiting_model.cdf()**, **waiting_model.sf()**, or a difference of CDF values.
    Run your cell, then expand **Show answers** to compare with the explanations, complete solution code, and numerical answers.
    """)
    return


@app.cell
def _():
    # 1. Calculate P(W < 20 minutes).
    exponential_answer_1 = None
    # 2. Calculate P(W > 40 minutes).
    exponential_answer_2 = None
    # 3. Calculate P(20 < W < 40 minutes).
    exponential_answer_3 = None
    print("Answer 1:", exponential_answer_1)
    print("Answer 2:", exponential_answer_2)
    print("Answer 3:", exponential_answer_3)
    return


@app.cell(hide_code=True)
def _(waiting_model, mo):
    _answers = f"""
1. **P(W < 20 minutes)** = {waiting_model.cdf(1 / 3):.4f}. Twenty minutes is 1/3 hour; the probability is 1 − exp(−1).
2. **P(W > 40 minutes)** = {waiting_model.sf(2 / 3):.4f}. Forty minutes is 2/3 hour; the probability is exp(−2).
3. **P(20 < W < 40 minutes)** = {waiting_model.cdf(2 / 3) - waiting_model.cdf(1 / 3):.4f}. The probability is exp(−1) − exp(−2).

```python
exponential_answer_1 = waiting_model.cdf(1 / 3)
exponential_answer_2 = waiting_model.sf(2 / 3)
exponential_answer_3 = waiting_model.cdf(2 / 3) - waiting_model.cdf(1 / 3)
```
"""
    mo.accordion({"Show answers": mo.md(_answers)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summaries and finite samples
    Observed variance below uses **ddof=0** to describe the generated observations. An unbiased population-variance estimator for independent, identically distributed observations uses **ddof=1** when there are at least two observations.
    Neither convention forces a finite sample to match theoretical variance.
    A plot shows a finite range; normal and exponential tails continue beyond it. The histogram range includes every displayed observation, so normalization does not discard sample observations.
    
    ## Conclusions
    | Model | Mean | Variance | SciPy scale |
    |---|---|---|---|
    | Uniform(a,b) | (a+b)/2 | (b−a)²/12 | b−a |
    | Normal N(μ,σ²) | μ | σ² | σ |
    | Shifted exponential c+W | c+1/λ | 1/λ² | 1/λ |
    
    - Probability is area under the PDF, not its height at a point.
    - Exact points have probability zero in these continuous models.
    - Specify parameter meanings and measurement units before calculating.
    - Simulated histograms estimate shape; theoretical curves and moments belong to the model.
    - CDFs, survival functions, and CDF differences answer lower-tail, upper-tail, and interval questions.
    - Select a model using assumptions and context rather than a convenient curve shape.
    
    ## Check your understanding
    1. Can a density exceed one? Can a probability exceed one?
    2. Which SciPy parameters produce Uniform(20,40)?
    3. What variance does **norm(loc=5, scale=2)** have?
    4. At a rate of four arrivals per hour, what exponential scale is needed?
    5. Is **pdf(15)** the probability of an exact waiting time of 15?
    6. Does changing histogram bins change the theoretical CDF?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Show answers": mo.md(r"""
1. **Density versus probability:** A density can exceed one; a probability cannot. Probability is an area, not a density height.
2. **Uniform(20,40):** Use loc=20 and scale=40−20=20.
3. **Normal variance:** scale=2 is the standard deviation, so the variance is 2²=4.
4. **Exponential scale:** 1/4 = 0.25 hour, or 15 minutes. The numerical scale must match the chosen time unit.
5. **PDF at 15:** No. It is a density height; P(X=15)=0 under the continuous model.
6. **Histogram bins:** No. They change the visual summary, not the theoretical CDF.
""")})
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
