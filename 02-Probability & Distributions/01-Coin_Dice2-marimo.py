import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Probability: Coins and Dice
    
    ## Learning goals
    By the end of this lesson, you should be able to:
    - Identify the possible outcomes of a coin toss or a die roll.
    - Distinguish theoretical probability from observed relative frequency.
    - Calculate relative frequency as count divided by number of trials.
    - Interpret graphs of running relative frequencies.
    - Explain the law of large numbers without assuming exact or steady agreement.
    - Compare fair and weighted dice and explain random variation.
    
    ## Before we simulate
    A **trial** is one repetition of an experiment: one toss or one roll.
    An **outcome** is the result of a trial. The **sample space** lists all possible outcomes.
    For a coin it is {Heads, Tails}; for a die it is {1, 2, 3, 4, 5, 6}.
    
    For a fair coin, P(Heads) = P(Tails) = 1/2. For a fair die, each face has probability 1/6.
    These are properties of our model, not numbers calculated from one experiment.
    
    **Relative frequency = number of occurrences / number of trials.**
    For example, 12 heads in 20 tosses gives a relative frequency of 12/20 = 0.60.
    The theoretical probability remains 0.50. A running relative frequency updates this calculation after every trial.
    
    We assume that successive trials are independent and that their probabilities stay the same.
    Independent means that the result of one trial does not change the probabilities on the next trial.
    
    ## Reproducible simulations
    NumPy generates pseudorandom outcomes. A **seed** initializes its generator.
    The same seed and inputs reproduce the same experiment in the same software environment.
    A different seed gives another experiment; it does not change the theoretical probabilities.
    
    We create a fresh generator inside each simulation function. This keeps the results independent of which notebook cell you run first.
    No external dataset is needed.
    """)
    return


@app.cell
def _(np):
    def simulate_outcomes(n, outcomes, probabilities, seed):
        """Generate n independent outcomes with the specified probabilities."""
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
            raise ValueError("The number of trials must be a positive integer.")
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 1 or len(outcomes) != len(probabilities):
            raise ValueError("Provide one probability for each outcome.")
        if len(set(outcomes)) != len(outcomes):
            raise ValueError("Outcome labels must be distinct.")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("Probabilities must be finite and nonnegative.")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Probabilities must sum to 1.")
        generator = np.random.default_rng(seed)
        return generator.choice(outcomes, size=n, p=probabilities)


    def running_frequencies(results, outcomes):
        """Count each outcome cumulatively, then divide by the trial numbers."""
        trial_numbers = np.arange(1, len(results) + 1)
        frequencies = []
        for outcome in outcomes:
            # True counts as 1; False counts as 0.
            matches = results == outcome
            cumulative_counts = np.cumsum(matches)
            frequencies.append(cumulative_counts / trial_numbers)
        return np.asarray(frequencies)
    return running_frequencies, simulate_outcomes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the code works
    `simulate_outcomes` receives ordinary probabilities, such as `[0.5, 0.5]`.
    There is no need to construct cumulative probability boundaries yourself.
    
    For results `["Heads", "Tails", "Heads"]`, the heads matches are `[True, False, True]`.
    `np.cumsum` adds these as it goes: `[1, 1, 2]`.
    Dividing by trial numbers `[1, 2, 3]` gives `[1, 0.5, 0.667]`.
    We scan the results once per possible outcome instead of repeatedly recounting an expanding list.
    
    The next helper displays a graph and a count summary. You can focus on the experiments before studying its plotting details.
    """)
    return


@app.cell
def _(np, plt, running_frequencies):
    def show_experiment(results, outcomes, probabilities, title):
        frequencies = running_frequencies(results, outcomes)
        trials = np.arange(1, len(results) + 1)
        colors = ["tab:green", "tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for index, outcome in enumerate(outcomes):
            color = colors[index]
            ax.plot(trials, frequencies[index], color=color, label=str(outcome),
                    marker="o" if len(results) <= 20 else None, markersize=4)
            ax.axhline(probabilities[index], color=color, linestyle="--", alpha=0.55)
        ax.set(xlabel="Number of trials", ylabel="Relative frequency",
               title=title, ylim=(-0.02, 1.02))
        ax.set_xlim(0.5, max(1.5, len(results)))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=len(outcomes))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        # Close the pyplot registration; the returned figure can still be displayed.
        plt.close(fig)
        print("Dashed lines show theoretical probabilities.")
        print(f"Trials: {len(results)}")
        print(f"{'Outcome':<10} {'Count':>8} {'Observed':>10} {'Theoretical':>12}")
        for outcome, probability in zip(outcomes, probabilities):
            count = int(np.count_nonzero(results == outcome))
            print(f"{str(outcome):<10} {count:>8} {count / len(results):>10.3f} {probability:>12.3f}")
        return fig
    return (show_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A fair coin: extend one experiment
    **Predict first:** Must ten tosses contain exactly five heads?
    
    Generate 10,000 tosses once. Every smaller view below uses the beginning of this same sequence.
    Changing the displayed number of trials therefore extends or shortens one experiment.
    The slider controls the number of results shown, not the probabilities.
    """)
    return


@app.cell
def _(simulate_outcomes):
    coin_results = simulate_outcomes(10_000, ["Heads", "Tails"], [0.5, 0.5], seed=2026)
    return (coin_results,)


@app.cell
def _(mo):
    coin_n = mo.ui.slider(steps=[1, 2, 10, 20, 50, 100, 500, 1000, 10000], value=20, label="Coin tosses displayed")
    coin_n
    return (coin_n,)


@app.cell
def _(coin_n, coin_results, show_experiment):
    show_experiment(coin_results[:coin_n.value], ["Heads", "Tails"], [0.5, 0.5], "Fair coin: one growing experiment")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself: predict, observe, explain
    1. Inspect 1, 2, 10, 100, and 10,000 tosses. Calculate the heads relative frequency from the printed count.
    2. Does the frequency get closer to 0.5 at every increase? Use your observations to explain.
    3. At the largest sample size, must the frequency be exactly 0.5?
    4. If tails appeared five times in a row, what would P(Heads) be on the next toss?
    
    **Discussion:** Ten tosses do not guarantee five heads. Relative frequencies fluctuate; a larger sample can sometimes be farther from 0.5. Under our independence assumption, P(Heads) on the next toss is still 0.5, whatever happened before.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Repeat the coin experiment
    Extending one sequence and starting a new experiment answer different questions.
    Here we compare five separate experiments at each sample size. Each row uses its own seed.
    
    **Predict:** Which column should usually have a smaller spread of relative frequencies?
    """)
    return


@app.cell
def _(mo, np, simulate_outcomes):
    repeat_rows = []
    for experiment_seed in [11, 22, 33, 44, 55]:
        repeated_coin = simulate_outcomes(2000, ["Heads", "Tails"], [0.5, 0.5], experiment_seed)
        small_frequency = np.count_nonzero(repeated_coin[:20] == "Heads") / 20
        large_frequency = np.count_nonzero(repeated_coin == "Heads") / 2000
        repeat_rows.append(f"| {experiment_seed} | {small_frequency:.3f} | {large_frequency:.3f} |")
    mo.md("| Seed | Heads: 20 tosses | Heads: 2,000 tosses |\n|---|---:|---:|\n" + "\n".join(repeat_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Change the five seeds in the preceding cell and compare again.
    Do all experiments give the same result? Is the larger experiment closer to 0.5 in every row?
    
    **Discussion:** Larger samples typically show less variation in relative frequency. Five experiments illustrate this idea; they do not establish a guarantee for every experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A fair die
    
    Each face has theoretical probability 1/6, approximately 0.167. Six rolls do not guarantee one occurrence of each face. Coin tosses and die rolls follow the same relative-frequency calculation.
    """)
    return


@app.cell
def _(simulate_outcomes):
    fair_results = simulate_outcomes(6000, [1, 2, 3, 4, 5, 6], [1 / 6] * 6, seed=2027)
    return (fair_results,)


@app.cell
def _(mo):
    fair_n = mo.ui.slider(steps=[1, 2, 6, 10, 30, 100, 600, 6000], value=6, label="Die rolls displayed")
    fair_n
    return (fair_n,)


@app.cell
def _(fair_n, fair_results, show_experiment):
    show_experiment(fair_results[:fair_n.value], [1, 2, 3, 4, 5, 6], [1 / 6] * 6, "A fair die: one growing experiment")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    
    1. Inspect six rolls. How many distinct faces appeared? Could a fair die show the same face six times?
    2. Inspect 600 and 6,000 rolls. Compare every relative frequency with 1/6.
    3. Explain why a face with observed frequency 0 still has theoretical probability 1/6.
    
    **Discussion:** A possible outcome need not appear in a short experiment. Even at 6,000 rolls, exactly 1,000 occurrences of each face are not required.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. A weighted die
    
    Here P(5) = 0.5 and every other face has probability 0.1. The probabilities sum to 1. More rolls help reveal this unequal distribution; they do not make the die fair.
    """)
    return


@app.cell
def _(simulate_outcomes):
    weighted_results = simulate_outcomes(6000, [1, 2, 3, 4, 5, 6], [0.1, 0.1, 0.1, 0.1, 0.5, 0.1], seed=2028)
    return (weighted_results,)


@app.cell
def _(mo):
    weighted_n = mo.ui.slider(steps=[1, 2, 6, 10, 30, 100, 600, 6000], value=6, label="Die rolls displayed")
    weighted_n
    return (weighted_n,)


@app.cell
def _(weighted_n, weighted_results, show_experiment):
    show_experiment(weighted_results[:weighted_n.value], [1, 2, 3, 4, 5, 6], [0.1, 0.1, 0.1, 0.1, 0.5, 0.1], "A weighted die: one growing experiment")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    
    1. Before displaying 100 rolls, predict the expected number of fives: 100 × 0.5 = 50. Compare with the observed count.
    2. Inspect 6,000 rolls. Which relative frequencies tend toward 0.1? Which tends toward 0.5?
    3. Does increasing the sample size make all faces equally frequent?
    
    **Discussion:** An expected count is a model-based average over repeated experiments, not a guaranteed count. The weighted die remains weighted regardless of sample size.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Design your own weighted die
    Give face 6 probability 0.40 and each other face probability 0.12.
    Check: 5 × 0.12 + 0.40 = 1.
    
    **Predict:** Which face will tend to appear most often? About how many sixes do you expect in 1,000 rolls?
    Change the probabilities below to another valid distribution. Keep six entries, all nonnegative, summing to 1.
    Change the seed to start another experiment with the same probabilities.
    """)
    return


@app.cell
def _(show_experiment, simulate_outcomes):
    custom_probabilities = [0.12, 0.12, 0.12, 0.12, 0.12, 0.40]
    custom_seed = 2029
    custom_results = simulate_outcomes(1000, [1, 2, 3, 4, 5, 6], custom_probabilities, custom_seed)
    show_experiment(custom_results, [1, 2, 3, 4, 5, 6], custom_probabilities, "Your weighted die")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Discussion:** The expected count of sixes is 1,000 × 0.40 = 400. The observed count can differ.
    A seed changes the realized outcomes, not the specified probabilities.
    
    ## Conclusions
    - Theoretical probability describes the model; relative frequency describes observed results.
    - Relative frequency is an outcome count divided by the number of trials.
    - Under independent trials with unchanged probabilities, the **law of large numbers** says that relative frequencies converge to theoretical probabilities as the number of trials grows without bound.
    - This does not guarantee exact agreement at a finite sample size or improvement at every additional trial.
    - A fair die has equal face probabilities. A weighted die has unequal probabilities; more rolls do not remove that weighting.
    - Previous results do not change the next trial's probabilities in these independent models.
    - Seeds make simulations reproducible. Different seeds help us explore variation between experiments.
    
    ## Check your understanding
    1. A fair coin produces 7 heads in 10 tosses. What are its observed heads frequency and theoretical heads probability?
    2. A fair die produces no sixes in 12 rolls. Has P(6) become zero?
    3. Does the law of large numbers imply that tails must follow a long run of heads?
    4. A weighted die has P(5) = 0.5. What is its expected number of fives in 200 rolls? Must that count occur?
    
    **Answers:** (1) 0.70 and 0.50. (2) No: P(6) remains 1/6. (3) No: trials are independent. (4) 100; no, the observed count can differ.
    
    ## Reference
    Unpingco, J. (2019). *Python for Probability, Statistics, and Machine Learning*. Springer, Chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
