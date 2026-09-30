import marimo

__generated_with = "0.25.0"

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Our first function: simulate outcomes
    A function is reusable code. Define it once, then call it with the experiment you want.
    The probabilities must match the outcomes in order, be nonnegative, and sum to 1.
    """)
    return


@app.cell
def _(np):
    def simulate_outcomes(n, outcomes, probabilities, seed):
        """Simulate n trials using the given outcomes and probabilities."""
        generator = np.random.default_rng(seed)
        return generator.choice(outcomes, size=n, p=probabilities)
    return (simulate_outcomes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How to use `simulate_outcomes`
    | Argument | Meaning | Coin example |
    |---|---|---|
    | `n` | Number of trials | `10` |
    | `outcomes` | Possible results | `["Heads", "Tails"]` |
    | `probabilities` | Probability of each result, in the same order | `[0.5, 0.5]` |
    | `seed` | Number used to reproduce the experiment | `2026` |

    Run the next cell to see ten tosses. Change `n` to 20 and run it again.
    Then change the seed: the probabilities stay the same, but the results can change.
    """)
    return


@app.cell
def _(simulate_outcomes):
    example_tosses = simulate_outcomes(
        n=10,
        outcomes=["Heads", "Tails"],
        probabilities=[0.5, 0.5],
        seed=2026,
    )
    print(example_tosses)
    return (example_tosses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Our second function: running relative frequencies
    After each trial, divide the count of an outcome so far by the number of trials so far.
    This function does that calculation for each possible outcome.
    """)
    return


@app.cell
def _(np):
    def running_frequencies(results, outcomes):
        """Calculate each outcome's relative frequency after every trial."""
        trial_numbers = np.arange(1, len(results) + 1)
        frequencies = []
        for outcome in outcomes:
            # True counts as 1; False counts as 0.
            matches = results == outcome
            cumulative_counts = np.cumsum(matches)
            frequencies.append(cumulative_counts / trial_numbers)
        return np.asarray(frequencies)
    return (running_frequencies,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How to use `running_frequencies`
    Pass the results and the possible outcomes in the order you want them displayed.
    For the sequence Heads, Tails, Heads:

    | Trial | Result | Heads so far | Heads relative frequency |
    |---|---|---:|---:|
    | 1 | Heads | 1 | 1/1 = 1.00 |
    | 2 | Tails | 1 | 1/2 = 0.50 |
    | 3 | Heads | 2 | 2/3 ≈ 0.67 |

    `results == "Heads"` gives `[True, False, True]`.
    `np.cumsum` adds these as it goes, giving `[1, 1, 2]`.
    Dividing by `[1, 2, 3]` gives the frequencies in the table.
    The first output row is Heads; the second is Tails.
    """)
    return


@app.cell
def _(np, running_frequencies):
    example_results = np.array(["Heads", "Tails", "Heads"])
    example_frequencies = running_frequencies(example_results, ["Heads", "Tails"])
    print("Heads:", example_frequencies[0])
    print("Tails:", example_frequencies[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Try it:** Add `"Tails"` to the sequence above. Before running it, calculate
    both final relative frequencies. Use the printed results to check your calculations.

    ## Our third function: display an experiment
    `show_experiment` draws the running frequencies and prints a count summary.
    You can expand its code if you want to explore how the graph is drawn.
    """)
    return


@app.cell(hide_code=True)
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
    ### How to use `show_experiment`
    Pass the simulated results, the possible outcomes, their theoretical probabilities,
    and a graph title. The outcomes and probabilities must use the same order.

    - Solid lines show running relative frequencies.
    - Dashed lines show theoretical probabilities.
    - The summary shows counts and final relative frequencies.

    Here we display the tosses from our first example.
    """)
    return


@app.cell
def _(example_tosses, show_experiment):
    show_experiment(example_tosses, ["Heads", "Tails"], [0.5, 0.5], "Our first coin experiment")
    return


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
    coin_n = mo.ui.slider(steps=[1, 2, 10, 20, 50, 100, 500, 1000, 10000], value=20, show_value=True, 
    label="Coin tosses displayed")
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
    
    <details>
    <summary>Show explanation</summary>
    
    Ten tosses do not guarantee five heads. Relative frequencies fluctuate; a larger sample can sometimes be farther from 0.5. Under our independence assumption, P(Heads) on the next toss is still 0.5, whatever happened before.
    
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Your workspace: count and calculate
    Edit `practice_coin_n` or `practice_coin_seed` below and run the cell.
    Count Heads in the displayed results. In the following cell, replace each `None`
    with your answer (a number or a Python calculation). Feedback updates when you run it.
    Use at most 100 tosses here so you can inspect the sequence.
    """)
    return


@app.cell
def _(simulate_outcomes):
    practice_coin_n = 20
    practice_coin_seed = 15
    practice_coin_results = simulate_outcomes(practice_coin_n, ["Heads", "Tails"], [0.5, 0.5], practice_coin_seed)
    print(practice_coin_results)
    return practice_coin_n, practice_coin_results


@app.cell
def _():
    coin_answer_count = None  # Replace with your count of Heads.
    coin_answer_frequency = None  # Replace with count / number of tosses.
    return coin_answer_count, coin_answer_frequency


@app.cell(hide_code=True)
def _(coin_answer_count, coin_answer_frequency, practice_coin_n, practice_coin_results, np):
    if coin_answer_count is None or coin_answer_frequency is None:
        print("Enter your count and relative frequency in the preceding cell.")
    else:
        _count = np.count_nonzero(practice_coin_results == "Heads")
        print("Heads count:", "Correct!" if coin_answer_count == _count else "Try counting Heads again.")
        print("Relative frequency:", "Correct!" if np.isclose(coin_answer_frequency, _count / practice_coin_n, atol=0.0005, rtol=0) else "Try count divided by number of tosses. Use at least three decimal places.")
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
    
    <details>
    <summary>Show explanation</summary>
    
    Larger samples typically show less variation in relative frequency. Five experiments illustrate this idea; they do not establish a guarantee for every experiment.
    
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Your workspace: compare two experiments
    Edit the seeds and number of tosses. Run the cell and calculate each Heads frequency.
    Then enter your answers below. Does changing the seed change the theoretical probability?
    """)
    return


@app.cell
def _(simulate_outcomes):
    repeat_practice_n = 20
    repeat_practice_a = simulate_outcomes(repeat_practice_n, ["Heads", "Tails"], [0.5, 0.5], seed=31)
    repeat_practice_b = simulate_outcomes(repeat_practice_n, ["Heads", "Tails"], [0.5, 0.5], seed=42)
    print("Experiment A:", repeat_practice_a)
    print("Experiment B:", repeat_practice_b)
    return repeat_practice_n, repeat_practice_a, repeat_practice_b


@app.cell
def _():
    repeat_answer_a = None  # Heads count in A / number of tosses.
    repeat_answer_b = None  # Heads count in B / number of tosses.
    return repeat_answer_a, repeat_answer_b


@app.cell(hide_code=True)
def _(np, repeat_answer_a, repeat_answer_b, repeat_practice_a, repeat_practice_b, repeat_practice_n):
    for _label, _answer, _results in [("A", repeat_answer_a, repeat_practice_a), ("B", repeat_answer_b, repeat_practice_b)]:
        if _answer is None:
            print(f"Experiment {_label}: enter your relative frequency.")
        else:
            _frequency = np.count_nonzero(_results == "Heads") / repeat_practice_n
            print(f"Experiment {_label}:", "Correct!" if np.isclose(_answer, _frequency, atol=0.0005, rtol=0) else "Try count divided by number of tosses.")
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
    fair_n = mo.ui.slider(steps=[1, 2, 6, 10, 30, 100, 600, 6000], value=6, show_value=True, 
    label="Die rolls displayed")
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
    
    <details>
    <summary>Show explanation</summary>
    
    A possible outcome need not appear in a short experiment. Even at 6,000 rolls, exactly 1,000 occurrences of each face are not required.
    
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Your workspace: choose a face
    Choose a face from 1 to 6 and edit the seed or number of rolls.
    Count that face in the printed sequence, then enter its count and relative frequency.
    """)
    return


@app.cell
def _(simulate_outcomes):
    practice_die_face = 6
    practice_die_n = 30
    practice_die_results = simulate_outcomes(practice_die_n, [1, 2, 3, 4, 5, 6], [1 / 6] * 6, seed=25)
    print(practice_die_results)
    print("Face to count:", practice_die_face)
    return practice_die_face, practice_die_n, practice_die_results


@app.cell
def _():
    die_answer_count = None
    die_answer_frequency = None
    return die_answer_count, die_answer_frequency


@app.cell(hide_code=True)
def _(np, practice_die_face, practice_die_n, practice_die_results, die_answer_count, die_answer_frequency):
    if die_answer_count is None or die_answer_frequency is None:
        print("Enter your count and relative frequency above.")
    else:
        _count = np.count_nonzero(practice_die_results == practice_die_face)
        print("Face count:", "Correct!" if die_answer_count == _count else "Count your selected face again.")
        print("Relative frequency:", "Correct!" if np.isclose(die_answer_frequency, _count / practice_die_n, atol=0.0005, rtol=0) else "Divide the count by the number of rolls. Use at least three decimal places.")
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
    weighted_n = mo.ui.slider(steps=[1, 2, 6, 10, 30, 100, 600, 6000], value=6, show_value=True, 
    label="Die rolls displayed")
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
    
    1. Before displaying 100 rolls, predict the expected number of fives. Compare with the observed count.
    2. Inspect 6,000 rolls. Which relative frequencies tend toward 0.1? Which tends toward 0.5?
    3. Does increasing the sample size make all faces equally frequent?
    
    <details>
    <summary>Show explanation</summary>
    
    An expected count is a model-based average over repeated experiments, not a guaranteed count. The weighted die remains weighted regardless of sample size.
    
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Your workspace: expected and observed counts
    Edit the number of rolls and seed. Enter your prediction before running the simulation.
    Use **expected count = number of trials × probability**.
    Then compare the observed count with your prediction. A difference is not a calculation error.
    """)
    return


@app.cell
def _():
    weighted_practice_n = 100
    weighted_practice_seed = 18
    weighted_answer_expected = None  # Your expected number of fives.
    return weighted_practice_n, weighted_practice_seed, weighted_answer_expected


@app.cell
def _(np, simulate_outcomes, weighted_practice_n, weighted_practice_seed, weighted_answer_expected):
    if weighted_answer_expected is None:
        print("Enter your expected count in the preceding cell to run the experiment.")
    else:
        print("Expected-count calculation:", "Correct!" if np.isclose(weighted_answer_expected, weighted_practice_n * 0.5) else "Try number of rolls multiplied by 0.5.")
        _results = simulate_outcomes(weighted_practice_n, [1, 2, 3, 4, 5, 6], [0.1, 0.1, 0.1, 0.1, 0.5, 0.1], weighted_practice_seed)
        print("Observed number of fives:", np.count_nonzero(_results == 5))
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Your workspace: design, predict, and check
    Edit the probabilities, number of rolls, seed, and face to investigate.
    Replace `None` with your expected count for that face.
    The next cell checks your distribution and calculation, then displays your experiment.
    """)
    return


@app.cell
def _():
    custom_probabilities = [0.12, 0.12, 0.12, 0.12, 0.12, 0.40]
    custom_n = 1000
    custom_seed = 2029
    custom_face = 6
    custom_answer_expected = None  # Number of rolls × probability of your face.
    return custom_probabilities, custom_n, custom_seed, custom_face, custom_answer_expected


@app.cell
def _(mo, np, show_experiment, simulate_outcomes, custom_probabilities, custom_n, custom_seed, custom_face, custom_answer_expected):
    print("Sum of probabilities:", sum(custom_probabilities))
    if len(custom_probabilities) != 6 or not np.all(np.isfinite(custom_probabilities)) or np.any(np.array(custom_probabilities) < 0) or not np.isclose(sum(custom_probabilities), 1):
        print("Use six nonnegative probabilities that sum to 1.")
    elif custom_face not in [1, 2, 3, 4, 5, 6]:
        print("Choose a face from 1 to 6.")
    elif custom_answer_expected is None:
        print("Enter your expected count above to run the experiment.")
    else:
        _expected = custom_n * custom_probabilities[custom_face - 1]
        print("Expected-count calculation:", "Correct!" if np.isclose(custom_answer_expected, _expected) else "Try number of rolls multiplied by your face's probability.")
        _custom_results = simulate_outcomes(custom_n, [1, 2, 3, 4, 5, 6], custom_probabilities, custom_seed)
        print("Observed count for your face:", np.count_nonzero(_custom_results == custom_face))
        mo.output.append(show_experiment(_custom_results, [1, 2, 3, 4, 5, 6], custom_probabilities, "Your weighted die"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <details>
    <summary>Show explanation</summary>
    
    With the original settings, the expected count of sixes is 1,000 × 0.40 = 400.
    If you change the settings, use your new number of rolls and probability. The observed count can differ.
    A seed changes the realized outcomes, not the specified probabilities.
    
    </details>
    
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
    
    <details>
    <summary>Show answers</summary>
    
    **Answers:** (1) 0.70 and 0.50. (2) No: P(6) remains 1/6. (3) No: trials are independent. (4) 100; no, the observed count can differ.
    
    </details>
    
    ## Reference
    Unpingco, J. (2019). *Python for Probability, Statistics, and Machine Learning*. Springer, Chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
