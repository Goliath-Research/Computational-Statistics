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
    - The same seed and inputs reproduce the same experiment in the same software environment.
    - A different seed gives another experiment; it does not change the theoretical probabilities.

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
    Add `"Tails"` to `example_results` above to explore another sequence.
    Calculate both final relative frequencies, then run the cell to check them.

    ## Our third function: display an experiment
    `show_experiment` draws the running frequencies and prints a count summary.
    """)
    return


@app.cell
def _(np, plt, running_frequencies):
    def show_experiment(results, outcomes, probabilities, title):
        """
        Plot running relative frequencies and print outcome counts,
        observed relative frequencies, and theoretical probabilities.
        """
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
    In each **Try it yourself** section, write your Python script in the code cell
    immediately below the instructions. Open this notebook in the **marimo editor**
    to edit and run these cells; a read-only page does not allow code editing.

    The cell contains named variables initialized to `None` so the notebook can run
    before you start. Replace those placeholders with your code, and add as many lines
    as you need. 

    For counting, `np.count_nonzero(results == outcome)` counts how often an outcome
    appears. You can also use a loop. Use `len(results)` for the number of trials.
    """)
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
    ### Try it yourself
    1. Inspect 1, 2, 10, 100, and 10,000 tosses with the slider.
    2. Does the frequency get closer to 0.5 at every increase? Must it be exactly 0.5 at the largest size?
    3. If Tails appeared five times in a row, what would P(Heads) be on the next toss?
    """)
    return

@app.cell(hide_code=True)
def _(mo):    
    mo.accordion({
    "Show answers": mo.md("""
    1. At one toss, the Heads frequency is 0 or 1.
       At two tosses, it can be 0, 0.5, or 1.
       With more tosses, it generally tends toward 0.5.

    2. No. The frequency does not necessarily get closer
       at every increase, nor must it equal exactly 0.5
       at 10,000 tosses.

    3. P(Heads) = 0.5. Independent tosses are unaffected
       by previous results.
    """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Write your script in the next cell:**
    - Use `coin_results[:coin_n.value]`, the results currently selected by the slider.
    - Calculate the number of Heads and their relative frequency.
    - Store your results in `student_coin_count` and `student_coin_frequency`.
    - Print your results. Move the slider and check that your calculations update.

    You may add variables, calculations, loops, and print statements to your script.
    """)
    return

@app.cell
def _(coin_n, coin_results):
    # Write your script here. Use the results selected by the slider.
    student_coin_results = coin_results[:coin_n.value]
    print(student_coin_results)

    # Replace None with code that counts Heads.
    student_coin_count = None

    # Replace None with code that calculates the Heads relative frequency.
    student_coin_frequency = None

    print("Heads count:", student_coin_count)
    print("Heads relative frequency:", student_coin_frequency)
    return student_coin_count, student_coin_frequency


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
    "Show answers": mo.md('''
```python	
student_coin_count = np.count_nonzero(student_coin_results == "Heads")
student_coin_frequency = student_coin_count / len(student_coin_results)
```
    ''')
    })
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Discussion:** 
    - Ten tosses do not guarantee five Heads. 
    - Relative frequencies fluctuate, a larger sample can sometimes be farther from 0.5. 
    - Under our independence assumption, P(Heads) on the next toss is still 0.5, whatever happened before.
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
    repeat_samples = []
    for experiment_seed in [11, 22, 33, 44, 55]:
        repeated_coin = simulate_outcomes(2000, ["Heads", "Tails"], [0.5, 0.5], experiment_seed)
        repeat_samples.append(repeated_coin)
        small_frequency = np.count_nonzero(repeated_coin[:20] == "Heads") / 20
        large_frequency = np.count_nonzero(repeated_coin == "Heads") / 2000
        repeat_rows.append(f"| {experiment_seed} | {small_frequency:.3f} | {large_frequency:.3f} |")
    mo.md("| Seed | Heads: 20 tosses | Heads: 2,000 tosses |\n|---|---:|---:|\n" + "\n".join(repeat_rows))
    return (repeat_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Change the five seeds in the preceding experiment cell and compare again.
    Do all experiments give the same result? Is the larger experiment closer to 0.5 in every row?

    **Write your script in the next cell:**
    `repeat_samples` contains the five sequences used in the table above.
    - For each sequence, calculate the Heads relative frequency for the first 20 tosses and for all 2,000 tosses.
    - Build two lists: `student_repeat_small` and `student_repeat_large`, in the same order as the experiments.
    - Use a loop or another method you know. Print both lists and compare with the table.

    To start a loop, use `for sample in repeat_samples:`. Within the loop,
    `sample[:20]` selects the first 20 tosses; `sample` contains all the tosses.
    """)
    return


@app.cell
def _():
    # Write your script here. Add your loop and calculations below.
    # Replace None with the lists your script calculates.
    student_repeat_small = None
    student_repeat_large = None

    # For each sample in repeat_samples, count Heads and calculate both frequencies.
    print("20-toss frequencies:", student_repeat_small)
    print("2000-toss frequencies:", student_repeat_large)
    return student_repeat_large, student_repeat_small

@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show answers": mo.md('''
```python
student_repeat_small = [float(np.count_nonzero(_sample[:20] == "Heads") / 20) for _sample in repeat_samples]
student_repeat_large = [float(np.count_nonzero(_sample == "Heads") / 2000) for _sample in repeat_samples]

```
''')
    })
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Discussion:** Larger samples typically show less variation in relative frequency.
    Five experiments illustrate this idea; they do not establish a guarantee for every experiment.
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
    fair_results = simulate_outcomes(6000, [1, 2, 3, 4, 5, 6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], seed=2027)
    return (fair_results,)


@app.cell
def _(mo):
    fair_n = mo.ui.slider(steps=[1, 2, 6, 10, 30, 100, 600, 6000], value=10, show_value=True, 
    label="Die rolls displayed")
    fair_n
    return (fair_n,)


@app.cell
def _(fair_n, fair_results, show_experiment):
    show_experiment(fair_results[:fair_n.value], [1, 2, 3, 4, 5, 6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], "A fair die: one growing experiment")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Inspect six rolls. How many distinct faces appeared?
       Could a fair die show the same face six times?
    2. Inspect 600 and 6,000 rolls. Compare each relative frequency with 1/6.
       Must every face appear exactly 1,000 times in 6,000 rolls?
    3. If a face has observed frequency 0, what is its theoretical probability?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show answers": mo.md("""
        1. Count the different faces shown in the six rolls.
           Yes, a fair die could show the same face six times, although this is unlikely.

        2. With more rolls, the relative frequencies generally tend toward 1/6.
           No, exactly 1,000 occurrences of each face are not required.

        3. Its theoretical probability remains 1/6.
           A possible outcome does not have to appear in a short experiment.
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Write your script in the next cell:**
    - Choose `student_die_face` from 1 to 6.
    - Use `fair_results[:fair_n.value]`, the rolls selected by the slider.
    - Calculate your face's count and relative frequency.
    - Calculate the number of distinct faces observed.
    - Print your results. Change the slider and your selected face.

    `set(student_die_results)` gives the different faces observed.
    Use `len(...)` to count them.
    """)
    return


@app.cell
def _(fair_n, fair_results):
    # Choose a face, then write your script below.
    student_die_face = 6
    student_die_results = fair_results[:fair_n.value]

    # Replace None with your calculations.
    student_die_count = None
    student_die_frequency = None
    student_die_distinct = None

    print("Face count:", student_die_count)
    print("Relative frequency:", student_die_frequency)
    print("Distinct faces:", student_die_distinct)
    return (
        student_die_count,
        student_die_distinct,
        student_die_face,
        student_die_frequency,
        student_die_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show answers": mo.md('''
```python
student_die_count = np.count_nonzero(student_die_results == student_die_face)
student_die_frequency = float(student_die_count / len(student_die_results))
student_die_distinct = len(set(student_die_results))
```
''')
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Discussion:** A possible outcome need not appear in a short experiment.
    Even at 6000 rolls, exactly 1000 occurrences of each face are not required.
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
    weighted_n = mo.ui.slider(steps=[1, 2, 6, 10, 30, 100, 600, 6000], value=10, show_value=True, 
    label="Die rolls displayed")
    weighted_n
    return (weighted_n,)


@app.cell
def _(show_experiment, weighted_n, weighted_results):
    show_experiment(weighted_results[:weighted_n.value], [1, 2, 3, 4, 5, 6], [0.1, 0.1, 0.1, 0.1, 0.5, 0.1], "A weighted die: one growing experiment")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    1. Before displaying 100 rolls, predict the expected number of fives. Compare with the observed count.
    2. Inspect 6,000 rolls. Which relative frequencies tend toward 0.1? Which tends toward 0.5?
    3. Does increasing the sample size make all faces equally frequent?    
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show answers": mo.md("""
        1. The expected number of fives is 100 × 0.5 = 50.
           The observed count can differ from 50.

        2. The relative frequencies of faces 1, 2, 3, 4, and 6 tend toward 0.1.
           The relative frequency of face 5 tends toward 0.5.

        3. No. More rolls help reveal the unequal probabilities.
           The die remains weighted.
        """)
    })
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Write your script in the next cell:**
    - Use `weighted_results[:weighted_n.value]`, the rolls selected by the slider.
    - Calculate the expected number of fives: number of selected rolls × 0.5.
    - Calculate the observed number of fives and their relative frequency.
    - Print your results. Move the slider and compare the expected and observed counts.
    """)
    return


@app.cell
def _(weighted_n, weighted_results):
    # Write your script here.
    student_weighted_results = weighted_results[:weighted_n.value]

    # Replace None with your calculations.
    student_weighted_expected = None
    student_weighted_count = None
    student_weighted_frequency = None

    print("Expected fives:", student_weighted_expected)
    print("Observed fives:", student_weighted_count)
    print("Observed frequency:", student_weighted_frequency)
    return (
        student_weighted_count,
        student_weighted_expected,
        student_weighted_frequency,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show answers": mo.md('''
```python
student_weighted_expected = len(student_weighted_results) * 0.5
student_weighted_count = np.count_nonzero(student_weighted_results == 5)
student_weighted_frequency = float(student_weighted_count / len(student_weighted_results))
```
''')
    })
    return

@app.cell
def _(
    np,
    student_weighted_count,
    student_weighted_expected,
    student_weighted_frequency,
    weighted_n,
    weighted_results,
):
    if student_weighted_expected is None or student_weighted_count is None or student_weighted_frequency is None:
        print("Write and run your script above to calculate the expected count, observed count, and frequency.")
    else:
        _selected = weighted_results[:weighted_n.value]
        _expected = len(_selected) * 0.5
        _count = int(np.count_nonzero(_selected == 5))
        _frequency = _count / len(_selected)
        print("Expected count:", "Correct!" if np.isclose(student_weighted_expected, _expected) else "Multiply the number of selected rolls by 0.5.")
        print("Observed count:", "Correct!" if student_weighted_count == _count else "Count the fives in the selected rolls.")
        print("Relative frequency:", "Correct!" if np.isclose(student_weighted_frequency, _frequency) else "Divide the observed count by the number of selected rolls.")
        print(f"Check: expected fives = {_expected:g}; observed fives = {_count}; relative frequency = {_frequency:.4f}.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Discussion:** An expected count is a model-based average over repeated experiments,
    not a guaranteed count. The weighted die remains weighted regardless of sample size.
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
    custom_n = 1000
    custom_results = simulate_outcomes(custom_n, [1, 2, 3, 4, 5, 6], custom_probabilities, custom_seed)
    show_experiment(custom_results, [1, 2, 3, 4, 5, 6], custom_probabilities, "Your weighted die")
    return custom_n, custom_probabilities, custom_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself
    Change the probabilities in the experiment above to design another weighted die.
    Keep six nonnegative probabilities that sum to 1. Change the seed to repeat the experiment.

    **Write your script in the next cell:**
    - Choose `student_custom_face` from 1 to 6.
    - Calculate the sum of `custom_probabilities`.
    - Calculate the expected count for your selected face and its observed relative frequency in `custom_results`.
    - Store and print the calculations using the named variables below.

    The probability of face 1 is `custom_probabilities[0]`; the probability of face 6
    is `custom_probabilities[5]`. For a selected face, use index `student_custom_face - 1`.
    After changing the experiment, check that your script still gives correct results.
    """)
    return


@app.cell
def _():
    # Choose a face, then write your script below.
    student_custom_face = 6

    # Replace None with your calculations.
    student_custom_sum = None
    student_custom_expected = None
    student_custom_frequency = None

    print("Probability sum:", student_custom_sum)
    print("Expected count:", student_custom_expected)
    print("Observed frequency:", student_custom_frequency)
    return (
        student_custom_expected,
        student_custom_face,
        student_custom_frequency,
        student_custom_sum,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show answers": mo.md('''
```python
student_custom_sum = sum(custom_probabilities)
student_custom_expected = custom_n * custom_probabilities[student_custom_face - 1]
student_custom_frequency = float(np.count_nonzero(custom_results == student_custom_face) / custom_n)
```
''')
    })
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Discussion:** With the original settings, the expected count of sixes is
    1,000 × 0.40 = 400. The observed count can differ. A seed changes the realized
    outcomes, not the specified probabilities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
