import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng()
    # A lesson draws many figures. This stops the warning about how many are open.
    plt.rcParams["figure.max_open_warning"] = 0
    return mo, np, pd, plt, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Probability: Coins and Dice

    ## Objectives

    - Simulate coin tosses and watch the frequencies of heads and tails move toward 1/2 as the number of tosses grows.
    - Simulate fair dice and watch the frequency of each face move toward 1/6.
    - Simulate a weighted die, with probability 1/2 of showing 5, and compare it with the fair die.

    ## Background

    This lesson introduces probability through coin tosses and dice rolls. NumPy draws the random outcomes, and Matplotlib graphs the running frequency after each trial. The graphs illustrate the law of large numbers: as the number of trials grows, the experimental frequencies settle near the theoretical probabilities. A weighted die shows what changes when those probabilities are not equal.

    ## Datasets Used

    This lesson does not use an external dataset. The coin tosses and dice rolls are generated in the code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flipping coins

    `coin_trial` tosses one coin. It returns `True` for heads and `False` for tails.

    `coin_simulate(n)` tosses `n` coins. After each toss it records the frequency of heads so far and the frequency of tails so far, then graphs both series. Heads are green and tails are blue.
    """)
    return


@app.cell
def _(rng):
    def coin_trial():
        """
        Simulate one coin toss.
        Return True for heads and False for tails.
        """
        return rng.random() > 0.5

    return (coin_trial,)


@app.cell
def _(coin_trial, np, plt):
    def coin_simulate(n):
        """
        Simulate n coin tosses and graph the running frequencies.
        """
        # One result per toss: True is heads, False is tails.
        coins = np.array([coin_trial() for i in range(n)])

        # Running frequency of heads after each toss.
        heads: int = 0
        p_heads = np.ndarray(n)
        for i in range(n):
            if coins[i]:
                heads += 1
            p_heads[i] = 1.0 * heads / (i + 1)

        # The frequency of tails is the complement of the frequency of heads.
        p_tails = [1 - p_heads[i] for i in range(n)]

        x = np.arange(n)
        plt.plot(x, p_heads, "og", label="Head")
        plt.plot(x, p_tails, "ob", label="Tail")
        plt.legend()
        plt.ylabel("Frequency")
        plt.title("Coins")
        plt.show()
        print("Heads = %2i    Prob(Head) = %.3f" % (heads, heads / n))
        print("Tails = %2i    Prob(Tail) = %.3f" % (n - heads, (n - heads) / n))

    return (coin_simulate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One coin

    With one toss, the face that appears has frequency 1 and the other has frequency 0.
    """)
    return


@app.cell
def _(coin_simulate):
    # Tossing one coin
    coin_simulate(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two coins and ten coins

    Each dot is the frequency after that toss. The last two printed lines summarize the whole experiment.
    """)
    return


@app.cell
def _(coin_simulate):
    # Tossing 2 coins
    coin_simulate(2)
    return


@app.cell
def _(coin_simulate):
    # Tossing 10 coins
    coin_simulate(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### More tosses

    Run the next cells in order. The frequencies should move closer to 0.5.
    """)
    return


@app.cell
def _(coin_simulate):
    # Tossing 20 coins
    coin_simulate(20)
    return


@app.cell
def _(coin_simulate):
    # Tossing 50 coins
    coin_simulate(50)
    return


@app.cell
def _(coin_simulate):
    # Tossing 100 coins
    coin_simulate(100)
    return


@app.cell
def _(coin_simulate):
    # Tossing 500 coins
    coin_simulate(500)
    return


@app.cell
def _(coin_simulate):
    # Tossing 1000 coins
    coin_simulate(1000)
    return


@app.cell
def _(coin_simulate):
    # Tossing 10000 coins
    coin_simulate(10000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The frequencies of heads and tails converge to 0.5.

    ### Try it yourself

    Change `n_tosses` and run the cell. Compare a small number of tosses with a large one.
    """)
    return


@app.cell
def _(coin_simulate):
    n_tosses = 200
    coin_simulate(n_tosses)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rolling fair dice

    `die_trial` rolls one fair die. The list `pr` holds the cumulative probabilities of a fair die: each face has probability 1/6.

    `fair_die_simulate(n)` rolls `n` dice and graphs the running frequency of each face. The colors are green (1), cyan (2), blue (3), black (4), magenta (5), and red (6).
    """)
    return


@app.cell
def _(rng):
    def die_trial(pr=[0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]):
        """
        Simulate one die roll.
        pr holds the cumulative probabilities. The default is a fair die.
        Replace pr to simulate a weighted die.
        """
        rnd = rng.random()
        faces = [i for i in range(1, 7) if pr[i - 1] <= rnd < pr[i]]
        return faces[0]

    return (die_trial,)


@app.cell
def _(die_trial, np, pd, plt):
    def fair_die_simulate(n):
        """
        Simulate n rolls of a fair die and graph the running frequencies.
        """
        cols = ["1", "2", "3", "4", "5", "6"]
        dice = []
        new_die = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
        die_color = {
            "1": "og",
            "2": "oc",
            "3": "ob",
            "4": "ok",
            "5": "om",
            "6": "or",
        }
        fq = pd.DataFrame(columns=cols)
        for i in range(n):
            dice.append(die_trial())
            for key in new_die:
                new_die[key] = dice.count(int(key)) / (i + 1)
            fq.loc[i] = new_die
        x = np.arange(1, n + 1)
        for c in cols:
            plt.plot(x, fq[c], die_color[c], label=c)
        plt.legend()
        plt.ylabel("Frequency")
        plt.title("Dice")
        plt.show()
        for c in cols:
            print(
                "N%c = %3i    Prob(N%c) = %.3f"
                % (c, dice.count(int(c)), c, fq[c][n - 1])
            )

    return (fair_die_simulate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With one roll, the face that appears has frequency 1 and the other faces have frequency 0. A second roll still leaves most frequencies at 0.
    """)
    return


@app.cell
def _(fair_die_simulate):
    # Rolling one die
    fair_die_simulate(1)
    return


@app.cell
def _(fair_die_simulate):
    # Rolling two dice
    fair_die_simulate(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Keep increasing the number of rolls. For a fair die, each face has theoretical probability 1/6, about 0.167.
    """)
    return


@app.cell
def _(fair_die_simulate):
    # Rolling 5 dice
    fair_die_simulate(5)
    return


@app.cell
def _(fair_die_simulate):
    # Rolling 6 fair dice
    fair_die_simulate(6)
    return


@app.cell
def _(fair_die_simulate):
    # Rolling 30 fair dice
    fair_die_simulate(30)
    return


@app.cell
def _(fair_die_simulate):
    # Rolling 100 fair dice
    fair_die_simulate(100)
    return


@app.cell
def _(fair_die_simulate):
    # Rolling 600 fair dice
    fair_die_simulate(600)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After several hundred rolls, the six frequencies sit near 1/6 = 0.167.

    ### Try it yourself

    Change `n_fair_rolls` and run the cell.
    """)
    return


@app.cell
def _(fair_die_simulate):
    n_fair_rolls = 60
    fair_die_simulate(n_fair_rolls)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rolling unfair dice

    The next experiment uses a weighted die. The probability of 5 is 1/2. The probability of each other face is 1/10.

    The cumulative probabilities passed to `die_trial` are `[0, 0.1, 0.2, 0.3, 0.4, 0.9, 1]`. The wide gap from 0.4 to 0.9 is the face 5.
    """)
    return


@app.cell
def _(die_trial, np, pd, plt):
    def unfair_dice_simulate(n):
        """
        Simulate n rolls of a weighted die, with P(5) = 1/2,
        and graph the running frequencies.
        """
        cols = ["1", "2", "3", "4", "5", "6"]
        dice = []
        new_die = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
        die_color = {
            "1": "og",
            "2": "oc",
            "3": "ob",
            "4": "ok",
            "5": "om",
            "6": "or",
        }
        fq = pd.DataFrame(columns=cols)
        pr = [0, 1 / 10, 2 / 10, 3 / 10, 4 / 10, 9 / 10, 1]
        for i in range(n):
            dice.append(die_trial(pr))
            for key in new_die:
                new_die[key] = dice.count(int(key)) / (i + 1)
            fq.loc[i] = new_die
        x = np.arange(1, n + 1)
        for c in cols:
            plt.plot(x, fq[c], die_color[c], label=c)
        plt.legend()
        plt.ylabel("Frequency")
        plt.title("Dice")
        plt.show()
        for c in cols:
            print(
                "N%c = %3i    Prob(N%c) = %.3f"
                % (c, dice.count(int(c)), c, fq[c][n - 1])
            )

    return (unfair_dice_simulate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One weighted roll still shows a single face at frequency 1. With only two or ten rolls, several faces can still have frequency 0.
    """)
    return


@app.cell
def _(unfair_dice_simulate):
    # Rolling one unfair die
    unfair_dice_simulate(1)
    return


@app.cell
def _(unfair_dice_simulate):
    # Rolling two unfair dice
    unfair_dice_simulate(2)
    return


@app.cell
def _(unfair_dice_simulate):
    # Rolling ten unfair dice
    unfair_dice_simulate(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By 30 rolls, 5 (magenta) is usually ahead of the other faces. By 600 rolls, its frequency is near 1/2 and the others are near 1/10 = 0.1.
    """)
    return


@app.cell
def _(unfair_dice_simulate):
    # Rolling 30 unfair dice
    unfair_dice_simulate(30)
    return


@app.cell
def _(unfair_dice_simulate):
    # Rolling 120 unfair dice
    unfair_dice_simulate(120)
    return


@app.cell
def _(unfair_dice_simulate):
    # Rolling 600 unfair dice
    unfair_dice_simulate(600)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it yourself

    Change `n_unfair_rolls` and run the cell. Watch the magenta series, which is the face 5.
    """)
    return


@app.cell
def _(unfair_dice_simulate):
    n_unfair_rolls = 80
    unfair_dice_simulate(n_unfair_rolls)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    **Key takeaways:**

    - With more coin tosses, the frequencies of heads and tails settle near 1/2.
    - With more rolls of a fair die, the frequency of each face settles near 1/6.
    - On the weighted die, the frequency of 5 settles near 1/2 and the other faces settle near 1/10.
    - A small experiment can land far from the theoretical probability. A large experiment shows the law of large numbers.

    ## References

    - Unpingco, J. (2019) *Python for Probability, Statistics, and Machine Learning*, USA: Springer, chapter 2.
    """)
    return


if __name__ == "__main__":
    app.run()
