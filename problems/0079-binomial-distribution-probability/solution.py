import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.

    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial

    Returns:
        Probability of k successes
    """

    # Number of ways to choose k successes from n trials
    combinations = math.comb(n, k)

    # Binomial probability formula
    probability = (
        combinations
        * (p ** k)
        * ((1 - p) ** (n - k))
    )

    return probability