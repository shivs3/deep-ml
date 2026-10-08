def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.

    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis

    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """

    # P(H_i) * P(E|H_i)
    numerator = [
        prior * likelihood
        for prior, likelihood in zip(priors, likelihoods)
    ]

    # P(E) = sum of P(H_i)P(E|H_i)
    evidence = sum(numerator)

    # Avoid division by zero
    if evidence == 0:
        return [0.0] * len(priors)

    # P(H_i|E)
    posterior = [
        value / evidence
        for value in numerator
    ]

    return posterior