import numpy as np

def cross_entropy_derivative(
    logits: list[float],
    target: int
) -> list[float]:
    """
    Compute the derivative of cross-entropy loss
    with respect to logits.

    Args:
        logits: Raw model outputs (before softmax)
        target: Index of the true class (0-indexed)

    Returns:
        Gradient vector where
        gradient[i] = dL/d(logits[i])
    """

    logits = np.array(logits, dtype=float)

    # Compute softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    # Gradient = softmax probabilities
    gradient = probabilities.copy()

    # Subtract 1 from the true class
    gradient[target] -= 1

    return gradient.tolist()