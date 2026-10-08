import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.

    Args:
        x: Input vector of real numbers

    Returns:
        Jacobian matrix J where
        J[i][j] = d(softmax_i) / d(x_j)
    """

    x = np.array(x, dtype=float)

    # Softmax
    exp_x = np.exp(x - np.max(x))
    q = exp_x / np.sum(exp_x)

    # Jacobian: diag(q) - q q^T
    J = np.diag(q) - np.outer(q, q)

    return J.tolist()

