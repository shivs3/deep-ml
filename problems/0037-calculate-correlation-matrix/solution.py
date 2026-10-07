import numpy as np


def calculate_correlation_matrix(X, Y = None):
    """
    Calculate the correlation matrix between X and Y.

    If Y is None, calculates the correlation matrix of X
    with itself.
    """

    X = np.asarray(X, dtype=float)

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)

    # Center the variables
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    # Covariance numerator
    covariance = X_centered.T @ Y_centered

    # Number of observations
    n = X.shape[0]

    # Sample covariance
    covariance = covariance / (n - 1)

    # Standard deviations
    std_X = np.std(X, axis=0, ddof=1)
    std_Y = np.std(Y, axis=0, ddof=1)

    # Correlation
    correlation = covariance / np.outer(std_X, std_Y)

    return correlation