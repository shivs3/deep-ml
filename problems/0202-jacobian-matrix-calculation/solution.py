import numpy as np

def jacobian_matrix(
    f,
    x: list[float],
    h: float = 1e-5
) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.

    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences

    Returns:
        Jacobian matrix as list of lists
    """

    x = np.array(x, dtype=float)

    # Evaluate function once to determine number of outputs
    f_x = np.array(f(x.tolist()), dtype=float)

    n_outputs = len(f_x)
    n_inputs = len(x)

    # Create Jacobian matrix
    J = np.zeros((n_outputs, n_inputs))

    # Calculate each partial derivative
    for j in range(n_inputs):

        # Create x + h and x - h
        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[j] += h
        x_backward[j] -= h

        # Evaluate function
        f_forward = np.array(f(x_forward.tolist()), dtype=float)
        f_backward = np.array(f(x_backward.tolist()), dtype=float)

        # Central difference
        J[:, j] = (
            f_forward - f_backward
        ) / (2 * h)

    return J.tolist()