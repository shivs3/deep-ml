import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """

    x = np.asarray(x, dtype=float)
    analytical_grad = np.asarray(analytical_grad, dtype=float)

    numerical_grad = np.zeros_like(x)

    # Calculate numerical gradient for each element
    for i in range(x.size):

        # Create copies so the original x is not modified
        x_plus = x.copy()
        x_minus = x.copy()

        # Perturb only the ith element
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        # Centered finite difference
        numerical_grad[i] = (
            f(x_plus) - f(x_minus)
        ) / (2 * epsilon)

    # Calculate relative error
    numerator = np.linalg.norm(
        numerical_grad - analytical_grad
    )

    denominator = (
        np.linalg.norm(numerical_grad)
        + np.linalg.norm(analytical_grad)
        + 1e-12
    )

    relative_error = numerator / denominator

    return numerical_grad, relative_error