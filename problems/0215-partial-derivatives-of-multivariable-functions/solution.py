import numpy as np

def compute_partial_derivatives(
    func_name: str,
    point: tuple[float, ...]
) -> tuple[float, ...]:
    """
    Compute partial derivatives of multivariable functions.

    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²

        point: Point (x, y) or (x, y, z)

    Returns:
        Tuple of partial derivatives
        (∂f/∂x, ∂f/∂y, ...)
    """

    # Define the function based on its name
    if func_name == "poly2d":
        def func(p):
            x, y = p
            return x**2 * y + x * y**2

    elif func_name == "exp_sum":
        def func(p):
            x, y = p
            return np.exp(x + y)

    elif func_name == "product_sin":
        def func(p):
            x, y = p
            return x * np.sin(y)

    elif func_name == "poly3d":
        def func(p):
            x, y, z = p
            return x**2 * y + y * z**2

    elif func_name == "squared_error":
        def func(p):
            x, y = p
            return (x - y)**2

    else:
        raise ValueError(f"Unknown function: {func_name}")

    # Numerical differentiation
    h = 1e-5
    point = np.array(point, dtype=float)

    derivatives = []

    for i in range(len(point)):

        # Copy the point
        point_forward = point.copy()
        point_backward = point.copy()

        # Change only variable i
        point_forward[i] += h
        point_backward[i] -= h

        # Central difference formula
        derivative = (
            func(point_forward) - func(point_backward)
        ) / (2 * h)

        derivatives.append(derivative)

    return tuple(derivatives)