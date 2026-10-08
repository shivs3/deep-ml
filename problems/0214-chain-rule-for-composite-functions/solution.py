import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.

    Args:
        functions: List of function names (applied right to left)
                   Available: 'square', 'sin', 'exp', 'log'

        x: Point at which to evaluate derivative

    Returns:
        Derivative value at x

    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """

    # Current value
    value = x

    # Start with derivative of the innermost function
    derivative = 1.0

    # Functions are applied from right to left
    for func in reversed(functions):

        if func == "square":
            # f(x) = x²
            derivative *= 2 * value
            value = value ** 2

        elif func == "sin":
            # f(x) = sin(x)
            derivative *= np.cos(value)
            value = np.sin(value)

        elif func == "exp":
            # f(x) = e^x
            derivative *= np.exp(value)
            value = np.exp(value)

        elif func == "log":
            # f(x) = ln(x)
            derivative *= 1 / value
            value = np.log(value)

        else:
            raise ValueError(f"Unknown function: {func}")

    return derivative