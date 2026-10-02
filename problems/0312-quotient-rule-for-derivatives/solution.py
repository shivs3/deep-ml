import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    if x == 0:
        return -1
    n1, n2 = len(g_coeffs) - 1 , len(h_coeffs) - 1

    d1, d2 = 0, 0
    g, h = 0, 0
    for i in g_coeffs:
        g += i * (x ** n1)
        d1 += i * n1 * (x ** (n1 - 1))
        n1 -= 1
    for i in h_coeffs:
        h += i * (x ** n2)
        d2 += i * n2 * (x ** (n2 - 1))
        n2 -= 1 

    derivative = (h * d1 - g * d2)/(h**2)
    return derivative