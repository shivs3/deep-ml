import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:

    n1 = len(g_coeffs) - 1
    n2 = len(h_coeffs) - 1

    g = 0
    h = 0
    d1 = 0
    d2 = 0

    for i in g_coeffs:
        g += i * x ** n1

        if n1 > 0:
            d1 += i * n1 * x ** (n1 - 1)

        n1 -= 1

    for i in h_coeffs:
        h += i * x ** n2

        if n2 > 0:
            d2 += i * n2 * x ** (n2 - 1)

        n2 -= 1

    # Denominator cannot be zero
    if h == 0:
        return -1

    return float((h * d1 - g * d2) / h ** 2)