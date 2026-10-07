import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination
    with partial pivoting.

    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(A)

    # Forward elimination
    for i in range(n):

        # Find row with largest pivot
        max_row = i + np.argmax(np.abs(A[i:, i]))

        # Swap rows
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Eliminate values below pivot
        for j in range(i + 1, n):

            factor = A[j, i] / A[i, i]

            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):

        x[i] = (
            b[i] - np.dot(A[i, i + 1:], x[i + 1:])
        ) / A[i, i]

    return x