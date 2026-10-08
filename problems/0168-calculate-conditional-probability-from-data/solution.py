def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.

    Args:
        data: List of (X, Y) tuples
        x: Value of X to condition on
        y: Value of Y to check

    Returns:
        float: Conditional probability, rounded to 4 decimal places
    """

    # Count how many times X = x
    x_count = 0

    # Count how many times X = x AND Y = y
    xy_count = 0

    for X, Y in data:
        if X == x:
            x_count += 1

            if Y == y:
                xy_count += 1

    # Avoid division by zero
    if x_count == 0:
        return 0.0

    probability = xy_count / x_count

    return round(probability, 4)