def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    count = {}
    for i in samples:
        count[i] = count.get(i, 0) + 1
    res = []
    for values, counts in count.items():
        res.append((values, counts/len(samples)))
    return res