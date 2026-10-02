import numpy as np
def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	# Your code here
	arr = np.array([i for i in range(1, n+1)])
	mean = np.mean(arr)
	var = np.var(arr)
	return (mean, var)