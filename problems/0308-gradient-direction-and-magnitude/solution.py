import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	mag = np.sqrt(np.sum(np.array(gradient)**2))
	direc = np.array(gradient)/mag if mag != 0 else np.array([0, 0])
	des = -1 * direc
	return {'magnitude': mag, 'direction': direc.tolist(), 'descent_direction': des.tolist()}