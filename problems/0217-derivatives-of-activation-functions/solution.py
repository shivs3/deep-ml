import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	z = 1/(1 + np.exp(-x))
	sigmoid_der = z * (1 - z)
	tanh_val = np.tanh(x)
	tanh_der = 1 - tanh_val**2
	relu_der = 1 if x>0 else 0
	res = {
		'sigmoid': sigmoid_der,
		'tanh' : tanh_der,
		'relu': relu_der
	}
	return res