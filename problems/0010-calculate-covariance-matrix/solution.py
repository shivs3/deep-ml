import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	vectors = np.array(vectors, dtype = float)
	mean = np.mean(vectors, axis=1, keepdims = True)
	X_centered = vectors - mean

	covariance = (X_centered @ X_centered.T)/(vectors.shape[1] - 1)
	return covariance.tolist()