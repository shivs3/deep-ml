import numpy as np

def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> (np.ndarray, float, list[float]):

    # Make copies so the original weights are not modified
    weights = np.array(initial_weights, dtype=float).copy()
    bias = float(initial_bias)

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)

    n = X.shape[0]

    mse_values = []

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for _ in range(epochs):

        # -------------------------
        # 1. Forward pass
        # -------------------------
        z = X @ weights + bias
        predictions = sigmoid(z)

        # -------------------------
        # 2. Compute MSE
        # -------------------------
        errors = predictions - y
        mse = np.mean(errors ** 2)

        # Record MSE BEFORE the update
        mse_values.append(mse)

        # -------------------------
        # 3. Backpropagation
        # -------------------------
        sigmoid_derivative = predictions * (1 - predictions)

        # dL/dz
        dz = 2 * errors * sigmoid_derivative

        # Average gradients over entire batch
        weight_gradient = (X.T @ dz) / n
        bias_gradient = np.mean(dz)

        # -------------------------
        # 4. Update weights/bias
        # -------------------------
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    # Round final results
    updated_weights = np.round(weights, 4)
    updated_bias = round(bias, 4)
    mse_values = [round(mse, 4) for mse in mse_values]

    return updated_weights, updated_bias, mse_values