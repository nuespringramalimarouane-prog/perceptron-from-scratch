import numpy as np

# perceptron as linear classifier
class Perceptron:
    def __init__(self, epochs: int = 1000) -> None:

        self.epochs = epochs

    def fit(self, X: np.ndarray, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0

        learning_rate = 0.1
        y_signed = np.where(y == 0, -1, 1)

        for _ in range(self.epochs):
            for i in range(n_samples):
                X_i = X[i]
                y_i = y_signed[i]
                score = self.W @ X_i + self.b
                if y_i * score <= 0:

                    self.W +=  learning_rate * y_i * X_i
                    self.b += learning_rate * y_i

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat = np.dot(X, self.W) + self.b
        return np.where(y_hat > 0, 1, 0)
