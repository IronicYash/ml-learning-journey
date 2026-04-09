import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.losses = []
        self.epochs = epochs
        self.W = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape

        # Initialize weights
        self.W = np.zeros((n, 1))
        self.b = 0

        for _ in range(self.epochs):
            # Forward pass
            y_pred = np.dot(X, self.W) + self.b

            # Compute error
            error = y_pred - y

            # Loss (Mean Squared Error)
            loss = np.mean(error ** 2)
            self.losses.append(loss)

            if _ % 100 == 0:
                print(f"Epoch {_}, Loss: {loss}")

            # Gradients
            dW = (2/m) * np.dot(X.T, error)
            db = (2/m) * np.sum(error)

            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.W) + self.b