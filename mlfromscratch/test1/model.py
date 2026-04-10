import numpy as np
class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        m, n = X.shape

        # Initialize weights
        self.W = np.zeros((n, 1))
        self.b = 0

        for epoch in range(self.epochs):

            for i in range(0, m, self.batch_size):

                # Mini-batch selection
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                # Forward pass
                y_pred = np.dot(X_batch, self.W) + self.b

                # Error
                error = y_pred - y_batch

                # Loss (MSE)
                loss = np.mean(error ** 2)

                # Gradients
                dW = (2 / len(X_batch)) * np.dot(X_batch.T, error)
                db = (2 / len(X_batch)) * np.sum(error)

                # Update weights
                self.W -= self.lr * dW
                self.b -= self.lr * db

            # Store loss after each epoch
            self.losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return np.dot(X, self.W) + self.b