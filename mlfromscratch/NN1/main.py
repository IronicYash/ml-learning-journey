import numpy as np
from neural_network import NeuralNetwork

# Simple dataset (non-linear)
X = np.array([
    [0],
    [1],
    [2],
    [3]
])

y = np.array([
    [0],
    [1],
    [4],
    [9]
])  # y = x^2

nn = NeuralNetwork(input_size=1, hidden_size=4, output_size=1, lr=0.01)

nn.train(X, y, epochs=1000)

predictions = nn.forward(X)
print("Predictions:\n", predictions)