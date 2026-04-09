import numpy as np
from model import LinearRegression
from utils import normalize

# Multi-feature dataset
X = np.array([
    [1000, 3, 10],
    [1500, 4, 5],
    [800,  2, 20],
    [1200, 3, 7],
    [1800, 5, 2]
])
X = normalize(X)

y = np.array([
    [200],
    [300],
    [150],
    [220],
    [350]
])

model = LinearRegression(lr=0.0000001, epochs=1000)
model.fit(X, y)

predictions = model.predict(X)

print("Predictions:\n", predictions)
print("Weights:\n", model.W)
print("Bias:\n", model.b)