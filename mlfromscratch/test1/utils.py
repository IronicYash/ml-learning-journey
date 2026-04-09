import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)