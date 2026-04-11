import numpy as np

# =========================
# ACTIVATION FUNCTIONS
# =========================

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


# =========================
# INITIALIZATION
# =========================

def initialize_parameters(layer_sizes):
    parameters = {}
    L = len(layer_sizes)

    for l in range(1, L):
        parameters["W" + str(l)] = (
            np.random.randn(layer_sizes[l], layer_sizes[l-1])
            * np.sqrt(2 / layer_sizes[l-1])
        )
        parameters["b" + str(l)] = np.zeros((layer_sizes[l], 1))

    return parameters


# =========================
# FORWARD PROPAGATION
# =========================

def forward_propagation(X, parameters):
    cache = {}
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        Z = np.dot(parameters["W"+str(l)], A) + parameters["b"+str(l)]
        A = relu(Z)

        cache["Z"+str(l)] = Z
        cache["A"+str(l)] = A

    # Output layer (Softmax)
    ZL = np.dot(parameters["W"+str(L)], A) + parameters["b"+str(L)]
    AL = softmax(ZL)

    cache["Z"+str(L)] = ZL
    cache["A"+str(L)] = AL

    return AL, cache


# =========================
# LOSS FUNCTION
# =========================

def compute_loss(AL, Y):
    m = Y.shape[1]
    return -(1/m) * np.sum(Y * np.log(AL + 1e-8))


# =========================
# BACKPROPAGATION
# =========================

def backward_propagation(X, Y, parameters, cache):
    grads = {}
    m = X.shape[1]
    L = len(parameters) // 2

    # Output layer
    AL = cache["A"+str(L)]
    dZL = AL - Y

    grads["dW"+str(L)] = (1/m) * np.dot(dZL, cache["A"+str(L-1)].T)
    grads["db"+str(L)] = (1/m) * np.sum(dZL, axis=1, keepdims=True)

    dA_prev = np.dot(parameters["W"+str(L)].T, dZL)

    # Hidden layers
    for l in reversed(range(1, L)):
        Z = cache["Z"+str(l)]
        dZ = dA_prev * relu_derivative(Z)

        A_prev = X if l == 1 else cache["A"+str(l-1)]

        grads["dW"+str(l)] = (1/m) * np.dot(dZ, A_prev.T)
        grads["db"+str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(parameters["W"+str(l)].T, dZ)

    return grads


# =========================
# UPDATE PARAMETERS
# =========================

def update_parameters(parameters, grads, lr):
    L = len(parameters) // 2

    for l in range(1, L+1):
        parameters["W"+str(l)] -= lr * grads["dW"+str(l)]
        parameters["b"+str(l)] -= lr * grads["db"+str(l)]

    return parameters


# =========================
# TRAINING LOOP
# =========================

def train(X, Y, layer_sizes, epochs=2000, lr=0.1):
    parameters = initialize_parameters(layer_sizes)

    for i in range(epochs):
        AL, cache = forward_propagation(X, parameters)
        loss = compute_loss(AL, Y)
        grads = backward_propagation(X, Y, parameters, cache)
        parameters = update_parameters(parameters, grads, lr)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return parameters


# =========================
# PREDICTION
# =========================

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    return np.argmax(AL, axis=0)


# =========================
# DATA (3-CLASS EXAMPLE)
# =========================

X = np.array([[1,0,0,1],
              [0,1,0,1]])

# One-hot labels (3 classes)
Y = np.array([[1,0,0,1],
              [0,1,0,0],
              [0,0,1,0]])


# =========================
# TRAIN MODEL
# =========================

layer_sizes = [2, 6, 6, 3]

parameters = train(X, Y, layer_sizes, epochs=2000, lr=0.1)


# =========================
# TEST
# =========================

preds = predict(X, parameters)
print("Predictions:", preds)