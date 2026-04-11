import numpy as np
    
# =========================
# ACTIVATION FUNCTIONS
# =========================

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def tanh(Z):
    return np.tanh(Z)

def tanh_derivative(Z):
    return 1 - np.tanh(Z)**2


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)


# =========================
# INITIALIZATION
# =========================

def initialize_parameters(layer_sizes):
    parameters = {}
    L = len(layer_sizes)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2 / layer_sizes[l-1])
        parameters["b" + str(l)] = np.zeros((layer_sizes[l], 1))

    return parameters

def initialize_adam(parameters):
    v = {}
    s = {}
    L = len(parameters) // 2

    for l in range(1, L+1):
        v["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
        v["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

        s["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
        s["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

    return v, s

def update_parameters_adam(parameters, grads, v, s, t,
                           learning_rate=0.01,
                           beta1=0.9, beta2=0.999, epsilon=1e-8):

    L = len(parameters) // 2

    for l in range(1, L+1):

        # Momentum
        v["dW"+str(l)] = beta1 * v["dW"+str(l)] + (1-beta1) * grads["dW"+str(l)]
        v["db"+str(l)] = beta1 * v["db"+str(l)] + (1-beta1) * grads["db"+str(l)]

        # RMSProp
        s["dW"+str(l)] = beta2 * s["dW"+str(l)] + (1-beta2) * (grads["dW"+str(l)]**2)
        s["db"+str(l)] = beta2 * s["db"+str(l)] + (1-beta2) * (grads["db"+str(l)]**2)

        # Bias correction
        v_corrected_dW = v["dW"+str(l)] / (1 - beta1**t)
        v_corrected_db = v["db"+str(l)] / (1 - beta1**t)

        s_corrected_dW = s["dW"+str(l)] / (1 - beta2**t)
        s_corrected_db = s["db"+str(l)] / (1 - beta2**t)

        # Update
        parameters["W"+str(l)] -= learning_rate * (
            v_corrected_dW / (np.sqrt(s_corrected_dW) + epsilon)
        )

        parameters["b"+str(l)] -= learning_rate * (
            v_corrected_db / (np.sqrt(s_corrected_db) + epsilon)
        )

    return parameters, v, s

# =========================
# FORWARD PROPAGATION
# =========================

def forward_propagation(X, parameters, hidden_activation="tanh"):
    cache = {}
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]

        Z = np.dot(W, A) + b

        if hidden_activation == "relu":
            A = relu(Z)
        elif hidden_activation == "tanh":
            A = tanh(Z)

        cache["Z"+str(l)] = Z
        cache["A"+str(l)] = A

    # Output layer (Sigmoid for binary classification)
    WL = parameters["W"+str(L)]
    bL = parameters["b"+str(L)]

    ZL = np.dot(WL, A) + bL
    AL = tanh(ZL)

    cache["Z"+str(L)] = ZL
    cache["A"+str(L)] = AL

    return AL, cache


# =========================
# LOSS FUNCTION (Binary Cross-Entropy)
# =========================

# def compute_loss(AL, Y): #for sigmoid 
#     m = Y.shape[1]
#     loss = -(1/m) * np.sum(Y*np.log(AL + 1e-8) + (1-Y)*np.log(1-AL + 1e-8))
#     return loss

def compute_loss(AL, Y):
    m = Y.shape[1]
    return (1/m) * np.sum((AL - Y) ** 2)

# =========================
# BACKPROPAGATION
# =========================

def backward_propagation(X, Y, parameters, cache, hidden_activation="relu"):
    grads = {}
    m = X.shape[1]
    L = len(parameters) // 2

    # Output layer
    AL = cache["A"+str(L)]
    dZL = (AL - Y) * (1 - AL**2)

    grads["dW"+str(L)] = (1/m) * np.dot(dZL, cache["A"+str(L-1)].T)
    grads["db"+str(L)] = (1/m) * np.sum(dZL, axis=1, keepdims=True)

    dA_prev = np.dot(parameters["W"+str(L)].T, dZL)

    # Hidden layers
    for l in reversed(range(1, L)):
        Z = cache["Z"+str(l)]

        if hidden_activation == "relu":
            dZ = dA_prev * relu_derivative(Z)
        elif hidden_activation == "tanh":
            dZ = dA_prev * tanh_derivative(Z)

        A_prev = X if l == 1 else cache["A"+str(l-1)]

        grads["dW"+str(l)] = (1/m) * np.dot(dZ, A_prev.T)
        grads["db"+str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(parameters["W"+str(l)].T, dZ)

    return grads


# =========================
# UPDATE PARAMETERS
# =========================

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L+1):
        parameters["W"+str(l)] -= learning_rate * grads["dW"+str(l)]
        parameters["b"+str(l)] -= learning_rate * grads["db"+str(l)]

    return parameters


# =========================
# TRAINING LOOP
# =========================

def train_adam(X, Y, layer_sizes, learning_rate=0.01, epochs=2000):

    parameters = initialize_parameters(layer_sizes)
    v, s = initialize_adam(parameters)

    t = 0  # timestep

    for i in range(epochs):
        t += 1

        AL, cache = forward_propagation(X, parameters, hidden_activation="tanh")
        loss = compute_loss(AL, Y)
        grads = backward_propagation(X, Y, parameters, cache, hidden_activation="tanh")

        parameters, v, s = update_parameters_adam(
            parameters, grads, v, s, t, learning_rate
        )

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return parameters

def train(X, Y, layer_sizes, learning_rate=0.1, epochs=1000, hidden_activation="relu"):
    parameters = initialize_parameters(layer_sizes)

    for i in range(epochs):
        AL, cache = forward_propagation(X, parameters, hidden_activation)
        loss = compute_loss(AL, Y)
        grads = backward_propagation(X, Y, parameters, cache, hidden_activation)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return parameters


# =========================
# PREDICTION FUNCTION
# =========================

def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    print("Raw outputs:", AL)
    return np.where(AL > 0, 1, -1)

# =========================
# TEST DATA (XOR PROBLEM)
# =========================

X = np.array([[0,0,1,1],
              [0,1,0,1]])

Y = np.array([[-1, 1, 1, -1]])

# =========================
# TRAIN MODEL
# =========================

layer_sizes = [2, 8, 8, 1]

#parameters = train(X, Y, layer_sizes, learning_rate=0.1, epochs=2000)
parameters = train_adam(X, Y, layer_sizes, learning_rate=0.01, epochs=3000)
# =========================
# TEST PREDICTION
# =========================

preds = predict(X, parameters)
print("Predictions:", preds)
print("Actual:", Y)