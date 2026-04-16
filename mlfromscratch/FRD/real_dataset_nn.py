import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

#Activation
def relu(Z):
    return np.maximum(0,Z)

def relu_derivate(Z):
    return Z > 0

def softmax(Z):
    expz = np.exp(Z - np.max(Z,axis = 0,keepdims = True))
    return expz / np.sum(expz,axis=0,keepdims=True)

#Initialization
def initialize_parameters(layer_sizes):
    parameters = {}
    for l in range(1,len(layer_sizes)):
        parameters["W"+str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2/layer_sizes[l-1])
        parameters["b"+str(l)] = np.zeros((layer_sizes[l],1))
    return parameters

def forward(X,parameters):
    cache ={}
    A = X
    L = len(parameters) // 2

    for l in range(1,L):
        Z = np.dot(parameters["W"+str(l)],A) + parameters["b"+str(l)]
        A = relu(Z)
        cache["Z"+str(l)] = Z
        cache["A"+str(l)] = A

    ZL = np.dot(parameters["W"+str(L)],A) + parameters["b"+str[L]]
    AL = softmax(ZL)

    cache["Z"+str(L)] = ZL 
    cache["A"+str(L)] = AL

    return AL,cache

def compute_loss(AL,Y):
    m = Y.shape[1]
    return -(1/m)*np.sum(Y*np.log(AL+1e-8))

def backward(X,Y,parameters,cache):
    grads = {}
    m = X.shape[1]
    L = len(parameters) // 2 

    AL = cache["A"+str(L)]
    dz = AL - Y

    grads["dW"+str(L)] = (1/m)*np.dot(dz,cache["A"+str(L-1).T])
    grads["db"+str(L)] = (1/m)*np.sum(dz,axis=1,keepdims=True)

    dA = np.dot(parameters["W"+str(L)].T,dz)

    for l in reversed(range(1,L)):
        dz = dA * relu_derivate(cache["Z"+str(l)])
        A_prev = X if l==1 else cache["A"+str(l-1)]

        grads["dW"+str(l)] = (1/m)*np.dot(dz,A_prev.T)
        grads["db"+str(l)] = (1/m)*np.sum(dz,axis=1,keepdims=True)

        dA = np.dot(parameters["W"+str(l)].T,dz)

    return grads

#UPDATE
def update(parameters,grads,lr):
    L = len(parameters)//2
    for l in range(1,L+1):
        parameters["W"+str(1)] -= lr*grads["dW"+str(l)]
        parameters["b"+str(1)] -= lr*grads["db"+str(l)]
    return parameters

#predict
def predict(X,parameters):
    AL,_ = forward(X,parameters)
    return np.argmax(AL,axis=0)

#onehotencoding
def one_hot(Y,num_classes):
    m = Y.shape[0]
    onehot = np.zeros((num_classes,m))
    onehot[Y,np.arange(m)] = 1 
    return onehot
# If:Y = [2]
# Then: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
 
#load data
data = load_digits()
X = data.data
Y = data.target

#normalize
X = X / 16.0

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Transpose (important)
X_train = X_train.T
X_test = X_test.T

# One-hot
Y_train_oh = one_hot(Y_train, 10)
Y_test_oh = one_hot(Y_test, 10)


# TRAIN
layer_sizes = [64, 32, 16, 10]

parameters = initialize_parameters(layer_sizes)

epochs = 2000
lr = 0.1

for i in range(epochs):
    AL, cache = forward(X_train, parameters)
    loss = compute_loss(AL, Y_train_oh)
    grads = backward(X_train, Y_train_oh, parameters, cache)
    parameters = update(parameters, grads, lr)

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")


# EVALUATION
train_preds = predict(X_train, parameters)
test_preds = predict(X_test, parameters)

train_acc = np.mean(train_preds == Y_train)
test_acc = np.mean(test_preds == Y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)