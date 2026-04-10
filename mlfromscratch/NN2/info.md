#DEEP NEURAL NETWORK

# Right now your model is:
# Input → Hidden (ReLU) → Output
# We will upgrade it to:
# Input → Hidden1 → Hidden2 → … → Output

< Visual  representation > 
Layer 1: A1 = ReLU(W1X + b1)
Layer 2: A2 = ReLU(W2A1 + b2)
Layer 3: A3 = ReLU(W3A2 + b3)
...
Output: ŷ

