import numpy as np
from neural_network import Neural_Network

class ChatNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = 3  # Input, Hidden, Output

        # Xavier Initialization
        self.w = {
            0: np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size),
            1: np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        }
        self.b = {
            1: np.zeros((1, hidden_size)),
            2: np.zeros((1, output_size))
        }

        self.z = {}
        self.a = {}
        self.dz = {}
        self.dw = {}
        self.db = {}

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # Assuming x is already sigmoid(x)

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_loss_derivative(self, y_true, y_pred):
        return y_pred - y_true

    def forward(self, input):
        self.z[1] = np.dot(input, self.w[0]) + self.b[1]
        self.a[1] = self.tanh(self.z[1])  # FIX: Use tanh in hidden layer
        self.z[2] = np.dot(self.a[1], self.w[1]) + self.b[2]
        self.a[2] = self.sigmoid(self.z[2])  # Keep sigmoid for output layer
        return self.a[2]

    def back_prop(self, y_true, input, alpha=0.5):  # Increased LR
        batch_size = y_true.shape[0]

        # Compute loss derivative
        self.dz[2] = self.mse_loss_derivative(y_true, self.a[2]) * self.sigmoid_derivative(self.a[2])
        self.db[2] = np.sum(self.dz[2], axis=0) / batch_size
        self.dw[1] = np.dot(self.a[1].T, self.dz[2]) / batch_size

        # Hidden layer error
        self.dz[1] = np.dot(self.dz[2], self.w[1].T) * self.tanh_derivative(self.a[1])  # FIX: Use tanh derivative
        self.db[1] = np.sum(self.dz[1], axis=0) / batch_size
        self.dw[0] = np.dot(input.T, self.dz[1]) / batch_size

        # Update weights and biases
        for i in range(2):
            self.w[i] -= alpha * self.dw[i]
        for i in range(1, 3):
            self.b[i] -= alpha * self.db[i]

    def train(self, X, y, epochs=5000):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.mse_loss(y, output)
            self.back_prop(y, X)

# Sample dataset (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])  # Expected outputs

# Initialize and train the network
nn = ChatNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
mnn = Neural_Network((2, 3, 1), hidden_activation="tanh", output_activation="sigmoid", loss="mean_squared_error")

def train_mnn():
    for i in range(5000):
        output = mnn.forward(X)
        mnn.back_prop(y, X, 0.5)
        if (i+1) % 500 == 0:
            print(f"Epoch {i+1}, Loss: {mnn.loss_func(output, y):.6f}")
    return

# train
nn.train(X, y, epochs=5000)
train_mnn()

# Final Predictions
print("Chat's Final Predictions:")
print(nn.forward(X))
print(48*'-')
print("My Final Predictions:")
print(mnn.forward(X))