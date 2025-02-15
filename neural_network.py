import numpy as np

class Neural_Network:

    def __init__(self, structure, hidden_activation="relu", output_activation="relu", loss="mean_squarred_error"):
        self.structure = structure
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.layers = len(structure)
        self.build()
        self.loss = "mse"
        pass

    def build(self):
        self.w = [self.gen_weight(self.structure[i-1], self.structure[i]) for i in range(1, self.layers, 1)]
        self.z = [None for i in range(self.layers+1)]
        self.a = [None for i in range(self.layers+1)]
        self.dw = [None for i in range(self.layers)]
        self.dz = [None for i in range(self.layers+1)]
        self.hidden_func = self.hidden_dfunc = self.output_func = self.output_dfunc = None
        self.setActivation(self.hidden_func, self.hidden_dfunc, self.hidden_activation)
        self.setActivation(self.output_func, self.output_dfunc, self.output_activation)
        self.loss_func = self.loss_dfunc = None
        return
    
    def setLoss(self, loss_func, loss_dfunc, loss):
        if(loss == "mean_squared_error"):
            loss_func = self.mse_loss
            loss_dfunc = self.mse_derivative
        elif(loss == "categorical_crossentropy"):
            loss_func = self.categorical_crossentropy_loss
            loss_dfunc = self.categorical_crossentropy_derivative
        else:
            print("Invalid Loss Function.")
            exit(0)
        return
    
    def setActivation(self, act_func, act_dfunc, activation):
        if(activation == "relu"):
            act_func = self.relu
            act_dfunc = self.relu_derivative
        elif(activation == "sigmoid"):
            act_func = self.sigmoid
            act_dfunc = self.sigmoid_derivative
        elif(activation == "softmax"):
            act_func = self.softmax
            act_dfunc = self.softmax_derivative
        else:
            print("Invalid Activation Function.")
            exit(0)
        return
    
    def getWeight(self):
        return self.w
    
    def copy_network(self, w):
        self.w = w.copy()
        return

    def update_network(self, weight, tau=0.001):
        for i in range(len(weight)): self.w[i] = tau*weight[i]+(1-tau)*self.w[i]
        return
    
    def gen_weight(self, l1, l2):
        return np.random.randn(l1, l2)*np.sqrt(1/l1)
    
    def relu(self, x):
        return x*(x>0).astype(float)

    def relu_derivative(self, x):
        return np.where(x>0, 1.0, 0.0)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig*(1-sig)
    
    def softmax(self, x):
        return
    
    def softmax_derivative(self, x):
        return
    
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred-y_true)**2)
    
    def mse_derivative(self, y_pred, y_true):
        return 2*(y_pred-y_true)/(y_true.size)
    
    def categorical_crossentropy_loss(self, y_pred, y_true):

        return
    
    def categorical_crossentropy_derivative(self, y_pred, y_true):

        return
    
    def value_clipping(self, x, clip_value=1e5):
        return np.clip(x, -clip_value, clip_value)
    
    def back_prop(self, sample, input, batch_size, alpha=0.001):
        self.dz[self.layers-1] = self.mse_derivative(self.a[self.layers-1], sample)*self.deriv_act_func(self.z[self.layers-1])
        print(self.dz[self.layers-1].shape)
        for i in range(self.layers-2, 0, -1): self.dz[i] = np.dot(self.dz[i+1], self.w[i].T)*self.deriv_act_func(self.z[i])
        self.dw[0] = np.dot(input.T, self.dz[1])/batch_size
        for i in range(1, self.layers-1): self.dw[i] = np.dot(self.a[i].T, self.dz[i+1])/batch_size
        for i in range(0, self.layers-1): self.w[i] -= alpha*self.dw[i]
        return
    
    def forward(self, input):
        self.z[1] = np.dot(input, self.w[0])
        self.a[1] = self.act_func(self.z[1])
        for i in range(2, self.layers, 1):
            self.z[i] = np.dot(self.a[i-1], self.w[i-1])
            self.a[i] = self.act_func(self.z[i])
        return self.a[self.layers-1]